import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env import FixedWingUAVEnv

# ============================================================
# === UNSEEN PATH ENVIRONMENT ================================
# ============================================================

class UnseenPathEnv(FixedWingUAVEnv):
    """
    Extension of FixedWingUAVEnv that replaces the default training path
    with unseen 3D geometries for generalization testing.
    """

    def __init__(self, config="waypoint", path_type="helix", **kwargs):
        super().__init__(config=config, **kwargs)
        self.path_type = path_type
        self.max_steps = 2000  # allow longer episodes for complex paths

    def reset(self, seed=None, options=None):
        """Reset environment with a new unseen path geometry."""
        # Don't call super().reset() as it overwrites our custom path
        # Instead, do our own initialization
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # === SELECT UNSEEN PATH TYPE ===
        if self.path_type == "helix":
            self._generate_helix_path()
        elif self.path_type == "descending_s":
            self._generate_descending_s_path()
        else:
            raise ValueError(f"Unknown path type: {self.path_type}")

        # === CREATE CHECKPOINTS ===
        self.checkpoint_positions = []
        self.checkpoints_reached = []
        checkpoint_indices = np.linspace(0, len(self.path_x_array) - 1, self.num_checkpoints + 1, dtype=int)
        for i in range(self.num_checkpoints):
            idx = checkpoint_indices[i + 1]
            cp_x = self.path_x_array[idx]
            cp_y = self.path_y_array[idx]
            cp_z = self.path_z_array[idx]
            self.checkpoint_positions.append((cp_x, cp_y, cp_z))
            self.checkpoints_reached.append(False)

        # Add goal
        self.checkpoint_positions.append((self.x_end, self.y_end, self.z_end))
        self.checkpoints_reached.append(False)
        self.next_checkpoint_idx = 0
        
        # Checkpoints created

        # === RESET AIRCRAFT STATE ===
        self.x, self.y, self.z = self.x0, self.y0, self.z0
        if len(self.path_x_array) > 1:
            dx = self.path_x_array[1] - self.path_x_array[0]
            dy = self.path_y_array[1] - self.path_y_array[0]
            self.heading_deg = np.degrees(np.arctan2(dy, dx))
        else:
            self.heading_deg = 0.0

        self.pitch_deg = 0.0
        self.current_speed = self.speed
        self.current_step = 0
        self._d_prev = float(np.hypot(self.x - self.x_end, self.y - self.y_end))
        _, self._along_track_prev, _ = self._cross_and_along_track()
        self._prev_cmd[:] = 0.0
        self._prev_prev_action[:] = 0.0
        
        # Clear any cached path segment lengths
        if hasattr(self, '_path_segment_lengths'):
            delattr(self, '_path_segment_lengths')
        if hasattr(self, '_cumulative_path_length'):
            delattr(self, '_cumulative_path_length')
            
        return self._obs(), {}

    # === Path Generators ===

    def _generate_helix_path(self):
        num_points = 1001
        turns = 3.0
        theta = np.linspace(0, turns * 2 * np.pi, num_points)
        radius = 150.0
        climb_height = 80.0
        self.path_x_array = 500.0 + radius * np.cos(theta)
        self.path_y_array = radius * np.sin(theta)
        self.path_z_array = 100.0 + (climb_height / (turns * 2 * np.pi)) * theta
        self.x0, self.y0, self.z0 = self.path_x_array[0], self.path_y_array[0], self.path_z_array[0]
        self.x_end, self.y_end, self.z_end = self.path_x_array[-1], self.path_y_array[-1], self.path_z_array[-1]
        
        # Increase number of checkpoints for helix path
        self.num_checkpoints = 16  # Increased from default 16

    def _generate_descending_s_path(self):
        """Generate descending S curve: straight descent + S-shaped lateral maneuver."""
        num_points = 1001
        
        # Path parameters
        straight_length = 200.0   # Length of initial straight segment
        descent_length = 400.0    # Length of the descent portion
        s_length = 1000.0         # Length of the S curve portion (increased for more peaks)
        altitude_drop = 50.0       # Total altitude lost during descent (150m to 100m)
        s_altitude = 150.0 - altitude_drop  # Constant altitude for S curve (100m)
        s_amplitude = 120.0       # Amplitude of S curve (lateral deviation)
        s_frequency = 0.1        # Frequency of S curve (cycles per 100m) - much lower for gentler curves
        
        # Increase checkpoint radius for this path to make it easier
        self.checkpoint_radius = 20.0  # Increased from default 10.0
        
        # Total path length
        total_path_length = straight_length + descent_length + s_length
        
        # Distribute points proportionally
        straight_points = int(num_points * straight_length / total_path_length)
        descent_points = int(num_points * descent_length / total_path_length)
        s_points = num_points - straight_points - descent_points
        
        # Create the path in segments
        self.path_x_array = np.zeros(num_points)
        self.path_y_array = np.zeros(num_points)
        self.path_z_array = np.zeros(num_points)
        
        # Segment 1: Initial straight segment (Y=0, constant Z at 150m)
        self.path_x_array[:straight_points] = np.linspace(0, straight_length, straight_points)
        self.path_y_array[:straight_points] = 0.0
        self.path_z_array[:straight_points] = 150.0
        
        # Segment 2: Smooth descent (Y=0, decreasing Z with smooth curve)
        descent_start_idx = straight_points
        descent_end_idx = straight_points + descent_points
        self.path_x_array[descent_start_idx:descent_end_idx] = np.linspace(straight_length, straight_length + descent_length, descent_points)
        self.path_y_array[descent_start_idx:descent_end_idx] = 0.0
        
        # Smooth descent using a sigmoid-like curve for gradual transition
        descent_x_local = self.path_x_array[descent_start_idx:descent_end_idx] - straight_length
        descent_ratio = descent_x_local / descent_length
        
        # Use smooth curve: starts slow, accelerates in middle, slows at end
        smooth_curve = 3 * descent_ratio**2 - 2 * descent_ratio**3  # Smooth S-curve
        self.path_z_array[descent_start_idx:descent_end_idx] = 150.0 - altitude_drop * smooth_curve
        
        # Segment 3: S curve at constant altitude with smooth transition
        s_start_idx = straight_points + descent_points
        s_x = np.linspace(straight_length + descent_length, straight_length + descent_length + s_length, s_points)
        
        # Create smooth transition from straight line to S curve
        transition_length = min(100.0, s_length * 0.2)  # 20% of S length or 100m, whichever is smaller
        transition_points = int(s_points * transition_length / s_length)
        
        # Calculate S curve (double sine for S shape)
        s_y_full = s_amplitude * np.sin(2 * np.pi * s_frequency * (s_x - straight_length - descent_length) / 100.0) * \
                   np.sin(np.pi * (s_x - straight_length - descent_length) / s_length)
        
        # Apply smooth transition using a sigmoid function
        transition_x = s_x[:transition_points] - straight_length - descent_length
        transition_weight = 1.0 / (1.0 + np.exp(-10.0 * (transition_x - transition_length/2) / transition_length))
        
        # Blend from straight line (y=0) to full S curve
        s_y = np.zeros_like(s_x)
        s_y[:transition_points] = s_y_full[:transition_points] * transition_weight
        s_y[transition_points:] = s_y_full[transition_points:]
        
        self.path_x_array[s_start_idx:] = s_x
        self.path_y_array[s_start_idx:] = s_y
        self.path_z_array[s_start_idx:] = s_altitude
        
        # Set start/end points
        self.x0, self.y0, self.z0 = self.path_x_array[0], self.path_y_array[0], self.path_z_array[0]
        self.x_end, self.y_end, self.z_end = self.path_x_array[-1], self.path_y_array[-1], self.path_z_array[-1]


# ============================================================
# === VALIDATION UTILITIES ===================================
# ============================================================

from validation import (
    visualize_episode, _normalize_obs_if_needed,
    create_observation_adapter, load_model, find_all_models
)

UNSEEN_PATHS = ["helix", "descending_s"]
NUM_EPISODES = 1


def plot_all_algorithms_3d_comparison(config, path_type, save_path=None, episode_seed=42,
                                     wind_enabled=True, wind_gust_magnitude=1.0,
                                     wind_gust_duration=5.0, wind_transition_time=1.0):
    """Plot all three algorithms (PPO, SAC, TD3) on the same 3D plot for comparison"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Color scheme for different algorithms
    colors = {
        'PPO': '#d62728',      # Red
        'SAC': '#2ca02c',      # Green  
        'TD3': '#1f77b4'       # Blue
    }
    
    algorithms = ['PPO', 'SAC', 'TD3']
    
    # Create figure with 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create environment to get the desired path
    env = UnseenPathEnv(config=config, path_type=path_type,
                       wind_enabled=wind_enabled,
                       wind_speed_mean=wind_gust_magnitude,
                       wind_speed_std=wind_gust_duration,
                       wind_heading_std=wind_transition_time)
    env.reset(seed=episode_seed)
    
    # Plot desired path first
    ax.plot(env.path_x_array, env.path_y_array, env.path_z_array, 
           color='#2c2c2c', linestyle='--', linewidth=3, 
           label='Desired Path', alpha=0.8)
    
    # Mark start and goal
    ax.scatter(env.x0, env.y0, env.z0, marker='*', color='#2ca02c', 
              s=300, label='Start', edgecolors='black', linewidths=0.5, alpha=0.9)
    ax.scatter(env.x_end, env.y_end, env.z_end, marker='X', color='#d62728', 
              s=200, label='Goal', edgecolors='black', linewidths=0.5, alpha=0.9)
    
    # Plot each algorithm's trajectory
    for algorithm in algorithms:
        try:
            # Find the model
            model_pattern = f"trained_models/{algorithm.lower()}_fixedwing_{config}_*.zip"
            model_files = glob.glob(model_pattern)
            if not model_files:
                print(f"   ⚠️  No model found for {algorithm} {config}")
                continue
            
            latest_model = max(model_files, key=os.path.getmtime)
            model = load_model(latest_model)
            
            # Load VecNormalize stats if available
            vecnorm_stats = None
            stats_path = latest_model.replace('.zip', '') + "_vn.pkl"
            if os.path.exists(stats_path):
                tmp_venv = DummyVecEnv([lambda: UnseenPathEnv(config=config, path_type=path_type,
                                                              wind_enabled=wind_enabled,
                                                              wind_speed_mean=wind_gust_magnitude,
                                                              wind_speed_std=wind_gust_duration,
                                                              wind_heading_std=wind_transition_time)])
                try:
                    vecnorm_stats = VecNormalize.load(stats_path, tmp_venv)
                    vecnorm_stats.training = False
                    vecnorm_stats.norm_reward = False
                except:
                    pass
            
            # Create observation adapter if needed
            test_env = UnseenPathEnv(config=config, path_type=path_type,
                                    wind_enabled=wind_enabled,
                                    wind_speed_mean=wind_gust_magnitude,
                                    wind_speed_std=wind_gust_duration,
                                    wind_heading_std=wind_transition_time)
            obs_adapter = create_observation_adapter(model.observation_space, test_env.observation_space, config)
            
            # Run episode to get trajectory
            obs, _ = test_env.reset(seed=episode_seed)
            if obs_adapter is not None:
                obs = obs_adapter(obs)
            obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
            
            xs = [test_env.x]
            ys = [test_env.y]
            zs = [test_env.z]
            done = False
            termination_reason = "unknown"
            
            while not done:
                action, _ = model.predict(obs_for_policy, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                if obs_adapter is not None:
                    obs = obs_adapter(obs)
                obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
                done = terminated or truncated
                xs.append(test_env.x)
                ys.append(test_env.y)
                zs.append(test_env.z)
                
                if done:
                    termination_reason = info.get("termination_reason", "unknown")
            
            # Plot this algorithm's trajectory
            color = colors.get(algorithm, '#666666')
            ax.plot(xs, ys, zs, color=color, linewidth=2.5, 
                   label=f'{algorithm} Path', alpha=0.9)
            
            # Add termination marker
            if termination_reason == "success":
                marker = 'o'  # Circle for success
                marker_label = f'{algorithm} (Success)'
            else:
                marker = 'x'  # X for failure
                marker_label = f'{algorithm} ({termination_reason})'
            
            ax.scatter(xs[-1], ys[-1], zs[-1], marker=marker, color=color, 
                      s=150, edgecolors='black', linewidths=1, alpha=0.9)
            
            print(f"   ✅ {algorithm}: {termination_reason} - {len(xs)} steps")
            
        except Exception as e:
            print(f"   ❌ Failed to load/run {algorithm}: {e}")
            import traceback
            traceback.print_exc()
    
    # Styling
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png', transparent=False)
        print(f"   💾 Saved comparison plot: {save_path}")
        plt.close()
    else:
        plt.show()



def run_unseen_validation(model_path, algorithm, config, path_type, vecnorm_stats=None, obs_adapter=None,
                          wind_enabled=False, wind_gust_magnitude=1.0, 
                          wind_gust_duration=5.0, wind_transition_time=1.0):
    """Validate model performance on unseen path type."""
    env = UnseenPathEnv(config=config, path_type=path_type,
                       wind_enabled=wind_enabled,
                       wind_speed_mean=wind_gust_magnitude,
                       wind_speed_std=wind_gust_duration,
                       wind_heading_std=wind_transition_time)
    model = load_model(model_path)

    print(f"\n🧭 Validating on unseen path: {path_type.upper()}")
    
    # Track episode details for U-turn debugging

    results_dir = f"validation_results/{algorithm.lower()}_{config}_unseen_{path_type}"
    os.makedirs(results_dir, exist_ok=True)

    success_count = 0
    total_rewards, avg_dpaths, path_efficiencies, completion_times = [], [], [], []
    oscillation_indices_roll, oscillation_indices_pitch = [], []
    deviation_correlations = []

    np.random.seed(42)
    seeds = [np.random.randint(0, 1000000) for _ in range(NUM_EPISODES)]

    for i in range(NUM_EPISODES):
        start_time = time.time()
        info = visualize_episode(
            env, model, episode=i,
            save_path=f"{results_dir}/episode_{i+1}.png",
            vecnorm_stats=vecnorm_stats,
            obs_adapter=obs_adapter,
            episode_seed=seeds[i]
        )
        duration = time.time() - start_time

        if info["termination_reason"] == "success":
            success_count += 1

        # Track termination details for descending_s
        if path_type == "descending_s":
            print(f"  Episode {i+1} termination: {info.get('termination_reason', 'unknown')}")
            print(f"  Checkpoints reached: {info.get('checkpoints_reached', 0)}/{info.get('total_checkpoints', 0)}")
            print(f"  Final position: ({env.x:.1f}, {env.y:.1f}, {env.z:.1f})")
            print(f"  Goal position: ({env.x_end:.1f}, {env.y_end:.1f}, {env.z_end:.1f})")
            print(f"  Distance to goal: {np.sqrt((env.x - env.x_end)**2 + (env.y - env.y_end)**2 + (env.z - env.z_end)**2):.1f}m")

        total_rewards.append(info.get("total_reward", np.nan))
        avg_dpaths.append(info.get("avg_dist_to_path", np.nan))
        path_efficiencies.append(info.get("path_efficiency", np.nan))
        oscillation_indices_roll.append(info.get("oscillation_index_roll", np.nan))
        oscillation_indices_pitch.append(info.get("oscillation_index_pitch", np.nan))
        deviation_correlations.append(info.get("deviation_reward_correlation", np.nan))
        completion_times.append(duration)

    summary = {
        "algorithm": algorithm,
        "config": config,
        "path_type": path_type,
        "success_rate": success_count / NUM_EPISODES * 100,
        "avg_reward": np.nanmean(total_rewards),
        "avg_dist_to_path": np.nanmean(avg_dpaths),
        "path_efficiency": np.nanmean(path_efficiencies),
        "oscillation_index_roll": np.nanmean(oscillation_indices_roll),
        "oscillation_index_pitch": np.nanmean(oscillation_indices_pitch),
        "deviation_reward_correlation": np.nanmean(deviation_correlations),
        "avg_time": np.nanmean(completion_times),
    }

    # Save summary
    with open(f"{results_dir}/summary.txt", "w") as f:
        f.write(f"UNSEEN PATH VALIDATION ({path_type.upper()})\n")
        f.write("=" * 60 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"   ✅ {path_type.upper():<15} | Success {summary['success_rate']:.1f}% | "
          f"Reward {summary['avg_reward']:.1f} | "
          f"OI-Roll {summary['oscillation_index_roll']:.3f} | OI-Pitch {summary['oscillation_index_pitch']:.3f}")

    return summary


def validate_unseen_paths_all_models(wind_enabled=True, wind_gust_magnitude=2.0,
                                    wind_gust_duration=5.0, wind_transition_time=1.0):
    """Run unseen-path validation across all trained models."""
    models = find_all_models()
    if not models:
        print("❌ No trained models found in trained_models/. Please train first.")
        return

    print(f"\n🚀 Starting unseen-path robustness validation on {len(models)} algorithms...")
    
    # Print wind configuration
    if wind_enabled:
        print(f"🌬️  Wind gusts enabled: magnitude={wind_gust_magnitude} m/s, duration={wind_gust_duration}s, transition={wind_transition_time}s")
    else:
        print("🌤️  Wind disabled for validation")
    
    # Create 3D comparison plots FIRST (before running full validation)
    print(f"\n📊 Creating 3D algorithm comparison plots...")
    os.makedirs("validation_results", exist_ok=True)
    
    for path_type in UNSEEN_PATHS:
        print(f"\n🎨 Generating comparison for {path_type.upper()}...")
        comparison_save_path = f"validation_results/algorithm_comparison_3d_{path_type}.png"
        try:
            plot_all_algorithms_3d_comparison(
                config="waypoint",
                path_type=path_type,
                save_path=comparison_save_path,
                episode_seed=42,
                wind_enabled=wind_enabled,
                wind_gust_magnitude=wind_gust_magnitude,
                wind_gust_duration=wind_gust_duration,
                wind_transition_time=wind_transition_time
            )
        except Exception as e:
            print(f"   ❌ Failed to create comparison plot for {path_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Starting detailed validation for each algorithm...")
    print(f"{'='*60}")
    
    np.random.seed(42)

    all_summaries = []

    for alg in models:
        for cfg in models[alg]:
            if cfg != "waypoint":
                continue
            model_path = models[alg][cfg]
            print(f"\n🔍 {alg.upper()} ({cfg}) → {model_path}")

            stats_path = model_path.replace(".zip", "_vn.pkl")
            vecnorm_stats = None
            if os.path.exists(stats_path):
                try:
                    tmp_env = DummyVecEnv([lambda: FixedWingUAVEnv(config=cfg,
                                                                   wind_enabled=wind_enabled,
                                                                   wind_speed_mean=wind_gust_magnitude,
                                                                   wind_speed_std=wind_gust_duration,
                                                                   wind_heading_std=wind_transition_time)])
                    vecnorm_stats = VecNormalize.load(stats_path, tmp_env)
                    vecnorm_stats.training = False
                    vecnorm_stats.norm_reward = False
                    print(f"   ✅ VecNormalize stats loaded")
                except Exception as e:
                    print(f"   ⚠️ Failed to load VecNormalize: {e}")

            tmp_env = FixedWingUAVEnv(config=cfg)
            model = load_model(model_path)
            obs_adapter = create_observation_adapter(model.observation_space, tmp_env.observation_space, cfg)

            for path_type in UNSEEN_PATHS:
                summary = run_unseen_validation(model_path, alg, cfg, path_type, vecnorm_stats, obs_adapter,
                                               wind_enabled=wind_enabled, wind_gust_magnitude=wind_gust_magnitude,
                                               wind_gust_duration=wind_gust_duration, wind_transition_time=wind_transition_time)
                all_summaries.append(summary)

    # Print summary table
    print("\n" + "=" * 100)
    print("UNSEEN PATH ROBUSTNESS COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Algorithm':<8} {'Config':<8} {'Path':<15} {'Succ%':<8} {'Reward':<10} {'DistPath':<10} "
          f"{'OI-Roll':<10} {'OI-Pitch':<10}")
    print("-" * 90)

    for s in all_summaries:
        print(f"{s['algorithm']:<8} {s['config']:<8} {s['path_type']:<15} "
              f"{s['success_rate']:<8.1f} {s['avg_reward']:<10.1f} {s['avg_dist_to_path']:<10.2f} "
              f"{s['oscillation_index_roll']:<10.3f} {s['oscillation_index_pitch']:<10.3f}")

    # Save comparison summary
    os.makedirs("validation_results", exist_ok=True)
    with open("validation_results/unseen_paths_summary.txt", "w") as f:
        f.write("UNSEEN PATH ROBUSTNESS SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Algorithm':<8} {'Config':<8} {'Path':<15} {'Succ%':<8} {'Reward':<10} {'DistPath':<10} "
                f"{'OI-Roll':<10} {'OI-Pitch':<10}\n")
        f.write("-" * 90 + "\n")
        for s in all_summaries:
            f.write(f"{s['algorithm']:<8} {s['config']:<8} {s['path_type']:<15} "
                    f"{s['success_rate']:<8.1f} {s['avg_reward']:<10.1f} {s['avg_dist_to_path']:<10.2f} "
                    f"{s['oscillation_index_roll']:<10.3f} {s['oscillation_index_pitch']:<10.3f}\n")

    print(f"\n📊 Results saved to validation_results/unseen_paths_summary.txt")
    print("📈 Individual episode plots are in each algorithm/config/path directory.")


if __name__ == "__main__":
    validate_unseen_paths_all_models(wind_enabled=True,
                                    wind_gust_magnitude=1.0,
                                    wind_gust_duration=10.0,
                                    wind_transition_time=1.0)