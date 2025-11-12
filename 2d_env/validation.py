import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env import FixedWingUAVEnv
import time
import glob

# === VISUALIZATION HELPERS ===
def _draw_start_finish(ax, env, title_suffix=""):
    """Draw start, goal, and checkpoint markers on the plot."""
    # Professional start marker
    ax.plot(env.x0, env.y0, marker='*', color='#2ca02c', markersize=16, 
            label='Start', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)

    # Professional goal center marker
    ax.plot(env.x_end, env.y_end, marker='X', color='#d62728', markersize=12, 
            label='Goal Center', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
    
def draw_2d_trajectory_with_checkpoints(ax, trajectory_x, trajectory_y, env):
    """Helper function to draw 2D trajectory with checkpoints."""
    # Draw checkpoints for waypoint config (circles only, no numbering or checkmarks)
    if (hasattr(env, 'checkpoint_positions') and len(env.checkpoint_positions) > 0 and
        hasattr(env, 'config') and env.config == 'waypoint'):
        for i, (cp_x, cp_y) in enumerate(env.checkpoint_positions):
            # Draw checkpoint circle only (no status indicators)
            cp_circle = Circle((cp_x, cp_y), env.checkpoint_radius, fill=False, 
                             linestyle=':', linewidth=1, alpha=0.4, color='orange')
            ax.add_patch(cp_circle)
            ax.plot(cp_x, cp_y, 'o', color='orange', markersize=8, alpha=0.8)

    # No title for cleaner appearance
    ax.legend(loc='best')

# === OBSERVATION PROCESSING ===
def _normalize_obs_if_needed(obs, vn):
    """Normalize observation using VecNormalize stats."""
    if vn is None or not hasattr(vn, "obs_rms"):
        return obs
    
    mean = np.array(vn.obs_rms.mean)
    var = np.array(vn.obs_rms.var)
    eps = 1e-8
    clip = getattr(vn, "clip_obs", 10.0)
    
    normalized = (obs - mean) / np.sqrt(var + eps)
    return np.clip(normalized, -clip, clip).astype(np.float32)

# === VISUALIZATION FUNCTIONS ===
def plot_path_comparison(env, model, vecnorm_stats=None):
    """Plot agent path vs desired path."""
    obs, _ = env.reset()
    obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
    done = False
    xs = [env.x]
    ys = [env.y]
    
    while not done:
        action, _ = model.predict(obs_for_policy, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
        done = terminated or truncated
        xs.append(env.x)
        ys.append(env.y)

    plt.figure(figsize=(15, 8))
    # Professional styling for desired path
    plt.plot(env.path_x_array, env.path_y_array, color='#2c2c2c', linestyle='--', 
             linewidth=3, label='Desired Path', alpha=0.8, solid_capstyle='round')
    # Professional styling for agent path
    plt.plot(xs, ys, color='#d62728', linestyle='-', linewidth=2.5, 
             label='Agent Path', alpha=0.9, solid_capstyle='round')
    _draw_start_finish(plt.gca(), env, title_suffix="(Sine Wave Path)")
    
    # Professional axis styling
    plt.xlabel('X (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Y (m)', fontsize=12, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.axis('equal')
    plt.show()

def _run_episode_simulation(env, model, vecnorm_stats, obs_adapter, episode_seed):
    """Run episode simulation and collect data."""
    # Initialize episode
    if episode_seed is not None:
        obs, _ = env.reset(seed=episode_seed)
    else:
        obs, _ = env.reset()
    
    if obs_adapter is not None:
        obs = obs_adapter(obs)
    obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
    
    # Data collection arrays
    xs, ys = [env.x], [env.y]
    headings = [env.heading_deg]
    heading_errors = []
    roll_cmds_deg = []
    roll_rates_deg_per_step = []
    control_actions = []
    step_rewards = []
    step_dist_to_path = []
    step_dist_to_goal = []
    path_segments = []
    
    total_reward = 0.0
    dpath_samples = []
    
    info = {
        "termination_reason": "running",
        "heading_diff_deg": 0.0,
        "dist_to_path_abs": 0.0,
        "reached_cp_count": 0,
        "total_cps": 0
    }

    done = False
    while not done:
        action, _ = model.predict(obs_for_policy, deterministic=True)
        control_actions.append(float(action[0]))
        
        # Handle different configs for roll command visualization
        if env.config == "waypoint":
            roll_cmds_deg.append(float(env._prev_action) * env.max_roll_deg)
        else:
            roll_cmds_deg.append(float(action[0]) * env.max_roll_deg)

        obs, reward, terminated, truncated, info = env.step(action)
        roll_rates_deg_per_step.append(float(info.get("roll_rate_deg_per_step", 0.0)))
        
        if obs_adapter is not None:
            obs = obs_adapter(obs)
        obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
        done = terminated or truncated

        # Accumulate data
        total_reward += float(reward)
        dpath_samples.append(float(info["dist_to_path_abs"]))
        step_rewards.append(float(reward))
        step_dist_to_path.append(float(info["dist_to_path_abs"]))
        
        dist_to_goal = np.hypot(env.x - env.x_end, env.y - env.y_end)
        step_dist_to_goal.append(float(dist_to_goal))

        xs.append(env.x)
        ys.append(env.y)
        headings.append(env.heading_deg)
        heading_errors.append(info["heading_diff_deg"])
        
        # Track path segments
        if len(xs) > 1:
            segment_length = np.hypot(xs[-1] - xs[-2], ys[-1] - ys[-2])
            path_segments.append(segment_length)
    
    return (xs, ys, headings, heading_errors, roll_cmds_deg, roll_rates_deg_per_step,
            control_actions, step_rewards, step_dist_to_path, step_dist_to_goal,
            path_segments, total_reward, dpath_samples, info)

def _calculate_episode_metrics(env, xs, ys, control_actions, step_rewards, step_dist_to_path, 
                            step_dist_to_goal, path_segments, total_reward, dpath_samples):
    """Calculate episode performance metrics."""
    # Convert to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Basic metrics
    avg_dist_to_path = float(np.mean(dpath_samples)) if len(dpath_samples) else float("nan")
    episode_length = len(xs) - 1
    
    # Path efficiency calculation
    if hasattr(env, 'path_x_array') and hasattr(env, 'path_y_array') and env.path_x_array is not None:
        ideal_path_length = 0.0
        for i in range(1, len(env.path_x_array)):
            segment_length = np.hypot(env.path_x_array[i] - env.path_x_array[i-1], 
                                    env.path_y_array[i] - env.path_y_array[i-1])
            ideal_path_length += segment_length
    else:
        ideal_path_length = np.hypot(env.x_end - env.x0, env.y_end - env.y0)
    
    actual_path_length = float(np.sum(path_segments)) if len(path_segments) > 0 else 0.0
    path_efficiency = ideal_path_length / actual_path_length if actual_path_length > 0 else 0.0
    
    # Oscillation index calculation
    if len(control_actions) > 1:
        actions_array = np.array(control_actions)
        action_variance = float(np.var(actions_array))
        action_changes = np.diff(actions_array)
        sign_changes = np.sum(np.diff(np.sign(action_changes)) != 0)
        max_possible_changes = len(action_changes) - 1
        change_frequency = (sign_changes / max_possible_changes) if max_possible_changes > 0 else 0.0
        
        normalized_variance = min(1.0, action_variance / 0.25)
        oscillation_index = 0.7 * normalized_variance + 0.3 * change_frequency
    else:
        oscillation_index = 0.0
    
    # Deviation-reward correlation
    if len(step_rewards) > 1 and len(step_dist_to_path) > 1:
        step_rewards_array = np.array(step_rewards)
        step_dist_to_path_array = np.array(step_dist_to_path)
        step_dist_to_goal_array = np.array(step_dist_to_goal)
        
        path_corr = np.corrcoef(step_rewards_array, step_dist_to_path_array)[0, 1]
        goal_corr = np.corrcoef(step_rewards_array, step_dist_to_goal_array)[0, 1]
        
        if np.isnan(path_corr):
            path_corr = 0.0
        if np.isnan(goal_corr):
            goal_corr = 0.0
            
        deviation_reward_correlation = max(path_corr, goal_corr)
    else:
        deviation_reward_correlation = 0.0
    
    return {
        "total_reward": total_reward,
        "avg_dist_to_path": avg_dist_to_path,
        "episode_length": episode_length,
        "path_efficiency": path_efficiency,
        "ideal_path_length": ideal_path_length,
        "actual_path_length": actual_path_length,
        "oscillation_index": oscillation_index,
        "deviation_reward_correlation": deviation_reward_correlation
    }

def visualize_episode(env, model, episode=0, save_path=None, vecnorm_stats=None, obs_adapter=None, episode_seed=None):
    """Visualize a single episode with comprehensive analysis."""
    # Run episode simulation
    (xs, ys, headings, heading_errors, roll_cmds_deg, roll_rates_deg_per_step,
     control_actions, step_rewards, step_dist_to_path, step_dist_to_goal,
     path_segments, total_reward, dpath_samples, info) = _run_episode_simulation(
         env, model, vecnorm_stats, obs_adapter, episode_seed)
    
    # Calculate metrics
    metrics = _calculate_episode_metrics(env, xs, ys, control_actions, step_rewards, 
                                       step_dist_to_path, step_dist_to_goal, path_segments,
                                       total_reward, dpath_samples)

    # Create visualization
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 20))

    # Plot 1: XY path with professional styling
    ax1.plot(env.path_x_array, env.path_y_array, color='#2c2c2c', linestyle='--', 
             linewidth=3, label='Desired Path', alpha=0.8, solid_capstyle='round')
    ax1.plot(xs, ys, color='#d62728', linestyle='-', linewidth=2.5, 
             label='Actual Path', alpha=0.9, solid_capstyle='round')
    _draw_start_finish(ax1, env, title_suffix="")
    
    # Professional axis styling
    ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.axis('equal')

    # Add termination condition in top left
    term_reason = info.get("termination_reason", "unknown")
    term_color = '#2ca02c' if term_reason == "success" else '#d62728'
    ax1.text(0.02, 0.98, f'Status: {term_reason}', 
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', color=term_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=term_color))

    # Plot 2: Heading with professional styling
    time_points = np.arange(len(headings)) * env.dt
    ax2.plot(time_points, headings, color='#2ca02c', linestyle='-', linewidth=2.5, 
             label='Heading (deg)', alpha=0.9, solid_capstyle='round')
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Heading (deg)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=11)

    # Plot 3: Roll command with professional styling
    if len(roll_cmds_deg) > 0:
        ax3.plot(time_points[:len(roll_cmds_deg)], roll_cmds_deg, color='#d62728', 
                linestyle='-', linewidth=2.5, label='Roll cmd (deg)', alpha=0.9, solid_capstyle='round')
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Roll cmd (deg)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.legend(fontsize=11)

    # Plot 4: Heading error with professional styling
    time_points_error = np.arange(len(heading_errors)) * env.dt
    ax4.plot(time_points_error, heading_errors, color='#1f77b4', linestyle='-', 
             linewidth=2.5, label='Heading Error (deg)', alpha=0.9, solid_capstyle='round')
    ax4.axhline(y=0, color='#2c2c2c', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Heading Error (deg)', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.legend(fontsize=11)

    # Plot 5: Roll rate change with professional styling
    if len(roll_rates_deg_per_step) > 0:
        time_points_roll_rate = np.arange(len(roll_rates_deg_per_step)) * env.dt
        ax5.plot(time_points_roll_rate, roll_rates_deg_per_step, color='#ff7f0e', 
                linestyle='-', linewidth=2.5, label='Roll Rate Change (deg/step)', 
                alpha=0.9, solid_capstyle='round')
        ax5.axhline(y=env.max_roll_rate_deg_per_step, color='#d62728', linestyle='--', 
                   alpha=0.7, linewidth=2, label=f'Hard Cap ({env.max_roll_rate_deg_per_step}°/step)')
        ax5.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Roll Rate Change (deg/step)', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='both', which='major', labelsize=10)
        ax5.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax5.legend(fontsize=11)
        ax5.set_ylim(0, max(env.max_roll_rate_deg_per_step * 1.5, max(roll_rates_deg_per_step) * 1.1))

    plt.tight_layout()
    if save_path:
        # Save with high quality for journal publication
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', format='png', transparent=False)
        plt.close()
    else:
        plt.show()

    # Return combined info and metrics
    out = dict(info)
    out.update(metrics)
    return out

# === MODEL MANAGEMENT ===
def find_all_models():
    """Find all trained models organized by algorithm and config."""
    model_patterns = [
        "trained_models/ppo_fixedwing_*.zip",
        "trained_models/sac_fixedwing_*.zip", 
        "trained_models/td3_fixedwing_*.zip"
    ]
    
    models = {}
    for pattern in model_patterns:
        files = glob.glob(pattern)
        for file in files:
            basename = os.path.basename(file)
            parts = basename.replace('.zip', '').split('_')
            if len(parts) >= 3:
                algorithm = parts[0].upper()
                # Handle goal_based config (has underscore in name)
                if len(parts) >= 4 and parts[2] == 'goal' and parts[3] == 'based':
                    config = 'goal_based'
                else:
                    config = parts[2]
                
                if algorithm not in models:
                    models[algorithm] = {}
                if config not in models[algorithm]:
                    models[algorithm][config] = []
                models[algorithm][config].append(file)
    
    # Get latest model for each combination
    latest_models = {}
    for alg in models:
        latest_models[alg] = {}
        for cfg in models[alg]:
            latest_models[alg][cfg] = max(models[alg][cfg], key=os.path.getmtime)
    
    return latest_models

def load_model(model_path):
    """Load model based on algorithm type."""
    basename = os.path.basename(model_path)
    algorithm = basename.split('_')[0].lower()
    
    try:
        if algorithm == 'ppo':
            return PPO.load(model_path)
        elif algorithm == 'sac':
            return SAC.load(model_path)
        elif algorithm == 'td3':
            return TD3.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    except Exception as e:
        print(f"   Error loading model: {e}")
        raise

def create_observation_adapter(model_obs_space, env_obs_space, config=None):
    """Create observation adapter for different observation spaces."""
    model_shape = model_obs_space.shape[0] if hasattr(model_obs_space, 'shape') else None
    env_shape = env_obs_space.shape[0] if hasattr(env_obs_space, 'shape') else None
    
    if model_shape == env_shape:
        return None  # No adapter needed
    elif model_shape == 3 and env_shape == 4:
        return lambda obs: obs[:3]  # Take first 3 elements
    elif model_shape == 4 and env_shape == 3:
        return lambda obs: np.concatenate([obs, [0.0]])  # Pad with zeros
    else:
        return None

# === VALIDATION FUNCTIONS ===
def run_validation(model_path, config, num_episodes=5, fixed_seeds=None, wind_enabled=True, 
                   wind_gust_magnitude=2.0, wind_gust_duration=10.0, wind_transition_time=1.0):
    """Run validation for a specific model.
    
    Args:
        wind_enabled: Enable wind gusts
        wind_gust_magnitude: Peak wind speed in m/s (default: 2.0)
        wind_gust_duration: How long gusts last in seconds (default: 10.0)
        wind_transition_time: Ramp up/down time in seconds (default: 1.0)
    """
    env = FixedWingUAVEnv(config=config, wind_enabled=wind_enabled, 
                          wind_speed_mean=wind_gust_magnitude,  # Used as gust magnitude
                          wind_speed_std=0.5,  # Unused in gust model
                          wind_heading_std=15.0)  # Unused in gust model
    model = load_model(model_path)
    obs_adapter = create_observation_adapter(model.observation_space, env.observation_space, config)
    
    # Load VecNormalize stats if available
    vecnorm_stats = None
    basename = os.path.basename(model_path)
    algorithm = basename.split('_')[0].lower()
    if algorithm in ['ppo', 'sac', 'td3']:
        stats_path = model_path.replace('.zip', '') + "_vn.pkl"
        if os.path.exists(stats_path):
            tmp_venv = DummyVecEnv([lambda: FixedWingUAVEnv(config=config)])
            try:
                vecnorm_stats = VecNormalize.load(stats_path, tmp_venv)
                vecnorm_stats.training = False
                vecnorm_stats.norm_reward = False
                print(f"   ✅ Loaded VecNormalize stats for {algorithm.upper()}")
            except Exception as e:
                print(f"   Warning: failed to load VecNormalize stats: {e}")
        else:
            print(f"   ⚠️  No VecNormalize stats found for {algorithm.upper()}")
    
    # Setup results directory
    algorithm = basename.split('_')[0].upper()
    results_dir = f"validation_results/{algorithm.lower()}_{config}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run validation episodes
    success_count = 0
    path_errors = []
    heading_errors = []
    completion_times = []
    total_rewards = []
    avg_dpaths = []
    oscillation_indices = []
    deviation_correlations = []
    episode_lengths = []
    path_efficiencies = []
    
    for episode in range(num_episodes):
        start_time = time.time()
        episode_seed = fixed_seeds[episode] if fixed_seeds and episode < len(fixed_seeds) else None
        
        info = visualize_episode(
            env, model, episode=episode,
            save_path=f"{results_dir}/episode_{episode+1}.png",
            vecnorm_stats=vecnorm_stats, obs_adapter=obs_adapter,
            episode_seed=episode_seed
        )
        
        episode_time = time.time() - start_time
        if info["termination_reason"] == "success":
            success_count += 1
        
        # Collect metrics
        path_errors.append(info["dist_to_path_abs"])
        heading_errors.append(info["heading_diff_deg"])
        total_rewards.append(info.get("total_reward", np.nan))
        avg_dpaths.append(info.get("avg_dist_to_path", np.nan))
        oscillation_indices.append(info.get("oscillation_index", np.nan))
        deviation_correlations.append(info.get("deviation_reward_correlation", np.nan))
        episode_lengths.append(info.get("episode_length", np.nan))
        path_efficiencies.append(info.get("path_efficiency", np.nan))
        completion_times.append(episode_time)
    
    # Calculate summary statistics
    summary = {
        'algorithm': algorithm,
        'config': config,
        'success_rate': success_count/num_episodes*100,
        'avg_path_error': np.mean(path_errors),
        'avg_heading_error': np.mean(heading_errors),
        'avg_dist_to_path': np.mean(avg_dpaths),
        'std_dist_to_path': np.std(avg_dpaths),
        'avg_total_reward': np.mean(total_rewards),
        'std_total_reward': np.std(total_rewards),
        'avg_completion_time': np.mean(completion_times),
        'avg_oscillation_index': np.mean(oscillation_indices),
        'std_oscillation_index': np.std(oscillation_indices),
        'avg_deviation_correlation': np.mean(deviation_correlations),
        'std_deviation_correlation': np.std(deviation_correlations),
        'avg_episode_length': np.mean(episode_lengths),
        'episode_length_std': np.std(episode_lengths),
        'avg_path_efficiency': np.mean(path_efficiencies),
        'std_path_efficiency': np.std(path_efficiencies)
    }
    
    # Save summary
    with open(f"{results_dir}/summary.txt", "w") as f:
        f.write(f"{algorithm} {config.upper()} Validation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Success rate: {summary['success_rate']:.1f}%\n")
        f.write(f"Average FINAL heading error: {summary['avg_heading_error']:.2f} deg\n")
        f.write(f"Average EPISODE avg dist-to-path: {summary['avg_dist_to_path']:.2f} m\n")
        f.write(f"Average total reward: {summary['avg_total_reward']:.2f}\n")
        f.write(f"Average completion time: {summary['avg_completion_time']:.2f} s\n")
        f.write(f"Oscillation Index (OI): {summary['avg_oscillation_index']:.3f}\n")
        f.write(f"Deviation-Reward Correlation (DRC): {summary['avg_deviation_correlation']:.3f}\n")
        f.write(f"Path Efficiency (PE): {summary['avg_path_efficiency']:.3f}\n")
        f.write(f"Average episode length: {summary['avg_episode_length']:.1f} ± {summary['episode_length_std']:.1f} steps\n")
    
    return summary

def validate_all_models(num_episodes=5, wind_enabled=False, wind_gust_magnitude=2.0, 
                       wind_gust_duration=10.0, wind_transition_time=1.0):
    """Validate all trained models and show comparison plots.
    
    Args:
        wind_enabled: Enable smooth wind gusts
        wind_gust_magnitude: Peak wind speed in m/s (default: 2.0)
        wind_gust_duration: How long gusts last in seconds (default: 10.0)
        wind_transition_time: Ramp up/down time in seconds (default: 1.0)
    """
    models = find_all_models()
    
    if not models:
        print("No trained models found! Please run training first.")
        return
    
    os.makedirs("validation_results", exist_ok=True)
    
    # Generate fixed seeds for fair comparison across algorithms
    # Each algorithm will get the same 20 paths in the same order
    np.random.seed(42)  # Fixed seed for reproducible episode seeds
    fixed_seeds = [np.random.randint(0, 1000000) for _ in range(num_episodes)]
    print(f"🔧 Using fixed seeds for fair comparison: {fixed_seeds[:5]}... (showing first 5)")
    
    if wind_enabled:
        print(f"🌬️  Wind gusts enabled: magnitude={wind_gust_magnitude} m/s, duration={wind_gust_duration}s, transition={wind_transition_time}s")
    
    all_results = []
    
    # Validate each model
    for algorithm in models:
        for config in models[algorithm]:
            # Skip unsupported configs
            if config not in ['waypoint', 'goal_based', 'heuristic']:
                print(f"\n⚠️  Skipping {algorithm} {config}: unsupported config (only 'waypoint', 'goal_based', and 'heuristic' supported)")
                continue
                
            model_path = models[algorithm][config]
            print(f"\n🔍 Found {algorithm} {config}: {model_path}")
            
            try:
                summary = run_validation(model_path, config, num_episodes, fixed_seeds,
                                        wind_enabled=wind_enabled, wind_gust_magnitude=wind_gust_magnitude,
                                        wind_gust_duration=wind_gust_duration, wind_transition_time=wind_transition_time)
                all_results.append(summary)
            except Exception as e:
                print(f"❌ Failed to validate {algorithm} {config}: {e}")
    
    # Create comparison summary
    print(f"\n{'='*90}")
    print("VALIDATION COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"{'Algorithm':<8} {'Config':<8} {'Success%':<10} {'Reward':<18} {'PD':<18} {'OI':<14}")
    print(f"{'-'*90}")
    
    for result in all_results:
        # Display config name with space conservation
        if result['config'] == "goal_based":
            config_display = "goal_based"
        elif result['config'] == "heuristic":
            config_display = "heuristic"
        else:
            config_display = result['config']
        
        reward_str = f"{result['avg_total_reward']:.1f}±{result['std_total_reward']:.1f}"
        dist_str = f"{result['avg_dist_to_path']:.2f}±{result['std_dist_to_path']:.2f}"
        oi_str = f"{result['avg_oscillation_index']:.3f}±{result['std_oscillation_index']:.3f}"
        
        print(f"{result['algorithm']:<8} {config_display:<8} "
              f"{result['success_rate']:<10.1f} {reward_str:<18} "
              f"{dist_str:<18} {oi_str:<14}")
    
    # Save overall comparison
    with open("validation_results/comparison_summary.txt", "w") as f:
        f.write("Model Comparison Summary\n")
        f.write("="*90 + "\n")
        f.write("Performance Metrics:\n")
        f.write("- Success%: Percentage of episodes reaching goal successfully\n")
        f.write("- Reward: Average total reward per episode (± std)\n")
        f.write("- PD (Path Deviation): Average distance from desired path in meters (± std)\n")
        f.write("- OI (Oscillation Index): Control action variance + change frequency (± std)\n")
        f.write("Interpretation:\n")
        f.write("- Lower PD indicates better path following\n")
        f.write("- Higher OI suggests zig-zag or bang-bang control policies\n")
        f.write("="*90 + "\n")
        f.write(f"{'Algorithm':<8} {'Config':<8} {'Success%':<10} {'Reward':<18} {'PD':<18} {'OI':<14}\n")
        f.write("-"*90 + "\n")
        for result in all_results:
            # Display config name with space conservation
            if result['config'] == "goal_based":
                config_display = "goal_based"
            elif result['config'] == "heuristic":
                config_display = "heuristic"
            else:
                config_display = result['config']
            
            reward_str = f"{result['avg_total_reward']:.1f}±{result['std_total_reward']:.1f}"
            dist_str = f"{result['avg_dist_to_path']:.2f}±{result['std_dist_to_path']:.2f}"
            oi_str = f"{result['avg_oscillation_index']:.3f}±{result['std_oscillation_index']:.3f}"
            
            f.write(f"{result['algorithm']:<8} {config_display:<8} "
                   f"{result['success_rate']:<10.1f} {reward_str:<18} "
                   f"{dist_str:<18} {oi_str:<14}\n")
    
    return all_results

def plot_all_agents_comparison(config="waypoint", save_path=None, episode_seed=42,
                               wind_enabled=True, wind_gust_magnitude=5.0,
                               wind_gust_duration=10.0, wind_transition_time=1.0):
    """Plot all three agents (PPO, SAC, TD3) on the same path for direct comparison."""
    models = find_all_models()
    
    if not models:
        print("❌ No trained models found!")
        return
    
    # Create environment
    env = FixedWingUAVEnv(config=config, wind_enabled=wind_enabled,
                          wind_speed_mean=wind_gust_magnitude,
                          wind_speed_std=0.5,
                          wind_heading_std=15.0)
    
    # Color scheme for algorithms
    colors = {
        'PPO': '#d62728',   # Red
        'SAC': '#2ca02c',   # Green
        'TD3': '#1f77b4'    # Blue
    }
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot desired path first
    obs, _ = env.reset(seed=episode_seed)
    ax.plot(env.path_x_array, env.path_y_array, color='#2c2c2c', linestyle='--', 
            linewidth=3, label='Desired Path', alpha=0.8, solid_capstyle='round', zorder=1)
    
    # Plot start and goal markers
    ax.plot(env.x0, env.y0, marker='*', color='#2ca02c', markersize=18, 
            label='Start', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9, zorder=5)
    ax.plot(env.x_end, env.y_end, marker='X', color='#d62728', markersize=14, 
            label='Goal', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9, zorder=5)
    
    # Plot each algorithm's trajectory
    for algorithm in ['PPO', 'SAC', 'TD3']:
        if algorithm not in models or config not in models[algorithm]:
            print(f"⚠️  {algorithm} {config} model not found, skipping...")
            continue
            
        model_path = models[algorithm][config]
        print(f"🔍 Loading {algorithm} {config}: {model_path}")
        
        try:
            # Load model
            model = load_model(model_path)
            
            # Load VecNormalize stats
            vecnorm_stats = None
            stats_path = model_path.replace('.zip', '') + "_vn.pkl"
            if os.path.exists(stats_path):
                tmp_venv = DummyVecEnv([lambda: FixedWingUAVEnv(config=config)])
                try:
                    vecnorm_stats = VecNormalize.load(stats_path, tmp_venv)
                    vecnorm_stats.training = False
                    vecnorm_stats.norm_reward = False
                except Exception as e:
                    print(f"   Warning: failed to load VecNormalize for {algorithm}: {e}")
            
            # Create observation adapter
            obs_adapter = create_observation_adapter(model.observation_space, env.observation_space, config)
            
            # Run episode with same seed for fair comparison
            obs, _ = env.reset(seed=episode_seed)
            if obs_adapter is not None:
                obs = obs_adapter(obs)
            obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
            
            xs, ys = [env.x], [env.y]
            done = False
            
            while not done:
                action, _ = model.predict(obs_for_policy, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if obs_adapter is not None:
                    obs = obs_adapter(obs)
                obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
                done = terminated or truncated
                xs.append(env.x)
                ys.append(env.y)
            
            # Plot this algorithm's path
            ax.plot(xs, ys, color=colors[algorithm], linestyle='-', linewidth=2.5,
                   label=f'{algorithm} Path', alpha=0.85, solid_capstyle='round', zorder=3)
            
            # Add termination marker
            term_reason = info.get("termination_reason", "unknown")
            marker = 'o' if term_reason == "success" else 'x'
            ax.plot(xs[-1], ys[-1], marker=marker, color=colors[algorithm], 
                   markersize=12, markeredgecolor='black', markeredgewidth=1, zorder=4)
            
            print(f"   ✅ {algorithm}: {term_reason}, {len(xs)} steps")
            
        except Exception as e:
            print(f"   ❌ Failed to run {algorithm}: {e}")
    
    # Professional styling
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.axis('equal')
    
    # Add title with wind info
    if wind_enabled:
        title = f'Algorithm Comparison ({config.upper()}) - Wind: {wind_gust_magnitude} m/s gusts'
    else:
        title = f'Algorithm Comparison ({config.upper()}) - No Wind'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                   edgecolor='none', format='png', transparent=False)
        print(f"💾 Saved comparison plot to: {save_path}")
        plt.close()
    else:
        plt.show()

def main():
    """Main validation function - validates all trained models"""
    print("🚀 Starting comprehensive validation of all trained models...")
    print("This will validate all algorithm x configuration combinations.")
    
    try:
        # First, create comparison plot for all agents
        print("\n📊 Creating algorithm comparison plot...")
        os.makedirs("validation_results", exist_ok=True)
        plot_all_agents_comparison(config="waypoint", 
                                   save_path="validation_results/algorithm_comparison_waypoint.png",
                                   episode_seed=42,
                                   wind_enabled=True,
                                   wind_gust_magnitude=5.0,
                                   wind_gust_duration=10.0,
                                   wind_transition_time=1.0)
        
        # Validate all models and show comparison with wind gusts
        results = validate_all_models(num_episodes=20, wind_enabled=True,
                                      wind_gust_magnitude=5.0,
                                      wind_gust_duration=10.0,
                                      wind_transition_time=1.0)
        
        if results:
            print(f"\n✅ Validation complete! Results saved in validation_results/")
            print(f"📊 {len(results)} model combinations validated")
            
            # Show plots for each model (you can comment this out if too many)
            print(f"\n📈 Individual episode plots saved in validation_results/")
            
        else:
            print("❌ No models found to validate. Please run training first.")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    main()