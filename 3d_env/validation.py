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
    # Start and goal markers
    ax.plot(env.x0, env.y0, marker='*', color='green', markersize=14, label='Start')
    ax.plot(env.x_end, env.y_end, 'rx', markersize=10, label='Goal Center')
    
    def draw_2d_trajectory_with_checkpoints(ax, trajectory_x, trajectory_y, env):
    """Helper function to draw 2D trajectory with checkpoints."""
    # Draw checkpoints for waypoint config
    if (hasattr(env, 'checkpoint_positions') and len(env.checkpoint_positions) > 0 and 
        hasattr(env, 'config') and env.config == 'waypoint'):
        next_cp_idx = getattr(env, 'next_checkpoint_idx', 0)
        for i, (cp_x, cp_y, cp_z) in enumerate(env.checkpoint_positions):
            # Determine checkpoint status
            if i < len(env.checkpoints_reached) and env.checkpoints_reached[i]:
                color = 'green'
                symbol = '✓'
            elif i == next_cp_idx:
                color = 'blue'
                symbol = '→'
            else:
                color = 'orange'
                symbol = str(i + 1)
            
            # Draw checkpoint circle and marker
            cp_circle = Circle((cp_x, cp_y), env.checkpoint_radius, fill=False, 
                             linestyle=':', linewidth=1, alpha=0.4, color=color)
            ax.add_patch(cp_circle)
            ax.plot(cp_x, cp_y, 'o', color=color, markersize=8, alpha=0.8)
            ax.text(cp_x, cp_y + 15, f'{symbol}{i+1}', ha='center', fontsize=8, 
                   color=color, fontweight='bold')

    if title_suffix:
        ax.set_title(f'Path Following {title_suffix}')
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
    plt.plot(env.path_x_array, env.path_y_array, 'b--', label='Desired Path', linewidth=2)
    plt.plot(xs, ys, 'r-', label='Agent Path', linewidth=2)
    _draw_start_finish(plt.gca(), env, title_suffix="(Sine Wave Path)")
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
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
    xs, ys, zs = [env.x], [env.y], [env.z]
    headings = [env.heading_deg]
    yaw_cmds_deg = []  # yaw command (heading control)
    roll_cmds_deg = []
    pitch_cmds_deg = []  # pitch command
    roll_rates_deg_per_step = []
    pitch_rates_deg_per_step = []
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
        
        # Handle different configs for roll, yaw, and pitch command visualization
        if env.config == "waypoint":
            roll_cmds_deg.append(float(env._prev_cmd[0]) * env.max_roll_deg)
            pitch_cmds_deg.append(float(env._prev_cmd[1]) * env.max_pitch_deg)
            # Yaw command is the roll command (affects heading)
            yaw_cmds_deg.append(float(env._prev_cmd[0]) * env.max_roll_deg)
        else:
            roll_cmds_deg.append(float(action[0]) * env.max_roll_deg)
            pitch_cmds_deg.append(float(action[1]) * env.max_pitch_deg)
            yaw_cmds_deg.append(float(action[0]) * env.max_roll_deg)

        obs, reward, terminated, truncated, info = env.step(action)
        roll_rates_deg_per_step.append(float(info.get("roll_rate_deg_per_step", 0.0)))
        pitch_rates_deg_per_step.append(float(info.get("pitch_rate_deg_per_step", 0.0)))
        
        if obs_adapter is not None:
            obs = obs_adapter(obs)
        obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
        done = terminated or truncated

        # Accumulate data
        reward_scalar = float(reward) if np.isscalar(reward) else float(reward[0])
        total_reward += reward_scalar
        dpath_samples.append(float(info["dist_to_path_abs"]))
        step_rewards.append(reward_scalar)
        step_dist_to_path.append(float(info["dist_to_path_abs"]))
        
        dist_to_goal = np.sqrt((env.x - env.x_end)**2 + (env.y - env.y_end)**2 + (env.z - env.z_end)**2)
        step_dist_to_goal.append(float(dist_to_goal))

        xs.append(env.x)
        ys.append(env.y)
        zs.append(env.z)
        headings.append(env.heading_deg)
        
        # Track path segments
        if len(xs) > 1:
            segment_length = np.sqrt((xs[-1] - xs[-2])**2 + (ys[-1] - ys[-2])**2 + (zs[-1] - zs[-2])**2)
            path_segments.append(segment_length)
    
    return (xs, ys, zs, headings, yaw_cmds_deg, roll_cmds_deg, pitch_cmds_deg, roll_rates_deg_per_step,
            pitch_rates_deg_per_step, control_actions, step_rewards, step_dist_to_path, step_dist_to_goal,
            path_segments, total_reward, dpath_samples, info)

def _calculate_episode_metrics(env, xs, ys, roll_cmds_deg, pitch_cmds_deg, step_rewards, step_dist_to_path, 
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
    
    # Roll Oscillation Index calculation
    if len(roll_cmds_deg) > 1:
        roll_array = np.array(roll_cmds_deg)
        roll_variance = float(np.var(roll_array))
        roll_changes = np.diff(roll_array)
        roll_sign_changes = np.sum(np.diff(np.sign(roll_changes)) != 0)
        max_possible_changes = len(roll_changes) - 1
        roll_change_frequency = (roll_sign_changes / max_possible_changes) if max_possible_changes > 0 else 0.0
        
        normalized_roll_variance = min(1.0, roll_variance / 0.25)
        oscillation_index_roll = 0.7 * normalized_roll_variance + 0.3 * roll_change_frequency
    else:
        oscillation_index_roll = 0.0
    
    # Pitch Oscillation Index calculation
    if len(pitch_cmds_deg) > 1:
        pitch_array = np.array(pitch_cmds_deg)
        pitch_variance = float(np.var(pitch_array))
        pitch_changes = np.diff(pitch_array)
        pitch_sign_changes = np.sum(np.diff(np.sign(pitch_changes)) != 0)
        max_possible_changes = len(pitch_changes) - 1
        pitch_change_frequency = (pitch_sign_changes / max_possible_changes) if max_possible_changes > 0 else 0.0
        
        normalized_pitch_variance = min(1.0, pitch_variance / 0.25)
        oscillation_index_pitch = 0.7 * normalized_pitch_variance + 0.3 * pitch_change_frequency
    else:
        oscillation_index_pitch = 0.0
    
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
        "oscillation_index_roll": oscillation_index_roll,
        "oscillation_index_pitch": oscillation_index_pitch,
        "deviation_reward_correlation": deviation_reward_correlation
    }

def visualize_episode(env, model, episode=0, save_path=None, vecnorm_stats=None, obs_adapter=None, episode_seed=None):
    """Visualize a single episode with comprehensive analysis."""
    # Run episode simulation
    (xs, ys, zs, headings, yaw_cmds_deg, roll_cmds_deg, pitch_cmds_deg, roll_rates_deg_per_step,
     pitch_rates_deg_per_step, control_actions, step_rewards, step_dist_to_path, step_dist_to_goal,
     path_segments, total_reward, dpath_samples, info) = _run_episode_simulation(
         env, model, vecnorm_stats, obs_adapter, episode_seed)
    
    # Calculate metrics
    metrics = _calculate_episode_metrics(env, xs, ys, roll_cmds_deg, pitch_cmds_deg, step_rewards, 
                                       step_dist_to_path, step_dist_to_goal, path_segments,
                                       total_reward, dpath_samples)

    # Determine save paths
    if save_path:
        path_save = save_path.replace('.png', '_paths.png')
        control_save = save_path.replace('.png', '_control.png')
    else:
        path_save = None
        control_save = None

    # ========== FIGURE 1: PATH VISUALIZATION ==========
    fig1 = plt.figure(figsize=(18, 6))
    
    # 3D plot
    ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
    
    # 2D path projections
    ax2 = plt.subplot(1, 3, 2)  # XY path (top view)
    ax3 = plt.subplot(1, 3, 3)  # XZ path (side view)

    # Plot 1: 3D path
    ax1.plot(env.path_x_array, env.path_y_array, env.path_z_array, 'b--', label='Desired Path', linewidth=2)
    ax1.plot(xs, ys, zs, 'r-', label='Actual Path', linewidth=2)
    
    # Add waypoints (checkpoints) to 3D plot
    if hasattr(env, 'checkpoint_positions') and len(env.checkpoint_positions) > 0:
        cp_xs = [cp[0] for cp in env.checkpoint_positions]
        cp_ys = [cp[1] for cp in env.checkpoint_positions]
        cp_zs = [cp[2] for cp in env.checkpoint_positions]
        
        # Plot checkpoints with different colors based on status
        next_cp_idx = info.get("next_checkpoint_idx", 0)
        for i, (cp_x, cp_y, cp_z) in enumerate(env.checkpoint_positions):
            if i < len(env.checkpoints_reached) and env.checkpoints_reached[i]:
                # Reached checkpoints - green
                ax1.scatter(cp_x, cp_y, cp_z, c='green', s=100, marker='o', alpha=0.8, label='Reached CP' if i == 0 else "")
            elif i == next_cp_idx:
                # Next checkpoint - blue
                ax1.scatter(cp_x, cp_y, cp_z, c='blue', s=120, marker='^', alpha=0.8, label='Next CP' if i == next_cp_idx else "")
            else:
                # Future checkpoints - orange
                ax1.scatter(cp_x, cp_y, cp_z, c='orange', s=80, marker='s', alpha=0.6, label='Future CP' if i == next_cp_idx + 1 else "")
        
        # Add goal marker (final checkpoint)
        goal_x, goal_y, goal_z = env.x_end, env.y_end, env.z_end
        ax1.scatter(goal_x, goal_y, goal_z, c='red', s=150, marker='*', alpha=0.9, label='Goal')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # Plot 2: XY path (top view)
    ax2.plot(env.path_x_array, env.path_y_array, 'b--', label='Desired Path', linewidth=2)
    ax2.plot(xs, ys, 'r-', label='Actual Path', linewidth=2)
    
    # Add waypoints to XY plot
    if hasattr(env, 'checkpoint_positions') and len(env.checkpoint_positions) > 0:
        next_cp_idx = info.get("next_checkpoint_idx", 0)
        for i, (cp_x, cp_y, cp_z) in enumerate(env.checkpoint_positions):
            if i < len(env.checkpoints_reached) and env.checkpoints_reached[i]:
                ax2.scatter(cp_x, cp_y, c='green', s=100, marker='o', alpha=0.8, label='Reached CP' if i == 0 else "")
            elif i == next_cp_idx:
                ax2.scatter(cp_x, cp_y, c='blue', s=120, marker='^', alpha=0.8, label='Next CP' if i == next_cp_idx else "")
            else:
                ax2.scatter(cp_x, cp_y, c='orange', s=80, marker='s', alpha=0.6, label='Future CP' if i == next_cp_idx + 1 else "")
        
        # Add goal marker
        ax2.scatter(env.x_end, env.y_end, c='red', s=150, marker='*', alpha=0.9, label='Goal')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')

    # Plot 3: XZ path (side view)
    ax3.plot(env.path_x_array, env.path_z_array, 'b--', label='Desired Path', linewidth=2)
    ax3.plot(xs, zs, 'r-', label='Actual Path', linewidth=2)
    
    # Add waypoints to XZ plot
    if hasattr(env, 'checkpoint_positions') and len(env.checkpoint_positions) > 0:
        next_cp_idx = info.get("next_checkpoint_idx", 0)
        for i, (cp_x, cp_y, cp_z) in enumerate(env.checkpoint_positions):
            if i < len(env.checkpoints_reached) and env.checkpoints_reached[i]:
                ax3.scatter(cp_x, cp_z, c='green', s=100, marker='o', alpha=0.8, label='Reached CP' if i == 0 else "")
            elif i == next_cp_idx:
                ax3.scatter(cp_x, cp_z, c='blue', s=120, marker='^', alpha=0.8, label='Next CP' if i == next_cp_idx else "")
            else:
                ax3.scatter(cp_x, cp_z, c='orange', s=80, marker='s', alpha=0.6, label='Future CP' if i == next_cp_idx + 1 else "")
        
        # Add goal marker
        ax3.scatter(env.x_end, env.z_end, c='red', s=150, marker='*', alpha=0.9, label='Goal')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.grid(True)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # ========== FIGURE 2: CONTROL SIGNALS ==========
    fig2, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Roll command
    time_points = np.arange(len(roll_cmds_deg)) * env.dt
    if len(roll_cmds_deg) > 0:
        ax4.plot(time_points, roll_cmds_deg, 'r-', label='Roll cmd (deg)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Roll cmd (deg)')
    ax4.grid(True)
    ax4.legend()

    # Plot 2: Roll rate change
    if len(roll_rates_deg_per_step) > 0:
        time_points_roll_rate = np.arange(len(roll_rates_deg_per_step)) * env.dt
        ax5.plot(time_points_roll_rate, roll_rates_deg_per_step, 'b-', 
                label='Roll Rate Change (deg/step)', linewidth=2)
        ax5.axhline(y=env.max_roll_rate_deg_per_step, color='red', linestyle='--', alpha=0.7, 
                   label=f'Hard Cap ({env.max_roll_rate_deg_per_step}°/step)')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Roll Rate Change (deg/step)')
        ax5.grid(True)
        ax5.legend()
        ax5.set_ylim(0, max(env.max_roll_rate_deg_per_step * 1.5, max(roll_rates_deg_per_step) * 1.1))

    # Plot 3: Pitch command
    if len(pitch_cmds_deg) > 0:
        time_points_pitch = np.arange(len(pitch_cmds_deg)) * env.dt
        ax6.plot(time_points_pitch, pitch_cmds_deg, 'm-', label='Pitch cmd (deg)')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Pitch cmd (deg)')
    ax6.grid(True)
    ax6.legend()

    # Plot 4: Pitch rate change
    if len(pitch_rates_deg_per_step) > 0:
        time_points_pitch_rate = np.arange(len(pitch_rates_deg_per_step)) * env.dt
        ax7.plot(time_points_pitch_rate, pitch_rates_deg_per_step, 'orange', 
                label='Pitch Rate Change (deg/step)', linewidth=2)
        ax7.axhline(y=env.max_pitch_rate_deg_per_step, color='red', linestyle='--', alpha=0.7, 
                   label=f'Pitch Rate Cap ({env.max_pitch_rate_deg_per_step}°/step)')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Pitch Rate Change (deg/step)')
        ax7.grid(True)
        ax7.legend()
        ax7.set_ylim(0, max(env.max_pitch_rate_deg_per_step * 1.5, max(pitch_rates_deg_per_step) * 1.1))

    plt.tight_layout()
    if control_save:
        plt.savefig(control_save, dpi=300, bbox_inches='tight')
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
    elif model_shape == 5 and env_shape == 6:
        return lambda obs: obs[:5]  # Take first 5 elements (remove altitude_error)
    elif model_shape == 6 and env_shape == 5:
        return lambda obs: np.concatenate([obs, [0.0]])  # Pad with altitude_error
    elif model_shape == 3 and env_shape == 6:
        return lambda obs: obs[:3]  # Take first 3 elements
    elif model_shape == 6 and env_shape == 3:
        return lambda obs: np.concatenate([obs, [0.0, 0.0, 0.0]])  # Pad with zeros
    else:
        return None

# === VALIDATION FUNCTIONS ===
def run_validation(model_path, config, num_episodes=5, fixed_seeds=None,
                   wind_enabled=False, wind_gust_magnitude=1.0, 
                   wind_gust_duration=10.0, wind_transition_time=1.0):
    """Run validation for a specific model."""
    env = FixedWingUAVEnv(config=config,
                          wind_enabled=wind_enabled,
                          wind_speed_mean=wind_gust_magnitude,  # Used as gust magnitude
                          wind_speed_std=wind_gust_duration,    # Used as gust duration
                          wind_heading_std=wind_transition_time)  # Used as transition time
    model = load_model(model_path)
    obs_adapter = create_observation_adapter(model.observation_space, env.observation_space, config)
    
    # Load VecNormalize stats if available
    vecnorm_stats = None
    basename = os.path.basename(model_path)
    algorithm = basename.split('_')[0].lower()
    if algorithm in ['ppo', 'sac', 'td3']:
        stats_path = model_path.replace('.zip', '') + "_vn.pkl"
        if os.path.exists(stats_path):
            tmp_venv = DummyVecEnv([lambda: FixedWingUAVEnv(config=config,
                                                            wind_enabled=wind_enabled,
                                                            wind_speed_mean=wind_gust_magnitude,
                                                            wind_speed_std=wind_gust_duration,
                                                            wind_heading_std=wind_transition_time)])
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
    oscillation_indices_roll = []
    oscillation_indices_pitch = []
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
        oscillation_indices_roll.append(info.get("oscillation_index_roll", np.nan))
        oscillation_indices_pitch.append(info.get("oscillation_index_pitch", np.nan))
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
        'avg_total_reward': np.mean(total_rewards),
        'avg_completion_time': np.mean(completion_times),
        'avg_oscillation_index_roll': np.mean(oscillation_indices_roll),
        'avg_oscillation_index_pitch': np.mean(oscillation_indices_pitch),
        'avg_deviation_correlation': np.mean(deviation_correlations),
        'avg_episode_length': np.mean(episode_lengths),
        'episode_length_std': np.std(episode_lengths),
        'avg_path_efficiency': np.mean(path_efficiencies)
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
        f.write(f"Oscillation Index Roll (OI-Roll): {summary['avg_oscillation_index_roll']:.3f}\n")
        f.write(f"Oscillation Index Pitch (OI-Pitch): {summary['avg_oscillation_index_pitch']:.3f}\n")
        f.write(f"Deviation-Reward Correlation (DRC): {summary['avg_deviation_correlation']:.3f}\n")
        f.write(f"Path Efficiency (PE): {summary['avg_path_efficiency']:.3f}\n")
        f.write(f"Average episode length: {summary['avg_episode_length']:.1f} ± {summary['episode_length_std']:.1f} steps\n")
    
    return summary

def validate_all_models(num_episodes=5, wind_enabled=False, wind_gust_magnitude=1.0, 
                       wind_gust_duration=10.0, wind_transition_time=1.0):
    """Validate all trained models and show comparison plots"""
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
    
    # Print wind configuration
    if wind_enabled:
        print(f"🌬️  Wind gusts enabled: magnitude={wind_gust_magnitude} m/s, duration={wind_gust_duration}s, transition={wind_transition_time}s")
    else:
        print("🌤️  Wind disabled for validation")
    
    all_results = []
    
    # Validate each model
    for algorithm in models:
        for config in models[algorithm]:
            # Skip unsupported configs
            if config != 'waypoint':
                print(f"\n⚠️  Skipping {algorithm} {config}: unsupported config (only 'waypoint' supported)")
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
    print(f"{'Algorithm':<8} {'Config':<8} {'Success%':<10} {'Avg Reward':<12} {'Dist Path':<10} {'OI-Roll':<10} {'OI-Pitch':<10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        # Display config name
        config_display = result['config']
        print(f"{result['algorithm']:<8} {config_display:<8} "
              f"{result['success_rate']:<10.1f} {result['avg_total_reward']:<12.2f} "
              f"{result['avg_dist_to_path']:<10.2f} "
              f"{result['avg_oscillation_index_roll']:<10.3f} {result['avg_oscillation_index_pitch']:<10.3f}")
    
    # Save overall comparison
    with open("validation_results/comparison_summary.txt", "w") as f:
        f.write("Model Comparison Summary\n")
        f.write("="*90 + "\n")
        f.write("Performance Metrics:\n")
        f.write("- Success%: Percentage of episodes reaching goal successfully\n")
        f.write("- Avg Reward: Average total reward per episode\n")
        f.write("- Dist Path: Average distance from desired path (meters)\n")
        f.write("Reward Hacking Detection Metrics:\n")
        f.write("- OI-Roll (Oscillation Index Roll) = Roll control variance + change frequency\n")
        f.write("- OI-Pitch (Oscillation Index Pitch) = Pitch control variance + change frequency\n")
        f.write("- Higher OI suggests zig-zag or bang-bang control policies\n")
        f.write("="*80 + "\n")
        f.write(f"{'Algorithm':<8} {'Config':<8} {'Success%':<10} {'Avg Reward':<12} {'Dist Path':<10} {'OI-Roll':<10} {'OI-Pitch':<10}\n")
        f.write("-"*80 + "\n")
        for result in all_results:
            # Display config name
            config_display = result['config']
            f.write(f"{result['algorithm']:<8} {config_display:<8} "
                   f"{result['success_rate']:<10.1f} {result['avg_total_reward']:<12.2f} "
                   f"{result['avg_dist_to_path']:<10.2f} "
                   f"{result['avg_oscillation_index_roll']:<10.3f} {result['avg_oscillation_index_pitch']:<10.3f}\n")
    
    return all_results

def main():
    """Main validation function - validates all trained models"""
    print("🚀 Starting comprehensive validation of all trained models...")
    print("This will validate all algorithm x configuration combinations.")
    
    try:
        # Validate all models and show comparison
        results = validate_all_models(num_episodes=1,
                                      wind_enabled=True,
                                      wind_gust_magnitude=4.0,
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