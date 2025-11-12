import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import PPO, SAC, TD3
from env import FixedWingUAVEnv
import time
import glob
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

def _draw_start_finish(ax, env, title_suffix=""):
    # Professional start marker
    ax.plot(env.x0, env.y0, marker='*', color='#2ca02c', markersize=16, 
            label='Start', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)

    # Professional goal center marker
    ax.plot(env.x_end, env.y_end, marker='X', color='#d62728', markersize=12, 
            label='Goal Center', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)

    # Title amend (only if title_suffix is not empty)
    if title_suffix:
        ax.set_title(f'Path Following {title_suffix}')

def _normalize_obs_if_needed(obs, vn):
    """Normalize a single observation using loaded VecNormalize stats (inference-only)."""
    if vn is None or not hasattr(vn, "obs_rms"):
        return obs
    mean = np.array(vn.obs_rms.mean)
    var = np.array(vn.obs_rms.var)
    eps = 1e-8
    clip = getattr(vn, "clip_obs", 10.0)
    
    # Handle dimension mismatch: if model was trained with 4D obs but env now uses 5D
    # We need to match dimensions by truncating the normalization stats
    if len(mean) < len(obs):
        print(f"   ⚠️  Dimension mismatch: model expects {len(mean)}D obs, but env provides {len(obs)}D")
        print(f"   Padding normalization stats to match current observation space")
        # Pad with zeros for mean and ones for variance for the new dimension
        mean = np.concatenate([mean, np.zeros(len(obs) - len(mean))])
        var = np.concatenate([var, np.ones(len(obs) - len(mean))])
    elif len(mean) > len(obs):
        print(f"   ⚠️  Dimension mismatch: model expects {len(mean)}D obs, but env provides {len(obs)}D")
        print(f"   Truncating normalization stats to match current observation space")
        mean = mean[:len(obs)]
        var = var[:len(obs)]
    
    normalized = (obs - mean) / np.sqrt(var + eps)
    return np.clip(normalized, -clip, clip).astype(np.float32)

def plot_path_comparison(env, model, vecnorm_stats=None):
    obs, _ = env.reset()
    obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
    done = False
    xs = [env.x]; ys = [env.y]
    while not done:
        action, _ = model.predict(obs_for_policy, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
        done = terminated or truncated
        xs.append(env.x); ys.append(env.y)

    plt.figure(figsize=(15, 8))
    # Episode-specific straight line
    plt.plot(env.path_x_array, env.path_y_array, 'b--', label='Desired Path', linewidth=2)
    # Agent path
    plt.plot(xs, ys, 'r-', label='Agent Path', linewidth=2)

    # Start/finish visuals
    _draw_start_finish(plt.gca(), env, title_suffix=": Straight Line")

    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.grid(True); plt.axis('equal')
    plt.show()

def visualize_episode(env, model, episode=0, save_path=None, vecnorm_stats=None, seed=None):
    # Use fixed seed for reproducible validation
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
    done = False
    xs = [env.x]; ys = [env.y]
    headings = [env.heading_deg]   # degrees
    heading_errors = []
    roll_cmds_deg = []

    # NEW: accumulate stats for the banner
    total_reward = 0.0
    dpath_samples = []
    control_actions = []  # Track control actions for oscillation analysis

    info = {
        "termination_reason": "running",
        "heading_diff_deg": 0.0,
        "dist_to_path_abs": 0.0,
        "reached_cp_count": 0,
        "total_cps": 0
    }

    while not done:
        action, _ = model.predict(obs_for_policy, deterministic=True)

        # Store raw action for oscillation analysis
        control_actions.append(float(action[0]))
        
        # roll command in degrees (normalized action * max_roll_deg)
        roll_cmds_deg.append(float(action[0]) * env.max_roll_deg)

        obs, reward, terminated, truncated, info = env.step(action)
        obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
        done = terminated or truncated

        # accumulate
        total_reward += float(reward)
        dpath_samples.append(float(info["dist_to_path_abs"]))

        xs.append(env.x); ys.append(env.y)
        headings.append(env.heading_deg)                     # degrees
        heading_errors.append(info["heading_diff_deg"])      # degrees

    xs = np.array(xs); ys = np.array(ys)
    headings = np.array(headings)
    heading_errors = np.array(heading_errors)
    roll_cmds_deg = np.array(roll_cmds_deg)

    # NEW: compute episode stats
    avg_dist_to_path = float(np.mean(dpath_samples)) if len(dpath_samples) else float("nan")
    total_steps = len(xs) - 1  # Subtract 1 because we start with initial position
    episode_length = total_steps  # Track episode length for distribution analysis
    
    # Compute Oscillation Index (OI)
    if len(control_actions) > 1:
        # Convert to numpy array for easier computation
        actions_array = np.array(control_actions)
        
        # Variance of control actions (measures spread/variability)
        action_variance = float(np.var(actions_array))
        
        # Action change frequency (number of sign changes / total possible changes)
        action_changes = np.diff(actions_array)
        sign_changes = np.sum(np.diff(np.sign(action_changes)) != 0)
        max_possible_changes = len(action_changes) - 1
        change_frequency = (sign_changes / max_possible_changes) if max_possible_changes > 0 else 0.0
        
        # Combined Oscillation Index (weighted sum of variance and frequency)
        # Scale variance to [0,1] range assuming actions are in [-1,1]
        normalized_variance = min(1.0, action_variance / 0.25)  # 0.25 is max variance for [-1,1] uniform
        oscillation_index = 0.7 * normalized_variance + 0.3 * change_frequency
    else:
        oscillation_index = 0.0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))

    # 1) XY path with start/finish visuals
    ax1.plot(env.path_x_array, env.path_y_array, 'b--', label='Desired Path', linewidth=2)
    ax1.plot(xs, ys, 'r-', label='Actual Path', linewidth=2)
    _draw_start_finish(ax1, env, title_suffix=": Straight Line")
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.grid(True); ax1.axis('equal')

    # success/fail + stats banner
    term = info.get("termination_reason", "running")
    ax1.text(
        0.02, 0.95,
        f"Termination: {term}\nTotal reward: {total_reward:.2f}\nAvg dist-to-path: {avg_dist_to_path:.2f} m",
        transform=ax1.transAxes, fontsize=11,
        bbox=dict(
            facecolor=('lightgreen' if term == 'success' else 'mistyrose'),
            edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.9
        )
    )

    # 2) Heading (deg)
    time_points = np.arange(len(headings)) * env.dt
    ax2.plot(time_points, headings, 'g-', label='Heading (deg)')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Heading (deg)')
    ax2.grid(True); ax2.legend()

    # 3) Roll command (deg)
    if len(roll_cmds_deg) > 0:
        ax3.plot(time_points[:len(roll_cmds_deg)], roll_cmds_deg, 'r-', label='Roll cmd (deg)')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Roll cmd (deg)')
    ax3.grid(True); ax3.legend()

    # 4) Heading error (deg)
    time_points_error = np.arange(len(heading_errors)) * env.dt
    ax4.plot(time_points_error, heading_errors, 'b-', label='Heading Error (deg)')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Heading Error (deg)')
    ax4.grid(True); ax4.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path); plt.close()
    else:
        plt.show()

    # NEW: return stats for console logging
    out = dict(info)
    out["total_reward"] = total_reward
    out["avg_dist_to_path"] = avg_dist_to_path
    out["oscillation_index"] = oscillation_index
    out["episode_length"] = episode_length
    return out

def find_all_models():
    """Find all trained models organized by algorithm"""
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
            # Parse: algorithm_fixedwing_waypoint_timestamp.zip
            parts = basename.replace('.zip', '').split('_')
            if len(parts) >= 3:
                algorithm = parts[0].upper()
                
                if algorithm not in models:
                    models[algorithm] = []
                    
                models[algorithm].append(file)
    
    # Get latest model for each algorithm
    latest_models = {}
    for alg in models:
        latest_models[alg] = max(models[alg], key=os.path.getmtime)
    
    return latest_models

def load_model(model_path):
    """Load model based on algorithm type with error handling for observation space mismatches"""
    basename = os.path.basename(model_path)
    algorithm = basename.split('_')[0].lower()
    
    try:
        if algorithm == 'ppo':
            model = PPO.load(model_path)
        elif algorithm == 'sac':
            model = SAC.load(model_path)
        elif algorithm == 'td3':
            model = TD3.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Check if the model's observation space is compatible
        print(f"   Model observation space: {model.observation_space}")
        return model
        
    except Exception as e:
        print(f"   Error loading model: {e}")
        raise

def run_validation(model_path, num_episodes=20, seed=42):
    """Run validation for a specific model"""
    # Set seeds for reproducible validation
    np.random.seed(seed)
    random.seed(seed)
    
    env = FixedWingUAVEnv()
    print(f"   Environment observation space: {env.observation_space}")
    print(f"   Environment action space: {env.action_space}")
    print(f"   Using seed {seed} for reproducible validation")
    
    model = load_model(model_path)
    print(f"   Model action space: {model.action_space}")
    vecnorm_stats = None
    # Load VecNormalize stats for all algorithms (PPO, SAC, TD3 all use VecNormalize)
    basename = os.path.basename(model_path)
    algorithm = basename.split('_')[0].lower()
    stats_path = model_path.replace('.zip', '') + "_vn.pkl"
    if os.path.exists(stats_path):
        tmp_venv = DummyVecEnv([lambda: FixedWingUAVEnv()])
        try:
            vecnorm_stats = VecNormalize.load(stats_path, tmp_venv)
            vecnorm_stats.training = False
            vecnorm_stats.norm_reward = False
            print(f"   ✅ Loaded VecNormalize stats for {algorithm.upper()}")
        except Exception as e:
            print(f"   ⚠️  Warning: failed to load VecNormalize stats for {algorithm.upper()}: {e}")
    else:
        print(f"   ⚠️  No VecNormalize stats found for {algorithm.upper()} - using raw observations")
    
    # Extract algorithm from path for naming
    basename = os.path.basename(model_path).replace('.zip', '')
    algorithm = basename.split('_')[0].upper()
    
    results_dir = f"validation_results/{algorithm.lower()}_waypoint"
    os.makedirs(results_dir, exist_ok=True)
    
    success_count = 0
    path_errors = []
    heading_errors = []
    completion_times = []
    total_rewards = []
    avg_dpaths = []
    oscillation_indices = []
    episode_lengths = []

    print(f"\n{'='*60}")
    print(f"Validating {algorithm} - {num_episodes} episodes")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        start_time = time.time()
        
        # Debug: Show first observation and normalization
        if episode == 0:
            obs, _ = env.reset(seed=seed + episode)  # Use deterministic seed per episode
            print(f"   Raw observation shape: {obs.shape}")
            print(f"   Raw observation: {obs}")
            if vecnorm_stats is not None:
                obs_norm = _normalize_obs_if_needed(obs, vecnorm_stats)
                print(f"   Normalized observation: {obs_norm}")
            else:
                print(f"   No normalization applied")
        
        info = visualize_episode(
            env,
            model,
            episode=episode,
            save_path=f"{results_dir}/episode_{episode+1}.png",
            vecnorm_stats=vecnorm_stats,
            seed=seed + episode,  # Use deterministic seed per episode
        )
        episode_time = time.time() - start_time
        if info["termination_reason"] == "success":
            success_count += 1

        path_errors.append(info["dist_to_path_abs"])
        heading_errors.append(info["heading_diff_deg"])
        total_rewards.append(info.get("total_reward", np.nan))
        avg_dpaths.append(info.get("avg_dist_to_path", np.nan))
        oscillation_indices.append(info.get("oscillation_index", np.nan))
        episode_lengths.append(info.get("episode_length", np.nan))
        completion_times.append(episode_time)

        print(f"Episode {episode+1}: reason={info['termination_reason']}, "
              f"final path err={info['dist_to_path_abs']:.2f} m, "
              f"final heading err={info['heading_diff_deg']:.2f} deg, "
              f"avg dist-to-path={info.get('avg_dist_to_path', float('nan')):.2f} m, "
              f"total reward={info.get('total_reward', float('nan')):.2f}, "
              f"oscillation={info.get('oscillation_index', float('nan')):.3f}, "
              f"steps={info.get('episode_length', 0)}, "
              f"time={episode_time:.2f}s")

    # Summary statistics with standard deviations (only for PD and OI)
    # Convert to numpy arrays for std calculations
    avg_dpaths_array = np.array(avg_dpaths)
    oscillation_indices_array = np.array(oscillation_indices)
    
    # Calculate means (all metrics)
    avg_path_error = float(np.mean(path_errors))
    avg_heading_error = float(np.mean(heading_errors))
    avg_total_reward = float(np.mean(total_rewards))
    avg_completion_time = float(np.mean(completion_times))
    avg_episode_length = float(np.mean(episode_lengths))
    
    # Calculate means and standard deviations (only for PD and OI)
    avg_dist_to_path = float(np.mean(avg_dpaths_array))
    dist_to_path_std = float(np.std(avg_dpaths_array))
    avg_oscillation_index = float(np.mean(oscillation_indices_array))
    oscillation_index_std = float(np.std(oscillation_indices_array))
    
    summary = {
        'algorithm': algorithm,
        'success_rate': success_count/num_episodes*100,
        'avg_path_error': avg_path_error,
        'avg_heading_error': avg_heading_error,
        'avg_dist_to_path': avg_dist_to_path,
        'dist_to_path_std': dist_to_path_std,
        'avg_total_reward': avg_total_reward,
        'avg_completion_time': avg_completion_time,
        'avg_oscillation_index': avg_oscillation_index,
        'oscillation_index_std': oscillation_index_std,
        'avg_episode_length': avg_episode_length,
    }

    print(f"\n{algorithm} Summary:")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Avg path error: {summary['avg_path_error']:.2f} m")
    print(f"Avg heading error: {summary['avg_heading_error']:.2f} deg")
    print(f"Avg total reward: {summary['avg_total_reward']:.2f}")
    print(f"Path Deviation (PD): {summary['avg_dist_to_path']:.2f}±{summary['dist_to_path_std']:.2f} m")
    print(f"Oscillation Index (OI): {summary['avg_oscillation_index']:.3f}±{summary['oscillation_index_std']:.3f}")
    print(f"Avg episode length: {summary['avg_episode_length']:.1f} steps")
    print(f"Avg completion time: {summary['avg_completion_time']:.2f} s")

    # Save summary
    with open(f"{results_dir}/summary.txt", "w") as f:
        f.write(f"{algorithm} Validation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Number of episodes: {num_episodes}\n")
        f.write(f"Success rate: {summary['success_rate']:.1f}%\n")
        f.write(f"Average FINAL path error: {summary['avg_path_error']:.2f} m\n")
        f.write(f"Average FINAL heading error: {summary['avg_heading_error']:.2f} deg\n")
        f.write(f"Average EPISODE avg dist-to-path (PD): {summary['avg_dist_to_path']:.2f} ± {summary['dist_to_path_std']:.2f} m\n")
        f.write(f"Average total reward: {summary['avg_total_reward']:.2f}\n")
        f.write(f"Average completion time: {summary['avg_completion_time']:.2f} s\n")
        f.write(f"Oscillation Index (OI): {summary['avg_oscillation_index']:.3f} ± {summary['oscillation_index_std']:.3f}\n")
        f.write(f"Average episode length: {summary['avg_episode_length']:.1f} steps\n")
    
    return summary

def validate_all_models(num_episodes=20, seed=42):
    """Validate all trained models and show comparison plots"""
    # Set global seed for reproducible validation
    np.random.seed(seed)
    random.seed(seed)
    
    models = find_all_models()
    
    if not models:
        print("No trained models found! Please run training first.")
        return
    
    print(f"🌱 Using seed {seed} for reproducible validation")
    os.makedirs("validation_results", exist_ok=True)
    
    all_results = []
    
    # Validate each model
    for algorithm in models:
        model_path = models[algorithm]
        print(f"\n🔍 Found {algorithm}: {model_path}")
        
        try:
            summary = run_validation(model_path, num_episodes, seed=seed)
            all_results.append(summary)
        except Exception as e:
            print(f"❌ Failed to validate {algorithm}: {e}")
    
    # Create comparison summary
    print(f"\n{'='*80}")
    print("VALIDATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Algorithm':<8} {'Success%':<10} {'Avg Reward':<12} {'OI':<12} {'PD':<12}")
    print(f"{'-'*80}")
    
    for result in all_results:
        print(f"{result['algorithm']:<8} "
              f"{result['success_rate']:<10.1f} "
              f"{result['avg_total_reward']:.2f} "
              f"{result['avg_oscillation_index']:.3f}±{result.get('oscillation_index_std', 0):.3f} "
              f"{result['avg_dist_to_path']:.2f}±{result.get('dist_to_path_std', 0):.2f}")
    
    # Save overall comparison
    with open("validation_results/comparison_summary.txt", "w") as f:
        f.write("Model Comparison Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Number of episodes per algorithm: {num_episodes}\n")
        f.write("Metrics:\n")
        f.write("- OI (Oscillation Index) = Control action variance + change frequency\n")
        f.write("- PD (Path Deviation) = Average distance from desired path (meters)\n")
        f.write("- Higher OI suggests zig-zag or bang-bang control policies\n")
        f.write("- Lower PD indicates better path-following performance\n")
        f.write("="*80 + "\n")
        f.write(f"{'Algorithm':<8} {'Success%':<10} {'Avg Reward':<20} {'OI':<20} {'PD':<20}\n")
        f.write("-"*80 + "\n")
        for result in all_results:
            f.write(f"{result['algorithm']:<8} "
                   f"{result['success_rate']:<10.1f} "
                   f"{result['avg_total_reward']:.2f} "
                   f"{result['avg_oscillation_index']:.3f}±{result.get('oscillation_index_std', 0):.3f} "
                   f"{result['avg_dist_to_path']:.2f}±{result.get('dist_to_path_std', 0):.2f}\n")
    
    return all_results

def plot_algorithm_comparison(models, num_episodes=1, fixed_seed=42):
    """Plot first episode paths for all algorithms in a single comparison plot."""
    if not models:
        print("No trained models found for comparison plot!")
        return
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Professional colors for journal publication
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    line_styles = ['-', '--', '-.', ':']
    line_widths = [2.5, 2.5, 2.5, 2.5]
    color_idx = 0
    
    # Use fixed seed for fair comparison
    print(f"🔧 Using fixed seed {fixed_seed} for fair algorithm comparison")
    
    for algorithm in models:
        model_path = models[algorithm]
        print(f"🔍 Plotting {algorithm}: {model_path}")
        
        try:
            # Setup environment and model
            env = FixedWingUAVEnv()
            model = load_model(model_path)
            
            # Load VecNormalize stats if available
            vecnorm_stats = None
            basename = os.path.basename(model_path)
            algorithm_lower = basename.split('_')[0].lower()
            if algorithm_lower in ['ppo', 'sac', 'td3']:
                stats_path = model_path.replace('.zip', '') + "_vn.pkl"
                if os.path.exists(stats_path):
                    tmp_venv = DummyVecEnv([lambda: FixedWingUAVEnv()])
                    try:
                        vecnorm_stats = VecNormalize.load(stats_path, tmp_venv)
                        vecnorm_stats.training = False
                        vecnorm_stats.norm_reward = False
                    except Exception as e:
                        print(f"   Warning: failed to load VecNormalize stats: {e}")
            
            # Run single episode with FIXED SEED for fair comparison
            obs, _ = env.reset(seed=fixed_seed)
            obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
            done = False
            xs = [env.x]
            ys = [env.y]
            
            while not done:
                action, _ = model.predict(obs_for_policy, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
                done = terminated or truncated
                xs.append(env.x)
                ys.append(env.y)
            
            # Plot this algorithm's path with professional styling
            color = colors[color_idx % len(colors)]
            line_style = line_styles[color_idx % len(line_styles)]
            line_width = line_widths[color_idx % len(line_widths)]
            label = f"{algorithm}"
            ax.plot(xs, ys, color=color, linestyle=line_style, linewidth=line_width, 
                   label=label, alpha=0.9, solid_capstyle='round')
            color_idx += 1
            
        except Exception as e:
            print(f"❌ Failed to plot {algorithm}: {e}")
    
    # Plot desired path (only once) with professional styling
    if 'env' in locals():
        ax.plot(env.path_x_array, env.path_y_array, color='#2c2c2c', linestyle='--', 
                linewidth=3, label='Desired Path', alpha=0.8, solid_capstyle='round')
        _draw_start_finish(ax, env, title_suffix="")
    
    # Professional axis styling
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Professional legend styling with double size
    legend = ax.legend(loc='upper right', framealpha=0.95, fancybox=True, shadow=True, 
                      fontsize=22, frameon=True, edgecolor='black', markerscale=2.4)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0.5)
    
    # Set Y-axis limits between -300 and 100, keep X-axis auto-scaled
    ax.set_ylim(-300, 50)
    ax.margins(x=0.05)  # 5% margin on X-axis only
    
    plt.tight_layout()
    
    # Save comparison plot with high quality for journal publication
    os.makedirs("validation_results", exist_ok=True)
    save_path = "validation_results/algorithm_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png', transparent=False)
    plt.show()
    
    print(f"✅ Algorithm comparison plot saved to {save_path}")

def create_comparison_plot():
    """Standalone function to create algorithm comparison plot."""
    print("🔍 Creating algorithm comparison plot...")
    models = find_all_models()
    
    if not models:
        print("❌ No models found!")
        return
    
    plot_algorithm_comparison(models)

def main():
    """Main validation function - validates all trained models"""
    print("🚀 Starting comprehensive validation of all trained models...")
    print("This will validate all algorithm x configuration combinations.")
    
    # Use same seed as training for consistency
    seed = 42
    print(f"🌱 Using seed {seed} for reproducible validation")
    
    try:
        # Find all models first
        models = find_all_models()
        
        if not models:
            print("❌ No models found to validate. Please run training first.")
            return
        
        # Create algorithm comparison plot
        print("\n📊 Creating algorithm comparison plot...")
        plot_algorithm_comparison(models)
        
        # Validate all models and show comparison
        results = validate_all_models(num_episodes=20, seed=seed)
        
        if results:
            print(f"\n✅ Validation complete! Results saved in validation_results/")
            print(f"📊 {len(results)} model combinations validated")
            print(f"📈 Individual episode plots saved in validation_results/")
            print(f"📈 Algorithm comparison plot saved as algorithm_comparison.png")
            
        else:
            print("❌ No models found to validate. Please run training first.")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    main()