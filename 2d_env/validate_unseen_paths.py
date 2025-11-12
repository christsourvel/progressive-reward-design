import os
import numpy as np
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
            label='Goal', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
    
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

def _normalize_obs_if_needed(obs, vn):
    """Normalize a single observation using loaded VecNormalize stats (inference-only)."""
    if vn is None or not hasattr(vn, "obs_rms"):
        return obs
    mean = np.array(vn.obs_rms.mean)
    var = np.array(vn.obs_rms.var)
    eps = 1e-8
    clip = getattr(vn, "clip_obs", 10.0)
    normalized = (obs - mean) / np.sqrt(var + eps)
    return np.clip(normalized, -clip, clip).astype(np.float32)

class UnseenPathEnv(FixedWingUAVEnv):
    """Environment that generates unseen paths for testing generalization"""
    
    def __init__(self, config="waypoint", path_type="circular", wind_enabled=False, 
                 wind_speed_mean=2.0, wind_speed_std=1.0, wind_heading_std=30.0, **kwargs):
        super().__init__(config=config, wind_enabled=wind_enabled, 
                        wind_speed_mean=wind_speed_mean, wind_speed_std=wind_speed_std, 
                        wind_heading_std=wind_heading_std, **kwargs)
        self.path_type = path_type
        self.path_len_m = 1000.0  # Keep same length for consistency
        self.max_steps = 2000  # Increased from 1000 for longer episodes
    
    def _cross_and_along_track(self):
        """Return (signed_cross_track_m, along_track_m, path_length) using general parametric approach."""
        # Find closest point on path to agent
        distances = np.hypot(self.path_x_array - self.x, self.path_y_array - self.y)
        closest_idx = np.argmin(distances)
        
        # Get closest point on path
        closest_x = self.path_x_array[closest_idx]
        closest_y = self.path_y_array[closest_idx]
        
        # Calculate path tangent at closest point
        if closest_idx < len(self.path_x_array) - 1:
            # Use forward difference
            tangent_x = self.path_x_array[closest_idx + 1] - closest_x
            tangent_y = self.path_y_array[closest_idx + 1] - closest_y
        elif closest_idx > 0:
            # Use backward difference
            tangent_x = closest_x - self.path_x_array[closest_idx - 1]
            tangent_y = closest_y - self.path_y_array[closest_idx - 1]
        else:
            # Single point or edge case
            tangent_x = 1.0
            tangent_y = 0.0
        
        # Normalize tangent vector
        tangent_norm = np.hypot(tangent_x, tangent_y)
        if tangent_norm > 0:
            tangent_x /= tangent_norm
            tangent_y /= tangent_norm
        
        # Vector from closest path point to agent
        to_agent_x = self.x - closest_x
        to_agent_y = self.y - closest_y
        
        # Cross product for signed distance (positive = right of path, negative = left)
        # Cross product: tangent × to_agent (2D cross product = scalar)
        cross_product = tangent_x * to_agent_y - tangent_y * to_agent_x
        signed_ct = float(cross_product)
        
        # Calculate along-track distance (arc length to closest point)
        along_track_m = self._calculate_arc_length_to_index(closest_idx)
        
        # Calculate total path length
        path_length = self._calculate_total_path_length()
        
        return signed_ct, along_track_m, path_length
    
    def _calculate_arc_length_to_index(self, target_idx):
        """Calculate arc length from start of path to given index."""
        if target_idx <= 0:
            return 0.0
        
        # Sum segment lengths from start to target index
        arc_length = 0.0
        for i in range(target_idx):
            dx = self.path_x_array[i + 1] - self.path_x_array[i]
            dy = self.path_y_array[i + 1] - self.path_y_array[i]
            segment_length = np.hypot(dx, dy)
            arc_length += segment_length
        
        return arc_length
    
    def _calculate_total_path_length(self):
        """Calculate total arc length of the path."""
        if len(self.path_x_array) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self.path_x_array) - 1):
            dx = self.path_x_array[i + 1] - self.path_x_array[i]
            dy = self.path_y_array[i + 1] - self.path_y_array[i]
            segment_length = np.hypot(dx, dy)
            total_length += segment_length
        
        return total_length
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        # Generate unseen path
        if self.path_type == "circular":
            self._generate_circular_path()
        elif self.path_type == "figure8":
            self._generate_figure8_path()
        elif self.path_type == "spiral":
            self._generate_spiral_path()
        else:
            raise ValueError(f"Unknown path type: {self.path_type}")
        
        # Create checkpoints along the new path
        self._create_checkpoints_for_path()
        
        # Reset agent state
        self.x = self.x0
        self.y = self.y0
        
        # Calculate path direction at start point and align agent perfectly
        if len(self.path_x_array) > 1:
            # Calculate tangent direction at start point
            dx = self.path_x_array[1] - self.path_x_array[0]
            dy = self.path_y_array[1] - self.path_y_array[0]
            path_heading_deg = np.degrees(np.arctan2(dy, dx))
            self.heading_deg = path_heading_deg
        else:
            self.heading_deg = 0.0
        self.current_step = 0
        self._d_prev = float(np.hypot(self.x - self.x_end, self.y - self.y_end))
        self._goal_dist_prev = self._d_prev
        _, self._along_track_prev, _ = self._cross_and_along_track()
        
        # Reset oscillation tracking
        self._prev_action = 0.0
        self._consecutive_oscillations = 0
        
        # Initialize previous action tracking for heuristic config
        self._prev_prev_action = 0.0
        
        # Reset speed to default
        self.current_speed = self.speed
        
        return self._obs(), {}
    
    def _generate_circular_path(self):
        """Generate a circular path"""
        num_points = 1001
        t = np.linspace(0, 2*np.pi, num_points)
        radius = 150.0  # meters
        center_x = 500.0
        center_y = 0.0
        
        self.path_x_array = center_x + radius * np.cos(t)
        self.path_y_array = center_y + radius * np.sin(t)
        
        # Start point on circle
        self.x0 = self.path_x_array[0]
        self.y0 = self.path_y_array[0]
        
        # End point outside the circle (extended radius)
        end_angle = 2*np.pi  # Same angle as last point on circle
        extended_radius = radius * 1.3  # 30% larger radius
        self.x_end = center_x + extended_radius * np.cos(end_angle)
        self.y_end = center_y + extended_radius * np.sin(end_angle)
    
    
    def _generate_spiral_path(self):
        """Generate a spiral path"""
        num_points = 1001
        t = np.linspace(0, 3*np.pi, num_points)  # Reduced from 4π to 3π for less tight spiral
        radius = 200.0 + 120.0 * t / (3*np.pi)  # Doubled starting radius and growth
        
        self.path_x_array = 500.0 + radius * np.cos(t)
        self.path_y_array = radius * np.sin(t)
        
        # Start and end points - spiral naturally has different start/end
        self.x0 = self.path_x_array[0]
        self.y0 = self.path_y_array[0]
        # End point is at the very end of the spiral (after all checkpoints)
        self.x_end = self.path_x_array[-1]
        self.y_end = self.path_y_array[-1]
    
    def _generate_figure8_path(self):
        """Generate a figure-8 path"""
        num_points = 1001
        t = np.linspace(0, 4*np.pi, num_points)
        scale = 300.0  # Increased from 200.0 for wider figure-8
        
        self.path_x_array = scale * np.sin(t) + 500.0
        self.path_y_array = scale * np.sin(2*t) / 2
        
        # Start point: beginning of figure-8
        self.x0 = self.path_x_array[0]
        self.y0 = self.path_y_array[0]
        
        # End point: at the last checkpoint (CP20) using absolute position
        # The final checkpoint will be the goal
        scale = 300.0  # Updated to match the path scale
        end_t = 6.0  # Same as the last checkpoint
        self.x_end = scale * np.sin(end_t) + 500.0
        self.y_end = scale * np.sin(2*end_t) / 2
    
    def _create_checkpoints_for_path(self):
        """Create checkpoints along the current path with smart spacing for complex paths"""
        self.checkpoint_positions = []
        self.checkpoints_reached = []

        # Ensure start and end points are different (minimum distance)
        min_distance = 50.0  # meters
        start_end_distance = np.hypot(self.x_end - self.x0, self.y_end - self.y0)
        if start_end_distance < min_distance:
            path_length = len(self.path_x_array)
            end_idx = int(0.75 * path_length)
            self.x_end = self.path_x_array[end_idx]
            self.y_end = self.path_y_array[end_idx]
            print(f"   Adjusted end point to ensure minimum distance: "
                  f"{np.hypot(self.x_end - self.x0, self.y_end - self.y0):.1f}m")

        # >>> special-cases per-path
        if self.path_type == "spiral":
            self._create_spiral_checkpoints()
        elif self.path_type == "figure8":
            self._create_figure8_checkpoints()
        else:
            self._create_standard_checkpoints()

        self.next_checkpoint_idx = 0
        
        # Initialize center checkpoint toggle for figure-8
        if self.path_type == "figure8":
            self.center_checkpoint_active = False
        else:
            self.center_checkpoint_active = True  # Always active for other paths
    
    def _create_spiral_checkpoints(self):
        """Create more checkpoints for spiral paths to better follow the spiral"""
        # Create checkpoints with uniform distribution for spiral paths
        num_checkpoints = 16  # Even more checkpoints for better accuracy
        
        # Use uniform distribution along the spiral path
        for i in range(num_checkpoints):
            # Distribute checkpoints evenly along the spiral
            idx = int((i + 1) * len(self.path_x_array) / (num_checkpoints + 1))
            cp_x = self.path_x_array[idx]
            cp_y = self.path_y_array[idx]
            self.checkpoint_positions.append((cp_x, cp_y))
            self.checkpoints_reached.append(False)
        
        # Add goal as final checkpoint
        self.checkpoint_positions.append((self.x_end, self.y_end))
        self.checkpoints_reached.append(False)
        
        # Increase checkpoint radius for spiral paths
        self.checkpoint_radius = 25.0  # Larger radius for spiral paths
        
        # print(f"   Spiral checkpoints created: {len(self.checkpoint_positions)} total")
        # for i, (cp_x, cp_y) in enumerate(self.checkpoint_positions):
        #     print(f"     CP{i+1}: ({cp_x:.1f}, {cp_y:.1f})")
    
    def _create_figure8_checkpoints(self):
        """
        Create checkpoints for figure-8 using absolute positions.
        Strategy: Use absolute positions to avoid path length dependencies
        """
        # Define checkpoints using absolute positions along the figure-8 path
        # These positions are calculated based on the parametric figure-8 equation
        t_values = [
            0.3,   # CP1: Early in first loop
            0.6,   # CP2: Early first loop  
            0.9,   # CP3: Mid first loop
            1.2,   # CP4: Mid first loop
            1.5,   # CP5: End of first loop (center area)
            1.8,   # CP6: Late first loop
            2.1,   # CP7: End of first loop
            2.4,   # CP8: Final first loop
            2.7,   # CP9: Start of second loop
            3.0,   # CP10: Early second loop
            3.3,   # CP11: Mid second loop
            3.6,   # CP12: Mid second loop
            3.9,   # CP13: Late second loop
            4.2,   # CP14: End of second loop
            # Additional checkpoints after CP14
            4.5,   # CP15: Extended second loop
            4.8,   # CP16: Extended second loop
            5.1,   # CP17: Extended second loop
            5.4,   # CP18: Extended second loop
            5.7,   # CP19: Extended second loop
            6.0,   # CP20: Extended second loop
        ]
        
        # Add checkpoints at these absolute positions
        for i, t in enumerate(t_values):
            # Calculate position using the same parametric equation as the path
            scale = 300.0  # Updated to match the path scale
            cp_x = scale * np.sin(t) + 500.0
            cp_y = scale * np.sin(2*t) / 2
            self.checkpoint_positions.append((cp_x, cp_y))
            self.checkpoints_reached.append(False)
        
        # No external goal - CP20 is the final goal
        
        # print(f"   Figure-8 checkpoints created: {len(self.checkpoint_positions)} total (absolute positions)")
        # for i, (cp_x, cp_y) in enumerate(self.checkpoint_positions):
        #     print(f"     CP{i+1}: ({cp_x:.1f}, {cp_y:.1f})")
    
    def step(self, action):
        """Override step method to handle center checkpoint toggle for figure-8"""
        if hasattr(self, 'path_type') and self.path_type == "figure8":
            # For figure-8, use larger checkpoint radius for better detection
            original_radius = self.checkpoint_radius
            self.checkpoint_radius = 20.0  # Larger radius for figure-8
            
            # Call parent step method (simplified - no special center checkpoint handling)
            obs, reward, terminated, truncated, info = super().step(action)
            
            # Restore original radius
            self.checkpoint_radius = original_radius
            
            return obs, reward, terminated, truncated, info
        else:
            # Normal behavior for other paths
            return super().step(action)
    
    def _create_standard_checkpoints(self):
        """Create checkpoints for non-figure-8 paths"""
        # Create checkpoints with smart spacing to avoid issues with self-crossing paths
        num_checkpoints = 8  # Restored to original number
        min_checkpoint_distance = 60.0  # Reduced minimum distance for more checkpoints
        
        # Start with first checkpoint
        first_idx = len(self.path_x_array) // 8  # Start 1/8 of the way through
        self.checkpoint_positions.append((self.path_x_array[first_idx], self.path_y_array[first_idx]))
        self.checkpoints_reached.append(False)
        
        # Add remaining checkpoints with distance checking
        last_checkpoint_idx = first_idx
        for i in range(1, num_checkpoints):
            # Find next suitable checkpoint position
            for candidate_idx in range(last_checkpoint_idx + 20, len(self.path_x_array) - 20, 10):
                candidate_x = self.path_x_array[candidate_idx]
                candidate_y = self.path_y_array[candidate_idx]
                
                # Check distance from last checkpoint
                last_cp_x, last_cp_y = self.checkpoint_positions[-1]
                distance = np.hypot(candidate_x - last_cp_x, candidate_y - last_cp_y)
                
                if distance >= min_checkpoint_distance:
                    self.checkpoint_positions.append((candidate_x, candidate_y))
                    self.checkpoints_reached.append(False)
                    last_checkpoint_idx = candidate_idx
                    break
            else:
                # If no suitable position found, place at regular interval
                idx = int((i + 1) * len(self.path_x_array) / (num_checkpoints + 1))
                self.checkpoint_positions.append((self.path_x_array[idx], self.path_y_array[idx]))
                self.checkpoints_reached.append(False)
        
        # Add goal as final checkpoint
        self.checkpoint_positions.append((self.x_end, self.y_end))
        self.checkpoints_reached.append(False)

def visualize_episode_unseen_path(env, model, episode=0, save_path=None, vecnorm_stats=None, obs_adapter=None):
    """Visualize episode on unseen path"""
    obs, _ = env.reset()
    
    # Apply observation adapter if needed
    if obs_adapter is not None:
        obs = obs_adapter(obs)
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
    step_rewards = []  # Track individual step rewards
    step_dist_to_path = []  # Track distance to path per step
    step_dist_to_goal = []  # Track distance to goal per step
    path_segments = []  # Track path segments for actual path length calculation

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
        # Apply observation adapter if needed
        if obs_adapter is not None:
            obs = obs_adapter(obs)
        obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
        done = terminated or truncated

        # accumulate
        total_reward += float(reward)
        dpath_samples.append(float(info["dist_to_path_abs"]))
        
        # Store step-wise data for correlation analysis
        step_rewards.append(float(reward))
        step_dist_to_path.append(float(info["dist_to_path_abs"]))
        
        # Track distance to goal
        dist_to_goal = np.hypot(env.x - env.x_end, env.y - env.y_end)
        step_dist_to_goal.append(float(dist_to_goal))

        xs.append(env.x); ys.append(env.y)
        headings.append(env.heading_deg)                     # degrees
        heading_errors.append(info["heading_diff_deg"])      # degrees
        
        # Track path segments for actual path length calculation
        if len(xs) > 1:
            segment_length = np.hypot(xs[-1] - xs[-2], ys[-1] - ys[-2])
            path_segments.append(segment_length)

    xs = np.array(xs); ys = np.array(ys)
    headings = np.array(headings)
    heading_errors = np.array(heading_errors)
    roll_cmds_deg = np.array(roll_cmds_deg)

    # NEW: compute episode stats
    avg_dist_to_path = float(np.mean(dpath_samples)) if len(dpath_samples) else float("nan")
    total_steps = len(xs) - 1  # Subtract 1 because we start with initial position
    episode_length = total_steps  # Track episode length for distribution analysis
    
    # Calculate Path Efficiency (PE) = L_ideal / L_actual
    # Ideal path length: length of the path
    if hasattr(env, 'path_x_array') and hasattr(env, 'path_y_array') and env.path_x_array is not None:
        # Calculate ideal path length by summing segments of the path
        ideal_path_length = 0.0
        for i in range(1, len(env.path_x_array)):
            segment_length = np.hypot(env.path_x_array[i] - env.path_x_array[i-1], 
                                    env.path_y_array[i] - env.path_y_array[i-1])
            ideal_path_length += segment_length
    else:
        # Fallback: straight line distance from start to goal
        ideal_path_length = np.hypot(env.x_end - env.x0, env.y_end - env.y0)
    
    # Actual path length: sum of all flown segments
    actual_path_length = float(np.sum(path_segments)) if len(path_segments) > 0 else 0.0
    
    # Path Efficiency ratio (ideal/actual)
    path_efficiency = ideal_path_length / actual_path_length if actual_path_length > 0 else 0.0
    
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
    
    # Compute Path Deviation vs Reward Correlation
    if len(step_rewards) > 1 and len(step_dist_to_path) > 1:
        step_rewards_array = np.array(step_rewards)
        step_dist_to_path_array = np.array(step_dist_to_path)
        step_dist_to_goal_array = np.array(step_dist_to_goal)
        
        # Correlation between rewards and distance to path (should be negative for normal behavior)
        path_corr = np.corrcoef(step_rewards_array, step_dist_to_path_array)[0, 1]
        if np.isnan(path_corr):
            path_corr = 0.0
            
        # Correlation between rewards and distance to goal (should be negative for normal behavior)
        goal_corr = np.corrcoef(step_rewards_array, step_dist_to_goal_array)[0, 1]
        if np.isnan(goal_corr):
            goal_corr = 0.0
            
        # Positive correlations are suspicious (high reward while far from path/goal)
        # Take the maximum of the two correlations as the deviation correlation metric
        deviation_reward_correlation = max(path_corr, goal_corr)
    else:
        deviation_reward_correlation = 0.0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))

    # 1) XY path with professional styling
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

    # 2) Heading with professional styling
    time_points = np.arange(len(headings)) * env.dt
    ax2.plot(time_points, headings, color='#2ca02c', linestyle='-', linewidth=2.5, 
             label='Heading (deg)', alpha=0.9, solid_capstyle='round')
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Heading (deg)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=11)

    # 3) Roll command with professional styling
    if len(roll_cmds_deg) > 0:
        ax3.plot(time_points[:len(roll_cmds_deg)], roll_cmds_deg, color='#d62728', 
                linestyle='-', linewidth=2.5, label='Roll cmd (deg)', alpha=0.9, solid_capstyle='round')
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Roll cmd (deg)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.legend(fontsize=11)

    # 4) Heading error with professional styling
    time_points_error = np.arange(len(heading_errors)) * env.dt
    ax4.plot(time_points_error, heading_errors, color='#1f77b4', linestyle='-', 
             linewidth=2.5, label='Heading Error (deg)', alpha=0.9, solid_capstyle='round')
    ax4.axhline(y=0, color='#2c2c2c', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Heading Error (deg)', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        # Save with high quality for journal publication
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', format='png', transparent=False)
        plt.close()
        
        # Also create standalone path plot
        path_only_save_path = save_path.replace('.png', '_path_only.png')
        plot_path_only(env, xs, ys, f"{env.path_type.upper()} Path - {info.get('path_type', 'Unknown')}", path_only_save_path)
    else:
        plt.show()

    # NEW: return stats for console logging
    out = dict(info)
    out["total_reward"] = total_reward
    out["avg_dist_to_path"] = avg_dist_to_path
    out["oscillation_index"] = oscillation_index
    out["deviation_reward_correlation"] = deviation_reward_correlation
    out["episode_length"] = episode_length
    out["path_efficiency"] = path_efficiency
    out["ideal_path_length"] = ideal_path_length
    out["actual_path_length"] = actual_path_length
    out["path_type"] = env.path_type
    return out

def plot_path_only(env, xs, ys, title, save_path=None):
    """Create a standalone path plot showing only trajectory and legend"""
    plt.figure(figsize=(12, 8))
    
    # Plot desired path
    plt.plot(env.path_x_array, env.path_y_array, color='#2c2c2c', linestyle='--', 
             linewidth=3, label='Desired Path', alpha=0.8, solid_capstyle='round')
    
    # Plot agent path
    plt.plot(xs, ys, color='#d62728', linestyle='-', linewidth=2.5, 
             label='Agent Path', alpha=0.9, solid_capstyle='round')
    
    # Mark start and goal
    plt.plot(env.x0, env.y0, marker='*', color='#2ca02c', markersize=16, 
             label='Start', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
    plt.plot(env.x_end, env.y_end, marker='X', color='#d62728', markersize=12, 
             label='Goal', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
    
    # Draw checkpoints for waypoint config
    if (hasattr(env, 'checkpoint_positions') and len(env.checkpoint_positions) > 0 and 
        hasattr(env, 'config') and env.config == 'waypoint'):
        for i, (cp_x, cp_y) in enumerate(env.checkpoint_positions):
            # Draw checkpoint circle only (no status indicators)
            cp_circle = Circle((cp_x, cp_y), env.checkpoint_radius, fill=False, 
                             linestyle=':', linewidth=1, alpha=0.4, color='orange')
            plt.gca().add_patch(cp_circle)
            plt.plot(cp_x, cp_y, 'o', color='orange', markersize=8, alpha=0.8)
    
    # Styling
    plt.xlabel('X (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Y (m)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', format='png', transparent=False)
        plt.close()
    else:
        plt.show()

def find_all_models():
    """Find all trained models organized by algorithm and config"""
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
            # Parse: algorithm_fixedwing_config_timestamp.zip
            parts = basename.replace('.zip', '').split('_')
            if len(parts) >= 3:
                algorithm = parts[0].upper()
                # Handle goal_based config (has underscore in name)
                if len(parts) >= 4 and parts[2] == 'goal' and parts[3] == 'based':
                    config = 'goal_based'
                else:
                    config = parts[2]  # waypoint, goal_based, or heuristic
                
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
        # print(f"   Model observation space: {model.observation_space}")
        return model
        
    except Exception as e:
        print(f"   Error loading model: {e}")
        raise

def create_observation_adapter(model_obs_space, env_obs_space):
    """Create observation adapter if needed for different observation spaces"""
    model_shape = model_obs_space.shape[0] if hasattr(model_obs_space, 'shape') else None
    env_shape = env_obs_space.shape[0] if hasattr(env_obs_space, 'shape') else None
    
    # print(f"   Model obs shape: {model_shape}, Environment obs shape: {env_shape}")
    
    if model_shape == env_shape:
        return None  # No adapter needed
    elif model_shape == 3 and env_shape == 4:
        # Model expects 3D obs, env provides 4D - take first 3 elements
        print("   Creating 3D observation adapter (taking first 3 elements)")
        return lambda obs: obs[:3]
    elif model_shape == 4 and env_shape == 3:
        # Model expects 4D obs, env provides 3D - pad with zeros
        print("   Creating 4D observation adapter (padding with zeros)")
        return lambda obs: np.concatenate([obs, [0.0]])
    else:
        print(f"   Warning: Incompatible observation spaces - Model: {model_shape}D, Env: {env_shape}D")
        return None

def run_validation_unseen_path(model_path, config, path_type, num_episodes=1,
                               wind_enabled=False, wind_gust_magnitude=2.0, 
                               wind_gust_duration=10.0, wind_transition_time=1.0):
    """Run validation for a specific model on unseen path"""
    env = UnseenPathEnv(config=config, path_type=path_type, 
                       wind_enabled=wind_enabled,
                       wind_speed_mean=wind_gust_magnitude,  # Used as gust magnitude
                       wind_speed_std=wind_gust_duration,    # Used as gust duration
                       wind_heading_std=wind_transition_time)  # Used as transition time
    # print(f"   Environment observation space: {env.observation_space}")
    # print(f"   🆕 Testing on UNSEEN PATH: {path_type.upper()}")
    
    model = load_model(model_path)
    
    # Create observation adapter if needed
    obs_adapter = create_observation_adapter(model.observation_space, env.observation_space)
    
    vecnorm_stats = None
    # Load VecNormalize stats for all algorithms (PPO, SAC, TD3 all use VecNormalize)
    basename = os.path.basename(model_path)
    algorithm = basename.split('_')[0].lower()
    if algorithm in ['ppo', 'sac', 'td3']:
        stats_path = model_path.replace('.zip', '') + "_vn.pkl"
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
                print(f"   ✅ Loaded VecNormalize stats for {algorithm.upper()}")
            except Exception as e:
                print(f"   Warning: failed to load VecNormalize stats: {e}")
        else:
            print(f"   ⚠️  No VecNormalize stats found for {algorithm.upper()}")
    
    # Extract algorithm and config from path for naming
    basename = os.path.basename(model_path).replace('.zip', '')
    algorithm = basename.split('_')[0].upper()
    
    results_dir = f"validation_results_paths/{algorithm.lower()}_{config}_{path_type}"
    os.makedirs(results_dir, exist_ok=True)
    
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

    print(f"\n{'='*60}")
    print(f"Validating {algorithm} {config.upper()} on {path_type.upper()} - {num_episodes} episode")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        start_time = time.time()
        info = visualize_episode_unseen_path(
            env,
            model,
            episode=episode,
            save_path=f"{results_dir}/episode_{episode+1}.png",
            vecnorm_stats=vecnorm_stats,
            obs_adapter=obs_adapter,
        )
        episode_time = time.time() - start_time
        if info["termination_reason"] == "success":
            success_count += 1

        path_errors.append(info["dist_to_path_abs"])
        heading_errors.append(info["heading_diff_deg"])
        total_rewards.append(info.get("total_reward", np.nan))
        avg_dpaths.append(info.get("avg_dist_to_path", np.nan))
        oscillation_indices.append(info.get("oscillation_index", np.nan))
        deviation_correlations.append(info.get("deviation_reward_correlation", np.nan))
        episode_lengths.append(info.get("episode_length", np.nan))
        path_efficiencies.append(info.get("path_efficiency", np.nan))
        completion_times.append(episode_time)

        # print(f"Episode {episode+1}: reason={info['termination_reason']}, "
        #       f"final path err={info['dist_to_path_abs']:.2f} m, "
        #       f"final heading err={info['heading_diff_deg']:.2f} deg, "
        #       f"avg dist-to-path={info.get('avg_dist_to_path', float('nan')):.2f} m, "
        #       f"total reward={info.get('total_reward', float('nan')):.2f}, "
        #       f"oscillation={info.get('oscillation_index', float('nan')):.3f}, "
        #       f"dev-corr={info.get('deviation_reward_correlation', float('nan')):.3f}, "
        #       f"path-eff={info.get('path_efficiency', float('nan')):.3f}, "
        #       f"steps={info.get('episode_length', 0)}, "
        #       f"time={episode_time:.2f}s")

    # Summary statistics
    success_rate_normalized = success_count/num_episodes  # [0, 1]
    avg_reward = np.mean(total_rewards)
    
    summary = {
        'algorithm': algorithm,
        'config': config,
        'path_type': path_type,
        'success_rate': success_count/num_episodes*100,
        'avg_path_error': np.mean(path_errors),
        'avg_heading_error': np.mean(heading_errors),
        'avg_dist_to_path': np.mean(avg_dpaths),
        'avg_total_reward': avg_reward,
        'avg_completion_time': np.mean(completion_times),
        'avg_oscillation_index': np.mean(oscillation_indices),
        'avg_deviation_correlation': np.mean(deviation_correlations),
        'avg_episode_length': np.mean(episode_lengths),
        'episode_length_std': np.std(episode_lengths),
        'avg_path_efficiency': np.mean(path_efficiencies)
    }

    # print(f"\n{algorithm} {config.upper()} {path_type.upper()} Summary:")
    # print(f"Success rate: {summary['success_rate']:.1f}%")
    # print(f"Avg path error: {summary['avg_path_error']:.2f} m")
    # print(f"Avg total reward: {summary['avg_total_reward']:.2f}")
    # print(f"Oscillation Index (OI): {summary['avg_oscillation_index']:.3f}")
    # print(f"Deviation-Reward Correlation (DRC): {summary['avg_deviation_correlation']:.3f}")
    # print(f"Path Efficiency (PE): {summary['avg_path_efficiency']:.3f}")
    # print(f"Avg episode length: {summary['avg_episode_length']:.1f}±{summary['episode_length_std']:.1f} steps")

    # Save summary
    with open(f"{results_dir}/summary_{path_type}.txt", "w") as f:
        f.write(f"{algorithm} {config.upper()} {path_type.upper()} Validation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Path Type: {path_type}\n")
        f.write(f"Success rate: {summary['success_rate']:.1f}%\n")
        f.write(f"Average FINAL path error: {summary['avg_path_error']:.2f} m\n")
        f.write(f"Average FINAL heading error: {summary['avg_heading_error']:.2f} deg\n")
        f.write(f"Average EPISODE avg dist-to-path: {summary['avg_dist_to_path']:.2f} m\n")
        f.write(f"Average total reward: {summary['avg_total_reward']:.2f}\n")
        f.write(f"Average completion time: {summary['avg_completion_time']:.2f} s\n")
        f.write(f"Oscillation Index (OI): {summary['avg_oscillation_index']:.3f}\n")
        f.write(f"Deviation-Reward Correlation (DRC): {summary['avg_deviation_correlation']:.3f}\n")
        f.write(f"Path Efficiency (PE): {summary['avg_path_efficiency']:.3f}\n")
        f.write(f"Average episode length: {summary['avg_episode_length']:.1f} ± {summary['episode_length_std']:.1f} steps\n")
    
    return summary

def validate_all_models_unseen_paths(num_episodes=1, wind_enabled=False, 
                                     wind_gust_magnitude=2.0, wind_gust_duration=10.0, 
                                     wind_transition_time=1.0):
    """Validate all trained models on various unseen paths"""
    
    # =============================================================================
    # EASY TOGGLE SYSTEM - Comment out what you don't want to test
    # =============================================================================
    
    # Algorithms to test (comment out any you don't want)
    ALGORITHMS_TO_TEST = [
        'PPO',
        'SAC', 
        'TD3'
    ]
    
    # Configurations to test (comment out any you don't want)
    CONFIGS_TO_TEST = [
        'waypoint',
        'heuristic',
        'goal_based'
    ]
    
    # Path types to test (comment out any you don't want)
    PATH_TYPES_TO_TEST = [
        #'circular',
        'figure8',
        'spiral'
    ]
    
    # =============================================================================
    
    models = find_all_models()
    
    if not models:
        print("No trained models found! Please run training first.")
        return
    
    # Use the toggle settings
    path_types = PATH_TYPES_TO_TEST
    
    os.makedirs("validation_results_paths", exist_ok=True)
    
    # Print wind configuration
    if wind_enabled:
        print(f"🌬️  Wind gusts enabled: magnitude={wind_gust_magnitude} m/s, duration={wind_gust_duration}s, transition={wind_transition_time}s")
    else:
        print("🌤️  Wind disabled for validation")
    
    all_results = []
    
    # Validate each model on each path type using toggle settings
    for algorithm in models:
        # Skip algorithms not in the toggle list
        if algorithm not in ALGORITHMS_TO_TEST:
            print(f"\n⚠️  Skipping {algorithm}: not in ALGORITHMS_TO_TEST")
            continue
            
        for config in models[algorithm]:
            # Skip configs not in the toggle list
            if config not in CONFIGS_TO_TEST:
                print(f"\n⚠️  Skipping {algorithm} {config}: not in CONFIGS_TO_TEST")
                continue
                
            model_path = models[algorithm][config]
            # print(f"\n🔍 Found {algorithm} {config}: {model_path}")
            
            for path_type in PATH_TYPES_TO_TEST:
                try:
                    # print(f"\n{'='*80}")
                    # print(f"Testing {algorithm} {config.upper()} on {path_type.upper()} path")
                    # print(f"{'='*80}")
                    
                    summary = run_validation_unseen_path(model_path, config, path_type, num_episodes,
                                                            wind_enabled=wind_enabled, 
                                                            wind_gust_magnitude=wind_gust_magnitude,
                                                            wind_gust_duration=wind_gust_duration, 
                                                            wind_transition_time=wind_transition_time)
                    all_results.append(summary)
                except Exception as e:
                    print(f"❌ Failed to validate {algorithm} {config} on {path_type}: {e}")
    
    # Create comparison summary
    print(f"\n{'='*150}")
    print("UNSEEN PATHS VALIDATION COMPARISON SUMMARY")
    print(f"{'='*150}")
    print(f"{'Algorithm':<8} {'Config':<8} {'Path':<10} {'Success%':<10} {'Avg Reward':<12} {'Dist Path':<10} {'PE':<8} {'OI':<8} {'DRC':<8}")
    print(f"{'-'*120}")
    
    for result in all_results:
        # Display config name with space conservation
        if result['config'] == "goal_based":
            config_display = "goal_based"
        elif result['config'] == "heuristic":
            config_display = "heuristic"
        else:
            config_display = result['config']
        print(f"{result['algorithm']:<8} {config_display:<8} {result['path_type']:<10} "
              f"{result['success_rate']:<10.1f} {result['avg_total_reward']:<12.2f} "
              f"{result['avg_dist_to_path']:<10.2f} {result['avg_path_efficiency']:<8.3f} "
              f"{result['avg_oscillation_index']:<8.3f} {result['avg_deviation_correlation']:<8.3f}")
    
    # Save overall comparison
    with open("validation_results_paths/comparison_summary_paths.txt", "w") as f:
        f.write("UNSEEN PATHS Model Comparison Summary\n")
        f.write("="*150 + "\n")
        f.write("Performance Metrics (on never-before-seen paths):\n")
        f.write("- Success%: Percentage of episodes reaching goal successfully\n")
        f.write("- Avg Reward: Average total reward per episode\n")
        f.write("- Dist Path: Average distance from desired path (meters)\n")
        f.write("Performance Metrics:\n")
        f.write("- PE (Path Efficiency) = L_ideal / L_actual (ratio of ideal to actual path length)\n")
        f.write("Reward Hacking Detection Metrics:\n")
        f.write("- OI (Oscillation Index) = Control action variance + change frequency\n")
        f.write("- DRC (Deviation-Reward Correlation) = Max correlation between rewards and distances\n")
        f.write("- Higher PE indicates more efficient path following (closer to ideal path)\n")
        f.write("- Higher OI suggests zig-zag or bang-bang control policies\n")
        f.write("- Positive DRC suggests high rewards while far from path/goal (suspicious)\n")
        f.write("="*120 + "\n")
        f.write(f"{'Algorithm':<8} {'Config':<8} {'Path':<10} {'Success%':<10} {'Avg Reward':<12} {'Dist Path':<10} {'PE':<8} {'OI':<8} {'DRC':<8}\n")
        f.write("-"*120 + "\n")
        for result in all_results:
            # Display config name with space conservation
            if result['config'] == "goal_based":
                config_display = "goal_based"
            elif result['config'] == "heuristic":
                config_display = "heuristic"
            else:
                config_display = result['config']
            f.write(f"{result['algorithm']:<8} {config_display:<8} {result['path_type']:<10} "
                   f"{result['success_rate']:<10.1f} {result['avg_total_reward']:<12.2f} "
                   f"{result['avg_dist_to_path']:<10.2f} {result['avg_path_efficiency']:<8.3f} "
                   f"{result['avg_oscillation_index']:<8.3f} {result['avg_deviation_correlation']:<8.3f}\n")
    
    return all_results

def plot_algorithm_comparison(algorithm_results, path_type, config, save_path=None):
    """Plot all algorithms on the same trajectory plot for comparison"""
    plt.figure(figsize=(12, 8))
    
    # Color scheme for different algorithms
    colors = {
        'PPO': '#d62728',      # Red
        'SAC': '#2ca02c',      # Green  
        'TD3': '#1f77b4'       # Blue
    }
    
    # Create environment to get the desired path
    env = UnseenPathEnv(config=config, path_type=path_type)
    env.reset()  # Generate the path
    
    # Plot desired path first
    plt.plot(env.path_x_array, env.path_y_array, color='#2c2c2c', linestyle='--', 
             linewidth=3, label='Desired Path', alpha=0.8, solid_capstyle='round')
    
    # Mark start and goal
    plt.plot(env.x0, env.y0, marker='*', color='#2ca02c', markersize=16, 
             label='Start', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
    plt.plot(env.x_end, env.y_end, marker='X', color='#d62728', markersize=12, 
             label='Goal', markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
    
    # Plot each algorithm's trajectory by re-running episodes
    for result in algorithm_results:
        algorithm = result['algorithm']
        color = colors.get(algorithm, '#666666')
        
        try:
            # Load the model and run a single episode to get trajectory
            model_path = f"trained_models/{algorithm.lower()}_fixedwing_{config}_*.zip"
            import glob
            model_files = glob.glob(model_path)
            if model_files:
                latest_model = max(model_files, key=os.path.getmtime)
                model = load_model(latest_model)
                
                # Create environment for this algorithm
                test_env = UnseenPathEnv(config=config, path_type=path_type)
                obs_adapter = create_observation_adapter(model.observation_space, test_env.observation_space)
                
                # Load VecNormalize stats if available
                vecnorm_stats = None
                stats_path = latest_model.replace('.zip', '') + "_vn.pkl"
                if os.path.exists(stats_path):
                    tmp_venv = DummyVecEnv([lambda: UnseenPathEnv(config=config, path_type=path_type)])
                    try:
                        vecnorm_stats = VecNormalize.load(stats_path, tmp_venv)
                        vecnorm_stats.training = False
                        vecnorm_stats.norm_reward = False
                    except:
                        pass
                
                # Run episode to get trajectory
                obs, _ = test_env.reset()
                if obs_adapter is not None:
                    obs = obs_adapter(obs)
                obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
                
                xs = [test_env.x]
                ys = [test_env.y]
                done = False
                
                while not done:
                    action, _ = model.predict(obs_for_policy, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    if obs_adapter is not None:
                        obs = obs_adapter(obs)
                    obs_for_policy = _normalize_obs_if_needed(obs, vecnorm_stats)
                    done = terminated or truncated
                    xs.append(test_env.x)
                    ys.append(test_env.y)
                
                # Plot this algorithm's trajectory
                plt.plot(xs, ys, color=color, linewidth=2.5, 
                        label=f'{algorithm} Path', alpha=0.9, solid_capstyle='round')
                
        except Exception as e:
            print(f"Warning: Could not load trajectory for {algorithm}: {e}")
            # Plot empty line for legend
            plt.plot([], [], color=color, linewidth=2.5, 
                    label=f'{algorithm} Path (Failed)', alpha=0.9, solid_capstyle='round')
    
    # Styling
    plt.xlabel('X (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Y (m)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png', transparent=False)
        plt.close()
    else:
        plt.show()

def create_comparison_plots(all_results):
    """Create comparison plots for all algorithm-path combinations"""
    # Group results by path_type and config
    grouped_results = {}
    for result in all_results:
        key = (result['path_type'], result['config'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Create comparison plot for each path type
    for (path_type, config), results in grouped_results.items():
        if len(results) > 1:  # Only create comparison if we have multiple algorithms
            save_path = f"validation_results_paths/{config}_{path_type}_comparison.png"
            plot_algorithm_comparison(results, path_type, config, save_path)
            print(f"✅ Saved algorithm comparison: {save_path}")

def main():
    """Main validation function - validates all trained models on unseen paths"""
    print("🚀 Starting UNSEEN PATHS validation of all trained models...")
    print("This will test models on completely different paths:")
    print("  - Circular trajectories")
    print("  - Figure-8 patterns") 
    print("  - Spiral paths")
    print("  - Zigzag patterns")
    print("  - Straight lines")
    print("This evaluates generalization to different path types.")
    
    try:
        # Validate all models and show comparison
        results = validate_all_models_unseen_paths(num_episodes=1,
                                                        wind_enabled=True,
                                                        wind_gust_magnitude=4.0,
                                                        wind_gust_duration=10.0,
                                                        wind_transition_time=1.0)
        
        if results:
            print(f"\n✅ UNSEEN PATHS validation complete! Results saved in validation_results_paths/")
            print(f"📊 {len(results)} model-path combinations validated")
            
            # Create comparison plots
            print(f"\n📈 Creating algorithm comparison plots...")
            create_comparison_plots(results)
            
            # Show plots for each model (you can comment this out if too many)
            print(f"\n📈 Individual episode plots saved in validation_results_paths/")
            
        else:
            print("❌ No models found to validate. Please run training first.")
            
    except Exception as e:
        print(f"❌ UNSEEN PATHS validation failed: {e}")

if __name__ == "__main__":
    main()
