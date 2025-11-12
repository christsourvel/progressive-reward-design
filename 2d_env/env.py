import gymnasium as gym
import numpy as np


def wrap_deg(a):
    """Wrap angle in degrees to [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


class WindGustModel:
    """Smooth, intermittent wind-gust model for UAV simulation."""

    def __init__(self, seed=None, wind_gust_magnitude=8.0, wind_gust_duration=10.0, 
                 wind_transition_time=1.0):
        self.rng = np.random.default_rng(seed)

        # --- Parameters ---
        self.wind_gust_magnitude = wind_gust_magnitude      # m/s peak strength
        self.wind_gust_duration = wind_gust_duration        # seconds gust lasts
        self.wind_transition_time = wind_transition_time    # seconds to ramp up/down
        self.time = 0.0                                     # current simulation time

        # --- Internal state ---
        self.next_gust_time = self.rng.uniform(5.0, 10.0)
        self.current_gust = {"x": 0.0, "y": 0.0}
        self.target_gust  = {"x": 0.0, "y": 0.0}
        self.gust_transition_start = 0.0
        self.gust_end_time = 0.0

    def reset(self):
        """Reset the wind model to initial state."""
        self.time = 0.0
        self.next_gust_time = self.rng.uniform(5.0, 10.0)
        self.current_gust = {"x": 0.0, "y": 0.0}
        self.target_gust  = {"x": 0.0, "y": 0.0}
        self.gust_transition_start = 0.0
        self.gust_end_time = 0.0

    def step(self, dt=0.1):
        """Advance the wind model by one timestep."""
        self.time += dt

        # 1️⃣ Start a new gust
        if self.time >= self.next_gust_time:
            gust_dir = self.rng.uniform(0, 2 * np.pi)
            gust_mag = self.wind_gust_magnitude
            self.target_gust = {
                "x": gust_mag * np.cos(gust_dir),
                "y": gust_mag * np.sin(gust_dir)
            }
            self.gust_transition_start = self.time
            self.gust_end_time = self.time + self.wind_gust_duration
            self.next_gust_time = self.gust_end_time + self.rng.uniform(10.0, 20.0)

        # 2️⃣ End the current gust (fade out)
        elif self.time >= self.gust_end_time and (
            self.current_gust["x"] != 0.0 or self.current_gust["y"] != 0.0
        ):
            self.target_gust = {"x": 0.0, "y": 0.0}
            self.gust_transition_start = self.time

        # 3️⃣ Smooth transition (cosine ramp)
        if self.time < self.gust_transition_start + self.wind_transition_time:
            progress = (self.time - self.gust_transition_start) / self.wind_transition_time
            progress = np.clip(progress, 0.0, 1.0)
            ease = 0.5 * (1 - np.cos(np.pi * progress))
            self.current_gust["x"] = (1 - ease) * self.current_gust["x"] + ease * self.target_gust["x"]
            self.current_gust["y"] = (1 - ease) * self.current_gust["y"] + ease * self.target_gust["y"]

        return self.current_gust["x"], self.current_gust["y"]


class FixedWingUAVEnv(gym.Env):
    """
    Fixed Wing UAV Environment for Reward Hacking Studies
    
    Configurations:
    - waypoint: Sequential checkpoint system with path following rewards
    - goal_based: Basic distance shaping with path penalty
    - Heuristic: Research paper reward structure
    
    Action: Normalized roll rate change [-1.0, +1.0]
    Observation: Varies by configuration (3D or 4D)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 config="waypoint",
                 dt=0.1,
                 speed=20.0,
                 max_roll_deg=45.0,
                 path_len_m=1000.0,
                 goal_center_radius_m=20.0,
                 bounds=((-500.0, 1500.0), (-500.0, 500.0)),
                 g=9.81,
                 wind_enabled=False,
                 wind_speed_mean=2.0,
                 wind_speed_std=0.5,
                 wind_heading_std=15.0
                 ):
        super().__init__()

        # Configuration validation
        self.config = config.lower()
        if self.config not in ["waypoint", "goal_based", "heuristic"]:
            raise ValueError(f"Config must be 'waypoint', 'goal_based', or 'heuristic', got '{config}'")

        # === PHYSICS PARAMETERS ===
        self.dt = float(dt)
        self.speed = float(speed)
        self.max_roll_deg = float(max_roll_deg)
        self.max_roll_rad = np.deg2rad(self.max_roll_deg)
        self.g = float(g)
        self.path_len_m = float(path_len_m)
        self.goal_center_radius_m = float(goal_center_radius_m)
        self.max_roll_rate_deg_per_step = 2.0  # realistic roll rate limit

        # === WIND PARAMETERS ===
        self.wind_enabled = bool(wind_enabled)
        self.wind_speed_mean = float(wind_speed_mean)  # m/s (used as gust magnitude)
        self.wind_speed_std = float(wind_speed_std)  # m/s (unused in gust model)
        self.wind_heading_std = float(wind_heading_std)  # degrees (unused in gust model)
        
        # Initialize wind gust model
        if self.wind_enabled:
            self.wind_model = WindGustModel(
                seed=None,  # Will be set in reset()
                wind_gust_magnitude=self.wind_speed_mean,
                wind_gust_duration=10.0,
                wind_transition_time=1.0
            )
        else:
            self.wind_model = None

        # === REWARD PARAMETERS ===
        self.alpha_delta_d = 0.1  # distance-change shaping
        self.beta_path = 1.0  # path deviation penalty
        self.path_following_reward = 0.5
        self.path_deviation_threshold = 50.0
        self.forward_progress_reward = 0.05
        self.checkpoint_reward = 25.0
        self.checkpoint_radius = 25.0
        self.num_checkpoints = 16

        # === ENVIRONMENT BOUNDS ===
        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]
        self.max_steps = 1000

        # === ACTION/OBSERVATION SPACES ===
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Observation space varies by configuration
        if self.config == "waypoint":
            # [range_to_next_cp, bearing_to_next_cp, cross_track_error, prev_roll_cmd]
            self.observation_space = gym.spaces.Box(
                low=np.array([0.0, -180.0, -np.inf, -1.0], dtype=np.float32),
                high=np.array([np.inf, np.inf, 180.0, np.inf, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.config == "goal_based":
            # [dx, dy, heading_error_deg, cross_track, prev_roll_cmd]
            self.observation_space = gym.spaces.Box(
                low=np.array([-np.inf, -np.inf, -180.0, -np.inf, -1.0], dtype=np.float32),
                high=np.array([np.inf, np.inf, 180.0, np.inf, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.config == "heuristic":
            # [eψ (deg), eCT (m), Δϕcmd]
            self.observation_space = gym.spaces.Box(
                low=np.array([-180.0, -np.inf, -1.0], dtype=np.float32),
                high=np.array([180.0, np.inf, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        # === PATH PARAMETERS ===
        self.sine_amplitude = 120.0
        self.sine_frequency = 1.5

        # === STATE VARIABLES ===
        self.x = self.y = self.heading_deg = 0.0
        self.current_step = 0
        self.x0 = self.y0 = self.x_end = self.y_end = 0.0
        self.path_x_array = self.path_y_array = None
        self.current_speed = self.speed
        
        # Tracking variables
        self._d_prev = 0.0
        self._along_track_prev = 0.0
        self._prev_action = 0.0
        self._prev_prev_action = 0.0
        self._prev_roll_rate = 0.0
        
        # Checkpoint system
        self.checkpoint_positions = []
        self.checkpoints_reached = []
        self.next_checkpoint_idx = 0

    # === HELPER METHODS ===
    def _goal_dx_dy(self):
        """Get distance to goal."""
        return (self.x_end - self.x, self.y_end - self.y)

    def _cross_and_along_track(self):
        """Calculate cross-track error and along-track distance."""
        x_clamped = np.clip(self.x, 0.0, self.path_len_m)
        path_y_at_x = np.interp(x_clamped, self.path_x_array, self.path_y_array)
        
        # Calculate path tangent using finite differences
        dx = 1.0
        if x_clamped + dx <= self.path_len_m:
            y_ahead = np.interp(x_clamped + dx, self.path_x_array, self.path_y_array)
            dy_dx = (y_ahead - path_y_at_x) / dx
        elif x_clamped - dx >= 0.0:
            y_behind = np.interp(x_clamped - dx, self.path_x_array, self.path_y_array)
            dy_dx = (path_y_at_x - y_behind) / dx
        else:
            dy_dx = 0.0
        
        # Vectorized tangent normalization
        tangent = np.array([1.0, dy_dx])
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent /= tangent_norm
        
        # Vectorized cross-track error calculation
        to_agent = np.array([self.x - x_clamped, self.y - path_y_at_x])
        signed_ct = float(np.cross(tangent, to_agent))  # 2D cross product
        
        # Along-track distance (approximation)
        along_track_m = x_clamped
        
        # Cache path length calculation
        if not hasattr(self, '_cached_path_length'):
            self._cached_path_length = self._calculate_total_path_length()
        path_length = self._cached_path_length
        
        return signed_ct, along_track_m, path_length
    
    def _calculate_total_path_length(self):
        """Calculate total path length using vectorized operations."""
        if len(self.path_x_array) < 2:
            return 0.0
        
        # Vectorized calculation: much faster than loop
        dx = np.diff(self.path_x_array)
        dy = np.diff(self.path_y_array)
        segment_lengths = np.hypot(dx, dy)
        return float(np.sum(segment_lengths))

    def _obs(self):
        """Generate observation based on configuration."""
        if self.config == "waypoint":
            # [range_to_next_cp, bearing_to_next_cp, cross_track_error, prev_roll_cmd]
            if self.next_checkpoint_idx < len(self.checkpoint_positions):
                cp_x, cp_y = self.checkpoint_positions[self.next_checkpoint_idx]
                dx = cp_x - self.x
                dy = cp_y - self.y
            else:
                dx, dy = self._goal_dx_dy()
            
            range_to_target = np.hypot(dx, dy)
            target_angle_deg = np.degrees(np.arctan2(dy, dx))
            bearing_to_target = wrap_deg(target_angle_deg - self.heading_deg)
            signed_ct, _, _ = self._cross_and_along_track()
            
            return np.array([range_to_target, bearing_to_target, signed_ct, self._prev_action], dtype=np.float32)
            
        elif self.config == "goal_based":
            # [dx, dy, heading_error_deg, cross_track, prev_roll_cmd]
            dx, dy = self._goal_dx_dy()
            signed_ct, _, _ = self._cross_and_along_track()
            
            # Calculate path tangent heading (same as in step method)
            x_clamped = np.clip(self.x, 0.0, self.path_len_m)
            path_y_at_x = np.interp(x_clamped, self.path_x_array, self.path_y_array)
            
            dx_tangent = 1.0
            if x_clamped + dx_tangent <= self.path_len_m:
                y_ahead = np.interp(x_clamped + dx_tangent, self.path_x_array, self.path_y_array)
                dy_dx = (y_ahead - path_y_at_x) / dx_tangent
            elif x_clamped - dx_tangent >= 0.0:
                y_behind = np.interp(x_clamped - dx_tangent, self.path_x_array, self.path_y_array)
                dy_dx = (path_y_at_x - y_behind) / dx_tangent
            else:
                dy_dx = 0.0
            
            tangent = np.array([1.0, dy_dx])
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 0:
                tangent /= tangent_norm
            
            desired_heading_deg = np.degrees(np.arctan2(tangent[1], tangent[0]))
            heading_error_deg = wrap_deg(self.heading_deg - desired_heading_deg)
            
            return np.array([dx, dy, heading_error_deg, signed_ct, self._prev_action], dtype=np.float32)
            
        elif self.config == "heuristic":
            # [eψ (deg), eCT (m), Δϕcmd]
            signed_ct, _, _ = self._cross_and_along_track()
            
            # Calculate path tangent direction
            x_clamped = np.clip(self.x, 0.0, self.path_len_m)
            path_y_at_x = np.interp(x_clamped, self.path_x_array, self.path_y_array)
            
            dx = 1.0
            if x_clamped + dx <= self.path_len_m:
                y_ahead = np.interp(x_clamped + dx, self.path_x_array, self.path_y_array)
                dy_dx = (y_ahead - path_y_at_x) / dx
            elif x_clamped - dx >= 0.0:
                y_behind = np.interp(x_clamped - dx, self.path_x_array, self.path_y_array)
                dy_dx = (path_y_at_x - y_behind) / dx
            else:
                dy_dx = 0.0
            
            tangent_x = 1.0
            tangent_y = dy_dx
            tangent_norm = np.hypot(tangent_x, tangent_y)
            if tangent_norm > 0:
                tangent_x /= tangent_norm
                tangent_y /= tangent_norm
            
            desired_heading_deg = np.degrees(np.arctan2(tangent_y, tangent_x))
            epsi_deg = wrap_deg(self.heading_deg - desired_heading_deg)
            delta_phi_cmd = np.clip(self._prev_action - self._prev_prev_action, -1.0, 1.0)
            
            return np.array([epsi_deg, signed_ct, delta_phi_cmd], dtype=np.float32)

    def _in_bounds(self):
        """Check if agent is within environment bounds."""
        return (self.xmin <= self.x <= self.xmax) and (self.ymin <= self.y <= self.ymax)

    # === GYM API ===
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # === BUILD SINE WAVE PATH ===
        num_points = 1001
        self.path_x_array = np.linspace(0, self.path_len_m, num_points)
        base_sine = np.sin(2 * np.pi * self.sine_frequency * self.path_x_array / self.path_len_m)
        
        # Randomize amplitude for each peak
        num_peaks = int(2 * self.sine_frequency)
        amplitude_modulation = np.ones(num_points)
        
        for i in range(num_peaks):
            peak_amplitude = float(self.np_random.uniform(30.0, 150.0))
            segment_start = int(i * num_points / num_peaks)
            segment_end = int((i + 1) * num_points / num_peaks)
            
            for j in range(segment_end - segment_start):
                amplitude_modulation[segment_start + j] = peak_amplitude
        
        self.path_y_array = amplitude_modulation * base_sine
        
        # === SET START/END POINTS ===
        self.x0 = self.y0 = 0.0
        self.x_end = self.path_len_m
        self.y_end = 0.0
        
        # === CREATE CHECKPOINTS ===
        self.checkpoint_positions = []
        self.checkpoints_reached = []
        checkpoint_indices = np.linspace(0, len(self.path_x_array) - 1, self.num_checkpoints + 1, dtype=int)
        
        for i in range(self.num_checkpoints):
            checkpoint_idx = checkpoint_indices[i + 1]
            cp_x = self.path_x_array[checkpoint_idx]
            cp_y = self.path_y_array[checkpoint_idx]
            self.checkpoint_positions.append((cp_x, cp_y))
            self.checkpoints_reached.append(False)
        
        # Add goal as final checkpoint
        self.checkpoint_positions.append((self.x_end, self.y_end))
        self.checkpoints_reached.append(False)
        self.next_checkpoint_idx = 0
        
        # === INITIALIZE AGENT STATE ===
        self.x = self.x0
        self.y = self.y0
        
        # Calculate initial heading aligned with path direction
        if len(self.path_x_array) > 1:
            path_dx = self.path_x_array[1] - self.path_x_array[0]
            path_dy = self.path_y_array[1] - self.path_y_array[0]
            path_heading_deg = np.degrees(np.arctan2(path_dy, path_dx))
        else:
            path_heading_deg = 0.0
        
        heading_variation = float(self.np_random.uniform(-5.0, 5.0))
        self.heading_deg = wrap_deg(path_heading_deg + heading_variation)
        
        # === RESET TRACKING VARIABLES ===
        self.current_step = 0
        self.current_speed = self.speed
        self._d_prev = float(np.hypot(self.x - self.x_end, self.y - self.y_end))
        _, self._along_track_prev, _ = self._cross_and_along_track()
        self._prev_action = 0.0
        self._prev_prev_action = 0.0
        self._prev_roll_rate = 0.0
        
        # Reset wind model
        if self.wind_enabled and self.wind_model is not None:
            # Re-initialize with proper seed from environment's RNG
            wind_seed = self.np_random.integers(0, 2**31)
            self.wind_model = WindGustModel(
                seed=wind_seed,
                wind_gust_magnitude=self.wind_speed_mean,
                wind_gust_duration=10.0,
                wind_transition_time=1.0
            )
        
        return self._obs(), {}

    def step(self, action):
        """Execute one step in the environment."""
        # === APPLY ACTION ===
        self.current_speed = self.speed
        
        # Convert action to roll command (rate-based control)
        roll_rate_change_deg = float(np.clip(action[0], -1.0, 1.0)) * self.max_roll_rate_deg_per_step
        roll_rate_change_cmd = roll_rate_change_deg / self.max_roll_deg
        roll_cmd = np.clip(self._prev_action + roll_rate_change_cmd, -1.0, 1.0)
        roll_rate_deg_per_step = abs(roll_rate_change_deg)
        
        # Update heading based on roll command
        roll_deg = roll_cmd * self.max_roll_deg
        roll_rad = np.deg2rad(roll_deg)
        heading_rate_rad = (self.g / self.current_speed) * np.tan(roll_rad)
        heading_rate_deg = np.degrees(heading_rate_rad)
        self.heading_deg = wrap_deg(self.heading_deg + heading_rate_deg * self.dt)

        # === UPDATE POSITION ===
        # Cache trigonometric calculations for efficiency
        hdg_rad = np.deg2rad(self.heading_deg)
        cos_hdg = np.cos(hdg_rad)
        sin_hdg = np.sin(hdg_rad)
        
        # Vectorized position update
        velocity = self.current_speed * self.dt
        self.x += velocity * cos_hdg
        self.y += velocity * sin_hdg
        
        # === APPLY WIND DISTURBANCE ===
        if self.wind_enabled and self.wind_model is not None:
            wind_x, wind_y = self.wind_model.step(self.dt)
            self.x += wind_x * self.dt
            self.y += wind_y * self.dt
        
        self.current_step += 1

        # === CALCULATE ERRORS ===
        # Vectorized distance calculation
        goal_vector = np.array([self.x_end - self.x, self.y_end - self.y])
        dist_to_goal = np.linalg.norm(goal_vector)
        signed_ct, _, path_length = self._cross_and_along_track()
        
        # Calculate heading error using vectorized operations
        x_clamped = np.clip(self.x, 0.0, self.path_len_m)
        path_y_at_x = np.interp(x_clamped, self.path_x_array, self.path_y_array)
        
        dx = 1.0
        if x_clamped + dx <= self.path_len_m:
            y_ahead = np.interp(x_clamped + dx, self.path_x_array, self.path_y_array)
            dy_dx = (y_ahead - path_y_at_x) / dx
        elif x_clamped - dx >= 0.0:
            y_behind = np.interp(x_clamped - dx, self.path_x_array, self.path_y_array)
            dy_dx = (path_y_at_x - y_behind) / dx
        else:
            dy_dx = 0.0
        
        # Vectorized tangent calculation
        tangent = np.array([1.0, dy_dx])
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent /= tangent_norm
        
        desired_heading_deg = np.degrees(np.arctan2(tangent[1], tangent[0]))
        heading_err_deg = wrap_deg(self.heading_deg - desired_heading_deg)

        # === CALCULATE REWARD ===
        terminated = False
        truncated = False
        term_reason = "running"
        checkpoint_reached_this_step = None

        if self.config == "waypoint":
            reward = 0.0
            
            # Path-following reward
            cross_track_error = abs(signed_ct)
            path_reward = self.path_following_reward * np.exp(-cross_track_error / self.path_deviation_threshold)
            reward += path_reward
            
            # Forward progress reward
            _, along_track_m, _ = self._cross_and_along_track()
            forward_progress = along_track_m - self._along_track_prev
            if forward_progress > 0:
                reward += self.forward_progress_reward * forward_progress
            self._along_track_prev = along_track_m
            
            # Sequential checkpoint rewards (vectorized distance calculation)
            agent_pos = np.array([self.x, self.y])
            for i, (cp_x, cp_y) in enumerate(self.checkpoint_positions):
                if not self.checkpoints_reached[i]:
                    checkpoint_pos = np.array([cp_x, cp_y])
                    dist_to_checkpoint = np.linalg.norm(agent_pos - checkpoint_pos)
                    checkpoint_radius = self.goal_center_radius_m if i == len(self.checkpoint_positions) - 1 else self.checkpoint_radius
                    if dist_to_checkpoint <= checkpoint_radius:
                        if i == self.next_checkpoint_idx:
                            reward += self.checkpoint_reward
                            self.checkpoints_reached[i] = True
                            self.next_checkpoint_idx += 1
                            checkpoint_reached_this_step = i + 1
                        else:
                            terminated = True
                            term_reason = "wrong_checkpoint_order"
                            reward -= 10.0
                        break
            
            # Roll rate penalty
            if roll_rate_deg_per_step > 0.5:
                roll_rate_penalty = 0.2 * (roll_rate_deg_per_step - 0.5) ** 2
                reward -= roll_rate_penalty
            
            # Jerk penalty (rate of change of roll rate)
            roll_jerk = abs(roll_rate_deg_per_step - self._prev_roll_rate)
            if roll_jerk > 0.3:
                jerk_penalty = 0.1 * (roll_jerk - 0.3) ** 2
                reward -= jerk_penalty
            
            # Success condition
            if not terminated and self.next_checkpoint_idx >= len(self.checkpoint_positions):
                terminated = True
                term_reason = "success"

        elif self.config == "goal_based":
            # Basic distance shaping + path penalty
            reward = 0.0
            delta_d = self._d_prev - dist_to_goal
            reward += self.alpha_delta_d * delta_d
            
            if path_length > 0.0:
                reward -= self.beta_path * (abs(signed_ct) / path_length)
            
            if dist_to_goal <= self.goal_center_radius_m:
                reward += 10.0
                terminated = True
                term_reason = "success"

        elif self.config == "heuristic":
            # heuristic reward structure
            epsi_rad = np.deg2rad(wrap_deg(self.heading_deg - desired_heading_deg))
            r_heading = -40.0 * (1 - np.cos(epsi_rad))
            r_path = -0.5 * abs(signed_ct)
            r_progress = 5.0
            r_CP = 0.0
            
            # Checkpoint rewards (vectorized distance calculation)
            agent_pos = np.array([self.x, self.y])
            for i, (cp_x, cp_y) in enumerate(self.checkpoint_positions):
                if not self.checkpoints_reached[i]:
                    checkpoint_pos = np.array([cp_x, cp_y])
                    if np.linalg.norm(agent_pos - checkpoint_pos) <= 50.0:
                        r_CP = 200.0
                        self.checkpoints_reached[i] = True
                        break
            
            delta_phi_cmd = roll_cmd - self._prev_action
            r_smooth = -0.1 * (delta_phi_cmd ** 2)
            
            reward = r_heading + r_path + r_progress + r_CP + r_smooth
            
            if all(self.checkpoints_reached):
                reward += 1000.0
                terminated = True
                term_reason = "success"

        # === CHECK TERMINATION CONDITIONS ===
        if not terminated and not self._in_bounds():
            truncated = True
            term_reason = "out_of_bounds"
        if not terminated and not truncated and (self.current_step >= self.max_steps):
            truncated = True
            term_reason = "max_steps"

        # === BUILD INFO DICTIONARY ===
        info = {
            "dist_to_path_abs": abs(signed_ct),
            "dist_to_path_signed": signed_ct,
            "heading_diff_deg": heading_err_deg,
            "termination_reason": term_reason,
            "dist_to_goal": dist_to_goal,
            "heading_deg": self.heading_deg,
            "roll_deg": float(roll_deg),
            "current_speed": float(self.current_speed),
            "checkpoints_reached": sum(self.checkpoints_reached),
            "total_checkpoints": len(self.checkpoints_reached),
            "next_checkpoint_idx": self.next_checkpoint_idx,
            "checkpoints_completed": self.next_checkpoint_idx >= len(self.checkpoint_positions),
            "roll_rate_deg_per_step": float(roll_rate_deg_per_step),
        }
        
        if checkpoint_reached_this_step is not None:
            info["checkpoint_reached"] = checkpoint_reached_this_step

        # === UPDATE STATE ===
        obs = self._obs()
        self._d_prev = dist_to_goal
        self._prev_prev_action = self._prev_action
        self._prev_action = roll_cmd
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up environment."""
        pass
