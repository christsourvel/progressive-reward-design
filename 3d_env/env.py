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
    
    Configuration:
    - waypoint: Sequential checkpoint system with path following rewards
    
    Action: Normalized roll and pitch rate changes [-1.0, +1.0] x [-1.0, +1.0]
    Observation: [range_to_next_cp, bearing_to_next_cp, cross_track_error, prev_roll_cmd, prev_pitch_cmd]
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 config="waypoint",
                 dt=0.1,
                 speed=20.0,
                 max_roll_deg=45.0,
                 path_len_m=1000.0,
                 goal_center_radius_m=25.0,
                 bounds=((-500.0, 2000.0), (-500.0, 500.0)),
                 g=9.81,
                 wind_enabled=False,
                 wind_speed_mean=1.0,
                 wind_speed_std=0.5,
                 wind_heading_std=15.0
                 ):
        super().__init__()

        # Configuration validation
        self.config = config.lower()
        if self.config != "waypoint":
            raise ValueError(f"Config must be 'waypoint', got '{config}'")

        # === PHYSICS PARAMETERS ===
        self.dt = float(dt)
        self.speed = float(speed)
        self.max_roll_deg = float(max_roll_deg)
        self.max_roll_rad = np.deg2rad(self.max_roll_deg)
        self.max_pitch_deg = 30.0  # pitch angle limit
        self.max_pitch_rad = np.deg2rad(self.max_pitch_deg)
        self.max_altitude_m = 250.0  # maximum altitude (increased for safety margin)
        self.min_altitude_m = 50.0   # minimum altitude (path starts at 100m)
        self.g = float(g)
        self.path_len_m = float(path_len_m)  # Used for compatibility, actual path bounds calculated in reset()
        self.goal_center_radius_m = float(goal_center_radius_m)
        # Actual path bounds (set in reset())
        self.actual_path_x_min = 0.0
        self.actual_path_x_max = 1000.0
        self.max_roll_rate_deg_per_step = 2.0  # realistic roll rate limit
        self.max_pitch_rate_deg_per_step = 0.4  # pitch rate limit (reduced for stability)

        # === WIND PARAMETERS ===
        self.wind_enabled = bool(wind_enabled)
        self.wind_speed_mean = float(wind_speed_mean)  # m/s (used as gust magnitude)
        self.wind_speed_std = float(wind_speed_std)  # seconds (used as gust duration)
        self.wind_heading_std = float(wind_heading_std)  # seconds (used as transition time)
        
        # Initialize wind gust model
        if self.wind_enabled:
            self.wind_model = WindGustModel(
                seed=None,  # Will be set in reset()
                wind_gust_magnitude=self.wind_speed_mean,
                wind_gust_duration=self.wind_speed_std,  # Now using the actual parameter!
                wind_transition_time=self.wind_heading_std  # Now using the actual parameter!
            )
        else:
            self.wind_model = None

        # === REWARD PARAMETERS ===
        self.alpha_delta_d = 0.1  # distance-change shaping
        self.beta_path = 1.0  # path deviation penalty
        self.path_following_reward = 2.0
        self.path_deviation_threshold = 15.0
        self.forward_progress_reward = 0.01
        self.checkpoint_reward = 25.0
        self.checkpoint_radius = 10.0
        self.num_checkpoints = 16

        # === ENVIRONMENT BOUNDS ===
        # Default bounds adjusted for incline (600m) + circle (radius 200m)
        # X: circle goes from 400 to 800, so -100 to 900 with margin
        # Y: circle center at 200, radius 200, so Y goes from 0 to 400, use -100 to 500 with margin
        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]
        self.max_steps = 1300

        # === ACTION/OBSERVATION SPACES ===
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Observation space for waypoint configuration
        # [range_to_next_cp, bearing_to_next_cp, cross_track_error, altitude_error, prev_roll_cmd, prev_pitch_cmd]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -180.0, -np.inf, -np.inf, -1.0, -1.0], dtype=np.float32),
            high=np.array([np.inf, 180.0, np.inf, np.inf, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # === PATH PARAMETERS ===
        self.sine_amplitude = 60.0  # Reduced amplitude for gentler curves
        self.sine_frequency = 0.8   # Reduced frequency for fewer turns

        # === STATE VARIABLES ===
        self.x = self.y = self.z = self.heading_deg = 0.0
        self.pitch_deg = 0.0  # pitch angle
        self.current_step = 0
        self.x0 = self.y0 = self.z0 = self.x_end = self.y_end = self.z_end = 0.0
        self.path_x_array = self.path_y_array = self.path_z_array = None
        self.current_speed = self.speed
        
        # Tracking variables
        self._d_prev = 0.0
        self._along_track_prev = 0.0
        self._prev_cmd = np.array([0.0, 0.0])  # [roll, pitch]
        self._prev_prev_action = np.array([0.0, 0.0])
        self._prev_roll_rate = 0.0
        self._prev_pitch_rate = 0.0
        
        # Checkpoint system
        self.checkpoint_positions = []
        self.checkpoints_reached = []
        self.next_checkpoint_idx = 0

    # === HELPER METHODS ===
    def _goal_dx_dy(self):
        """Get distance to goal."""
        return (self.x_end - self.x, self.y_end - self.y)

    def _cross_and_along_track(self):
        """Calculate 3D cross-track error and along-track distance."""
        # Find the closest point on the path to the agent's current position
        agent_pos = np.array([self.x, self.y, self.z])
        
        # Calculate distances to all path points
        path_points = np.column_stack([self.path_x_array, self.path_y_array, self.path_z_array])
        distances = np.linalg.norm(path_points - agent_pos, axis=1)
        
        # Find the index of the closest point
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Get the closest point coordinates
        closest_x = self.path_x_array[closest_idx]
        closest_y = self.path_y_array[closest_idx]
        closest_z = self.path_z_array[closest_idx]
        
        # Calculate tangent at the closest point using neighboring points
        # Use a few points ahead and behind for better tangent estimation
        tangent_window = 5  # points to use for tangent calculation
        
        if closest_idx + tangent_window < len(self.path_x_array):
            # Use forward difference
            ahead_idx = closest_idx + tangent_window
            tangent_vec = np.array([
                self.path_x_array[ahead_idx] - closest_x,
                self.path_y_array[ahead_idx] - closest_y,
                self.path_z_array[ahead_idx] - closest_z
            ])
        elif closest_idx - tangent_window >= 0:
            # Use backward difference
            behind_idx = closest_idx - tangent_window
            tangent_vec = np.array([
                closest_x - self.path_x_array[behind_idx],
                closest_y - self.path_y_array[behind_idx],
                closest_z - self.path_z_array[behind_idx]
            ])
        else:
            # At the very start or end, use whatever is available
            if closest_idx > 0:
                tangent_vec = np.array([
                    closest_x - self.path_x_array[closest_idx - 1],
                    closest_y - self.path_y_array[closest_idx - 1],
                    closest_z - self.path_z_array[closest_idx - 1]
                ])
            elif closest_idx < len(self.path_x_array) - 1:
                tangent_vec = np.array([
                    self.path_x_array[closest_idx + 1] - closest_x,
                    self.path_y_array[closest_idx + 1] - closest_y,
                    self.path_z_array[closest_idx + 1] - closest_z
                ])
            else:
                tangent_vec = np.array([1.0, 0.0, 0.0])
        
        # Normalize tangent vector
        tangent_norm = np.linalg.norm(tangent_vec)
        if tangent_norm > 1e-6:
            tangent = tangent_vec / tangent_norm
        else:
            tangent = np.array([1.0, 0.0, 0.0])
        
        # Vector from closest path point to agent
        to_agent = agent_pos - np.array([closest_x, closest_y, closest_z])
        
        # Calculate signed cross-track error
        # Project to_agent onto the tangent, then get perpendicular component
        projection_length = np.dot(to_agent, tangent)
        perpendicular_vec = to_agent - projection_length * tangent
        cross_track_distance = np.linalg.norm(perpendicular_vec)
        
        # Determine sign using cross product in horizontal plane
        # Positive = right of path, negative = left of path
        tangent_2d = tangent[:2]  # x, y components
        to_agent_2d = to_agent[:2]
        
        if np.linalg.norm(tangent_2d) > 1e-6 and np.linalg.norm(to_agent_2d) > 1e-6:
            # 2D cross product (scalar result)
            cross_2d = tangent_2d[0] * to_agent_2d[1] - tangent_2d[1] * to_agent_2d[0]
            if cross_2d < 0:
                cross_track_distance = -cross_track_distance
        
        # Along-track distance: sum of path lengths up to closest point
        if not hasattr(self, '_path_segment_lengths'):
            # Cache segment lengths for efficiency
            dx = np.diff(self.path_x_array)
            dy = np.diff(self.path_y_array)
            dz = np.diff(self.path_z_array)
            self._path_segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
            self._cumulative_path_length = np.concatenate([[0], np.cumsum(self._path_segment_lengths)])
        
        along_track_m = self._cumulative_path_length[closest_idx]
        
        # Total path length
        path_length = self._cumulative_path_length[-1]
        
        return cross_track_distance, along_track_m, path_length

    def _obs(self):
        """Generate observation for waypoint configuration."""
        # [range_to_next_cp, bearing_to_next_cp, cross_track_error, altitude_error, prev_roll_cmd, prev_pitch_cmd]
        if self.next_checkpoint_idx < len(self.checkpoint_positions):
            cp_x, cp_y, cp_z = self.checkpoint_positions[self.next_checkpoint_idx]
            dx = cp_x - self.x
            dy = cp_y - self.y
            dz = cp_z - self.z
        else:
            dx, dy = self._goal_dx_dy()
            dz = self.z_end - self.z
        
        range_to_target = np.sqrt(dx**2 + dy**2 + dz**2)
        target_angle_deg = np.degrees(np.arctan2(dy, dx))
        bearing_to_target = wrap_deg(target_angle_deg - self.heading_deg)
        signed_ct, _, _ = self._cross_and_along_track()
        
        # Calculate altitude error
        altitude_error = dz
        
        return np.array([range_to_target, bearing_to_target, signed_ct, altitude_error, self._prev_cmd[0], self._prev_cmd[1]], dtype=np.float32)

    def _in_bounds(self):
        """Check if agent is within environment bounds."""
        return ((self.xmin <= self.x <= self.xmax) and 
                (self.ymin <= self.y <= self.ymax) and
                (self.min_altitude_m <= self.z <= self.max_altitude_m))

    # === GYM API ===
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # === BUILD 3D PATH: STRAIGHT + GENTLE DESCENT + INCLINE + CIRCULAR MANEUVER ===
        num_points = 1001
        
        # Path parameters
        straight_length = 300.0    # Length of initial straight segment (increased)
        descent_length = 400.0     # Length of gentle descent segment (increased)
        incline_length = 500.0     # Length of the incline portion (increased)
        circle_radius = 200.0      # Radius of the circular maneuver (reduced to stay within Y bounds)
        altitude_drop = 20.0       # Altitude lost during gentle descent
        altitude_gain = 40.0       # Total altitude gained during incline (reduced from 80m)
        circle_altitude = 100.0 - altitude_drop + altitude_gain  # Constant altitude for circle
        
        # Calculate approximate arc length for the circle portion (270 degrees = 3/2 * pi radians)
        circle_arc_angle = 1.5 * np.pi  # 270 degrees in radians
        circle_arc_length = circle_radius * circle_arc_angle
        
        # Total path length along the trajectory
        total_path_length = straight_length + descent_length + incline_length + circle_arc_length
        
        # Distribute points proportionally
        straight_points = int(num_points * straight_length / total_path_length)
        descent_points = int(num_points * descent_length / total_path_length)
        incline_points = int(num_points * incline_length / total_path_length)
        circle_points = num_points - straight_points - descent_points - incline_points
        
        # Create the path in segments
        self.path_x_array = np.zeros(num_points)
        self.path_y_array = np.zeros(num_points)
        self.path_z_array = np.zeros(num_points)
        
        # Segment 1: Initial straight segment (Y=0, constant Z at 100m)
        self.path_x_array[:straight_points] = np.linspace(0, straight_length, straight_points)
        self.path_y_array[:straight_points] = 0.0
        self.path_z_array[:straight_points] = 100.0
        
        # Segment 2: Gentle descent (Y=0, decreasing Z with smooth curve)
        descent_start_idx = straight_points
        descent_end_idx = straight_points + descent_points
        self.path_x_array[descent_start_idx:descent_end_idx] = np.linspace(straight_length, straight_length + descent_length, descent_points)
        self.path_y_array[descent_start_idx:descent_end_idx] = 0.0
        
        # Smooth descent using a sigmoid-like curve for gradual transition
        descent_x_local = self.path_x_array[descent_start_idx:descent_end_idx] - straight_length
        descent_ratio = descent_x_local / descent_length
        
        # Use smooth curve: starts slow, accelerates in middle, slows at end
        smooth_curve = 3 * descent_ratio**2 - 2 * descent_ratio**3  # Smooth S-curve
        self.path_z_array[descent_start_idx:descent_end_idx] = 100.0 - altitude_drop * smooth_curve
        
        # Segment 3: Gentle incline (straight line with Y=0, increasing Z)
        incline_start_idx = straight_points + descent_points
        incline_end_idx = incline_start_idx + incline_points
        self.path_x_array[incline_start_idx:incline_end_idx] = np.linspace(straight_length + descent_length, straight_length + descent_length + incline_length, incline_points)
        self.path_y_array[incline_start_idx:incline_end_idx] = 0.0
        
        # Smooth incline transition - make ascent much gentler
        incline_x_local = self.path_x_array[incline_start_idx:incline_end_idx] - straight_length - descent_length
        incline_ratio = incline_x_local / incline_length
        
        # Use a much gentler curve for ascent: starts very slow, gradually accelerates
        # This creates a smoother transition from descent to ascent
        incline_smooth_curve = incline_ratio**3  # Cubic curve - much gentler start
        self.path_z_array[incline_start_idx:incline_end_idx] = (100.0 - altitude_drop) + altitude_gain * incline_smooth_curve
        
        # Segment 4: Circular maneuver at constant altitude
        # Circle center is at (straight_length + descent_length + incline_length, circle_radius, circle_altitude)
        circle_start_idx = straight_points + descent_points + incline_points
        circle_center_x = straight_length + descent_length + incline_length
        circle_center_y = circle_radius
        
        # Parametric circle: start at theta = -pi/2 (bottom), go counterclockwise
        # End before completing the full circle (about 3/4 of the circle)
        theta = np.linspace(-np.pi/2, 1.0*np.pi, circle_points)  # About 270 degrees
        
        self.path_x_array[circle_start_idx:] = circle_center_x + circle_radius * np.cos(theta)
        self.path_y_array[circle_start_idx:] = circle_center_y + circle_radius * np.sin(theta)
        self.path_z_array[circle_start_idx:] = circle_altitude
        
        # Calculate actual path bounds for x-clamping
        self.actual_path_x_min = float(np.min(self.path_x_array))
        self.actual_path_x_max = float(np.max(self.path_x_array))
        
        # Add smooth transition from ascent to circle with level flight segment
        transition_points = min(30, incline_points // 8)  # More points for smoother transition
        if transition_points > 0:
            # Create level flight segment between ascent and circle
            level_start = incline_start_idx + incline_points - transition_points // 2
            level_end = circle_start_idx + transition_points // 2
            
            # Ensure we don't exceed array bounds
            level_start = max(incline_start_idx, level_start)
            level_end = min(circle_start_idx + circle_points, level_end)
            
            if level_end > level_start:
                # Create smooth transition using sigmoid-like curve
                for i in range(level_start, level_end):
                    if i < len(self.path_x_array):
                        # Calculate transition weight (0 to 1)
                        if i < incline_start_idx + incline_points:
                            # Still in ascent segment - gradual transition to level
                            transition_weight = (i - level_start) / (incline_start_idx + incline_points - level_start)
                            current_z = self.path_z_array[i]
                            target_z = circle_altitude  # Level flight altitude
                            self.path_z_array[i] = current_z + (target_z - current_z) * transition_weight
                        elif i >= circle_start_idx:
                            # In circle segment - gradual transition from level
                            transition_weight = (i - circle_start_idx) / (level_end - circle_start_idx)
                            # Keep circle altitude (already set correctly)
                            pass
                        else:
                            # In between - maintain level flight
                            self.path_z_array[i] = circle_altitude
        
        # === SET START/END POINTS ===
        self.x0 = self.y0 = 0.0
        self.z0 = 100.0  # Start at baseline altitude
        # Goal should be at the actual end of the generated path
        self.x_end = self.path_x_array[-1]
        self.y_end = self.path_y_array[-1]
        self.z_end = self.path_z_array[-1]
        
        # === CREATE CHECKPOINTS ===
        self.checkpoint_positions = []
        self.checkpoints_reached = []
        checkpoint_indices = np.linspace(0, len(self.path_x_array) - 1, self.num_checkpoints + 1, dtype=int)
        
        # Add all checkpoints from straight, descent, and ascent segments
        # Only skip even-numbered checkpoints in the circular segment
        circle_start_idx = straight_points + descent_points + incline_points
        
        for i in range(self.num_checkpoints):
            checkpoint_idx = checkpoint_indices[i + 1]
            
            # Check if this checkpoint is in the circular segment
            if checkpoint_idx >= circle_start_idx:
                # In circular segment - only add odd-numbered checkpoints
                if (i + 1) % 2 == 1:  # Only odd-numbered checkpoints
                    cp_x = self.path_x_array[checkpoint_idx]
                    cp_y = self.path_y_array[checkpoint_idx]
                    cp_z = self.path_z_array[checkpoint_idx]
                    self.checkpoint_positions.append((cp_x, cp_y, cp_z))
                    self.checkpoints_reached.append(False)
            else:
                # In straight/descent/ascent segments - add all checkpoints
                cp_x = self.path_x_array[checkpoint_idx]
                cp_y = self.path_y_array[checkpoint_idx]
                cp_z = self.path_z_array[checkpoint_idx]
                self.checkpoint_positions.append((cp_x, cp_y, cp_z))
                self.checkpoints_reached.append(False)
        
        
        # Add goal as final checkpoint
        self.checkpoint_positions.append((self.x_end, self.y_end, self.z_end))
        self.checkpoints_reached.append(False)
        self.next_checkpoint_idx = 0
        
        # === INITIALIZE AGENT STATE ===
        self.x = self.x0
        self.y = self.y0
        self.z = self.z0
        
        # Calculate initial heading aligned with path direction
        if len(self.path_x_array) > 1:
            path_dx = self.path_x_array[1] - self.path_x_array[0]
            path_dy = self.path_y_array[1] - self.path_y_array[0]
            path_heading_deg = np.degrees(np.arctan2(path_dy, path_dx))
        else:
            path_heading_deg = 0.0
        
        heading_variation = float(self.np_random.uniform(-5.0, 5.0))
        self.heading_deg = wrap_deg(path_heading_deg + heading_variation)
        self.pitch_deg = 0.0  # Initialize pitch to level flight
        
        # === RESET TRACKING VARIABLES ===
        self.current_step = 0
        self.current_speed = self.speed
        self._d_prev = float(np.hypot(self.x - self.x_end, self.y - self.y_end))
        _, self._along_track_prev, _ = self._cross_and_along_track()
        self._prev_cmd = np.array([0.0, 0.0], dtype=np.float32)     # last applied command [roll_cmd, pitch_cmd]
        self._prev_prev_action = np.array([0.0, 0.0])
        self._prev_roll_rate = 0.0
        self._prev_pitch_rate = 0.0
        
        # === RESET WIND MODEL ===
        if self.wind_enabled and self.wind_model is not None:
            # Re-seed wind model with environment's RNG
            self.wind_model.rng = np.random.default_rng(self.np_random.integers(0, 2**31))
            self.wind_model.reset()
        
        return self._obs(), {}

    def step(self, action):
        """Execute one step in the environment."""
        # === CHECK IF ALREADY AT GOAL ===
        # Early termination check before any movement
        if self.next_checkpoint_idx >= len(self.checkpoint_positions):
            # Already completed all checkpoints, return immediately
            obs = self._obs()
            info = {
                "dist_to_path_abs": 0.0,
                "dist_to_path_signed": 0.0,
                "heading_diff_deg": 0.0,
                "termination_reason": "success",
                "dist_to_goal": 0.0,
                "heading_deg": self.heading_deg,
                "pitch_deg": float(self.pitch_deg),
                "roll_deg": 0.0,
                "altitude_m": float(self.z),
                "current_speed": float(self.current_speed),
                "checkpoints_reached": sum(self.checkpoints_reached),
                "total_checkpoints": len(self.checkpoints_reached),
                "next_checkpoint_idx": self.next_checkpoint_idx,
                "checkpoints_completed": True,
                "roll_rate_deg_per_step": 0.0,
                "pitch_rate_deg_per_step": 0.0,
            }
            return obs, 0.0, True, False, info
        
        # === APPLY ACTION ===
        self.current_speed = self.speed
        
        # Convert action to roll and pitch commands (rate-based control)
        roll_rate_change_deg = float(np.clip(action[0], -1.0, 1.0)) * self.max_roll_rate_deg_per_step
        pitch_rate_change_deg = float(np.clip(action[1], -1.0, 1.0)) * self.max_pitch_rate_deg_per_step
        
        roll_rate_change_cmd = roll_rate_change_deg / self.max_roll_deg
        pitch_rate_change_cmd = pitch_rate_change_deg / self.max_pitch_deg
        
        roll_cmd = np.clip(self._prev_cmd[0] + roll_rate_change_cmd, -1.0, 1.0)
        pitch_cmd = np.clip(self._prev_cmd[1] + pitch_rate_change_cmd, -1.0, 1.0)
        
        roll_rate_deg_per_step = abs(roll_rate_change_deg)
        pitch_rate_deg_per_step = abs(pitch_rate_change_deg)
        
        # Update heading based on roll command
        roll_deg = roll_cmd * self.max_roll_deg
        roll_rad = np.deg2rad(roll_deg)
        heading_rate_rad = (self.g / self.current_speed) * np.tan(roll_rad)
        heading_rate_deg = np.degrees(heading_rate_rad)
        self.heading_deg = wrap_deg(self.heading_deg + heading_rate_deg * self.dt)
        
        # Update pitch angle
        self.pitch_deg = np.clip(self.pitch_deg + pitch_rate_change_deg, -self.max_pitch_deg, self.max_pitch_deg)

        # === UPDATE POSITION ===
        # Cache trigonometric calculations for efficiency
        hdg_rad = np.deg2rad(self.heading_deg)
        pitch_rad = np.deg2rad(self.pitch_deg)
        cos_hdg = np.cos(hdg_rad)
        sin_hdg = np.sin(hdg_rad)
        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        
        # 3D position update (pitch affects forward speed and altitude change)
        forward_velocity = self.current_speed * self.dt * cos_pitch
        vertical_velocity = self.current_speed * self.dt * sin_pitch * np.cos(roll_rad)
        
        self.x += forward_velocity * cos_hdg
        self.y += forward_velocity * sin_hdg
        self.z += vertical_velocity
        
        # === APPLY WIND DISTURBANCE ===
        if self.wind_enabled and self.wind_model is not None:
            wind_x, wind_y = self.wind_model.step(self.dt)
            self.x += wind_x * self.dt
            self.y += wind_y * self.dt
        
        self.current_step += 1

        # === CALCULATE ERRORS ===
        # 3D distance calculation
        goal_vector = np.array([self.x_end - self.x, self.y_end - self.y, self.z_end - self.z])
        dist_to_goal = np.linalg.norm(goal_vector)
        signed_ct, _, path_length = self._cross_and_along_track()
        
        # Calculate heading error using closest point on path
        agent_pos = np.array([self.x, self.y, self.z])
        path_points = np.column_stack([self.path_x_array, self.path_y_array, self.path_z_array])
        distances = np.linalg.norm(path_points - agent_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get tangent at closest point
        tangent_window = 5
        if closest_idx + tangent_window < len(self.path_x_array):
            ahead_idx = closest_idx + tangent_window
            tangent_x = self.path_x_array[ahead_idx] - self.path_x_array[closest_idx]
            tangent_y = self.path_y_array[ahead_idx] - self.path_y_array[closest_idx]
        elif closest_idx - tangent_window >= 0:
            behind_idx = closest_idx - tangent_window
            tangent_x = self.path_x_array[closest_idx] - self.path_x_array[behind_idx]
            tangent_y = self.path_y_array[closest_idx] - self.path_y_array[behind_idx]
        elif closest_idx < len(self.path_x_array) - 1:
            tangent_x = self.path_x_array[closest_idx + 1] - self.path_x_array[closest_idx]
            tangent_y = self.path_y_array[closest_idx + 1] - self.path_y_array[closest_idx]
        else:
            tangent_x = 1.0
            tangent_y = 0.0
        
        desired_heading_deg = np.degrees(np.arctan2(tangent_y, tangent_x))
        heading_err_deg = wrap_deg(self.heading_deg - desired_heading_deg)

        # === CALCULATE REWARD ===
        terminated = False
        truncated = False
        term_reason = "running"
        checkpoint_reached_this_step = None

        reward = 0.0
        
        # 3D Path-following reward with separate horizontal and vertical components
        # Use the closest point on path (already calculated in heading error)
        path_y_at_closest = self.path_y_array[closest_idx]
        path_z_at_closest = self.path_z_array[closest_idx]
        
        # Calculate separate horizontal and vertical errors
        horizontal_error = abs(self.y - path_y_at_closest)
        vertical_error = abs(self.z - path_z_at_closest)
        
        # Weight the errors differently (horizontal more important for path following)
        horizontal_weight = 1.0
        vertical_weight = 1.0  # Less weight for altitude (can be adjusted)
        
        # Combined weighted error
        weighted_error = horizontal_weight * horizontal_error + vertical_weight * vertical_error
        
        # Path-following reward with separate components
        horizontal_reward = self.path_following_reward * 0.4 * np.exp(-horizontal_error / self.path_deviation_threshold)
        vertical_reward = self.path_following_reward * 0.6 * np.exp(-vertical_error / (self.path_deviation_threshold * 0.5))  # More sensitive to altitude
        
        path_reward = horizontal_reward + vertical_reward
        reward += path_reward
        
        # Forward progress reward
        _, along_track_m, _ = self._cross_and_along_track()
        forward_progress = along_track_m - self._along_track_prev
        if forward_progress > 0:
            reward += self.forward_progress_reward * forward_progress
        self._along_track_prev = along_track_m
        
        # Sequential checkpoint rewards (3D distance calculation)
        agent_pos = np.array([self.x, self.y, self.z])
        for i, (cp_x, cp_y, cp_z) in enumerate(self.checkpoint_positions):
            if not self.checkpoints_reached[i]:
                checkpoint_pos = np.array([cp_x, cp_y, cp_z])
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
                        reward -= 30.0  # Strong penalty for wrong checkpoint order
                    break
        
        # Roll rate penalty
        if roll_rate_deg_per_step > 0.05:
            roll_rate_penalty = 1.0 * (roll_rate_deg_per_step - 0.5) ** 2
            reward -= roll_rate_penalty
        
        # Pitch rate penalty (increased penalty for excessive pitch changes)
        if pitch_rate_deg_per_step > 0.05:
            pitch_rate_penalty = 3.0 * (pitch_rate_deg_per_step - 0.4) ** 2
            reward -= pitch_rate_penalty
        
        # Jerk penalty (rate of change of roll rate)
        roll_jerk = abs(roll_rate_deg_per_step - self._prev_roll_rate)
        if roll_jerk > 0.3:
            roll_jerk_penalty = 0.1 * (roll_jerk - 0.3) ** 2
            reward -= roll_jerk_penalty
        
        # Pitch jerk penalty (rate of change of pitch rate)
        pitch_jerk = abs(pitch_rate_deg_per_step - self._prev_pitch_rate)
        if pitch_jerk > 0.2:
            pitch_jerk_penalty = 0.15 * (pitch_jerk - 0.2) ** 2
            reward -= pitch_jerk_penalty
        
        # Success condition
        if not terminated and self.next_checkpoint_idx >= len(self.checkpoint_positions):
            terminated = True
            term_reason = "success"

        # === CHECK TERMINATION CONDITIONS ===
        if not terminated and not self._in_bounds():
            truncated = True
            term_reason = "out_of_bounds"
            reward -= 50.0  # Strong penalty for going out of bounds
        if not terminated and not truncated and (self.current_step >= self.max_steps):
            truncated = True
            term_reason = "max_steps"
            reward -= 25.0  # Penalty for timeout (didn't complete in time)

        # === BUILD INFO DICTIONARY ===
        info = {
            "dist_to_path_abs": abs(signed_ct),
            "dist_to_path_signed": signed_ct,
            "horizontal_error": horizontal_error,
            "vertical_error": vertical_error,
            "weighted_error": weighted_error,
            "heading_diff_deg": heading_err_deg,
            "termination_reason": term_reason,
            "dist_to_goal": dist_to_goal,
            "heading_deg": self.heading_deg,
            "pitch_deg": float(self.pitch_deg),
            "roll_deg": float(roll_deg),
            "altitude_m": float(self.z),
            "current_speed": float(self.current_speed),
            "checkpoints_reached": sum(self.checkpoints_reached),
            "total_checkpoints": len(self.checkpoints_reached),
            "next_checkpoint_idx": self.next_checkpoint_idx,
            "checkpoints_completed": self.next_checkpoint_idx >= len(self.checkpoint_positions),
            "roll_rate_deg_per_step": float(roll_rate_deg_per_step),
            "pitch_rate_deg_per_step": float(pitch_rate_deg_per_step),
        }
        
        if checkpoint_reached_this_step is not None:
            info["checkpoint_reached"] = checkpoint_reached_this_step

        # === UPDATE STATE ===
        obs = self._obs()
        self._d_prev = dist_to_goal
        self._prev_prev_action = self._prev_cmd.copy()
        self._prev_cmd = np.array([roll_cmd, pitch_cmd])       # applied command state (for physics + obs)
        self._prev_roll_rate = roll_rate_deg_per_step
        self._prev_pitch_rate = pitch_rate_deg_per_step
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up environment."""
        pass
