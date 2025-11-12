import gymnasium as gym
import numpy as np


def wrap_deg(a):
    """Wrap angle in degrees to [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


class FixedWingUAVEnv(gym.Env):
    """
    Fixed Wing UAV Environment for Path Following
    
    Reward structure:
      - Distance shaping (signed) + path penalty
      - Terminates when reaching goal_center_radius_m
      - Success bonus: +10.0

    ACTION LIMITS:
      - Action space: [-1.0, +1.0] (normalized)
      - Action = roll rate change, scaled by max_roll_rate_deg_per_step (2.0°/step)
      - Roll angle limits: ±max_roll_deg (default: ±45.0°)
      - Roll rate limits: ±max_roll_rate_deg_per_step (2.0°/step)
      - Speed: Fixed at default speed (no velocity control)
    
    Observation: [dx, dy, heading_error_deg, cross_track, prev_roll_cmd]
    where heading_error_deg is the angle difference between UAV heading and path direction
    and prev_roll_cmd is the previous roll command (normalized [-1, +1])
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 dt=0.1,
                 speed=20.0,
                 max_roll_deg=45.0,
                 path_len_m=1000.0,
                 goal_radius_m=60.0,
                 goal_center_radius_m=15.0,
                 heading_tol_deg=10.0,
                 tau_step=0.01,
                 start_pos_spread_m=100.0,
                 start_heading_spread_deg=40.0,
                 bounds=((-500.0, 1500.0), (-500.0, 500.0)),
                 g=9.81
                 ):
        super().__init__()

        # --- Task / dynamics ---
        self.dt = float(dt)
        self.speed = float(speed)
        self.max_roll_deg = float(max_roll_deg)
        self.max_roll_rad = np.deg2rad(self.max_roll_deg)
        self.g = float(g)
        self.path_len_m = float(path_len_m)
        self.goal_radius_m = float(goal_radius_m)
        self.goal_center_radius_m = float(goal_center_radius_m)
        self.heading_tol_deg = float(heading_tol_deg)
        self.tau = float(tau_step)

        # Reward shaping coefficients
        self.alpha_delta_d = 0.1  # distance-change shaping
        self.beta_path = 1.0      # path deviation penalty (normalized)
        
        # Roll rate control parameters (same as RH2 waypoint config)
        self.max_roll_rate_deg_per_step = 2.0  # realistic (≈20 °/s)

        self.start_pos_spread_m = float(start_pos_spread_m)
        self.start_heading_spread_deg = float(start_heading_spread_deg)
        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]
        self.max_steps = int(70.0 / self.dt)

        # --- Spaces ---
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: [dx, dy, heading_error_deg, cross_track, prev_roll_cmd]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1000.0, -1000.0, -180.0, -1000.0, -1.0], dtype=np.float32),
            high=np.array([1000.0, 1000.0, 180.0, 1000.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self.x = self.y = self.heading_deg = 0.0
        self.current_step = 0
        self.x0 = self.y0 = self.x_end = self.y_end = 0.0
        self.path_x_array = self.path_y_array = None
        self._d_prev = 0.0
        
        # Previous action tracking for roll rate control
        self._prev_action = 0.0

    # --------- helpers ----------
    def _goal_dx_dy(self):
        return (self.x_end - self.x, self.y_end - self.y)

    def _cross_and_along_track(self):
        """Return (signed_cross_track_m, along_track_m, path_length)."""
        vx = self.x_end - self.x0
        vy = self.y_end - self.y0
        wx = self.x - self.x0
        wy = self.y - self.y0

        vv = vx * vx + vy * vy
        path_length = float(np.hypot(vx, vy))
        if vv == 0.0:
            return 0.0, 0.0, path_length

        # projection factor (unclipped)
        t_unclipped = (wx * vx + wy * vy) / vv
        # clamp to segment
        t = float(np.clip(t_unclipped, 0.0, 1.0))

        # closest point
        cx = self.x0 + t * vx
        cy = self.y0 + t * vy

        dx = self.x - cx
        dy = self.y - cy
        dist_to_seg = float(np.hypot(dx, dy))

        # signed via cross product
        cross = vx * (self.y - self.y0) - vy * (self.x - self.x0)
        sign = float(np.sign(cross)) if cross != 0.0 else 1.0
        signed_ct = sign * dist_to_seg

        along_track_m = t * path_length
        return signed_ct, along_track_m, path_length

    def _obs(self):
        dx, dy = self._goal_dx_dy()
        signed_ct, _, _ = self._cross_and_along_track()
        
        # Calculate angle difference between UAV heading and path direction
        # Path direction is from start to end of the path
        path_dx = self.x_end - self.x0
        path_dy = self.y_end - self.y0
        path_heading_deg = np.degrees(np.arctan2(path_dy, path_dx))
        heading_error_deg = wrap_deg(self.heading_deg - path_heading_deg)
        
        return np.array([dx, dy, heading_error_deg, signed_ct, self._prev_action], dtype=np.float32)

    def _in_bounds(self):
        return (self.xmin <= self.x <= self.xmax) and (self.ymin <= self.y <= self.ymax)

    def _rebuild_path_arrays(self):
        self.path_x_array = np.linspace(self.x0, self.x_end, 1001)
        self.path_y_array = np.linspace(self.y0, self.y_end, 1001)

    # --------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x0 = float(self.np_random.uniform(-self.start_pos_spread_m, self.start_pos_spread_m))
        self.y0 = float(self.np_random.uniform(-self.start_pos_spread_m, self.start_pos_spread_m))
        self.x = self.x0
        self.y = self.y0
        self.heading_deg = float(
            self.np_random.uniform(-self.start_heading_spread_deg, self.start_heading_spread_deg)
        )
        goal_dx = self.path_len_m + self.np_random.uniform(-200.0, 200.0)
        goal_dy = self.np_random.uniform(-300.0, 300.0)
        self.x_end = self.x0 + goal_dx
        self.y_end = self.y0 + goal_dy
        self._rebuild_path_arrays()

        self.current_step = 0
        self._d_prev = float(np.hypot(self.x - self.x_end, self.y - self.y_end))
        
        # Reset previous action tracking
        self._prev_action = 0.0
        
        return self._obs(), {}

    def step(self, action):
        # Roll rate control (same as RH2 waypoint config)
        # Action is normalized roll rate change [-1, +1]
        # Scale to actual roll rate change: action * max_roll_rate_deg_per_step
        roll_rate_change_deg = float(np.clip(action[0], -1.0, 1.0)) * self.max_roll_rate_deg_per_step
        roll_rate_change_cmd = roll_rate_change_deg / self.max_roll_deg
        roll_cmd = self._prev_action + roll_rate_change_cmd
        roll_cmd = np.clip(roll_cmd, -1.0, 1.0)  # Keep roll command in valid range
        
        # Roll command → heading change
        roll_deg = roll_cmd * self.max_roll_deg
        roll_rad = np.deg2rad(roll_deg)
        heading_rate_rad = (self.g / self.speed) * np.tan(roll_rad)
        heading_rate_deg = np.degrees(heading_rate_rad)
        self.heading_deg = wrap_deg(self.heading_deg + heading_rate_deg * self.dt)

        # Move
        hdg_rad = np.deg2rad(self.heading_deg)
        self.x += self.speed * np.cos(hdg_rad) * self.dt
        self.y += self.speed * np.sin(hdg_rad) * self.dt
        self.current_step += 1

        # Distances / errors
        dist_to_goal = np.hypot(self.x - self.x_end, self.y - self.y_end)
        dx_goal = self.x_end - self.x
        dy_goal = self.y_end - self.y
        desired_heading_deg = np.degrees(np.arctan2(dy_goal, dx_goal))
        heading_err_deg = wrap_deg(self.heading_deg - desired_heading_deg)

        signed_ct, _, path_length = self._cross_and_along_track()

        # ----- Rewards -----
        terminated = False
        truncated = False
        term_reason = "running"

        reward = 0.0
        #reward = -self.tau  # time penalty
        delta_d = self._d_prev - dist_to_goal
        reward += self.alpha_delta_d * delta_d
        # path penalty (normalized)
        if path_length > 0.0:
            reward -= self.beta_path * (abs(signed_ct) / path_length)
        if dist_to_goal <= self.goal_center_radius_m:
            reward += 10.0
            terminated = True
            term_reason = "success"

        # Bounds / timeout
        if not terminated and not self._in_bounds():
            truncated = True
            term_reason = "out_of_bounds"
        if not terminated and not truncated and (self.current_step >= self.max_steps):
            truncated = True
            term_reason = "max_steps"

        info = {
            "dist_to_path_abs": abs(signed_ct),
            "dist_to_path_signed": signed_ct,
            "heading_diff_deg": heading_err_deg,
            "termination_reason": term_reason,
            "dist_to_goal": dist_to_goal,
            "heading_deg": self.heading_deg,
            "roll_deg": float(roll_deg),
        }

        obs = self._obs()
        self._d_prev = dist_to_goal
        
        # Update previous action tracking for roll rate control
        self._prev_action = roll_cmd
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass