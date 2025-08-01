"""
Environment that will simulate agents and access there performances.
"""

# Typing
import abc

# Compute
import torch
import numpy as np

# Rendering
import matplotlib.pyplot as plt
import imageio


class Simulator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, weights, **kwargs):
        assert NotImplementedError
    
    @abc.abstractmethod
    def record_run(self, weights, **kwargs):
        assert NotImplementedError
        
    @property
    def weight_size(self):
        assert NotImplementedError

### CartPole
class TorchCartPole(Simulator):
    def __init__(self, device="cuda"):
        self.device = device

        # Physics constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # half pole length
        self.polemass_length = self.masspole * self.length
        self.tau = 0.02  # time step
        self.force_mag = 10.0

        # Limits for failure
        self.x_threshold = 2.4
        self.theta_threshold = 12 * 2 * np.pi / 360  # 12 degrees in radians

    
    
    @property
    def weight_size(self):
        return 4  # Number of parameter of the model
    
    def _physics_step(self, state, force):
        x, x_dot, theta, theta_dot = state.T
        costheta, sintheta = torch.cos(theta), torch.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return torch.stack((x, x_dot, theta, theta_dot), dim=1)

    def evaluate(self, weights, max_steps=500):
        """Evaluate many agents in parallel on GPU. Returns steps survived."""
        num_envs = weights.shape[0]
        state = (torch.rand((num_envs, 4), device=self.device) - 0.5) * 0.1
        steps_alive = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        alive = torch.ones(num_envs, device=self.device, dtype=torch.bool)

        for _ in range(max_steps):
            if not alive.any():
                break

            scores = (weights * state).sum(dim=1)
            force = torch.where(scores > 0, self.force_mag, -self.force_mag)

            state = self._physics_step(state, force)

            still_alive = (state[:, 0] > -self.x_threshold) & (state[:, 0] < self.x_threshold) & \
                          (state[:, 2] > -self.theta_threshold) & (state[:, 2] < self.theta_threshold)

            steps_alive[alive & still_alive] += 1
            alive = alive & still_alive

        return steps_alive

    def _setup_video(self):
        plt.ioff()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2.4, 2.4)
        self.ax.set_ylim(-1, 2)
        self.ax.axis('off')

        self.cart_line, = self.ax.plot([], [], "k-", lw=8)
        self.pole_line, = self.ax.plot([], [], "r-", lw=4)

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _render_frame(self, x, theta):
        self.fig.canvas.restore_region(self.background)

        cart_y = 0.5
        pole_x = x + self.length * np.sin(theta)
        pole_y = cart_y + self.length * np.cos(theta)

        self.cart_line.set_data([x - 0.2, x + 0.2], [cart_y, cart_y])
        self.pole_line.set_data([x, pole_x], [cart_y, pole_y])

        self.ax.draw_artist(self.cart_line)
        self.ax.draw_artist(self.pole_line)

        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

        buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        buf.shape = (h, w, 4)

        # Convert ARGB to RGBA
        buf = buf[:, :, [1, 2, 3, 0]]

        # Drop alpha channel
        rgb_buf = buf[:, :, :3].copy()

        return rgb_buf


    def record_run(self, weights, max_steps=200, video_path="cartpole_run.mp4"):
        import imageio
        """Run single agent and save video. No interactive rendering."""
        self._setup_video()

        state = (torch.rand((1, 4), device=self.device) - 0.5) * 0.1
        frames = []

        for _ in range(max_steps):
            score = (weights * state).sum()
            force = self.force_mag if score > 0 else -self.force_mag
            state = self._physics_step(state, torch.tensor([force], device=self.device))

            x, _, theta, _ = state[0]
            frame = self._render_frame(x.item(), theta.item())
            frames.append(frame)

            if abs(x) > self.x_threshold or abs(theta) > self.theta_threshold:
                break

        imageio.mimsave(video_path, frames, fps=int(1 / self.tau))
        plt.close(self.fig)





class TorchPendulum(Simulator):
    def __init__(self, device="cpu"):
        self.device = device

        self.max_speed = 15.0
        self.max_torque = 1.5
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.start_angle=15 * 2 * np.pi / 360  # 15 degrees in radians
        # For rendering
        self.fig, self.ax = None, None
        self.length = self.l  # rod length for drawing

    @property
    def weight_size(self):
        return 250  # Number of parameter of the model
    
    def _physics_step(self, state, torque):
        theta, theta_dot = state[:, 0], state[:, 1]
        # Discretization of \ddot{\theta} = -\frac{g}{l} \sin(\theta) + \dfrac{\tau}{m l^2} 
        new_theta_dot = theta_dot + self.dt *  ( - self.g / self.l * torch.sin(theta) + torque / (self.m * self.l**2))
        new_theta_dot = torch.clamp(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt
        return torch.stack((new_theta, new_theta_dot), dim=1)


    def evaluate(self, weights, max_steps=300):
        batch_size = weights.shape[0]
        weights = weights.to(self.device)

        # Layer 1: matrix (2,20) = 40 weights
        matrix1 = weights[:, :40].reshape(batch_size, 2, 20)

        # Layer 2: matrix (20,10) = 200 weights
        matrix2 = weights[:, 40:240].reshape(batch_size, 20, 10)

        # Output layer: matrix (10,1) = 10 weights
        last_layer = weights[:, 240:]  # shape: (batch_size, 10)

        # Initial state: start angle and zero velocity
        theta = torch.full((batch_size,), self.start_angle, device=self.device)
        theta_dot = torch.zeros(batch_size, device=self.device)
        state = torch.stack((theta, theta_dot), dim=1)

        total_reward = torch.zeros(batch_size, device=self.device)

        for i in range(max_steps):
            # Forward pass:
            hidden1 = torch.relu(torch.bmm(state.unsqueeze(1), matrix1).squeeze(1))  # (batch, 20)
            hidden2 = torch.relu(torch.bmm(hidden1.unsqueeze(1), matrix2).squeeze(1))  # (batch, 10)
            
            torque = torch.sum(hidden2 * last_layer, dim=1)  # (batch,)
            torque = torch.tanh(torque) * self.max_torque

            # Step physics forward
            state = self._physics_step(state, torque)
            theta, theta_dot = state[:, 0], state[:, 1]

            # Normalize angle to [-1, 1]
            norm_theta = ((theta % (2 * torch.pi)) - torch.pi) / torch.pi

            # Compute reward
            #reward = -(norm_theta ** 2
            #        + 0.1 * (theta_dot / self.max_speed) ** 2
            #        + 0.001 * (torque / self.max_torque) ** 2)
            reward = - i/max_steps * norm_theta ** 2

            total_reward += reward

        return total_reward


    def record_run(self, weights, max_steps=300, video_path="pendulum_run.mp4"):
        import imageio
        self._setup_render()

        weights = weights.to(self.device).unsqueeze(0)

        # Layer 1: matrix (2,20) = 40 weights
        matrix1 = weights[:, :40].reshape(-1, 2, 20)
        # Layer 2: matrix (20,10) = 200 weights
        matrix2 = weights[:, 40:240].reshape(-1, 20, 10)
        # Output layer: matrix (10,1) = 10 weights
        last_layer = weights[:, 240:]

        theta = torch.full((1,), self.start_angle, device=self.device)
        theta_dot = torch.zeros(1, device=self.device)
        state = torch.stack((theta, theta_dot), dim=1)

        frames = []

        for _ in range(max_steps):
            hidden1 = torch.relu(torch.bmm(state.unsqueeze(1), matrix1).squeeze(1))
            hidden2 = torch.relu(torch.bmm(hidden1.unsqueeze(1), matrix2).squeeze(1))

            torque = torch.sum(hidden2 * last_layer, dim=1)
            torque = torch.tanh(torque) * self.max_torque

            state = self._physics_step(state, torque)

            theta = state[0, 0].item()
            frame = self._render_frame(theta)
            frames.append(frame)

        imageio.mimsave(video_path, frames, fps=int(1 / self.dt))
        plt.close(self.fig)


    def _setup_render(self):
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Bigger pivot node above the rod
        self.origin_y_offset = 0.3  # offset above origin to draw the pivot node higher

        self.line, = self.ax.plot([], [], lw=4, color='r')
        self.node = plt.Circle((0, self.origin_y_offset), 0.1, color='k', zorder=5)
        self.ax.add_patch(self.node)

    def _render_frame(self, theta):
        theta_t = torch.tensor(theta, device=self.device)

        x_start, y_start = 0, self.origin_y_offset

        # Rod end position:
        x_end = x_start + self.length * torch.sin(theta_t).item()
        y_end = y_start - self.length * torch.cos(theta_t).item()

        self.line.set_data([x_start, x_end], [y_start, y_end])
        self.fig.canvas.draw()

        buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        buf.shape = (h, w, 4)

        # Convert ARGB to RGBA
        buf = buf[:, :, [1, 2, 3, 0]]

        rgb_buf = buf[:, :, :3].copy()

        return rgb_buf

    




def try_cartpole():
    env = TorchCartPole(device="cuda")

    weights = torch.randn((1000, 4), device="cuda")

    # Parallel evaluation, fast and no rendering
    scores = env.evaluate(weights)
    best_idx = scores.argmax()

    # Record video of best agent run
    env.record_run(weights[best_idx], video_path="best_cartpole.mp4")

def try_pendulum():
    device = "cuda"
    pendulum = TorchPendulum(device=device)

    # Example random linear weights: [w_theta, w_theta_dot]
    weights = torch.randn((100000, 250), device="cuda")


    # Evaluate cumulative reward for this agent
    scores = pendulum.evaluate(weights)
    best_idx = scores.argmax()
    print("best reward : ", scores[best_idx].item())
    # Save video of this agent running
    pendulum.record_run(weights[best_idx], video_path="pendulum_run.mp4")


#def try_cartpole():
#try_pendulum()