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
from matplotlib.patches import Arc, FancyArrowPatch
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
        self.friction = 0.05
        # For rendering
        self.fig = None
        self.ax = None
        self.length = self.l
        self.origin = (0, 0.3)
        self.arrow_base_offset = 1.2  # vertical offset for torque arrow
        self.arrow = None

    @property
    def weight_size(self):
        return 250  # Number of parameter of the model
    
    def _physics_step(self, state, torque):
        theta, theta_dot = state[:, 0], state[:, 1]
        # Discretization of \ddot{\theta} = -\frac{g}{l} \sin(\theta) + \dfrac{\tau}{m l^2} 
        new_theta_dot = theta_dot + self.dt *  ( - self.g / self.l * torch.sin(theta) + torque / (self.m * self.l**2)) \
                        - self.friction * theta_dot * self.dt

        new_theta_dot = torch.clamp(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt
        return torch.stack((new_theta, new_theta_dot), dim=1)


    def evaluate(self, weights, max_steps=400):
        batch_size = weights.shape[0]
        weights = weights.to(self.device)

        # Layer weights
        matrix1 = weights[:, :40].reshape(batch_size, 2, 20)
        matrix2 = weights[:, 40:240].reshape(batch_size, 20, 10)
        last_layer = weights[:, 240:]

        # Initial state
        theta = torch.full((batch_size,), self.start_angle, device=self.device)
        theta_dot = torch.zeros(batch_size, device=self.device)
        state = torch.stack((theta, theta_dot), dim=1)

        total_reward = torch.zeros(batch_size, device=self.device)

        for i in range(max_steps):
            # Forward pass
            hidden1 = torch.relu(torch.bmm(state.unsqueeze(1), matrix1).squeeze(1))
            hidden2 = torch.relu(torch.bmm(hidden1.unsqueeze(1), matrix2).squeeze(1))
            torque = torch.sum(hidden2 * last_layer, dim=1)
            torque = torch.tanh(torque) * self.max_torque

            # Physics update
            state = self._physics_step(state, torque)
            theta, theta_dot = state[:, 0], state[:, 1]

            # Normalize angle
            norm_theta = ((theta % (2 * torch.pi)) - torch.pi) / torch.pi
            reward = -  (norm_theta ** 2 + 0.1 * (torque / self.max_torque)**2)

            # Accumulate reward
            total_reward += reward


        return total_reward


    def record_run(self, weights, max_steps=400, video_path="pendulum_run.mp4"):
        import imageio
        self._setup_render()

        weights = weights.to(self.device).unsqueeze(0)
        matrix1 = weights[:, :40].reshape(-1, 2, 20)
        matrix2 = weights[:, 40:240].reshape(-1, 20, 10)
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
            frame = self._render_frame(theta, torque.item())
            frames.append(frame)

        imageio.mimsave(video_path, frames, fps=int(1 / self.dt), macro_block_size=1)
        plt.close(self.fig)

    def _setup_render(self):
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.ax.set_xlim(-2,2)
        self.ax.set_ylim(-2,2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        # draw pivot node
        self.ax.add_patch(plt.Circle(self.origin, 0.1, color='k', zorder=5))
        # create line handle
        self.line, = self.ax.plot([], [], lw=4, color='r')
        # placeholder for arrow
        self.arrow = None

    def _render_frame(self, theta, torque):
        # pendulum
        x0, y0 = self.origin
        x1 = x0 + self.length * np.sin(theta)
        y1 = y0 - self.length * np.cos(theta)
        self.line.set_data([x0, x1], [y0, y1])
        # remove old arrow
        if self.arrow:
            self.arrow.remove()
        # draw torque arrow below pivot
        direction = np.sign(torque)
        magnitude = abs(torque)/self.max_torque
        max_len = 0.8
        dx = direction * max_len * magnitude
        dy = 0
        ax = self.ax
        self.arrow = ax.arrow(
            x0, y0 - self.arrow_base_offset,
            dx, dy,
            head_width=0.1, head_length=0.1,
            length_includes_head=True,
            color='blue'
        )
        # finalize
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        w,h = self.fig.canvas.get_width_height()
        buf.shape = (h, w, 4)
        buf = buf[:,:, [1,2,3,0]]
        return buf[:,:,:3].copy()


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