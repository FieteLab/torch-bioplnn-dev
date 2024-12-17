import torch
from torch.utils.data import Dataset


class CorrelatedDots(Dataset):
    def __init__(
        self,
        resolution=(128, 128),
        n_frames=10,
        n_dots=100,
        correlation=1.0,
        max_speed=5,
        samples_per_epoch=10000,
    ):
        """
        Args:
            n_frames (int): Number of frames to generate.
            n_dots (int): Number of dots in each frame.
            direction (str): Direction of motion ('right', 'left', 'up', 'down').
            speed (int): Speed of the dot motion in pixels per frame.
            resolution (tuple): Size of each frame (height, width).
            correlation (float): Correlation of dots' motion (0 to 1, where 1 is fully correlated, 0 is fully random).
        """
        self.n_frames = n_frames
        self.n_dots = n_dots
        self.resolution = resolution
        self.correlation = correlation
        self.max_speed = max_speed
        self.samples_per_epoch = samples_per_epoch

        self.direction_vectors = {
            0: torch.tensor((1, 0)),  # Right
            1: torch.tensor((-1, 0)),  # Left
            2: torch.tensor((0, -1)),  # Up
            3: torch.tensor((0, 1)),  # Down
        }

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        height, width = self.resolution
        frames = []

        # Initialize random dot positions
        x = torch.rand(self.n_dots) * width
        y = torch.rand(self.n_dots) * height

        # Determine the direction and speed vector for correlated dots
        correlated_speed = 0
        while correlated_speed == 0:
            correlated_speed = (torch.rand(1) * self.max_speed).ceil()
        correlated_direction = torch.randint(4, (1,)).squeeze()

        correlated_dx, correlated_dy = (
            self.direction_vectors[correlated_direction.item()]
            * correlated_speed
        )

        # Precompute random directions for uncorrelated dots
        random_speed = torch.rand(self.n_dots) * self.max_speed
        random_directions = torch.rand(self.n_dots) * 2 * torch.pi
        random_dx = torch.cos(random_directions) * random_speed
        random_dy = torch.sin(random_directions) * random_speed

        # Generate each frame
        for _ in range(self.n_frames):
            # Create an empty frame
            frame = torch.zeros(height, width)

            # Determine which dots move in the correlated direction
            correlated_mask = torch.rand(self.n_dots) < self.correlation

            # Apply correlated movement
            x[correlated_mask] = (x[correlated_mask] + correlated_dx) % width
            y[correlated_mask] = (y[correlated_mask] + correlated_dy) % height

            # Apply random movement to uncorrelated dots
            x[~correlated_mask] = (
                x[~correlated_mask] + random_dx[~correlated_mask]
            ) % width
            y[~correlated_mask] = (
                y[~correlated_mask] + random_dy[~correlated_mask]
            ) % height

            # Draw dots on the frame
            x_int = x.long()
            y_int = y.long()
            frame[y_int, x_int] = 1  # Set the pixel to 1 (white dot)

            frames.append(frame)

        frames = torch.stack(frames).unsqueeze(1)

        assert frames.shape == (self.n_frames, 1, height, width)

        return frames, correlated_direction
