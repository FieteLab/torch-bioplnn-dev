import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import torch


def visualize_maze_activations(
    activations_path: str = "activations/ei/mazes/daily-armadillo-3361/activations.pt",
    save_dir: str = "visualizations",
    fps: int = 5,
):
    os.makedirs(save_dir, exist_ok=True)

    activations = torch.load(activations_path)

    # Convert activations (which is a list of dicts, where the inner dicts are column names) to a pandas df
    df = pd.DataFrame(activations)

    for i, sample in df.iterrows():
        plt.imsave(
            f"{save_dir}/x_maze_{i}.png",
            sample.x.squeeze()[:3].permute(1, 2, 0).numpy(),
        )
        plt.imsave(
            f"{save_dir}/x_start_end_{i}.png",
            sample.x.squeeze()[3].numpy(),
            cmap="viridis",
        )
        for activation in ("h_pyrs", "h_inters"):
            # Assuming 'sample' is your data object containing h_pyrs
            hs = sample[activation][-1].squeeze()

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(8, 8))

            # Initialize the plot with the first frame
            im = ax.imshow(hs[0].sum(dim=0).cpu().numpy(), cmap="viridis")

            # Function to update the frame
            def update(frame):
                im.set_array(hs[frame].sum(dim=0).cpu().numpy())
                return [im]

            # Create the animation
            anim = animation.FuncAnimation(
                fig, update, frames=len(hs), interval=1000 / fps, blit=True
            )

            # Save the animation as an MP4 file
            anim.save(
                f"{save_dir}/{activation}_{i}.mp4",
                writer="ffmpeg",
                fps=fps,
            )

            plt.close(fig)


def visualize_v1_activations(
    activations_path: str = "activations/ei/mazes/daily-armadillo-3361/activations.pt",
    save_dir: str = "visualizations",
    fps: int = 5,
    sheet_size: tuple[int, int] = (150, 300),
):
    """
    Visualizes the activations of the TopographicalRNN as an animation.

    Args:
        activations_path (str): Path to the activations file.
        save_dir (str, optional): Directory to save the visualizations. Defaults to "visualizations".
        fps (int, optional): Frames per second for the animation. Defaults to 5.
        sheet_size (tuple[int, int], optional): Size of each sheet. Defaults to (150, 300).
    """
    os.makedirs(save_dir, exist_ok=True)

    activations = torch.load(activations_path)

    for i in range(len(activations)):
        activations[i] = activations[i][0].reshape(*sheet_size)

    # Convert activations (which is a list of dicts, where the inner dicts are column names) to a pandas df
    df = pd.DataFrame(activations)

    for i, sample in df.iterrows():
        hs = sample["hs"][-1].squeeze()

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Initialize the plot with the first frame
        im = ax.imshow(hs[0].sum(dim=0).cpu().numpy(), cmap="viridis")

        # Function to update the frame
        def update(frame):
            im.set_array(hs[frame].sum(dim=0).cpu().numpy())
            return [im]

        # Create the animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(hs), interval=1000 / fps, blit=True
        )

        # Save the animation as an MP4 file
        anim.save(
            f"{save_dir}/v1_{i}.mp4",
            writer="ffmpeg",
            fps=fps,
        )

        plt.close(fig)
