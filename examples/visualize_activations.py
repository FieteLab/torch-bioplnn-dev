import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import torch


def visualize_crnn_activations(
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

            # Display the animation in the notebook
            # display(HTML(anim.to_jshtml()))

            # Save the animation as an MP4 file
            anim.save(
                f"{save_dir}/{activation}_{i}.mp4",
                writer="ffmpeg",
                fps=fps,
            )

            plt.close(fig)
