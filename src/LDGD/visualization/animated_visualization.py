import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import torch


def animate_train(point_history, labels, file_name, inverse_length_scale, save_path='', largest=True,
                  inducing_points_history=None):
    values, indices = torch.topk(torch.tensor(inverse_length_scale), k=2, largest=largest)

    l1 = indices.numpy().flatten()[0]
    l2 = indices.numpy().flatten()[1]

    colors = ['r', 'b', 'g']
    colors_list = [colors[int(label)] for label in labels]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')


    # Create an empty scatter plot (we'll update it in the animation)
    # Normalize labels for color mapping
    norm = Normalize(vmin=labels.min(), vmax=labels.max())
    cmap = cm.viridis  # You can choose any colormap
    scatter = ax.scatter(point_history[0][:, l1], point_history[0][:, l2], c=colors_list)

    if inducing_points_history is not None:
        inducing_points_hist_reg, inducing_points_hist_cls = inducing_points_history
        inducing_points_hist_reg = [point.reshape(-1, point.shape[-1]) for point in inducing_points_hist_reg]
        inducing_points_hist_cls = [point.reshape(-1, point.shape[-1]) for point in inducing_points_hist_cls]
        scatter_inducing_reg = ax.scatter(inducing_points_hist_reg[0][:, l1],
                                          inducing_points_hist_reg[0][:, l2], c='black', marker='x', label='Regression')
        scatter_inducing_cls = ax.scatter(inducing_points_hist_cls[0][:, l1],
                                          inducing_points_hist_cls[0][:, l2], c='purple', marker='x', label='Classification')

    # if len(point_history) > 100:
    #     point_history = [point_history[int(i)] for i in list((np.linspace(0, len(point_history) - 1, 100, dtype=int)))]

    all_data_poins = np.stack(point_history, axis=0)

    # Set up the axis limits according to your points
    min_x, max_x = np.min(all_data_poins[:, :, l1]), np.max(all_data_poins[:, :, l1])
    min_y, max_y = np.min(all_data_poins[:, :, l2]), np.max(all_data_poins[:, :, l2])

    if inducing_points_history is not None:
        inducing_points_reg = np.concatenate(inducing_points_hist_reg, axis=0)
        inducing_points_cls = np.concatenate(inducing_points_hist_cls, axis=0)
        all_inducing_poins = np.concatenate([inducing_points_reg, inducing_points_cls], axis=0)
        min_zx, max_zx = np.min(all_inducing_poins[:, l1]), np.max(all_inducing_poins[:, l1])
        min_zy, max_zy = np.min(all_inducing_poins[:, l2]), np.max(all_inducing_poins[:, l2])
        min_x, max_x = np.min([min_x, min_zx]), np.max([max_x, max_zx])
        min_y, max_y = np.min([min_y, min_zy]), np.max([max_y, max_zy])

    ax.legend()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    def update(frame):
        # Get the points for the current frame
        points = point_history[frame][:, [l1, l2]]

        # Update the scatter plot data
        scatter.set_offsets(points)
        if inducing_points_history is not None:
            inducing_points_reg_frame = inducing_points_hist_reg[frame][:, [l1, l2]]
            inducing_points_cls_frame = inducing_points_hist_cls[frame][:, [l1, l2]]
            scatter_inducing_reg.set_offsets(inducing_points_reg_frame)
            scatter_inducing_cls.set_offsets(inducing_points_cls_frame)
        # scatter.set_array(np.squeeze(labels))

        return scatter,
    # Create the animation
    animation = anim.FuncAnimation(fig, update, frames=len(point_history), interval=200, blit=True)

    # Set up the writer for saving the animation (you can change the format and settings as needed)
    writer = anim.PillowWriter(fps=30)

    # Save the animation to a video file
    animation.save(save_path + '/{}.gif'.format(file_name), writer=writer)
