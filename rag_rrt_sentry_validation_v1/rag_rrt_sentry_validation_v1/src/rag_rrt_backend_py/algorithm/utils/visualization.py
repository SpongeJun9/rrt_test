from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_map(ax, binary_mask):
    """
    binary_mask:
        1: free
        0: obstacle
    """
    ax.imshow(binary_mask, cmap="gray", origin="upper")
    ax.set_aspect("equal")


def plot_tree(ax, vertices, parents, num_vertices, linewidth=0.4, alpha=0.5):
    for i in range(1, num_vertices):
        parent = parents[i]
        if parent < 0:
            continue

        p = vertices[i]
        q = vertices[parent]
        ax.plot([p[0], q[0]], [p[1], q[1]], linewidth=linewidth, alpha=alpha)


def plot_path(ax, path, linewidth=2.0, label="path"):
    if path is None:
        return

    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return

    ax.plot(path[:, 0], path[:, 1], linewidth=linewidth, label=label)


def plot_points(ax, points, s=4, alpha=0.8, label=None):
    if points is None:
        return

    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return

    ax.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, label=label)


def save_planning_figure(
    filepath,
    binary_mask,
    planner,
    result,
    title=None,
):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    plot_map(ax, binary_mask)

    plot_tree(
        ax,
        planner.vertices,
        planner.parents,
        planner.num_vertices,
        linewidth=0.3,
        alpha=0.35,
    )

    if hasattr(planner, "guided_sampler"):
        original_path = planner.guided_sampler.get_original_path()
        prior_path = planner.guided_sampler.get_prior_path()

        if len(original_path) > 0:
            ax.plot(
                original_path[:, 0],
                original_path[:, 1],
                linestyle="--",
                linewidth=1.5,
                label="A* original prior",
            )

        if len(prior_path) > 0:
            ax.plot(
                prior_path[:, 0],
                prior_path[:, 1],
                linestyle=":",
                linewidth=2.0,
                label="used guided prior",
            )

    plot_path(ax, result.path, linewidth=2.5, label="final path")

    ax.scatter(planner.start[0], planner.start[1], s=60, marker="o", label="start")
    ax.scatter(planner.goal[0], planner.goal[1], s=60, marker="*", label="goal")

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(0, binary_mask.shape[1])
    ax.set_ylim(binary_mask.shape[0], 0)
    ax.legend(loc="best", fontsize=8)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def save_curve_figure(
    filepath,
    result,
    title=None,
):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)

    # Cost curve
    cost_history = np.asarray(result.cost_history, dtype=float)
    axes[0].plot(cost_history)
    axes[0].set_ylabel("Best cost")
    axes[0].set_title("Cost curve")
    axes[0].grid(True, alpha=0.3)

    # Alpha curve
    if len(result.alpha_history) > 0:
        axes[1].plot(result.alpha_history)
    axes[1].set_ylabel("Alpha")
    axes[1].set_title("Guided sampling ratio")
    axes[1].grid(True, alpha=0.3)

    # Reliability curve
    if len(result.reliability_history) > 0:
        axes[2].plot(result.reliability_history, label="relative reliability")

    if hasattr(result, "guided_progress_rate_history") and len(result.guided_progress_rate_history) > 0:
        axes[2].plot(result.guided_progress_rate_history, label="guided progress rate")

    axes[2].set_ylabel("Score / rate")
    axes[2].set_xlabel("Update step")
    axes[2].set_title("Reliability feedback")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)