import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================

def plot_heatmap(heat, labels, filename="heatmap.png"):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    im = ax.imshow(heat, aspect="auto", cmap="inferno", origin="lower",
                   norm=Normalize(vmin=np.min(heat), vmax=np.max(heat)))

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time step")
    ax.set_title("Internal Thermal and State Evolution")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Scaled intensity")

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Heatmap salvo como {filename}")

def plot_temporal_panels(results, filename="temporal_panels.png"):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    axs[0, 0].plot(results["t"], results["TempC"], color="crimson", lw=2)
    axs[0, 0].set_title("Temperature")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("C")

    axs[0, 1].plot(results["t"], results["x"], color="navy", lw=2)
    axs[0, 1].axhline(1.0, color="black", ls="--", lw=1)  # Critical line
    axs[0, 1].set_title("Critical index x")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("x")

    axs[1, 0].plot(results["t"], results["D_Pd"], color="darkgreen", lw=2)
    axs[1, 0].set_title("D/Pd proxy")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("D/Pd")

    axs[1, 1].plot(results["t"], results["Df"], color="purple", lw=2)
    axs[1, 1].set_title("Df")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Df")

    plt.suptitle("Thermal and Structural Reaction Evolution", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Painéis temporais salvos como {filename}")