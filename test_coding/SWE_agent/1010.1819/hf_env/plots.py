### Code evalution
import matplotlib.pyplot as plt

import numpy as np
import inspect


def plot_kspace(kspace):
    fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
    ax.scatter(*kspace.T, s=2)
    ax.set_aspect("equal")
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    return fig


def plot_matele(mat):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.spy((mat))
    ax.set_title("$H_0$")
    return fig


def plot_2d_bandstructure(ham, en):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(en.shape[0]):
        ax.plot_trisurf(ham.k_space[:, 0], ham.k_space[:, 1], en[i])
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_zlabel("E")
    return fig


def plot_high_symm_bandstructure(k_list, en, ax=None):  # DROP
    if ax is None:
        fig, ax = plt.subplots()
    for e in en:
        k_abs = np.sqrt(np.diff(k_list[:, 0]) ** 2 + np.diff(k_list[:, 1]) ** 2)
        k_abs = np.concatenate([[0], np.cumsum(k_abs)])
        ax.plot(k_abs, e, color="k")
    Nk = (k_list.shape[0] - 1) // 4
    for i in range(5):
        ax.axvline(k_abs[Nk * i], ls="--", color="r")
    ax.set_xticks(
        [k_abs[Nk * i] for i in range(5)], ["$\Gamma$", "K", "M", "$\Gamma$", "K'"]
    )
    return fig


def plot_2d_false_color_map(ham, en, kmax=6, width=3):
    from matplotlib.ticker import ScalarFormatter

    kmax = min(len(en), kmax)  # number of bands

    fig, ax = plt.subplots(1, kmax, figsize=(3 * kmax, 3), tight_layout=True)
    for idx in range(kmax):
        im = ax[idx].tripcolor(
            ham.k_space[:, 0], ham.k_space[:, 1], en[idx], shading="gouraud"
        )
        axins = ax[idx].inset_axes([1.03, 0., 0.05, 0.99])  
        ax[idx].tricontour(
            ham.k_space[:, 0],
            ham.k_space[:, 1],
            en[idx],
            levels=10,
            colors="k",
            linewidths=1,
        )
        ax[idx].set_title(f"Band {idx+1}")
        ax[idx].set_xlabel("$k_x$")
        ax[idx].set_ylabel("$k_y$")
        
        cbar = plt.colorbar(
            im,
            cax=axins,
            ax=ax[idx],
        )
        cbar.formatter = ScalarFormatter(useOffset=False, useMathText=False)
        cbar.formatter.set_scientific(False)
        cbar.update_ticks()
        ax[idx].set_aspect("equal")

    return fig


def print_gap(ham_int, exp_val, en_int,level=None):
    mean_U = np.abs(ham_int.generate_interacting(exp_val)).mean()
    mean_T = np.abs(ham_int.generate_non_interacting()).mean()
    levels = len(en_int)
    if level is None:
        level = levels // 2
    gap = en_int[level].min() - en_int[level - 1].max()
    str1 = f"Gap is {gap:.2f}"
    str2 = f"U/T is {mean_U/mean_T:.2f}"
    str3 = f"mean_U is {mean_U:.2f}"
    print(str1)
    print(str2)
    print(str3)
    with open('results/output.txt', 'w') as f:
        f.write('\n'.join([str1, str2, str3]))
    return

