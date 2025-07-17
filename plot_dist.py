import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def main():

    barriers = np.loadtxt("barriers.txt")
    plt.hist(barriers, color="purple", alpha=0.6, linewidth=2.0, edgecolor="black", zorder=7)
    plt.grid()
    plt.xlabel("migration barrier")
    plt.ylabel("counts")
    plt.savefig("barriers.pdf")


if __name__ == "__main__":

    mpl.use("TkAgg")
    main()
