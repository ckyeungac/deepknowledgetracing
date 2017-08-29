import matplotlib.pyplot as plt
import numpy as np
import math


# Perhaps we would use bokeh later on for the interactive visualization
# http://bokeh.pydata.org/en/0.10.0/docs/gallery/cat_heatmap_chart.html
def plot_heatmap(data, x_labels, y_labels, second_x_labels=None, fig_size_inches=[15, 5], cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])

    heatmap = ax.pcolor(data, cmap=cmap)

    # Format
    fig = plt.gcf()

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(len(x_labels)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(y_labels)) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # set the label
    ax.set_xticklabels(x_labels, minor=False)
    ax.set_yticklabels(y_labels, minor=False)
    ax.set_xlabel("the skill id answered at the time step")
    ax.set_ylabel("the skill id of the output layer")

    # second axis label
    if second_x_labels != None:
        ax2 = ax.twiny()
        ax2.set_xticks(np.arange(len(second_x_labels)) + 0.5, minor=False)
        ax2.set_xticklabels(second_x_labels, minor=False)
        ax2.set_xlabel("Correct Label")
        ax2.xaxis.tick_top()

    # Turn off all the ticks
    ax = plt.gca()
    fig.colorbar()

    # fig.colorbar(heatmap, fraction=0.02, pad=0.04)
    plt.show()
