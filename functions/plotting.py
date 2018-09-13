#! /usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np


def ctdmo_deployment():
    """
    Plot CTDMO data from an entire platform
    :return: None
    """


def save_fig(save_dir, file_name, res=150):
    """
    Save figure to a directory with a resolution of 150 DPI
    :param save_dir: Location of the directory to save the file
    :param file_name: The name of the file to save
    :param res: Resolution in DPI of the image to save
    :return: None
    """
    save_file = os.path.join(save_dir, file_name)
    plt.savefig(str(save_file) + '.png', dpi=res)
    plt.close()