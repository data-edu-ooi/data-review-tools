#! /usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os
import numpy as np


def format_date_axis(define_axis, figure):
    df = mdates.DateFormatter('%Y-%m-%d')
    define_axis.xaxis.set_major_formatter(df)
    figure.autofmt_xdate()
    

def reject_outliers(data, m=3):
    # function to reject outliers beyond 3 standard deviations of the mean.
    # data: numpy array containing data
    # m: the number of standard deviations from the mean. Default: 3
    return abs(data - np.nanmean(data)) < m * np.nanstd(data)
    

def save_fig(save_dir, file_name, res=300):
    """
    Save figure to a directory with a resolution of 150 DPI
    :param save_dir: Location of the directory to save the file
    :param file_name: The name of the file to save
    :param res: Resolution in DPI of the image to save
    :return: None
    """
    save_file = os.path.join(save_dir, file_name)
    plt.savefig(str(save_file), dpi=res)
    plt.close()


def y_axis_disable_offset(define_axis):
    y_formatter = ticker.ScalarFormatter(useOffset=False)
    define_axis.yaxis.set_major_formatter(y_formatter)