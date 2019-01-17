
import os
import pandas as pd
import itertools
import numpy as np
import xarray as xr
import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd
import functions.group_by_timerange as gt

from matplotlib import pyplot
from matplotlib import colors as mcolors

import matplotlib.gridspec as gridspec
import datetime
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import calendar
from calendar import monthrange
from matplotlib.ticker import MaxNLocator, LinearLocator


fig0 = pyplot.Figure()

gs = gridspec.GridSpec(2, 2, width_ratios=[5, 10], height_ratios=[10, 5])

ax1 = pyplot.subplot(gs[0])
ax2 = pyplot.subplot(gs[3])


clust_data = np.random.random((10,3))
collabel=("col 1", "col 2", "col 3")

ax2.axis('off')
ax2.axis('tight')

the_table = ax2.table(cellText=clust_data, colLabels=collabel, loc='center')

the_table.set_fontsize(5)

ax1.plot(clust_data[:,0], clust_data[:,1])

pyplot.show()

t=1
