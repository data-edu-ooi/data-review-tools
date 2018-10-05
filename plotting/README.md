# Plotting
This toolbox contains tools to plot data. 

### Scripts
- [ctdmo_platform_timeseries.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/ctdmo_platform_timeseries.py): Plot a timeseries of all CTDMO data from an entire platform. Outputs two plots of each science variable by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.

### Tools
- [plot_profiles.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/tools/plot_profiles.py): Creates two profile plots of raw and science variables for a mobile instrument (e.g. profilers and gliders) by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations. The user has the option of selecting a specific time range to plot.

- [plot_timeseries_all.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/tools/plot_timeseries_all.py): Creates two timeseries plots of raw and science variables for a reference designator by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.

- [plot_timeseries_panel.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/tools/plot_timeseries_panel.py): Creates timeseries panel plots of all science variables for an instrument, deployment, and delivery method. These plots omit data outside of 5 standard deviations. The user has the option of selecting a specific time range to plot.

- [plot_timeseries.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/tools/plot_timeseries.py): Create two timeseries plots of raw and science variables for all deployments of a reference designator by delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 3 standard deviations. The user has the option of selecting a specific time range to plot.

### Example files
- [ctdmo_data_request_summary.csv](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/example_files/ctdmo_data_request_summary.csv): Example csv file containing CTDMO datasets from one platform to plot. This can be an output from one of the [data download tools](https://github.com/data-edu-ooi/data-review-tools/tree/master/data_download) and must contain a column labeled 'outputUrl'.