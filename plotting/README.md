# Plotting
This toolbox contains tools to plot data. 

### Scripts
- [ctdmo_platform_timeseries.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/ctdmo_platform_timeseries.py): Plot a timeseries of all CTDMO data from an entire platform. Outputs two plots of each science variable by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.

### Tools
- [plot_timeseries_deployment.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/tools/plot_timeseries_deployment.py): Creates two timeseries plots of each science variable for a reference designator by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.

### Example files
- [ctdmo_data_request_summary.csv](https://github.com/data-edu-ooi/data-review-tools/blob/master/plotting/example_files/ctdmo_data_request_summary.csv): Example csv file containing CTDMO datasets from one platform to plot. This can be an output from one of the [data download tools](https://github.com/data-edu-ooi/data-review-tools/tree/master/data_download) and must contain a column labeled 'outputUrl'.