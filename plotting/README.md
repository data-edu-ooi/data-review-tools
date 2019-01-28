# Plotting
This toolbox contains tools to plot data. 

### Main Functions
- [ctdmo_platform_timeseries.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/ctdmo_platform_timeseries.py): Plot a timeseries of all CTDMO data from an entire platform. Outputs two plots of each science variable by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.

- [timeseries_plots.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/timeseries_plots.py): Wrapper script that imports tools to plot a variety of timeseries plots for an instrument/deployment. The user has the option of selecting a specific time range to plot and only plotting data from the preferred method/stream.

### Scripts
- [plot_compare_timeseries.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/scripts/plot_compare_timeseries.py): Creates two timeseries plots of telemetered and recovered science variables for a reference designator where the Long Names of the two variables are the same: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations. The user has the option of selecting a specific time range to plot.

- [plot_profiles.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/scripts/plot_profiles.py): Creates two profile plots of raw and science variables for a mobile instrument (e.g. profilers and gliders) by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations. The user has the option of selecting a specific time range to plot.

- [plot_timeseries_all.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/scripts/plot_timeseries_all.py): Creates two timeseries plots of raw and science variables for all deployments of a reference designator by delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.

- [plot_timeseries_panel.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/scripts/plot_timeseries_panel.py): Creates timeseries panel plots of all science variables for an instrument, deployment, and delivery method. These plots omit data outside of 5 standard deviations. The user has the option of selecting a specific time range to plot and only plotting data from the preferred method/stream.

- [plot_timeseries_pm_all.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/scripts/plot_timeseries_pm_all.py): Creates two timeseries plots of science variables for all deployments of a reference designator using the preferred stream for each deployment (plots may contain data from different delivery methods): 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.

- [plot_timeseries.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/scripts/plot_timeseries.py): Create two timeseries plots of raw and science variables for all deployments of a reference designator by delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations. The user has the option of selecting a specific time range to plot and only plotting data from the preferred method/stream.

- [plot_ts.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/scripts/plot_ts.py): Create temperature salinity plots with density contours by instrument and deployment, colored by time (cooler colors indicate earlier timepoints).

- [plot_xsection.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/scripts/plot_xsection.py): Create two timeseries plots of raw and science variables for all deployments of a reference designator by delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 3 standard deviations. When plotting glider data, zeros and negative values are excluded for both plots, and are not calculated as part of the standard deviation. The user has the option of selecting a specific time range to plot.

### Example files
- [ctdmo_data_request_summary.csv](https://github.com/ooi-data-lab/data-review-tools/blob/master/plotting/example_files/ctdmo_data_request_summary.csv): Example csv file containing CTDMO datasets from one platform to plot. This can be an output from one of the [data download tools](https://github.com/ooi-data-lab/data-review-tools/tree/master/data_download) and must contain a column labeled 'outputUrl'.