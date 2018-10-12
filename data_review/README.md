# Data Review
Tools for automated analysis of OOI netCDF files downloaded from uFrame.

### Scripts
- [analyze_nc_data.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/data_review/analyze_nc_data.py): Wrapper script that imports tools to analyze netCDF files and provide summary outputs.

### Tools
- [nc_file_analysis.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/data_review//tools/nc_file_analysis.py): Automated analysis of .nc files from a THREDDs server. Summarizes data by deployment, delivery method, data stream, and science parameter and provides a .json output by reference designator. Analysis includes: 
	- compare data start and end times to deployment start and end times
	- compare deployment depth from asset managment to pressure data in file 
	- check for data gaps >1 day
	- check that timestamps are unique
	- check that timestamps are in ascending order
	- compare variables in the file to variables defined in the [Data Review Database](http://datareview.marine.rutgers.edu/)
	- calculate statistics for science variables (defined in preload)

- [nc_file_summary.py](https://github.com/data-edu-ooi/data-review-tools/blob/master/data_review//tools/nc_file_summary.py): Provides human-readable .csv file summaries of the .json file output from the nc\\_file\\_analysis tool: 1) file summary and 2) science variable summary.

### Output
- Collection of summary output files, organized by platform.