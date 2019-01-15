# Data Download
This toolbox contains examples and tools to download data from uFrame via the OOI Machine to Machine (M2M) interface. 

### Main Functions
- [download_data_local.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/download_data_local.py): Imports tools that build netCDF data request urls for science streams of the instruments input by the user (does not check against the Datateam Database), sends those requests (if prompted), and downloads the netCDF, provenance, and annotation files to a local directory.

- [download_data_nc_interactive.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/download_data_nc_interactive.py): Imports tools that compare the [OOI Datateam Database](http://ooi.visualocean.net/) to [ooinet.oceanobservatories.org](https://ooinet.oceanobservatories.org/), builds netCDF data request urls, sends the requests, and writes a summary of the status of the data requests. This interactive tool prompts the user for inputs. Data can be downloaded for multiple platforms, instruments, data streams, etc.

- [download_data_ooi1_0.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/download_data_ooi1_0.py): Imports tools to use the [data_review_list](https://github.com/ooi-data-lab/data-review-tools/tree/master/review_list) to download OOI 1.0 datasets via the OOI M2M interface. If a file containing datasets to download is not provided, the script will be interactive. An example input file: [data_download.csv](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/example_files/data_download.csv)

### Scripts
- [data_request_tools.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/scripts/data_request_tools.py): A collection of tools used to create data request urls.

- [data_request_urls_nocheck.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/scripts/data_request_urls_nocheck.py): Identifies all data available for download listed in the OOI GUI data catalog (https://ooinet.oceanobservatories.org/api/uframe/stream) and builds data request urls (for netCDF files) for the science streams of the instruments input by the user. This script does not check against the Datateam Database.

- [data_request_urls_ooi1_0.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/scripts/data_request_urls_ooi1_0.py): Builds data request urls (for netCDF files) for OOI 1.0 science streams to download. The urls include time constraints so only deployments during OOI 1.0 will be downloaded.

- [data_request_urls.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/scripts/data_request_urls.py): Compares the reference designators, methods and streams in the [OOI Datateam Database](http://ooi.visualocean.net/) to those available in the OOI GUI data catalog (https://ooinet.oceanobservatories.org/api/uframe/stream), and builds data request urls (for netCDF files) for the science streams of the instruments input by the user

- [interactive_inputs.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/scripts/interactive_inputs.py): Filters the OOI Datateam Database and provides interactive inputs for data download.

- [send_data_requests_nc.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/scripts/send_data_requests_nc.py): Sends data request urls and provides a summary output that contains the links to the THREDDS data server.

- [thredds_download_nc.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/scripts/thredds_download_nc.py): Downloads netCDF, provenance, and annotation files from a THREDDS directory to a local directory.


### Example files
- [data_download.csv](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/example_files/data_download.csv): Example csv file for optional input to [download_data_ooi1_0.py](https://github.com/ooi-data-lab/data-review-tools/blob/master/data_download/download_data_ooi1_0.py)

### Notes
- In order to access OOI data through the OOI API, you will need to create a user account on [ooinet.oceanobservatories.org](https://ooinet.oceanobservatories.org/). Your API Username and Token can be found in your User Profile.