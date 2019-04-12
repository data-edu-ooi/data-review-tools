# Data Review Tools
This is a collection of tools to facilitate the review of OOI 1.0 data.

## Installation Instructions
Add the channel conda-forge to your .condarc. You can find out more about conda-forge from their website: https://conda-forge.org/

`conda config --add channels conda-forge`

Clone the data-review-tools repository

`git clone https://github.com/ooi-data-lab/data-review-tools.git`

Change your current working directory to the location that you downloaded data-review-tools. 

`cd /Users/lgarzio/Documents/repo/data-review-tools/`

Create conda environment from the included environment.yml file:

`conda env create -f environment.yml`

Once the environment is done building, activate the environment:

`conda activate data-review-tools`

Install the toolbox to the conda environment from the root directory of the data-review-tools toolbox:

`pip install .`

The toolbox should now be installed to your conda environment.


## Folders
### cruise\_data\_compare
Contains tools for comparing OOI platform data with data collected on OOI maintenance cruises.

### data_review
Tools for automated analysis of OOI netCDF files downloaded from uFrame.

### functions
Contains common functions used by multiple tools.

### plotting
Contains tools for plotting data.