# AKutils

AKUTILS is a Python package that contains functions to aid in the development of the AKVEG Map.

*Author*: Timm Nawrocki, Alaska Center for Conservation Science, University of Alaska Anchorage

*Created On*: 2023-10-10

*Last Updated*: 2026-02-03

*Description*: Functions for data acquisition, data manipulation, statistical modeling, results post-processing, and accuracy assessment for geospatial data development in Alaska.

## Getting Started
These instructions will enable you to import functions from the AKutils package into scripts. 

### Prerequisites

1. Python 3.9+
2. pip
3. arcpy
4. datetime
5. pandas
6. psycopg2
7. rasterio
8. time

### Installing

This section describes the purpose of each script and gives instructions for modifying parameters to reproduce results or adapt to another project.

```bash
pip install git+https://github.com/accs-uaa/akutils
```

## Usage
This section describes the purpose of each function and provides details of parameters and outputs.

### Functions

#### end_timing

The *end_timing* function is used to calculate the amount of time a process took to complete. The 

##### **Parameters that must be modified:**

* *iteration_start*: Specify a time captured using the time.time() function.

##### Example

```
print('Counting to 1,000,000...')
iteration_start = time.time()
count = 1
while count < 1000000:
	count += 1
end_timing(iteration_start)
```

## Credits

### Authors

* **Timm Nawrocki** - *Alaska Center for Conservation Science, University of Alaska Anchorage*

### Usage Requirements

Usage of the scripts, packages, tools, or routines included in this repository should be cited as follows:

Nawrocki, T.W. 2023. AKutils. Git Repository. Available: https://github.com/accs-uaa/akutils

### Citations

N/A

## License
[MIT](https://choosealicense.com/licenses/mit/)
