# Simulate correlated ground-motion IMs conditional on recordings

**Special example for the 2023 earthquake at the border of Turkey and Syria** 

key features:
- Import rupture information and station data (including recordings) from [USGS shakemap system](https://earthquake.usgs.gov/data/shakemap/).
- Uses the [openquake]() engine for geo-computations and ground-motion models.
- Illustration on how to generate spatially correlated intensity measures that take into account the recorded values from the earthquake.
- Shakemap results from USGS do not (yet) allow for this.
- Currently only considers spatial correlation and no cross-correlation. Therefore: Only one intensity measure at a time!

### Installation
I recommend to install the required packages in a virtual environment (for example using miniconda).

1. Clone the repo.
2. Create virtual conda environment: `conda create -f environment.yml`


### Limitation and ToDo's

...


### Licence
...

