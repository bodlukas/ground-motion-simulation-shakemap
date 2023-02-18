## ShakemapSim: Simulate spatially correlated ground-motion intensity measures conditional on recordings

**Example for the 2023 earthquake at the border of Turkey and Syria** 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7646888.svg)](https://doi.org/10.5281/zenodo.7646888)

ShakemapSim is a user-friendly tool to generate spatially correlated fields of ground-motion intensity measures (IMs) conditional on available recordings from a seismic network. This is particularly useful for validation and development of models for damage, loss and recovery predictions using data gathered after an event. 

If you use this tool in your work please please cite it as:
> Bodenmann, Lukas, and Stojadinović, Božidar. (2023). ShakemapSim: Simulate spatially correlated ground-motion intensity measures conditional on recordings (v1.0). Zenodo. https://doi.org/10.5281/zenodo.7646888

**Quick start** Open the notebook [ShakemapSim_Example.ipynb](ShakemapSim_Example.ipynb) on a hosted Jupyter notebook service (e.g., Google Colab). It does not require any local python setup and you can immediately start to customize the models and perform the computations yourself. It explains how to: 
1. import earthquake rupture information and recorded ground-motion IMs,
2. specify (and customize) which ground-motion models and spatial correlation models are used to compute the shakemap,
3. specify sites at which we would like to predict ground-motion IMs,
4. use **ShakemapSim** to predict and sample ground-motion IMs. 

The tool uses the [openquake engine](https://github.com/gem/oq-engine#openquake-engine) for geo-computations and implementations of ground-motion models. In the provided example we import rupture information and station data (including recorded IMs) from the [USGS shakemap system](https://earthquake.usgs.gov/data/shakemap/). 

![alt text](https://github.com/bodlukas/ground-motion-simulation-shakemap/blob/main/data/ShakemapSim.png)

The tool was developed by the research group of [Prof. Bozidar Stojadinovic](https://stojadinovic.ibk.ethz.ch/) at the Department of Civil, Environmental and Geomatic Engineering at ETH Zürich. 

### Local installation
The required dependencies for a local setup are listed in the `environment.yml` file. I recommend to create a virtual (mini-)conda environment. 

### Limitation
The current implementation only considers spatial correlation and no spatial cross-correlation. Therefore, it can only be used to simulate and predict the same intensity measure (e.g., PGA) at multiple sites.

**Important** Note that the provided [example data sets](data/) are only for illustrative purposes. Users should definitely check for updated rupture and station data from this [event](https://earthquake.usgs.gov/earthquakes/eventpage/us6000jllz/shakemap/metadata). The provided vs30 values were retrieved from [USGS](https://earthquake.usgs.gov/data/vs30/), and are based on geographic slope. Users should carefully assess whether these estimates are representative for that region. 

### Acknowledgments
We gratefully acknowledge support from the ETH Risk Center ("DynaRisk", Grant Nr. 395 2018-FE-213). 

### Licence
The code is licensed under the Apache2.0 license. Feel free to use it based on the terms and conditions listed in the LICENSE.md file and reference the doi stated above. I intend this code to be used for NON-COMMERCIAL uses, if you'd like to use it for commercial uses, please contact Lukas Bodenmann via bodenmann (at) ibk.baug.ethz.ch .

