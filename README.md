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

![Schema](https://github.com/bodlukas/ground-motion-simulation-shakemap/blob/main/data/ShakemapSim.png)

The tool uses the [openquake engine](https://github.com/gem/oq-engine#openquake-engine) for geo-computations and implementations of ground-motion models. In the provided example we import rupture information and station data (including recorded IMs) from the [USGS shakemap system](https://earthquake.usgs.gov/data/shakemap/). 

The tool was developed by the research group of [Prof. Bozidar Stojadinovic](https://stojadinovic.ibk.ethz.ch/) at the Department of Civil, Environmental and Geomatic Engineering at ETH Zürich. 

### Local installation
The required dependencies for a local setup are listed in the `environment.yml` file. We recommend to first create a new virtual (mini-)conda environment.

### Limitation
The current implementation only considers spatial correlation and no spatial cross-correlation. Therefore, it can only be used to simulate and predict the same intensity measure (e.g., PGA) at multiple sites.

>**_Important_** Note that the provided [example data sets](data/) are only for illustrative purposes. Users should definitely check for updated rupture and station data from this [event](https://earthquake.usgs.gov/earthquakes/eventpage/us6000jllz/shakemap/metadata). The provided vs30 values were retrieved from [USGS](https://earthquake.usgs.gov/data/vs30/), and are based on geographic slope. Users should carefully assess whether these estimates are representative for that region. 

### Structure
- [modules](modules/) contains the main objects, functions and methods to perform the computation including (relatively) rich documentation.
- [data](data/) contains some example data sets used to explain the method. When working with these data sets, please take into account the notes from the section above!
- [ShakemapSim_Example.ipynb](ShakemapSim_Example.ipynb) is the main notebook that explains the workflow on the example of the 2023 M7.8 earthquake at the border of Turkey and Syria. As mentioned above, the notebook can be opened in colab. 
- [theoretical_background.ipynb](theoretical_background.ipynb) provides a short mathematical overview on ground-motion models, spatial correlation models and the shakemap algorithm.
- [test.ipynb](test.ipynb) contains a short script that uses simulated data to test the implemented shakemap algorithm.
- [utils.py](utils.py) contains several functions to import rupture and station information from USGSs ShakeMap system.

### Acknowledgments
We gratefully acknowledge support from the ETH Risk Center ("DynaRisk", Grant Nr. 395 2018-FE-213). 

### Licence
The code written by us is licensed under the GNU Affero General Public License v3.0 (AGPL 3.0) license. Feel free to use it based on the terms and conditions listed in the LICENSE file and reference the doi stated above. I intend this code to be used for NON-COMMERCIAL uses, if you'd like to use it for commercial uses, please contact Lukas Bodenmann via bodenmann (at) ibk.baug.ethz.ch . Remember that the workflow heavily relies on OpenQuake, which is also licensed under the AGPL 3.0 license.
