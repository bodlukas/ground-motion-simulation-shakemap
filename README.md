# Simulate correlated ground-motion intensity measures conditional on recorded values

**Special example for the 2023 earthquake at the border of Turkey and Syria** 

key features:
- How to import rupture information and station data (including recordings) from USGS shakemap system.
- Illustration on how to generate spatially correlated intensity measures that take into account the recorded values from the earthquake.
- Shakemap results from USGS do not (yet) allow for this.
- Uses the openquake engine for geo-computations and ground-motion models.
- Currently only considers spatial correlation and no cross-correlation. Therefore: Only one intensity measure at a time!
