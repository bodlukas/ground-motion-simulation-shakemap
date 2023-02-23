# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v1.2] - 2023-02-23

### Added 

- Shakemap computation algorithm of [Engler et al. (2022)](https://doi.org/10.1785/0120210177). In v1.1 the method of [Worden et al. (2018)](https://doi.org/10.1785/0120170201) was the only option. 
- `rupture.json` and `stationlist.json` from USGS ShakeMap version 13. In v1.1 we used version 12 data which can still be accessed via `rupture_v12.json` and `stationlist_v12.json`.
- changelog

### Changed

- Name of shakemap computation object to accommodate both methods: `Shakemap_EnglerEtAl2022()` and `Shakemap_WordenEtAl2018()` instead of `Shakemap()`.
- Updated `utils.get_finite_fault()` for the version 4 finite fault geometry from USGS.
- Updated theoretical background doc with the newly added method `Shakemap_EnglerEtAl2022()` 
- Added the second method to the ShakemapSim_Example notebook.
- License to comply with OpenQuake license

### Fixed

- Bug in computation of mean parameter for HeresiMiranda2019 model. The bug only mattered for `mode='mean'`. Default was `mode='median'`.

## [v1.1] - 2023-02-19

### Added

- Theoretical background document `theoretical_background.md`.

### Changed

- Text changes in readme, documentation and notebook explanations.

## [v1.0] - 2023-02-18
- Initial version

[v1.2]: https://github.com/bodlukas/ground-motion-simulation-shakemap/compare/v1.0...v1.2
[v1.1]: https://github.com/bodlukas/ground-motion-simulation-shakemap/compare/v1.0...v1.1
[v1.0]: https://github.com/bodlukas/ground-motion-simulation-shakemap/releases/tag/v1.0 
