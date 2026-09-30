# Assessment of Coupled Finite-Volume Framework

OpenFOAM case files and results for assessing a coupled finite-volume (interFoam + beam-mooring) framework for modelling moored floating photovoltaic (FPV) platforms, developed for the Queen's University Belfast NEXSYS Semi-Sub FPV Collaboration.

The case models a moored semi-submersible floating photovoltaic platform under regular waves, with mooring lines represented as flexible beams pretensioned to ~19.5 N. Numerical results are compared against physical wave-tank experiments across a matrix of wave frequency/amplitude combinations.

## Repository layout

```
caseFiles/
  baseCase/          Full-length OpenFOAM case (mesh, dictionaries, run scripts)
  baseCase_short/     Shorter-duration variant of the case for quick testing
  wave_inputs/        CSV tables of wave cases (frequency, amplitude, period) run through each case

results/
  rawResults/
    experimentResults/       Wave-tank experiment data, one CSV per wave case
    numericalModelResults/   OpenFOAM output per wave case (e.g. F3A3, F5A6, ...)
  postProcessedResults/
    postProcessing_script.py  Post-processes raw results into comparison plots
    <waveCase>_plots/          Per-wave-case plots (heave, pitch, surge, tensions, wave elevation, FFTs)
    colourMaps/                Error/difference colour maps across the wave case matrix
    ResultsTables.pdf          Summary results tables
```

Wave case naming follows `F<frequency>A<amplitude>`, e.g. `F5A6` = 0.5 Hz wave frequency, 6 cm nominal wave amplitude (see `caseFiles/wave_inputs/wave_inputs.csv` for the full mapping to input wave height and period).

## Running a case

Each case directory (`caseFiles/baseCase`, `caseFiles/baseCase_short`) is a standard OpenFOAM case:

```
cd caseFiles/baseCase
./Allrun_postProcess
```

`Allrun_postProcess` builds the mesh (`blockMesh`, `snappyHexMesh`), sets the initial free surface, decomposes the domain (including the mooring beam regions), runs `interFoam` in parallel, and then runs post-processing. See `run.slurm` for the SLURM batch submission used on HPC, and each case's own `README.md` for case-specific setup notes (beam properties, wave settings).

## Post-processing

`results/postProcessedResults/postProcessing_script.py` reads raw experiment and simulation data from `results/rawResults/` and generates the comparison plots and colour maps under `results/postProcessedResults/`.

## Requirements

- OpenFOAM v2312 (as sourced in `run.slurm`; other versions are noted but commented out)
- Python 3 with `numpy`, `pandas`, `matplotlib`, `scipy` for post-processing
