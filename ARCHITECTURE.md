## Architecture Overview

PyMRI_PSO is structured into modular components that separate configuration,

PyMRI_PSO/

├──
├── pso_main_*.py           # Main PSO control scripts
├── pso_preparation_*.py    # PSO environment and system parameter preparation
├── pso_model_parameters.py # Definition of PSO model parameters and intervals
├── pso_core.py             # Core PSO implementation
├── pso_visualization.py    # Visualization utilities
├── mwf_modeling.py         # MWF modeling based on in-vivo or atlas MRI data
├── help_tools.py           # Helper utilities
├── pso_examples.ipynb      # Example usage and documentation

## Module Overview

1. pso_main_*.py  

  Main entry script for performing particle swarm optimization (PSO) on MRI invivo data.
  All functionality is controlled via a user-defined configuration dictionary. Contains:
  
    a) Preparation of the PSO environment based on the configuration
    b) Setup of parallel execution on multi-core systems
    c) Execution of PSO in the selected mode, including saving/returning results

2. pso_preparation_*.py  

  Preparation layer for PSO-based MRI inversion. Purpose:
  
    a) Loading observed MRI data (in-vivo and/or atlas-based)
    b) Loading or generating binary masks
    c) Construction of the full system model cube
    d) Data preprocessing and B1 handling
    d) Population of the system parameter dictionary for selected slices

3. pso_model_parameters.py  

  Definition of model and optimization parameters, including:
  	
	a) InversionParams - signal model parameters
	b) PSOParams       - literature-based weight factors
	c) T1/T2/T2SParams - parameter interval definitions of model vectors

4. pso_core.py
  
  Core implementation of particle swarm optimization, containing:
  
    a) PSO class implementation
	b) Optimization routines
	c) Internal PSO-related helper methods

5. mwf_modeling.py

  Modeling of Myelin Water Fraction (MWF) based on relaxation MRI data (T2,T2star).

  Provides three classes:
  
    a) signal_models   - T2 and T2* signal models (EPG and simple exponential)
    b) mwf_data        - data loading utilities
    c) mwf_analysis    - MWF estimation and analysis methods
	
6. pso_visualization.py   

  Exemplary visualization utilities for single and joint inversion results, including:

    a) Parameter maps (e.g., MWF, misfit)
    b) Convergence diagnostics
    c) Pareto plots

7. help_tools.py

  Helper functions supporting efficient PSO execution on in-vivo and atlas MRI data.
 
8. pso_examples.ipynb

	Comprehensive demonstration and documentation notebook, including:
	
    a) Detailed documentation cell explaining architecture, workflow, and running modes  
    a) Demonstrations of different execution modes (slice-parallel, pixel-iterative, iteration-test)  
    a) Example configurations for single and joint inversion  
    a) Visualization of inversion results and diagnostic plots  