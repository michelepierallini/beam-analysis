# Lumped Parameter Model

This folder models and simualtes the dynamics of flexible beams using a lumped parameter approach, whcih is one of the control-oriented modeling apporches. 

The model is heredidate from the classic rigid body formulation adjointed with spings and dapers at the (rotodoial) joints' level.

## Contents

- **`CONST.py`**  
  Contains physical and simulation constants (e.g., mass, length, stiffness, damping arrays) for the lumped parameter models.
  ToDo: replace with a `.yaml` file

- **`Fishing_class.py`**  
  Defines the `DataRobot` class, which encapsulates the state, dynamics, and integration routines for the lumped parameter robot/beam.

- **`get_dynamics.ipynb`**  
  Jupyter notebook for symbolic derivation and code generation of dynamic equations (inertia, Coriolis, gravity, stiffness, damping, Jacobian, etc.) using SymPy. 
  Also generates Python functions for fast numerical evaluation. You can use `np.`, `jnp.`, and `torch.` libraries for sppeding up the dynamics' computation.

- **`run_dyn_control_fl.py`**  
  Running dynamic simulations and control.

- **`test_dyn.py`**  
  Script for testing the dynamic functions.


- **`fishing_rod_12_joints_np/`**  
  Auto-generated Python files (e.g., `_inertia_matrix_np.py`, `_gravity_matrix_np.py`, etc.) containing fast numerical implementations of the dynamic model for a 12-joint beam.

## Usage 

`python run_dyn_control_fl.py`

## Workflow

1. **Generation**  
   Use [`get_dynamics.ipynb`](get_dynamics.ipynb) to derive and export Python functions for the dynamic model (inertia, Coriolis, gravity, etc.) for a specified number of segments.

2. **Configure**  
   Configure model parameters in [`CONST.py`](CONST.py) and initialize the robot model using [`Fishing_class.py`](Fishing_class.py).

3. **Simulations**  
   Use [`run_dyn_control_fl.py`](run_dyn_control_fl.py) or custom scripts to simulate the system, apply controls, and analyze results.


