# Beam Analysis

This repository contains a simple scratch Python-based structural analysis tool for evaluating mechanical properties of beams with variable cross-sections and tip forces. The implementation supports customizable parameters such as beam geometry, material properties, and applied loads.

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Features](#features)
- [Example Usage](#example-usage)
- [File Structure](#file-structure)

## Setup Instructions

### Prerequisites
- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) installed globally

### Project Initialization
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yursds/beam-analysis.git
   cd beam-analysis
   ```
2. **Install dependencies using `uv`**:
   ```bash
   uv sync
   ```
   This will install all required packages specified in the project's dependency configuration.

3. **Run the main script**:
   ```bash
   uv run python beam_analysis.py
   ```
   This will execute the example usage provided in the code.

## Features
- Calculation of cross-sectional area and moment of inertia along the beam.
- Computation of angular acceleration, internal forces (shear and bending moment), and stresses.
- Elastic curve (deflection) calculation using curvature integration.
- Strain energy absorption estimation.
- Plotting capabilities for visualizing results.

## Example Usage
The script includes an example at the bottom (`if __name__ == "__main__"`), which analyzes a beam with configurable materials like steel, aluminum, fiberglass, carbon fiber, and PVC.

Each material is evaluated and compared based on:
- Mass
- Stress distribution
- Deflection
- Energy absorption

Plots are generated to visualize:
1. Bending Moment $M(x)$
2. Maximum Stress $\sigma_{max}(x)$
3. Elastic Curve

## File Structure
- `scripts/beam_analysis.py`: Main module containing the `BeamAnalysis` class and example usage.
