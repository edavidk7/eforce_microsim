# eForce Driverless 2024 newbie task - DV μ-sim

## Introduction

Using this miniature simulator, you will be tasked with tuning and possibly implementing the autonomous algorithm stack that's required to successfully drive around a FS Driverless track.

## Installation

1. Clone this repository onto a machine with Linux or macOS
2. Make sure to have python >= 3.10 installed (ideally 3.11 - 3.12)
3. Activate you venv if applicable or make sure your python is aliased under `python`
4. Run `python -m pip install -r requirements.txt` to install the required packages
5. run `python run.py` to start the simulation with evaluation

## Structure

- `run.py` - Main runner script for the simulation
- `user_config.py` - User configuration file for the simulation, **which you can edit**
- `mission.py` - Mission file for the simulation, **which you will be implementing**
- `bin/` - Contains the binary files for the simulator
- `helpers/` - Contains all driverless algorithms and simulator code
- `maps/` - Contains all the maps for the simulator

## Task

1. Implement the `mission.py` file to successfully drive around the track
2. Tune the PID controller in the `mission.py` file to achieve the best performance
3. Be the fastest around the track! You will be compared with others

## Helpful resources

1. https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/ControlOverview.html
2. https://www.ni.com/en/shop/labview/pid-theory-explained.html
3. https://blogs.mathworks.com/student-lounge/2022/10/03/path-planning-for-formula-student-driverless-cars-using-delaunay-triangulation/
4. https://www.youtube.com/watch?v=U6vr3iNrwRA&list=PLgnQpQtFTOGQrZ4O5QzbIHgl3b1JHimN_
5. https://numpy.org/
6. https://docs.python.org/3.11/