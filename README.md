# Memory of Motion for Warm-starting Trajectory Optimization#

We use the concept of memory of motion to warm-start trajectory
optimization. In this work we use TrajOpt
(http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/) as the 
trajectory optimizer. A set of problems are solved online using TrajOpt, 
and we use function approximators to predict the initial guesses online 
for warm-starting TrajOpt.

## Installation Procedure ##
Install TrajOpt: 
```bash
see http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/
```
Install pbdlib-python for BGMR: 
```bash
see https://gitlab.idiap.ch/rli/pbdlib-python
```

Install GPy:
```bash
see https://github.com/SheffieldML/GPy
```

## Who do I talk to? ##

- Teguh Santoso Lembono - 
Idiap Research Institute, Martigny, Switzerland
Ecole Polytechnique Federale Lausanne
teguh.lembono@idiap.ch
