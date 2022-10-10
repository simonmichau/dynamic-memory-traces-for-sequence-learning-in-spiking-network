#!/bin/bash

echo 'Running NEST'
cd nest && python nest_network.py

echo 'Plotting'
cd ..
python plot_joint.py
