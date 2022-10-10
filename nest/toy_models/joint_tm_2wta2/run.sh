#!/bin/bash

echo 'Running Klampfl'
cd klampfl && python2 sem_liquid_clean.py pattern1

echo 'Running NEST'
cd ../nest/ && python nest_network.py

echo 'Plotting'
cd ..
python plot_joint.py
