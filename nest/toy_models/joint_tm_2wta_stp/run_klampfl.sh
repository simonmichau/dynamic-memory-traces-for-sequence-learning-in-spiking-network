#!/bin/bash

echo 'Running Klampfl'
cd klampfl && python2 sem_liquid_clean.py pattern1

echo 'Plotting'
cd ..
python plot_joint.py
