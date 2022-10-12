#!/bin/bash

conda init bash

echo 'Running Klampfl'
conda deactivate
conda activate dynamic-memory-traces-for-sequence-learning-in-spiking-networks
cd klampfl && python2 sem_liquid_clean.py pattern1

echo 'Running NEST'
conda deactivate
conda activate nestenv
cd ../nest/ && python nest_network.py

echo 'Plotting'
cd ..
python plot_joint.py
