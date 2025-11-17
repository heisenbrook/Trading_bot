#!/bin/bash

DURATION_MAX="30m"
PYTHON_EXE="/home/matteo-vannini/anaconda3/bin/python3"
SCRIPT_PATH="/home/matteo-vannini/Scrivania/Trading_bot/t_learning.py"

echo "tf" | timeout $DURATION_MAX $PYTHON_EXE $SCRIPT_PATH

sleep 5m

echo "lstm" | timeout $DURATION_MAX $PYTHON_EXE $SCRIPT_PATH