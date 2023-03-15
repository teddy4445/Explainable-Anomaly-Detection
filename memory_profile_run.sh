#!/bin/bash
echo "Running 'main.py' and generating memory profile analysis"
mprof run main.py
mprof plot
