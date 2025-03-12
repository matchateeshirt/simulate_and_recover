#!/bin/bash

echo "Running unit tests..."
python3 -m unittest discover -s test -p "test_*.py"

echo "Tests completed."
