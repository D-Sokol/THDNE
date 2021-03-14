#!/bin/bash

# Removes from active directory all regular files except itself and hidden ones.
# In the current case this script will remove all generated images and saved models.

find . -type f  -not -name '.*'  -not -samefile "$0"  -delete
