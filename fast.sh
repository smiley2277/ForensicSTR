#!/bin/bash

FILENAME=$(find ./testset -name "*.csv" -type f)
for element in $FILENAME ; do
    chmod +x $element
    FILENAME_1=${element##*/}
    FILENAME_2=${FILENAME_1%%.*}
    FILENAME_3=${FILENAME_2%%_predict}
    echo $FILENAME_1
    python3 main_v2.py --mode test --test_file ./testset/$FILENAME_1
    python3 organize.py --input $FILENAME_3
    done
