#!/bin/bash

# Python script path
SCRIPT="main.py"

# Directory to store the logs
LOG_DIR="logs"
mkdir -p $LOG_DIR

#Parameter combinations
TP_VALUES=(60 120 180)
TF_VALUE=180
MODEL_VERSIONS=(1 2)

# Iterate over the combinations and run the script
for TP in "${TP_VALUES[@]}"; do
    for MODEL_VERSION in "${MODEL_VERSIONS[@]}"; do
        LOG_FILE="$LOG_DIR/log_tp${TP}_tf${TF_VALUE}_mv${MODEL_VERSION}.txt"

        echo "Running: python $SCRIPT -tp $TP -tf $TF_VALUE -a $MODEL_VERSION"
        python $SCRIPT -tp $TP -tf $TF_VALUE -a $MODEL_VERSION > "$LOG_FILE" 2>&1

        echo "Output saved in $LOG_FILE"
    done
done
