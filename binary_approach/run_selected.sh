#!/bin/bash

SCRIPT="main.py"
LOG_DIR="logs"
mkdir -p $LOG_DIR


# PMs

### Exp. 1

TP=60
TF=120
MODEL_VERSION=2 # 1 Fv, 2 AFV
F=0 # feature codes
M=0 # 0 PM, 1 PDM, 2 HM
B=15 # bin size
CT=1 # classification strategy
S=0 # split code

LOG_FILE="$LOG_DIR/log_tp${TP}_tf${TF}_mv${MODEL_VERSION}_ct${CT}_f${F}_m${M}_bs${B}_sp${S}_PM_v2.txt"

echo "Running: python $SCRIPT -tp $TP -tf $TF -a $MODEL_VERSION -f $F -ct $CT -m $M -bs $B -sp $S"
python $SCRIPT -tp $TP -tf $TF -a $MODEL_VERSION -f $F -ct $CT -m $M -bs $B -sp $S > "$LOG_FILE" 2>&1
echo "Output saved to $LOG_FILE"


### Exp. 2

TP=120
TF=120
MODEL_VERSION=2 # 1 Fv, 2 AFV
F=0 # feature codes
M=0 # 0 PM, 1 PDM, 2 HM
B=15 # bin size
CT=1 # classification strategy
S=0 # split code

LOG_FILE="$LOG_DIR/log_tp${TP}_tf${TF}_mv${MODEL_VERSION}_ct${CT}_f${F}_m${M}_bs${B}_sp${S}_PM_v2.txt"

echo "Running: python $SCRIPT -tp $TP -tf $TF -a $MODEL_VERSION -f $F -ct $CT -m $M -bs $B -sp $S"
python $SCRIPT -tp $TP -tf $TF -a $MODEL_VERSION -f $F -ct $CT -m $M -bs $B -sp $S > "$LOG_FILE" 2>&1
echo "Output saved to $LOG_FILE"


### Exp. 3

TP=120
TF=180
MODEL_VERSION=2 # 1 Fv, 2 AFV
F=0 # feature codes
M=0 # 0 PM, 1 PDM, 2 HM
B=15 # bin size
CT=1 # classification strategy
S=0 # split code

LOG_FILE="$LOG_DIR/log_tp${TP}_tf${TF}_mv${MODEL_VERSION}_ct${CT}_f${F}_m${M}_bs${B}_sp${S}_PM_v2.txt"

echo "Running: python $SCRIPT -tp $TP -tf $TF -a $MODEL_VERSION -f $F -ct $CT -m $M -bs $B -sp $S"
python $SCRIPT -tp $TP -tf $TF -a $MODEL_VERSION -f $F -ct $CT -m $M -bs $B -sp $S > "$LOG_FILE" 2>&1
echo "Output saved to $LOG_FILE"