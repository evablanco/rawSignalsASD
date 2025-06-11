#!/bin/bash

SCRIPT="main.py"
LOG_DIR="normalized_sigs/logs"  # "normalized_sigs/logs"
mkdir -p $LOG_DIR

# Configuración de parámetros a probar
TP_VALUES=(60 180 300)
TF_VALUES=(60 180 300)
MODEL_VERSIONS=(1)      # Versiones del modelo
FEATS_CODES=(0 1 2 6)  # Códigos de características
MODEL_TYPES=(0 1 3)       # Tipo de modelo: 0 PM, 1 intra TL, 3 PDM
BIN_SIZES=(15)
SPLIT_CODES=(1)         # Tipos de split: 0 PM-LSS o 1 PM-SS
CW_TYPES=(1)          # Class weights: 0 ninguno, 1 balanceado, 2 custom

for TF in "${TF_VALUES[@]}"; do
  for TP in "${TP_VALUES[@]}"; do
    for MODEL_VERSION in "${MODEL_VERSIONS[@]}"; do
      for FEATS_CODE in "${FEATS_CODES[@]}"; do
        for MODEL in "${MODEL_TYPES[@]}"; do
          for BIN_SIZE in "${BIN_SIZES[@]}"; do
            for SPLIT_CODE in "${SPLIT_CODES[@]}"; do
              for CW_TYPE in "${CW_TYPES[@]}"; do
                LOG_FILE="$LOG_DIR/log_tp${TP}_m${MODEL}_tf${TF}_mv${MODEL_VERSION}_f${FEATS_CODE}_m${MODEL}_bs${BIN_SIZE}_sp${SPLIT_CODE}_cw${CW_TYPE}.txt"
                echo "Ejecutando: python $SCRIPT -tp $TP -tf $TF -v $MODEL_VERSION -f $FEATS_CODE -m $MODEL -bs $BIN_SIZE -sp $SPLIT_CODE -cw $CW_TYPE"
                python $SCRIPT -tp $TP -tf $TF -v $MODEL_VERSION -f $FEATS_CODE -m $MODEL -bs $BIN_SIZE -sp $SPLIT_CODE -cw $CW_TYPE > "$LOG_FILE" 2>&1
                echo "Resultado guardado en $LOG_FILE"
              done
            done
          done
        done
      done
    done
  done
done