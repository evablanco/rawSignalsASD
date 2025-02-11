#!/bin/bash

# Ruta al script de Python
SCRIPT="main.py"

# Directorio donde se guardarán los logs
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Combinaciones de parámetros
TP_VALUES=(180)
TF_VALUE=180
MODEL_VERSIONS=(1) # (1 2)
F_VALUES=(0) # (0 1 2 3 4 5 6 7 8)
M_VALUES=(0 1) # (0 1)

# Iterar sobre las combinaciones y ejecutar el script
for TP in "${TP_VALUES[@]}"; do
    for MODEL_VERSION in "${MODEL_VERSIONS[@]}"; do
        for F in "${F_VALUES[@]}"; do
            for M in "${M_VALUES[@]}"; do
                # Nombre del archivo de log
                LOG_FILE="$LOG_DIR/log_tp${TP}_tf${TF_VALUE}_mv${MODEL_VERSION}_f${F}_m${M}.txt"

                # Comando a ejecutar
                echo "Ejecutando: python $SCRIPT -tp $TP -tf $TF_VALUE -a $MODEL_VERSION -f $F -m $M"
                python $SCRIPT -tp $TP -tf $TF_VALUE -a $MODEL_VERSION -f $F -m $M > "$LOG_FILE" 2>&1

                # Mensaje de confirmación
                echo "Resultado guardado en $LOG_FILE"
            done
        done
    done
done