import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_utils

def get_data_example(data, subject_ID, session_ID, start_seconds=0, interval_seconds=3, tp=60, tf=180):
    """
    Genera una figura que visualiza las señales de un sujeto y sesión específicos,
    incluyendo las ventanas de observación (tp) y predicción (tf).

    Args:
        data (dict): Diccionario con los datos de todos los sujetos y sesiones.
        subject_ID (str): ID del sujeto a visualizar.
        session_ID (str): ID de la sesión a visualizar.
        start_seconds (int): Segundo inicial del segmento de interés.
        interval_seconds (int): Duración del segmento de interés (en segundos).
        tp (int): Tamaño de la ventana de observación (en segundos).
        tf (int): Tamaño de la ventana de predicción (en segundos).

    """
    # Extraer datos del sujeto y sesión
    subject_data = data.get(subject_ID, {})
    session_data = subject_data.get(session_ID, {})

    if session_data.empty:
        raise ValueError(f"No se encontraron datos para el sujeto {subject_ID}, sesión {session_ID}.")

    # Extraer señales y etiquetas
    eda_data = session_data['EDA']
    bvp_data = session_data['BVP']
    acc_x_data = session_data['ACC_X']
    acc_y_data = session_data['ACC_Y']
    acc_z_data = session_data['ACC_Z']

    # Calcular el intervalo temporal usando pd.Timedelta
    start_time = eda_data.index[0] + pd.Timedelta(seconds=start_seconds)
    end_time = start_time + pd.Timedelta(seconds=interval_seconds)
    print(f"Intervalo de tiempo seleccionado: {start_time} - {end_time}")

    # Filtrar los datos dentro del intervalo de tiempo
    eda_filtered = eda_data[(eda_data.index >= start_time) & (eda_data.index < end_time)]
    eda_filtered.columns = ['EDA']
    bvp_filtered = bvp_data[(bvp_data.index >= start_time) & (bvp_data.index < end_time)]
    bvp_filtered.columns = ['BVP']
    acc_x_filtered = acc_x_data[(acc_x_data.index >= start_time) & (acc_x_data.index < end_time)]
    acc_x_filtered.columns = ['ACC_X']
    acc_y_filtered = acc_y_data[(acc_y_data.index >= start_time) & (acc_y_data.index < end_time)]
    acc_y_filtered.columns = ['ACC_Y']
    acc_z_filtered = acc_z_data[(acc_z_data.index >= start_time) & (acc_z_data.index < end_time)]
    acc_z_filtered.columns = ['ACC_Z']

    if eda_filtered.empty or bvp_filtered.empty or acc_x_filtered.empty or acc_y_filtered.empty or acc_z_filtered.empty:
        raise ValueError("No hay datos en el intervalo especificado.")

    # Crear la figura
    plt.figure(figsize=(12, 10))

    # Subplot para EDA
    plt.subplot(5, 1, 1)
    plt.plot(eda_filtered.index, eda_filtered, label="EDA", color="blue")
    plt.axvspan(start_time, start_time + pd.Timedelta(seconds=tp), color='green', alpha=0.3, label="Observación (tp)")
    plt.axvspan(start_time + pd.Timedelta(seconds=tp), start_time + pd.Timedelta(seconds=tp + tf), color='red', alpha=0.3, label="Predicción (tf)")
    plt.title("EDA")
    plt.ylabel("Amplitude")
    plt.legend()

    # Subplot para BVP
    plt.subplot(5, 1, 2)
    plt.plot(bvp_filtered.index, bvp_filtered, label="BVP", color="orange")
    plt.axvspan(start_time, start_time + pd.Timedelta(seconds=tp), color='green', alpha=0.3)
    plt.axvspan(start_time + pd.Timedelta(seconds=tp), start_time + pd.Timedelta(seconds=tp + tf), color='red', alpha=0.3)
    plt.title("BVP")
    plt.ylabel("Amplitude")
    plt.legend()

    # Subplot para ACC_X
    plt.subplot(5, 1, 3)
    plt.plot(acc_x_filtered.index, acc_x_filtered, label="ACC_X", color="purple")
    plt.axvspan(start_time, start_time + pd.Timedelta(seconds=tp), color='green', alpha=0.3)
    plt.axvspan(start_time + pd.Timedelta(seconds=tp), start_time + pd.Timedelta(seconds=tp + tf), color='red', alpha=0.3)
    plt.title("ACC_X")
    plt.ylabel("Acceleration")
    plt.legend()

    # Subplot para ACC_Y
    plt.subplot(5, 1, 4)
    plt.plot(acc_y_filtered.index, acc_y_filtered, label="ACC_Y", color="brown")
    plt.axvspan(start_time, start_time + pd.Timedelta(seconds=tp), color='green', alpha=0.3)
    plt.axvspan(start_time + pd.Timedelta(seconds=tp), start_time + pd.Timedelta(seconds=tp + tf), color='red', alpha=0.3)
    plt.title("ACC_Y")
    plt.ylabel("Acceleration")
    plt.legend()

    # Subplot para ACC_Z
    plt.subplot(5, 1, 5)
    plt.plot(acc_z_filtered.index, acc_z_filtered, label="ACC_Z", color="cyan")
    plt.axvspan(start_time, start_time + pd.Timedelta(seconds=tp), color='green', alpha=0.3)
    plt.axvspan(start_time + pd.Timedelta(seconds=tp), start_time + pd.Timedelta(seconds=tp + tf), color='red', alpha=0.3)
    plt.title("ACC_Z")
    plt.ylabel("Acceleration")
    plt.xlabel("Timestamp (ms)")
    plt.legend()

    plt.tight_layout()
    plt.show()


### TEST ###
freq = 32
data_path_resampled = './dataset_resampled/'
ds_path = data_path_resampled + "dataset_" + str(freq) + "Hz.csv"
data_dict = data_utils.load_data_to_dict(ds_path)
get_data_example(
    data=data_dict,
    subject_ID="1223.01",
    session_ID="01",
    start_seconds=60,  # Segmento desde 10 segundos
    interval_seconds=10,  # Intervalo de 5 segundos
    tp=60,  # Ventana de observación de 3 segundos
    tf=180   # Ventana de predicción de 2 segundos
)