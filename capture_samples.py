import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime
import time  # Importamos el módulo time


def capture_samples(path, margin_frame=2, min_cant_frames=5, hold_time=0.3):
    '''
    ### CAPTURA DE MUESTRAS PARA UNA PALABRA
    Recibe como parámetro la ubicación de guardado y guarda los frames
    
    `path` ruta de la carpeta de la palabra \n
    `margin_frame` cantidad de frames que se ignoran al comienzo y al final \n
    `min_cant_frames` cantidad de frames minimos para cada muestra \n
    `hold_time` tiempo en segundos para mantener la captura si la mano desaparece momentáneamente
    '''
    create_folder(path)
    
    count_frame = 0
    frames = []
    num_captures = 0
    last_detection_time = None  # Variable para almacenar el tiempo de la última detección
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            image = frame.copy()
            if not ret: break
            
            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results):
                count_frame += 1
                last_detection_time = time.time()  # Actualiza el tiempo de la última detección
                if count_frame > margin_frame:
                    cv2.putText(image, f'Capturando... ({count_frame} capturas)', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))
                
            else:
                if last_detection_time and (time.time() - last_detection_time <= hold_time):
                    # Continuar capturando frames si no ha pasado más de 'hold_time' segundos desde la última detección
                    count_frame += 1
                    if count_frame > margin_frame:
                        cv2.putText(image, f'Capturando... ({count_frame} capturas)', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                        frames.append(np.asarray(frame))
                else:
                    if len(frames) > min_cant_frames + margin_frame:
                        frames = frames[:-margin_frame]
                        today = datetime.now().strftime('%y%m%d%H%M%S%f')
                        output_folder = os.path.join(path, f"sample_{today}")
                        create_folder(output_folder)
                        save_frames(frames, output_folder)
                        num_captures += 1
                    
                    frames = []
                    count_frame = 0
                    cv2.putText(image, f'Listo para capturar... ({num_captures} muestras)', FONT_POS, FONT, FONT_SIZE, (0,220, 100))
            
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

# Cuenta el número de carpetas generadas al final de la toma de capturas
def count_generated_folders(path):
    # Lista todos los elementos en el directorio y filtra solo carpetas
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return len(folders)

if __name__ == "__main__":
    word_name = "software" # Palabra a capturar
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
    # Llamada a la función para contar las carpetas generadas
    num_folders = count_generated_folders(word_path)
    print(f"Carpetas generadas en '{word_name}': {num_folders}")
