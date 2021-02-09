import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Configurações Iniciais
video_source = ('source.mp4')
model = ResNet50(weights='imagenet')
font = cv2.FONT_HERSHEY_DUPLEX

# Lista de objetos que devem ser classificados como carros
cars = [
    'tractor',
    'ambulance',
    'tow_truck',
    'sports_car',
    'pickup',
    'school_bus',
    'garbage_truck',
    'minibus',
    'minivan',
    'streetcar',
    'recreational_vehicle',
    'moving_van',
    'passenger_car',
    'Model_T',
    'trolleybus',
    'police_van',
    'limousine'
]

# Abrindo o vídeo para obter informações
capture = cv2.VideoCapture(video_source)
conectado, video = capture.read()
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
fps = capture.get(cv2.CAP_PROP_FPS)

# Definindo Região de Interesse
bounding_box = cv2.selectROI("Regiao de interesse", video, False)
x1, y1, w, h = (bounding_box[i] for i in range(4))
x2 = x1 + w
y2 = y1 + h
cv2.destroyAllWindows()

# Saída
nome_arquivo = 'resultado.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_altura = video.shape[0]
video_largura = video.shape[1]
saida_video = cv2.VideoWriter(nome_arquivo, fourcc, fps, (video_largura, video_altura))

# Inicializando Variável
c = 0

# Loop Principal
for c in tqdm(range(frame_count)):
    #Checa se arquivo já terminou de rodar e capta frame como imagem
    conectado, frame = capture.read()

    if not conectado:
        break

    else: 
        #Definindo e desenhando regiao de interesse
        interest = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        #Processamento da imagem
        img_array = cv2.cvtColor(interest, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        predict = model.predict(img_array)
        result = decode_predictions(predict, top = 1)
        
        #Retirando resultados da regiao de interesse
        confianca = result[0][0][2]
        categoria = result[0][0][1]
        
        #Analisa se Regiao de Interesse possui ou não um carro
        if categoria in cars:
            cv2.putText(frame, "Carro", (x1, y1-5), font, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(round(confianca * 100)) + "%", (x2-60, y1-5), font, 1, (0, 255, 0), 2)
        
        else:
            cv2.putText(frame, "Outros", (x1, y1-5), font, 1, (0, 0, 255), 2)

        #Escreve o frame na saída
        saida_video.write(frame)

        #Contador para a barra de progresso
        c += 1
        
    if cv2.waitKey(1) == 27:
        break

# Liberando Arquivos e Finalizando
capture.release()
saida_video.release()
cv2.destroyAllWindows()
