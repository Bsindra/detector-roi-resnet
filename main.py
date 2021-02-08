import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

video_source = ('source.mp4')
capture = cv2.VideoCapture(video_source)
model = ResNet50(weights='imagenet')
threshold = 0.5
font = cv2.FONT_HERSHEY_DUPLEX

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

conectado, video = capture.read()

bounding_box = cv2.selectROI("Classificador", video, False)

x1, y1, w, h = (bounding_box[i] for i in range(4))

x2 = x1 + w
y2 = y1 + h

# Saída

frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
fps = capture.get(cv2.CAP_PROP_FPS)

nome_arquivo = 'resultado.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_altura = video.shape[0]
video_largura = video.shape[1]
saida_video = cv2.VideoWriter(nome_arquivo, fourcc, fps, (video_largura, video_altura))
c = 0

for c in tqdm(range(frame_count)):
    conectado, frame = capture.read()
    if conectado == False:
        break
    if conectado == True:
        
        interest = frame[y1:y2, x1:x2]
        
        img_array = cv2.cvtColor(interest, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = np.expand_dims(img_array, axis=0)

        img_array = preprocess_input(img_array)
        
        predict = model.predict(img_array)
        result = decode_predictions(predict, top = 1)
        
        ths = result[0][0][2]
        label = result[0][0][1]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        if label in cars:
            cv2.putText(frame, "Carro", (x1, y1-5), font, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(round(ths * 100)) + "%", (x2-60, y1-5), font, 1, (0, 255, 0), 2)
        
        else:
            cv2.putText(frame, 'Outro', (x1, y1-5), font, 1, (0, 0, 255), 2)

        saida_video.write(frame)

        c += 1
        
    if cv2.waitKey(1) == 27:
        break
            
capture.release()
saida_video.release()
cv2.destroyAllWindows()
