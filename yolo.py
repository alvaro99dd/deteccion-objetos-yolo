from ultralytics import YOLO
import cv2
import math

# Iniciamos la cámara y establecemos la resolución de la imagen
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Cargamos nuestro modelo
model = YOLO("yolo-Weights/yolov8n.pt")

# Definimos las clases que queremos detectar
classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Bucle de captura
while True:
    succes, img = cap.read() # Capturamos la imagen
    results = model(img, stream=True) # Enviamos a yolo la imagen para detectar objetos
    
    # Iteramos sobre lo que detecte yolo
    for r in results:
        boxes = r.boxes # Obtenemos las cajas de los objetos detectados
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0] # Obtenemos las coordenadas de la caja
            x1, y1, x2, y2 = map(int, [x1,y1,x2,y2]) # Convertimos a enteros
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # Dibujamos la caja en la imagen
            
            # Calculamos la confianza del objeto detectado
            confidence = math.ceil(box.conf[0] * 100)
            print(confidence)
            
            # Detectamos el nombre
            cls = int(box.cls[0])
            print(classNames[cls])
            
            # Escribimos el nombre del objeto en la imagen
            org = (x1,y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", org, font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Creamos una ventana para mostrar la imagen
    cv2.imshow("Webcam", img)
    # Bucle de salida
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()