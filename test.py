import cv2
import time


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


def detect_and_recognize_smile(gray, frame):
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        if len(smiles) > 0:
            
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            print("Smile recognized!")
            return True  
    return False  


video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    
    frame = cv2.resize(frame, (640, 480))

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    if detect_and_recognize_smile(gray, frame):
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_filename = f"smile_capture_{timestamp}.jpg"
        cv2.imwrite(image_filename, frame)
        print(f"Photo saved as {image_filename}")
        break 

    
    cv2.imshow('Smile Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()