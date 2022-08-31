from tokenize import Double
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
 
mp_hands = mp.solutions.hands
#hands = mp_hands.Hands()
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

mp_draw = mp.solutions.drawing_utils

webcam = cv2.VideoCapture('videos/02_360p.mp4')
 
w,h = 540,360
while True: 
    success, image = webcam.read()
    if success == False :
        break
        
    image_resize = cv2.resize(image, (w, h))
    image_rgb = cv2.cvtColor(image_resize ,cv2.COLOR_BGR2RGB)  
    results = hands.process(image_rgb)

    #print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        # Both Hands are present in image(frame)
                     
        for landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image_resize, landmark, mp_hands.HAND_CONNECTIONS) 

        if len(results.multi_handedness) == 2:
            cv2.putText(image_resize, 'Both Hands', (250, 50),cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
 
        else:
            try:
                for i in results.multi_handedness:     
                    label = MessageToDict(i)['classification'][0]['label']

                    if label == 'Left':
                        cv2.putText(image_resize, label+' Hand',(20, 50), cv2.FONT_HERSHEY_COMPLEX,  0.9, (0, 255, 0), 2)
 
                    if label == 'Right':
                        cv2.putText(image_resize, label+' Hand', (460, 50), cv2.FONT_HERSHEY_COMPLEX,  0.9, (0, 255, 0), 2)
            except:
                print("")
        
    cv2.imshow("Image",image_resize)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break