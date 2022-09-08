import cv2
import time
import mediapipe as mp

class Position:
  def __init__(self, x, y):
    self.x = x
    self.y = y
 

def get_bbox_coordinates(handLandmark): 
    all_x, all_y = [], []
    for hnd in mp_hands.HandLandmark:
        all_x.append(handLandmark.landmark[hnd].x )
        all_y.append(handLandmark.landmark[hnd].y )

    position_min = Position(min(all_x), min(all_y))
    position_max = Position(max(all_x), max(all_y))
    return position_min, position_max

mp_hands = mp.solutions.mediapipe.python.solutions.hands 
mp_draw = mp.solutions.mediapipe.python.solutions.drawing_utils

hands = mp_hands.Hands()

file_location = 'videos/03_240p.mp4'
cap = cv2.VideoCapture(file_location)

cTime, pTime = 0,0

text_color = (255,255,255) 


width,height = 540,360
while True: 
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.") 
      break 

    #height, width, channels = image.shape 
    
    image = cv2.resize(image, (width, height)) 
    image_rgb = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)  
    results = hands.process(image_rgb)
 
    if results.multi_hand_landmarks:

        # green when 2 hands
        text_color = (0,255,0) if len(results.multi_handedness) == 2 else (0,0,0) 
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness): 
            mp_draw.draw_landmarks(image, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(thickness=1,circle_radius=1),
                mp_draw.DrawingSpec(thickness=1,circle_radius=1))

            pos_min, pos_max = get_bbox_coordinates(hand_landmarks)   
            cv2.rectangle(image, (int(pos_min.x * width),int(pos_min.y * height)), (int(pos_max.x * width),int(pos_max.y * height)), (255, 0, 0), 1)
                                   
            label = handedness.classification[0].label
            if label == 'Left':      
                cv2.putText(image, 'Right',  (int(pos_min.x * width),int(pos_min.y * height)), cv2.FONT_HERSHEY_COMPLEX,  0.4, text_color, 1)
                
            if label == 'Right': 
                cv2.putText(image, 'Left',  (int(pos_min.x * width),int(pos_min.y * height)), cv2.FONT_HERSHEY_COMPLEX,  0.4, text_color, 1)
               

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)),  (int(0.1 * width),int(0.1 * height)), cv2.FONT_HERSHEY_COMPLEX,  0.4, (255,255,255), 1)
 
    cv2.imshow("Image",image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break