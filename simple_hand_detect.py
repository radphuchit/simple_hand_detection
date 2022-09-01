import cv2
import mediapipe as mp

def get_bbox_coordinates(handLandmark, image_shape):
    """ 
    Get bounding box coordinates for a hand landmark.
    Args:
        handLadmark: A HandLandmark object.
        image_shape: A tuple of the form (height, width).
    Returns:
        A tuple of the form (xmin, ymin, xmax, ymax).
    """
    all_x, all_y = [], [] # store all x and y points in list
    for hnd in mp_hands.HandLandmark:
        all_x.append(int(handLandmark.landmark[hnd].x * image_shape[1])) # multiply x by image width
        all_y.append(int(handLandmark.landmark[hnd].y * image_shape[0])) # multiply y by image height

    return min(all_x), min(all_y), max(all_x), max(all_y)

mp_hands = mp.solutions.mediapipe.python.solutions.hands
#hands = mp_hands.Hands()
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

mp_draw = mp.solutions.mediapipe.python.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
 
w,h = 540,360
text_color = (0, 0, 0)
while True: 
    success, image = webcam.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue 
    
    image.flags.writeable = False
    image_resize = cv2.resize(image, (w, h))
    image_resize = cv2.flip(image_resize, 1)
    image_rgb = cv2.cvtColor(image_resize ,cv2.COLOR_BGR2RGB)  
    results = hands.process(image_rgb)

    #print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:

        if len(results.multi_handedness) == 2 : 
            text_color = (0,255,0)
        else :
            text_color = (0,0,0)
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            #print('hand_landmarks:', landmark)
            #print(f'Index finger tip coordinates: (',f'{landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w}, ' f'{landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h})'  )
            #mp_draw.draw_landmarks(image_resize, landmark)
            m = get_bbox_coordinates(hand_landmarks,(h,w))           
            image_resize = cv2.rectangle(image_resize, (m[0],m[1]), (m[2],m[3]), (255, 0, 0), 1)
                           
            index = handedness.classification[0].index
            label = handedness.classification[0].label
            if label == 'Left':     
                cv2.putText(image_resize, label+' Hand('+str(index)+')',(m[0],m[1]), cv2.FONT_HERSHEY_COMPLEX,  0.4, text_color, 1)

            if label == 'Right': 
                cv2.putText(image_resize, label+' Hand('+str(index)+')', (m[0],m[1]), cv2.FONT_HERSHEY_COMPLEX,  0.4, text_color, 1)
 
         
    cv2.imshow("Image",image_resize)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break