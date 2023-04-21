# pip install opencv-python
# pip install mediapipe

import cv2
import numpy as np
import time
from math import pi
import torchvision
import torch
import torch.nn as nn
import mediapipe as mp

class CameraException(Exception):
    "Could not read camera"
    pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()
MODEL_PATH = "VGG19_libras.pt"


def load_model(num_classes):    
    model = torchvision.models.vgg19_bn(weights="IMAGENET1K_V1")


    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, num_classes)]) # Add our layer with 4 outputs
    model.classifier = nn.Sequential(*features) # Replace the model classifier

    # Load fine tuned model
    model.load_state_dict(torch.load(MODEL_PATH))
    
    if USE_GPU:
        model.to(DEVICE)
        
    return model
    

def main():
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
    model = load_model(len(classes))
    
    cam = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    
    pTime, cTime = 0, 0
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    model.eval()

    while True:
        success, img = cam.read()
        if not success:
            raise CameraException()
        
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        
        height, width, c = img.shape

        if results.multi_hand_landmarks:
            start_point = (0, 0)
            end_point = (0, 0)
            for handLms in results.multi_hand_landmarks: # working with each hand
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    if id == 9:
                        start_point = (cx - 100 if cx - 100 > 0 else 0, cy - 100 if cy - 100 > 0 else 0)
                        end_point = (cx + 100 if cx + 100 < width else width, cy + 100 if cy + 100 < height else height)
            cv2.rectangle(img, start_point, end_point, color=(255, 0, 255), thickness=2)
            
            if start_point != (0, 0) and end_point != (0, 0):
                cropped_image = img[start_point[0]:end_point[0], start_point[1]:end_point[1]]
                img.flags.writeable = False        
                
                image = transform(cropped_image)
                
                if USE_GPU:
                    with torch.no_grad():
                        image = image.to(DEVICE)
                        
                output = model(image.unsqueeze(0))
                
                _, pred = torch.max(output.data, 1)
                
                print(pred)
                
                letter = classes[pred]
                
                del image, output, pred
                torch.cuda.empty_cache()
                
                img.flags.writeable = True

                cv2.putText(img, str(letter), (10, 120), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
            
        # prints FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        
        
        cv2.imshow("Video", img)
        
        k = cv2.waitKey(1)
        if k % 256 == 27: # Leaves with ESC
            break
        
    cv2.destroyAllWindows()
    cam.release()

if __name__=='__main__':
    main()