# pip install opencv-python

import cv2
import numpy as np
import time
from math import pi
import torchvision
import torch
import torch.nn as nn

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
    pTime, cTime = 0, 0
    
    transform = torchvision.transforms.ToTensor()

    model.eval()

    while 1:
        success, img = cam.read()
        if not success:
            raise CameraException()
        
        height, width, c = img.shape

        img.flags.writeable = False
        
        image = transform(img)
        
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

        # prints FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        
        cv2.putText(img, str(letter), (10, 120), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        
        cv2.imshow("Video", img)
        
        k = cv2.waitKey(1)
        if k % 256 == 27: # Leaves with ESC
            break
        
    cv2.destroyAllWindows()
    cam.release()

if __name__=='__main__':
    main()