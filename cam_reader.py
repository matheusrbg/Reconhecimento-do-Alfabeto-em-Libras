import cv2
import time
import torchvision
import torch
import sys
import torch.nn as nn
import mediapipe as mp
import torch.nn.functional as nnf

class CameraException(Exception):
    "Could not read camera"
    pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()
MODEL = sys.argv[1]
try:
    VALID = int(sys.argv[2])
except:
    TRANSF = None
    VALID = None
try:
    TRANSF = int(sys.argv[3])
except:
    TRANSF = None
    VERSION = str(VALID)

def load_vgg(num_classes):    
    model = torchvision.models.vgg19_bn(weights='IMAGENET1K_V1')

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
    
def load_resnet(num_classes):
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(model.fc.weight)

    # Load fine tuned model
    model.load_state_dict(torch.load(MODEL_PATH))
    
    if USE_GPU:
        model.to(DEVICE)

    return model

def load_googlenet(num_classes):
    
    model = torchvision.models.googlenet(weights='IMAGENET1K_V1')
    
    if MODEL_PATH == "googlenet_libras.pt" or MODEL_PATH == "googlenet2_libras.pt":
        model.fc=nn.Sequential(
            nn.Linear(in_features=1024,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=num_classes,bias=True)
        )
    else:
        model.fc=nn.Linear(in_features=1024,out_features=num_classes)

    # Load fine tuned model
    model.load_state_dict(torch.load(MODEL_PATH))
    
    if USE_GPU:
        model.to(DEVICE)

    return model    

def load_convnext(num_classes):
  
    model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
  
    n_inputs = None
    for name, child in model.named_children():
        if name == 'classifier':
            for sub_name, sub_child in child.named_children():
                if sub_name == '2':
                    n_inputs = sub_child.in_features

    model.classifier = nn.Sequential(
        nn.LayerNorm((768,1,1), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(n_inputs, 2048, bias=True),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, num_classes),
        nn.LogSoftmax(dim=1)

    )

    # Load fine tuned model
    model.load_state_dict(torch.load(MODEL_PATH))

    if USE_GPU:
        model.to(DEVICE)

    return model

def choiceByVersion():
    if MODEL.lower() == 'vgg':
        model_path = "VGG19_libras.pt"
        load = load_vgg
    elif MODEL.lower() == 'resnet':
        model_path = "resnet_libras.pt"
        load = load_resnet
    elif MODEL.lower() == 'googlenet':
        if VERSION == None or VERSION == '1':
            model_path = "googlenet_libras.pt"
        else:
            model_path = "googlenet" + VERSION + "_libras.pt"
        load = load_googlenet
    elif MODEL.lower() == 'convnext':
        if VERSION == None or VERSION == '1':
            model_path = "convnext_libras.pt"
        else:
            model_path = "convnext" + VERSION + "_libras.pt"
        load = load_convnext
    
    return model_path, load

def choiceBySettings():
    if MODEL.lower() == 'vgg':
        if VALID == 1:
            if TRANSF == 1:
                model_path = "VGG19_libras.pt"
            elif TRANSF == 2:
                model_path = "VGG19_2_libras.pt"
            else:
                raise ValueError("Transformada " + str(TRANSF) + "não existe")
        elif VALID == 0:
            if TRANSF == 1:
                raise NameError("Modelo ainda não existe")
            elif TRANSF == 2:
                raise NameError("Modelo ainda não existe")
            else:
                raise ValueError("Transformada " + str(TRANSF) + "não existe")
        else:
            raise ValueError("Validação deve ser 0 ou 1, " + str(VALID) + "foi recebido")
        load = load_vgg
    elif MODEL.lower() == 'resnet':
        if VALID == 1:
            if TRANSF == 1:
                model_path = "resnet3_libras.pt"
            elif TRANSF == 2:
                model_path = "resnet2_libras.pt"
            else:
                raise ValueError("Transformada " + str(TRANSF) + "não existe")
        elif VALID == 0:
            if TRANSF == 1:
                raise NameError("Modelo ainda não existe")
            elif TRANSF == 2:
                model_path = "resnet_libras.pt"
            else:
                raise ValueError("Transformada " + str(TRANSF) + "não existe")
        else:
            raise ValueError("Validação deve ser 0 ou 1, " + str(VALID) + "foi recebido")
        load = load_resnet
    elif MODEL.lower() == 'googlenet':
        if VALID == 1:
            if TRANSF == 1:
                model_path = "googlenet4_libras.pt"
            elif TRANSF == 2:
                model_path = "googlenet3_libras.pt"
            else:
                raise ValueError("Transformada " + str(TRANSF) + "não existe")
        elif VALID == 0:
            if TRANSF == 1:
                model_path = "googlenet_libras.pt"
            elif TRANSF == 2:
                model_path = "googlenet2_libras.pt"
            else:
                raise ValueError("Transformada " + str(TRANSF) + "não existe")
        else:
            raise ValueError("Validação deve ser 0 ou 1, " + str(VALID) + "foi recebido")
        load = load_googlenet
    elif MODEL.lower() == 'convnext':
        if VALID == 1:
            if TRANSF == 1:
                raise NameError("Modelo ainda não existe")
            elif TRANSF == 2:
                model_path = "convnext3_libras.pt"
            else:
                raise ValueError("Transformada " + str(TRANSF) + "não existe")
        elif VALID == 0:
            if TRANSF == 1:
                model_path = "convnext_libras.pt"
            elif TRANSF == 2:
                model_path = "convnext2_libras.pt"
            else:
                raise ValueError("Transformada " + str(TRANSF) + "não existe")
        else:
            raise ValueError("Validação deve ser 0 ou 1, " + str(VALID) + "foi recebido")
        load = load_convnext
    
    return model_path, load

SPACE = 30

def printAlignCenter(text):
    text_space = int(SPACE - len(text)/2)
    print("/"*text_space + text + "\\"*text_space) 

if TRANSF == None:
    print("\/"*SPACE)
    printAlignCenter(" Escolha por versão ")
    printAlignCenter(" Versão: " + VERSION + ' ')
    print("/\\"*SPACE)
    MODEL_PATH, LOAD = choiceByVersion()
else:
    print("\/"*SPACE)
    printAlignCenter(" Escolha por configuração ")
    printAlignCenter(" Validação: " + str(bool(VALID)) + ' ')
    printAlignCenter(" Trasnformada: " + str(TRANSF) + ' ')
    print("/\\"*SPACE)
    MODEL_PATH, LOAD = choiceBySettings()

def main():
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
    letter = '#'
    model = LOAD(len(classes))
  
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
                        delta_pixels = 140
                        start_point = (cx - delta_pixels if cx - delta_pixels > 0 else 0, cy - delta_pixels if cy - delta_pixels > 0 else 0)
                        end_point = (cx + delta_pixels if cx + delta_pixels < width else width, cy + delta_pixels if cy + delta_pixels < height else height)
            cv2.rectangle(img, start_point, end_point, color=(255, 0, 255), thickness=2)
            
            if start_point != (0, 0) and end_point != (0, 0):
                cropped_image = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
                img.flags.writeable = False        
                
                image = transform(cropped_image)
                cv2.imshow("Cropped Image", cropped_image)
                if USE_GPU:
                    with torch.no_grad():
                        image = image.to(DEVICE)
                        
                output = model(image.unsqueeze(0))
                prob = nnf.softmax(output, dim=1)
                score, pred = prob.topk(1, dim = 1)
                
                if score.item() > 0.8:
                    letter = classes[pred]
                else:
                    letter = '#'
                
                
                del image, output, pred
                torch.cuda.empty_cache()
                
                img.flags.writeable = True

                cv2.putText(img, "%.2f" % score.item(), (10, 170), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)

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