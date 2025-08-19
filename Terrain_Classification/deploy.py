import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import models
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights
import cv2
import time
from PIL import Image
from torchvision import transforms





# Define same model structure
allowed_classes = ['cement', 'dry_leaf', 'grass', 'rocks', 'sand', 'soil', 'wood_chips']
num_classes = len(allowed_classes)
model = models.mobilenet_v2(weights=None)  # no preloaded weights

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=model.last_channel, out_features=64, bias=True),
    nn.BatchNorm1d(64),
    nn.ReLU(),

   
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=64, out_features=num_classes, bias=True),

)

# Load weights
model.load_state_dict(torch.load("model_25.pth", map_location=torch.device('cpu')))
model.eval()





# Initialize camera
cap = cv2.VideoCapture(1)

# Define transform (must match training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Define class names
class_names = allowed_classes

try:
    print("Starting prediction loop. Press Ctrl+C to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert and preprocess for model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = transform(img_pil).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = class_names[pred.item()]
            print(f"[{time.strftime('%H:%M:%S')}] Predicted terrain: {predicted_class}")

        # Overlay predicted class name on frame
        cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Terrain Classification", frame)

        # Wait for 1 second or 'q' key
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()


