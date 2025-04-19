import cv2
import torch
import torch.nn as nn
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Ánh xạ nhãn
def map_label(predicted):
    if predicted < 10:
        return str(predicted)  # Số 0-9
    else:
        return chr(65 + (predicted - 10))  # Chữ A-Z

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=36).to(device)
model.load_state_dict(torch.load("./models/cnn_custom.pth", map_location=device))
model.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_size = 200
roi_x = (frame_width - roi_size) // 2
roi_y = (frame_height - roi_size) // 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình")
        break

    roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imshow("Processed ROI", resized)

    img = transform(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        label = map_label(predicted.item())
        confidence_score = confidence.item() * 100

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
    text = f"Predicted: {label} ({confidence_score:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam - Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()