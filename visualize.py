import torch
from torch.utils.data import DataLoader
from utils.data_loader import load_emnist_data
from utils.helper import plot_images
from models.cnn import CNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = load_emnist_data(split='letters')  # Sử dụng EMNIST letters

    # Khởi tạo mô hình với số lớp đầu ra là 26 (A-Z)
    model = CNN(num_classes=26).to(device)

    # Load trọng số từ checkpoint
    model.load_state_dict(torch.load('./models/cnn_emnist_letters.pth'))
    model.eval()  # Chuyển mô hình sang chế độ đánh giá

    # Lấy một batch dữ liệu từ test_loader
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Dự đoán
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # Hiển thị ảnh cùng với nhãn thực tế và nhãn dự đoán
    plot_images(images.cpu(), labels.cpu(), predictions.cpu())

if __name__ == "__main__":
    main()