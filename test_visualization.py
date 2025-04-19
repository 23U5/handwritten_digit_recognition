import sys
import os

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_mnist_data
from utils.helper import plot_images

# Load dữ liệu MNIST
_, test_loader = load_mnist_data()

# Lấy một batch dữ liệu từ test_loader
images, labels = next(iter(test_loader))

# Hiển thị ảnh và nhãn
plot_images(images, labels)