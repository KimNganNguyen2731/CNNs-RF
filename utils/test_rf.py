import torch
import numpy as np

def predict_rf(convNet_model, rf_classifier, test_loader):
    conv_net = convNet_model.eval()  # Đặt CNN ở chế độ đánh giá (evaluation)

    all_predictions = []
    true_labels = []

    # Lặp qua từng batch trong test_loader để trích xuất đặc trưng và dự đoán
    for inputs, targets in test_loader:
        with torch.no_grad():
            # Trích xuất đặc trưng từ CNN
            features = conv_net(inputs).numpy()

            # Dự đoán với mô hình Random Forest
            predictions = rf_classifier.predict(features)

            all_predictions.extend(predictions)
            true_labels.extend(targets.numpy())  # Lưu trữ nhãn thực tế để so sánh sau này

    return np.array(all_predictions), np.array(true_labels)