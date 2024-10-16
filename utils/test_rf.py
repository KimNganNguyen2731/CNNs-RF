import torch
import numpy as np
from tools.tools import get_fc_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, test_loader):
  model.eval()
  n_correct = 0
  n_samples = 0
  for inputs, labels in test_loader:
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    label_predicted = model(inputs)
    _, predicted = torch.max(label_predicted, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()
  print(f'Accuracy of ConvNet = {100*n_correct/n_samples:.2f}')
    

def predict_rf(model, rf_classifier, test_loader):
    model.eval()  # Set CNN model to evaluation module
    all_predictions = []
    n_correct = 0
    n_samples = 0

    # Predict on each batch in test_loader
    for inputs, true_label in test_loader:
        # Extract features from fully connected layer
        features = extract_fc_features(model, inputs)

        # Using Random Forest model to predict
        predictions = rf_classifier.predict(features)

        # Convert prediction result to tensor to compute accuracy
        all_predictions = torch.tensor(predictions)

        # Compare true and predicted labels
        true_label = true_label.cpu().numpy()  # Convert to numpy array 
        n_correct += (all_predictions == true_label).sum().item()

        n_samples += true_label.shape[0]

    accuracy = 100 * n_correct / n_samples
    print(f'Accuracy = {accuracy:.2f}%')
