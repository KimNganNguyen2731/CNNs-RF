import torch
import numpy as np

def predict_rf(convNet_model, rf_classifier, test_loader):
    conv_net = convNet_model.eval()  # set CNNs model to evaluation mode (evaluation)

    all_predictions = []
    true_labels = []

    # loop over each batch in test_loader to extract features and predict labels
    for inputs, targets in test_loader:
        with torch.no_grad():
            # Extract features with CNNs model
            features = conv_net(inputs).numpy()

            # Predict with Random Forest
            predictions = rf_classifier.predict(features)

            all_predictions.extend(predictions)
            true_labels.extend(targets.numpy())  # Save the real labels to compare with the predicted labels

    return np.array(all_predictions), np.array(true_labels)
