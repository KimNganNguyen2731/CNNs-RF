import torch
import numpy as np

from sklearn.ensemble import RandomForestClassifier


def train_rf(convNet_model, train_loader, n_estimators: int = 100):
    conv_net = convNet_model.eval()
    
    features = []
    labels = []
    
    # Loop over each batch to extract features
    for inputs, targets in train_loader:
        with torch.no_grad():
            feature = conv_net(inputs).numpy()
            features.append(feature)
            labels.append(targets.numpy())
            
    # Combine all features and labels to numpy array
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    # Training Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators)
    rf_classifier.fit(features, labels)
    
    return rf_classifier