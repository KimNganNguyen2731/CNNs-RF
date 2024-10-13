import torch
import torch.nn as nn
import numpy as np

from tools.tools import get_fc_features, loss_result
from sklearn.ensemble import RandomForestClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ConvNet(convNet_model, train_loader, num_epochs: int = 10):
    ConvNet = convNet_model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ConvNet.parameters(), lr=0.0001)
    
    loss_result = []
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = ConvNet(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}, step: {i}/{len(train_loader)}, loss: {loss.item()}")
                loss_result.append(loss.item())

    return ConvNet


def train_rf(model, train_loader, n_estimators):
    convNet = model.eval()  # Set model to evaluation module
    all_features = []
    all_labels = []

    # Loop through each batch to extract features from fully connected layer
    for inputs, targets in train_loader:
        features = get_fc_features(convNet, inputs)  # Extract features from FC
        all_features.append(features)
        all_labels.append(targets.numpy())

    # Combine all features and label to numpy array
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    # Training Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators)
    rf_classifier.fit(all_features, all_labels)

    return rf_classifier
