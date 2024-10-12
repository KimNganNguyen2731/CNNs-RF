import torch
import torch.nn as nn
import numpy as np

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
    convNet = model.eval()
    
    features = []
    labels = []
    
    # Loop over each batch to extract features
    for inputs, targets in train_loader:
        with torch.no_grad():
            feature = convNet(inputs).numpy()
            features.append(feature)
            labels.append(targets.numpy())
            
    # Combine all features and labels to numpy array
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    # Training Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators)
    rf_classifier.fit(features, labels)
    
    return rf_classifier
