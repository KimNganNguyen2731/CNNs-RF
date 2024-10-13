import torch
import numpy as np
import matplotlib.pyplot as plt

def get_fc_features(model, inputs):
    features = []

    # Identify hook to get output of fully connected layer
    def hook_fn(module, input, output):
        features.append(output.detach().cpu().numpy())  # Save features

    # Register hook into fully connected layer
    handle = model.fc3.register_forward_hook(hook_fn)

    # Run forward pass to extract features from fully connected layer
    with torch.no_grad():
        model(inputs.to(DEVICE))

    # Delete hook after extracting features
    handle.remove()

    return np.vstack(features)


def loss_curve(loss_result):
  plt.plot(loss_result, color = 'red')
  plt.xlabel('Step')
  plt.ylabel('Loss')
  plt.title('Loss Curve')
  plt.savefig('loss_curve.png')
  plt.show()
