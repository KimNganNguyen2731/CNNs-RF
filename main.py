from tools.dataloader_CIFAR10 import dataloader
from models.CNNs import ConvNet
from utils.train_rf import train_ConvNet, train_rf
from utils.test_rf import predict_rf, test


def main():
  train_loader, test_loader = dataloader()
  
  ConvNet_model = ConvNet(img_size = 32, out_channels = 128)
  train_ConvNet_model = train_ConvNet(ConvNet_model, train_loader, num_epochs = 50)
  test_ConvNet = test(train_ConvNet_model, test_loader)

  
  train_RF_model = train_rf(train_ConvNet_model, train_loader, 100)
  test_RF_model = predict_rf(train_ConvNet_model, train_RF_model, test_loader)


if __name__ = "__main__":
main()
