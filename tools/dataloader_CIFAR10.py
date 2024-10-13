# Data loading
def dataloader():
  train_dataset = datasets.CIFAR10(
      root = "data",
      train = True,
      download=True,
      transform = transforms.ToTensor()
  )

  test_dataset = datasets.CIFAR10(
      root = "data",
      train = False,
      download = True,
      transform = transforms.ToTensor()
  )
  # Data Loader
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
  return train_loader, test_loader
