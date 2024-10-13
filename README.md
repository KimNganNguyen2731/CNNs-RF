# CNNs-RF
Image classification was performed using the Convolutional Neural Network model with the CIFAR10 data set, resulting in a 66.80% correct classification.

## Convolutional Neural Network
![image](https://github.com/user-attachments/assets/d707af25-ffd2-41cd-ac50-ab848509da56)

Change the classification layer of CNNs using the Random Forest model to increase the performance of classification results.
## Random Forest
![image](https://github.com/user-attachments/assets/7e571901-e2ef-4058-9764-5d1bfd97f753)
## The idea for this combination
The CNN model was used to train the dataset, and after that, this pre-trained model was used to extract features for the input of the Random Forest.

When features from the second fully connected layer of the CNNs model were used for the input of Random Forest, a 67.79% correct classification was obtained. The features of the third fully connected layer for the input of the RF model, resulting a 66.84% correct classification.

