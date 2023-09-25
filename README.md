## Multi Label Face Emotion Recognition using Machine Learning

This repository contains a machine learning project for emotion recognition using deep learning techniques. The project is based on the recognition of six primary emotions: Surprise, Fear, Disgust, Happiness, Sadness, and Anger.

Data Set - http://www.whdeng.cn/raf/model2.html
<br>

### Average_Ensemble_Model.ipynb

**File Overview:**

- **Data**: The project uses a dataset of facial images, which are categorized into the six emotions mentioned above.

- **Models**: Two pre-trained deep learning models, ResNet50 and VGG19, are used for feature extraction from the facial images.

- **Ensemble Learning**: An ensemble model is constructed by combining the features extracted from ResNet50 and VGG19 models using an average ensemble technique.

- **Training**: The ensemble model is trained on the dataset to predict the presence of each of the six emotions in the images.

- **Evaluation**: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

- **Hamming Loss**: The Hamming loss, a measure of label-based accuracy, is also calculated.

- **Usage**: The trained ensemble model can be used for emotion recognition in new images.

**Results:**

The project achieved the following results:

- **Accuracy**: The model achieved an accuracy of approximately 70% on the validation dataset.

- **F1 Score**: The F1-score, which is a measure of precision and recall, was approximately 0.52 on average.

- **Hamming Loss**: The Hamming loss, a measure of label-based accuracy, was approximately 0.27.

**Usage:**

To use the trained ensemble model for emotion recognition in your own images:

1. Load the model using `load_model` from TensorFlow/Keras.
2. Preprocess your image, resizing it to 100x100 pixels.
3. Predict the emotions present in the image.
4. Set a threshold (e.g., 0.4) to determine the emotions with confidence scores above the threshold.
5. Display the recognized emotions on the image.
<br>

### Deep_Learning_Model.ipynb

**File Overview:**

- **Data**: The project utilizes a dataset of facial images, categorized into the six emotions mentioned above.

- **Model**: A convolutional neural network (CNN) is designed and trained to classify emotions from the facial images.

- **Training**: The CNN model is trained on the dataset to predict the presence of each of the six emotions in the images.

- **Evaluation**: The model's performance is evaluated using metrics such as accuracy and loss on a validation dataset.

- **Usage**: The trained model can be used for emotion recognition in new images.

**Results:**

The project achieved the following results:

- **Accuracy**: The model achieved an accuracy of approximately 60% on the validation dataset.
<br>

### Test_Model.ipynb

**File Overview:**

This file serves as a crucial tool for benchmarking transfer learning models, aiding in the selection of the top-performing models for the final model selection process.

1. **ResNet50 Model**
   - Model Type: Sequential
   - Base Pre-trained Model: ResNet50
   - Top Layers: Flatten -> Dense(128) -> BatchNormalization -> Dropout(0.5) -> Dense(64) -> BatchNormalization -> Dropout(0.5) -> Dense(32) -> BatchNormalization -> Dropout(0.5) -> Dense(6, activation='sigmoid')

2. **VGG19 Model**
   - Model Type: Sequential
   - Base Pre-trained Model: VGG19
   - Top Layers: Flatten -> Dense(256) -> BatchNormalization -> Dropout(0.4) -> Dense(128) -> BatchNormalization -> Dropout(0.4) -> Dense(64) -> BatchNormalization -> Dropout(0.4) -> Dense(32) -> BatchNormalization -> Dropout(0.4) -> Dense(6, activation='sigmoid')

3. **MobileNetV2 Model**
   - Model Type: Sequential
   - Base Pre-trained Model: MobileNetV2
   - Top Layers: Flatten -> Dense(128) -> BatchNormalization -> Dropout(0.4) -> Dense(64) -> BatchNormalization -> Dropout(0.4) -> Dense(32) -> BatchNormalization -> Dropout(0.4) -> Dense(6, activation='sigmoid')

4. **InceptionResNetV2 Model**
   - Model Type: Sequential
   - Base Pre-trained Model: InceptionResNetV2
   - Top Layers: Flatten -> Dense(256) -> BatchNormalization -> Dropout(0.4) -> Dense(128) -> BatchNormalization -> Dropout(0.4) -> Dense(64) -> BatchNormalization -> Dropout(0.4) -> Dense(32) -> BatchNormalization -> Dropout(0.4) -> Dense(6, activation='sigmoid')

**Note:**

- This project serves as a demonstration of using deep learning for emotion recognition.
- The accuracy and performance of the model can vary based on the quality and diversity of the input data.
- Ensure that you have the required libraries, such as TensorFlow, OpenCV, and Matplotlib, installed to run the code.

If you have any specific questions or need further assistance with these models or any related tasks, please feel free to ask!
tharindudilshan9944@gmail.com
