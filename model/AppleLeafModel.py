import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class AppleLeafModel:
    def __init__(self):
        self.model = None
        self.class_names = ["diseased", "healthy"]
        self.img_size = (128, 128)
        self.batch_size = 32 #32 images each time
        self.train_data = None
        self.test_data = None
        self.class_weight = None
        self.healthy_threshold = 0.5

    def load_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.2,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.08,
            shear_range=0.05,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
        #use datagen (datagenerator) to read all the images from directory "dataset"
        #label them based on the folder name, like diseased and healthy, and put them to train_data
        self.train_data = train_datagen.flow_from_directory(
            "Dataset",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary",
            subset="training",
            shuffle=True
        )

        self.test_data = val_datagen.flow_from_directory(
            "Dataset",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary",
            subset="validation",
            shuffle=False
        )

        self.class_weight = None
        print("Data loaded successfully.")

    def build_model(self):
        # Conv blocks + batch norm + dropout are usually stronger than only adding depth.
        self.model = models.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.15),

            layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.2),

            layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # binary classification
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                           loss='binary_crossentropy',
                           metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        print("Model built and compiled.")
#train the model 10 times by default, patience=3
    def train_model(self, epochs=10, patience=3):
        print("Training started...")
        #vau_loss means "validation loss", which tells the model to watch the validation loss 
        #when the model stops and restore the best weights with lowest loss value
        earlyStop=EarlyStopping(monitor="val_loss",patience=patience,restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
        
        # Define checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            'best_model_weights.h5',  # Path to save the best model weights
            monitor='val_loss',       # Monitor validation loss
            save_best_only=True,      # Only save the best weights
            mode='min',               # We want to minimize validation loss
            verbose=1
        )#training will stop if earlyStop occurs (patience up to 3)
        history=self.model.fit(
            self.train_data,
            epochs=epochs,
            validation_data=self.test_data,
            callbacks=[earlyStop, reduce_lr, checkpoint_callback]
        )
        
        print("Training finished.")
        
        self.model.load_weights("best_model_weights.h5")
        self.model.save("trained_model.h5")  # save the model to .h5 file
        
        print(" Model saved as trained_model.h5")
        self.plot_training_history(history)
        
        #plot training history for visualisation
    def plot_training_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))
    
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
    
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
    
        plt.tight_layout()
        plt.show()


    def evaluate_model(self):
        metrics_dict = self.model.evaluate(self.test_data, return_dict=True)
        accuracy = float(metrics_dict.get("accuracy", 0.0))
        print(f"Validation Accuracy: {accuracy:.4f}")
        if "auc" in metrics_dict:
            print(f"Validation AUC: {metrics_dict['auc']:.4f}")
        return accuracy
    
    #use comfusion matrix and performance marix derived from it to evaluate the model
    def tune_threshold(self):
        y_true = self.test_data.classes
        y_pred_probs = self.model.predict(self.test_data).flatten()

        best_threshold = self.healthy_threshold
        best_macro_f1 = -1.0

        for threshold in np.arange(0.30, 0.71, 0.01):
            y_pred = (y_pred_probs > threshold).astype("int32")
            macro_f1 = f1_score(y_true, y_pred, average="macro")
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_threshold = float(threshold)

        self.healthy_threshold = best_threshold
        print(f"Best healthy threshold from validation: {self.healthy_threshold:.2f} (macro F1={best_macro_f1:.4f})")

    def evaluate_with_confusion_matrix(self):
    # Get true labels and predicted labels
        y_true = self.test_data.classes
        y_pred_probs = self.model.predict(self.test_data)
        y_pred = (y_pred_probs > self.healthy_threshold).astype("int32").flatten()
    
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
    
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
    
        # Visualize confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()



    def predict_image(self, index):
        # Get one batch of images and labels
        images, labels = next(self.test_data)
        img = images[index]
        prediction = self.model.predict(np.expand_dims(img, axis=0))[0][0]
        predicted_label = 1 if prediction > self.healthy_threshold else 0
        true_label = int(labels[index])

        plt.imshow(img)
        plt.title(f"Predicted: {self.class_names[predicted_label]}, Actual: {self.class_names[true_label]}")
        plt.axis('off')
        plt.show()
        
    def predict_single_image(self, img_array):
        prediction = self.model.predict(img_array)[0][0]
        healthy_score = float(prediction)
        diseased_score = float(1 - prediction)
        predicted_label = 1 if prediction > self.healthy_threshold else 0
        confidence = healthy_score if predicted_label == 1 else diseased_score
        return predicted_label, confidence, healthy_score, diseased_score
    
    def load_model(self, model_path="trained_model.h5"):
       self.model = tf.keras.models.load_model(model_path)
       print(" Trained model loaded.")
       
    def run(self):
        self.load_data()
        self.build_model()
        self.train_model(epochs=15, patience=5)
        self.evaluate_model()
        self.tune_threshold()
        self.evaluate_with_confusion_matrix()
        self.predict_image(0)
    
    