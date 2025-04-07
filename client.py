import flwr as fl
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from typing import Dict, Tuple, List
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import time
import warnings
from collections import defaultdict
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LungCancerClient(fl.client.NumPyClient):
    def __init__(self, data_path, client_id):
        self.client_id = client_id
        self.data_path = data_path
        self.start_time = time.time()
        self.round_times = []
        self.val_acc_history = defaultdict(list)
        
        # Build both models
        self.densenet_model = self._build_densenet_model()
        self.mobilenet_model = self._build_mobilenet_model()
        self.ensemble_model = self._build_ensemble_model()
        
        # Calculate parameter counts for splitting
        self.densenet_param_count = len(self.densenet_model.get_weights())
        
        self.train_gen, self.test_gen = self._prepare_data()
        
        print(f"Client {self.client_id} ready with {self.train_gen.samples} training samples")
    
    def _build_densenet_model(self):
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(3, activation='softmax', name='densenet_output')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def _build_mobilenet_model(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(3, activation='softmax', name='mobilenet_output')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def _build_ensemble_model(self):
        """Build model that combines both architectures"""
        densenet_output = self.densenet_model.get_layer('densenet_output').output
        mobilenet_output = self.mobilenet_model.get_layer('mobilenet_output').output
        
        # Combine outputs with learnable weights
        combined = concatenate([densenet_output, mobilenet_output])
        weights = Dense(2, activation='softmax', name='ensemble_weights')(combined)
        weighted_output = weights[:, 0:1] * densenet_output + weights[:, 1:2] * mobilenet_output
        
        ensemble_model = Model(
            inputs=[self.densenet_model.input, self.mobilenet_model.input],
            outputs=weighted_output
        )
        ensemble_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return ensemble_model
    
    def _normalize_image(self, img):
        """Custom normalization for images (replaces preprocess_input)"""
        img = img.astype('float32') / 255.0  # Scale to [0,1]
        return img
    
    def _prepare_data(self):
        datagen = ImageDataGenerator(
            preprocessing_function=self._normalize_image,
            validation_split=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            zoom_range=0.2
        )
        train_gen = datagen.flow_from_directory(
            self.data_path, 
            target_size=(224, 224), 
            batch_size=32,
            class_mode='categorical', 
            subset='training', 
            shuffle=True
        )
        test_gen = datagen.flow_from_directory(
            self.data_path, 
            target_size=(224, 224), 
            batch_size=32,
            class_mode='categorical', 
            subset='validation', 
            shuffle=False
        )
        if train_gen.num_classes != 3 or test_gen.num_classes != 3:
            raise ValueError(
                f"Expected 3 classes (lung_aca, lung_n, lung_scc), but found {train_gen.num_classes} classes. "
                f"Check data directory structure at {self.data_path}"
            )
        return train_gen, test_gen
    
    def get_parameters(self, config):
        # Return parameters as a single list (DenseNet params followed by MobileNet params)
        return self.densenet_model.get_weights() + self.mobilenet_model.get_weights()
    
    def set_parameters(self, parameters):
        # Split parameters into DenseNet and MobileNet parts
        densenet_params = parameters[:self.densenet_param_count]
        mobilenet_params = parameters[self.densenet_param_count:]
        
        self.densenet_model.set_weights(densenet_params)
        self.mobilenet_model.set_weights(mobilenet_params)
    
    def fit(self, parameters, config):
        round_start = time.time()
        self.set_parameters(parameters)
        
        # Train both models
        densenet_history = self.densenet_model.fit(
            self.train_gen, epochs=1, verbose=0, steps_per_epoch=len(self.train_gen)
        )
        mobilenet_history = self.mobilenet_model.fit(
            self.train_gen, epochs=1, verbose=0, steps_per_epoch=len(self.train_gen)
        )
        
        train_acc = {
            'densenet': densenet_history.history['accuracy'][0],
            'mobilenet': mobilenet_history.history['accuracy'][0]
        }
        
        round_time = time.time() - round_start
        self.round_times.append(round_time)
        
        print(f"[Client {self.client_id}] Round completed in {round_time:.2f}s | "
              f"Train Acc - DenseNet: {train_acc['densenet']:.4f}, MobileNet: {train_acc['mobilenet']:.4f}")
        
        return self.get_parameters(config), self.train_gen.samples, {
            "train_accuracy_densenet": train_acc['densenet'],
            "train_accuracy_mobilenet": train_acc['mobilenet'],
            "client_id": self.client_id,
            "round_time": round_time
        }
    
    def evaluate(self, parameters, config):
        eval_start = time.time()
        self.set_parameters(parameters)
        
        # Evaluate both models separately
        densenet_results = self.densenet_model.evaluate(self.test_gen, verbose=0, steps=len(self.test_gen))
        mobilenet_results = self.mobilenet_model.evaluate(self.test_gen, verbose=0, steps=len(self.test_gen))
        
        test_loss = {
            'densenet': densenet_results[0],
            'mobilenet': mobilenet_results[0]
        }
        
        test_acc = {
            'densenet': densenet_results[1],
            'mobilenet': mobilenet_results[1]
        }
        
        # Track validation accuracy history for ensemble weighting
        self.val_acc_history['densenet'].append(test_acc['densenet'])
        self.val_acc_history['mobilenet'].append(test_acc['mobilenet'])
        
        # Generate predictions for both models
        y_true = self.test_gen.classes
        y_pred_densenet = np.argmax(self.densenet_model.predict(self.test_gen, steps=len(self.test_gen)), axis=1)
        y_pred_mobilenet = np.argmax(self.mobilenet_model.predict(self.test_gen, steps=len(self.test_gen)), axis=1)
        
        # Generate ensemble predictions weighted by validation accuracy
        avg_densenet_acc = np.mean(self.val_acc_history['densenet'])
        avg_mobilenet_acc = np.mean(self.val_acc_history['mobilenet'])
        total_acc = avg_densenet_acc + avg_mobilenet_acc
        densenet_weight = avg_densenet_acc / total_acc
        mobilenet_weight = avg_mobilenet_acc / total_acc
        
        densenet_probs = self.densenet_model.predict(self.test_gen, steps=len(self.test_gen))
        mobilenet_probs = self.mobilenet_model.predict(self.test_gen, steps=len(self.test_gen))
        ensemble_probs = densenet_weight * densenet_probs + mobilenet_weight * mobilenet_probs
        y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
        
        # Generate metrics for all models
        self._generate_metrics(y_true, y_pred_densenet, densenet_probs, model_type='densenet')
        self._generate_metrics(y_true, y_pred_mobilenet, mobilenet_probs, model_type='mobilenet')
        self._generate_metrics(y_true, y_pred_ensemble, ensemble_probs, model_type='ensemble')
        
        # Save all models
        self.save_model()
        
        eval_time = time.time() - eval_start
        print(f"[Client {self.client_id}] Evaluation completed in {eval_time:.2f}s | "
              f"Test Acc - DenseNet: {test_acc['densenet']:.4f}, MobileNet: {test_acc['mobilenet']:.4f}")
        
        # Return both losses weighted by their validation accuracy
        weighted_loss = (densenet_weight * test_loss['densenet'] + 
                         mobilenet_weight * test_loss['mobilenet'])
        
        return float(weighted_loss), self.test_gen.samples, {
            "test_accuracy_densenet": test_acc['densenet'],
            "test_accuracy_mobilenet": test_acc['mobilenet'],
            "ensemble_accuracy": accuracy_score(y_true, y_pred_ensemble),
            "client_id": self.client_id,
            "eval_time": eval_time,
            "densenet_weight": densenet_weight,
            "mobilenet_weight": mobilenet_weight
        }
    
    def _generate_metrics(self, y_true, y_pred, y_probs, model_type):
        """Generate confusion matrix, classification report and ROC curve for a model"""
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["lung_aca", "lung_n", "lung_scc"],
                    yticklabels=["lung_aca", "lung_n", "lung_scc"])
        plt.title(f"Client {self.client_id} - {model_type.capitalize()} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"client_{self.client_id}_{model_type}_cm.png")
        plt.close()
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=["lung_aca", "lung_n", "lung_scc"])
        with open(f"client_{self.client_id}_{model_type}_report.txt", "w") as f:
            f.write(report)
        
        # ROC curve
        plt.figure(figsize=(8, 6))
        for i in range(3):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"Client {self.client_id} - {model_type.capitalize()} ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(f"client_{self.client_id}_{model_type}_roc.png")
        plt.close()
    
    def save_model(self, filename=None):
        """Save all models with appropriate naming"""
        if filename is None:
            base_name = f"client_{self.client_id}"
        else:
            base_name = os.path.splitext(filename)[0]
            
        self.densenet_model.save(f"{base_name}_densenet.h5")
        self.mobilenet_model.save(f"{base_name}_mobilenet.h5")
        self.ensemble_model.save(f"{base_name}_ensemble.h5")
        print(f"Models saved as {base_name}_densenet.h5, {base_name}_mobilenet.h5, {base_name}_ensemble.h5")

def start_client(server_address, data_path, client_id):
    try:
        print(f"Starting client {client_id} connecting to {server_address}")
        client = LungCancerClient(data_path, client_id)
        fl.client.start_numpy_client(server_address=server_address, client=client)
        total_time = time.time() - client.start_time
        avg_round = sum(client.round_times)/len(client.round_times) if client.round_times else 0
        print(f"\nClient {client_id} Summary:")
        print(f"Total runtime: {total_time:.2f}s")
        print(f"Avg round time: {avg_round:.2f}s")
        print(f"Rounds completed: {len(client.round_times)}")
        print(f"Final model weights - DenseNet: {np.mean(client.val_acc_history['densenet']):.4f}, "
              f"MobileNet: {np.mean(client.val_acc_history['mobilenet']):.4f}")
    except ConnectionRefusedError:
        print(f"Connection refused: Could not connect to server at {server_address}. Check server status and address.")
    except ValueError as ve:
        print(f"Data error: {str(ve)}")
    except Exception as e:
        print(f"Client error: {str(e)}. Verify network and server configuration.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="192.168.29.100:8080", help="Server address (e.g., 192.168.1.100:8080)")
    parser.add_argument("--data", default="C2", help="Path to data")
    parser.add_argument("--id", default="2", help="Client ID")
    args = parser.parse_args()
    start_client(args.server, args.data, args.id)