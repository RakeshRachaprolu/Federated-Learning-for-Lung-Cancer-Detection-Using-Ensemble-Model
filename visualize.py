import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, concatenate
from tensorflow.keras.applications import DenseNet121, MobileNetV2
import argparse
import os
import re

def build_ensemble_model():
    input_shape = (224, 224, 3)
    densenet_input = Input(shape=input_shape)
    mobilenet_input = Input(shape=input_shape)
    
    densenet_base = DenseNet121(weights='imagenet', include_top=False, input_tensor=densenet_input)
    mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=mobilenet_input)
    
    densenet_base.trainable = False
    mobilenet_base.trainable = False
    
    densenet_out = GlobalAveragePooling2D()(densenet_base.output)
    densenet_out = Dense(128, activation='relu')(densenet_out)
    densenet_out = Dense(3, activation='softmax', name='densenet_output')(densenet_out)
    
    mobilenet_out = GlobalAveragePooling2D()(mobilenet_base.output)
    mobilenet_out = Dense(128, activation='relu')(mobilenet_out)
    mobilenet_out = Dense(3, activation='softmax', name='mobilenet_output')(mobilenet_out)
    
    combined = concatenate([densenet_out, mobilenet_out])
    weights = Dense(2, activation='softmax', name='ensemble_weights')(combined)
    weighted_output = weights[:, 0:1] * densenet_out + weights[:, 1:2] * mobilenet_out
    
    model = Model(
        inputs=[densenet_input, mobilenet_input],
        outputs=weighted_output
    )
    return model

class LungCancerPredictor:
    def __init__(self, model_path):
        self.class_names = ["lung_aca", "lung_n", "lung_scc"]
        self.model = build_ensemble_model()
        
        try:
            self.model.load_weights(model_path)
            print("✅ Model weights loaded successfully")
        except Exception as e:
            print(f"❌ Error loading weights: {str(e)}")
            raise
        
        self._prepare_visualization_model()

    def _prepare_visualization_model(self):
        last_conv_layer_name = 'conv5_block16_2_conv'
        try:
            last_conv_layer = self.model.get_layer(last_conv_layer_name)
            self.last_conv_layer_name = last_conv_layer_name
            self.feature_model = Model(
                inputs=self.model.inputs,
                outputs=[
                    last_conv_layer.output,
                    self.model.output
                ]
            )
            print("✅ Grad-CAM visualization available")
        except Exception as e:
            self.feature_model = None
            print(f"⚠️ Feature visualization not available: {str(e)}")

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")
        
        img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pre = img_rgb.astype('float32') / 255.0
        return img_rgb, np.expand_dims(img_pre, axis=0)

    def generate_gradcam(self, img_pre, pred_class, threshold=0.5):
        if not self.feature_model:
            return None, None
        
        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.feature_model([img_pre, img_pre])
                loss = predictions[:, pred_class]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            heatmap = tf.maximum(heatmap, 0)
            max_heat = tf.reduce_max(heatmap)
            if max_heat != 0:
                heatmap = heatmap / max_heat
            
            heatmap = heatmap.numpy()
            heatmap = cv2.resize(heatmap, (224, 224))
            
            binary_mask = np.where(heatmap > threshold, 1, 0)
            binary_mask = binary_mask.astype(np.uint8)
            
            # Create green microscopic-style heatmap
            heatmap_green = np.zeros((224, 224, 3), dtype=np.float32)
            heatmap_green[..., 1] = heatmap  # Green channel
            heatmap_green = np.clip(heatmap_green, 0, 1)
            
            return heatmap_green, binary_mask
            
        except Exception as e:
            print(f"⚠️ Could not generate Grad-CAM: {str(e)}")
            return None, None

    def isolate_cancer_tissue(self, img_rgb, binary_mask):
        """Extract cancer tissue with green microscopic effect"""
        # Create a green-tinted version
        green_tissue = np.zeros_like(img_rgb, dtype=np.float32)
        mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
        
        # Apply green tint to cancer regions
        green_tissue[..., 1] = img_rgb[..., 1] * mask_3ch[..., 1]  # Green channel
        green_tissue[..., 0] = img_rgb[..., 0] * mask_3ch[..., 0] * 0.2  # Reduced red
        green_tissue[..., 2] = img_rgb[..., 2] * mask_3ch[..., 2] * 0.2  # Reduced blue
        
        # Enhance contrast
        green_tissue = np.clip(green_tissue * 1.5, 0, 255)
        
        return green_tissue

    def infer_actual_class(self, image_path):
        """Extract actual class from image filename if available"""
        filename = os.path.basename(image_path).lower()
        
        # Common patterns in medical image naming: look for class name in filename
        for class_name in self.class_names:
            if class_name in filename:
                return class_name, self.class_names.index(class_name)
        
        # Check if any of the class name parts appear in the filename
        # lung_aca detection
        if any(x in filename for x in ["aca", "adenocarcinoma"]):
            return "lung_aca", 0
            
        # lung_n detection
        if any(x in filename for x in ["normal", "_n_", "_n."]):
            return "lung_n", 1
            
        # lung_scc detection
        if any(x in filename for x in ["scc", "squamous"]):
            return "lung_scc", 2
        
        return "Unknown", -1

    def predict(self, image_path, threshold=0.5):
        img_rgb, img_pre = self.preprocess_image(image_path)
        
        pred_probs = self.model.predict([img_pre, img_pre])[0]
        pred_class = np.argmax(pred_probs)
        
        # Get actual class if it can be inferred from filename
        actual_class_name, actual_class_idx = self.infer_actual_class(image_path)
        
        heatmap, binary_mask = self.generate_gradcam(img_pre, pred_class, threshold)
        
        if binary_mask is not None:
            cancer_tissue = self.isolate_cancer_tissue(img_rgb, binary_mask)
        else:
            cancer_tissue = None
        
        return img_rgb, pred_class, pred_probs, heatmap, cancer_tissue, binary_mask, actual_class_name, actual_class_idx

    def visualize(self, image_path, actual_class=None, threshold=0.5, save_output=False, output_dir="output"):
        try:
            img_rgb, pred_class, pred_probs, heatmap, cancer_tissue, binary_mask, filename_class_name, filename_class_idx = self.predict(image_path, threshold)
            
            # Use provided actual_class if available, otherwise try to infer from filename
            if actual_class:
                actual_class_name = actual_class
                actual_class_idx = self.class_names.index(actual_class) if actual_class in self.class_names else -1
            else:
                actual_class_name, actual_class_idx = filename_class_name, filename_class_idx
            
            # For normal lung tissue, only show the original image
            if self.class_names[pred_class] == "lung_n":
                plt.figure(figsize=(8, 6))
                
                # Prepare title with both actual and predicted classes
                if actual_class_idx != -1:
                    title = f"Original\nActual: {actual_class_name}\nPredicted: {self.class_names[pred_class]}"
                    accuracy = "✓ Correct" if pred_class == actual_class_idx else "✗ Incorrect"
                    title += f"\n{accuracy}"
                else:
                    title = f"Original\nPredicted: {self.class_names[pred_class]}"
                
                # Original image
                plt.imshow(img_rgb)
                plt.title(title)
                plt.axis("off")
                
            else:
                # For cancer classes, show the full visualization
                plt.figure(figsize=(20, 5))
                
                # Prepare title with both actual and predicted classes
                if actual_class_idx != -1:
                    title = f"Original\nActual: {actual_class_name}\nPredicted: {self.class_names[pred_class]}"
                    accuracy = "✓ Correct" if pred_class == actual_class_idx else "✗ Incorrect"
                    title += f"\n{accuracy}"
                else:
                    title = f"Original\nPredicted: {self.class_names[pred_class]}"
                
                # Original image
                plt.subplot(1, 4, 1)
                plt.imshow(img_rgb)
                plt.title(title)
                plt.axis("off")
                
                if heatmap is not None:
                    # Green heatmap view
                    plt.subplot(1, 4, 2)
                    plt.imshow(heatmap)
                    plt.title(" Heatmap")
                    plt.axis("off")
                    
                    # Green cancer tissue view
                    plt.subplot(1, 4, 3)
                    plt.imshow(cancer_tissue.astype(np.uint8))
                    plt.title("Cancer Tissue ")
                    plt.axis("off")
                    
                    # Combined view with green overlay
                    superimposed_img = heatmap * 0.4 + img_rgb/255.0
                    superimposed_img = np.clip(superimposed_img, 0, 1)
                    
                    plt.subplot(1, 4, 4)
                    plt.imshow(superimposed_img)
                    plt.title("Combined  View")
                    plt.axis("off")
                else:
                    for i in [2, 3, 4]:
                        plt.subplot(1, 4, i)
                        plt.text(0.5, 0.5, "Visualization\nnot available", 
                              ha='center', va='center')
                        plt.axis("off")
            
            plt.tight_layout()
            
            if save_output:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                plt.savefig(f"{output_dir}/{base_name}_visualization.png")
                
                # Only save cancer tissue and mask for cancer classes
                if self.class_names[pred_class] != "lung_n":
                    if cancer_tissue is not None:
                        cancer_img = cancer_tissue.astype(np.uint8)
                        cv2.imwrite(f"{output_dir}/{base_name}_cancer_only.png", 
                                  cv2.cvtColor(cancer_img, cv2.COLOR_RGB2BGR))
                    
                    if binary_mask is not None:
                        cv2.imwrite(f"{output_dir}/{base_name}_mask.png", 
                                  binary_mask * 255)
                
                print(f"✅ Images saved to {output_dir}/")
            
            plt.show()
            
            print("\nPREDICTION RESULTS:")
            if actual_class_idx != -1:
                print(f"Actual Class: {actual_class_name}")
            else:
                print("Actual Class: Unknown (could not infer from filename)")
            print(f"Predicted Class: {self.class_names[pred_class]} ({pred_probs[pred_class]*100:.2f}% confidence)")
            
            if actual_class_idx != -1:
                if pred_class == actual_class_idx:
                    print("✓ CORRECT PREDICTION")
                else:
                    print("✗ INCORRECT PREDICTION")
                
            print("\nClass probabilities:")
            for name, prob in zip(self.class_names, pred_probs):
                print(f"{name}: {prob*100:.2f}%")
                
        except Exception as e:
            print(f"\n❌ Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, nargs='+', 
                       help="Path(s) to lung image(s) - can specify multiple paths")
    parser.add_argument("--actual", nargs='+',
                       help="Actual classes (if known): lung_aca, lung_n, or lung_scc - must match number of images")
    parser.add_argument("--client", required=True, help="Client ID (e.g., 2)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Threshold for cancer detection (0.0-1.0)")
    parser.add_argument("--save", action="store_true", 
                       help="Save output images")
    parser.add_argument("--output", default="output", 
                       help="Output directory for saved images")
    args = parser.parse_args()
    
    # Validate that if actual classes are provided, they match the number of images
    if args.actual and len(args.actual) != len(args.images):
        print("Error: Number of actual classes must match number of images")
        exit(1)
    
    model_path = f"client_{args.client}_ensemble.h5"
    
    print(f"\nLoading model weights from {model_path}")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        predictor = LungCancerPredictor(model_path)
        
        # Process multiple images
        for i, image_path in enumerate(args.images):
            if not os.path.exists(image_path):
                print(f"\n❌ Image not found: {image_path}")
                continue
                
            print(f"\nAnalyzing image: {image_path}")
            
            # If actual classes were provided, pass the corresponding one
            actual_class = args.actual[i] if args.actual else None
            
            predictor.visualize(
                image_path,
                actual_class=actual_class,
                threshold=args.threshold,
                save_output=args.save,
                output_dir=args.output
            )
            print("-" * 50)  # Separator between image results
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        print("\n💡 Final troubleshooting steps:")
        print("1. Verify the exact model architecture used during training")
        print("2. Check how the model was saved (model.save() vs model.save_weights())")
        print("3. Try retraining and saving the model with simpler architecture")
        print("4. Consider using TensorFlow 2.6.0 specifically")
        print("5. Share your model creation code for exact solution")