from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar, Metrics
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model
import warnings

warnings.filterwarnings("ignore")

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average of metrics."""
    num_samples = [n for n, _ in metrics]
    results = {}
    
    for key in metrics[0][1].keys():
        if key.startswith("_"):
            continue
            
        if key.endswith("time"):
            # For time metrics, use average
            values = [m[key] for _, m in metrics]
            results[key] = sum(values) / len(values)
        else:
            # For other metrics, use weighted average
            weighted_values = [m[key] * n for n, m in metrics]
            results[key] = sum(weighted_values) / sum(num_samples)
    
    # Add client info
    client_ids = [m["client_id"] for _, m in metrics]
    results["clients"] = ", ".join(str(cid) for cid in client_ids)
    
    return results

def get_densenet_model(num_classes=3):
    """Create DenseNet121 model with proper preprocessing."""
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax', name='densenet_output')(x)
    
    return Model(inputs=base_model.input, outputs=outputs)

def get_mobilenet_model(num_classes=3):
    """Create MobileNetV2 model with proper preprocessing."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax', name='mobilenet_output')(x)
    
    return Model(inputs=base_model.input, outputs=outputs)

def get_ensemble_model(densenet_model, mobilenet_model):
    """Create ensemble model combining DenseNet and MobileNet."""
    densenet_output = densenet_model.get_layer('densenet_output').output
    mobilenet_output = mobilenet_model.get_layer('mobilenet_output').output
    
    # Combine outputs with learnable weights
    combined = concatenate([densenet_output, mobilenet_output])
    weights = Dense(2, activation='softmax', name='ensemble_weights')(combined)
    weighted_output = weights[:, 0:1] * densenet_output + weights[:, 1:2] * mobilenet_output
    
    ensemble_model = Model(
        inputs=[densenet_model.input, mobilenet_model.input],
        outputs=weighted_output
    )
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return ensemble_model