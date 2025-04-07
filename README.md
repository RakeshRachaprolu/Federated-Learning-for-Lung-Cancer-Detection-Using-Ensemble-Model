# Lung Cancer Detection with Federated Learning and Ensemble Models

This repository contains a federated learning-based system for lung cancer detection using ensemble deep learning models (DenseNet121 and MobileNetV2). The project implements a client-server architecture using Flower (`flwr`) for distributed training, a Tkinter-based GUI for both server and client management, and a prediction module with Grad-CAM visualization for interpreting model results.

The system is designed to classify lung images into three categories:
- `lung_aca`: Lung Adenocarcinoma
- `lung_n`: Normal Lung Tissue
- `lung_scc`: Lung Squamous Cell Carcinoma

## Features
- **Federated Learning**: Distributed training across multiple clients with a central server aggregating model weights.
- **Ensemble Models**: Combines DenseNet121 and MobileNetV2 with learnable weights for improved accuracy.
- **GUI**: Tkinter-based interfaces for managing the server (`FlowerServerApp`) and clients (`LungCancerClientApp`).
- **Visualization**: Grad-CAM heatmaps and cancer tissue isolation for model interpretability.
- **Metrics**: Confusion matrices, classification reports, and ROC curves generated for each client.

