import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
import argparse
import socket

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Proper metric aggregation handling different types."""
    aggregated = {}
    
    # Numeric metrics to aggregate
    numeric_metrics = [
        'train_accuracy_densenet', 'train_accuracy_mobilenet',
        'test_accuracy_densenet', 'test_accuracy_mobilenet',
        'ensemble_accuracy', 'round_time', 'eval_time',
        'densenet_weight', 'mobilenet_weight'
    ]
    
    for key in numeric_metrics:
        if key in metrics[0][1]:
            weighted_values = [m[1][key] * m[0] for m in metrics]
            aggregated[key] = np.sum(weighted_values) / np.sum([m[0] for m in metrics])
    
    # Handle client IDs
    client_ids = [m[1]["client_id"] for m in metrics]
    aggregated["clients"] = ", ".join(str(cid) for cid in client_ids)
    
    return aggregated

def get_local_ip():
    """Get the local IP address of the server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"Error getting local IP: {str(e)}")
        return "127.0.0.1"

def start_server(port):
    local_ip = get_local_ip()
    server_address = f"0.0.0.0:{port}"
    print(f"Starting Flower server on {server_address} (Local IP: {local_ip})")
    print(f"Clients should connect to {local_ip}:{port}")
    
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server for Lung Cancer Detection")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    args = parser.parse_args()
    start_server(args.port)