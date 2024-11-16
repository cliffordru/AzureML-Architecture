"""
Azure ML workspace setup and configuration utilities.
"""

import os
import yaml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute

def load_config(config_path: str) -> dict:
    """Load Azure configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_ml_client(config: dict) -> MLClient:
    """Create Azure ML client with default credentials."""
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name']
    )

def setup_compute_target(ml_client: MLClient, config: dict):
    """Set up compute targets for training."""
    # CPU compute cluster
    cpu_compute_config = AmlCompute(
        name=f"{config['name']}-cpu-cluster",
        type="amlcompute",
        size=config['compute_target']['cpu_cluster']['vm_size'],
        min_instances=config['compute_target']['cpu_cluster']['min_nodes'],
        max_instances=config['compute_target']['cpu_cluster']['max_nodes'],
        idle_time_before_scale_down=config['compute_target']['cpu_cluster']['idle_seconds_before_scaledown']
    )
    ml_client.begin_create_or_update(cpu_compute_config).result()

    # GPU compute cluster (if needed)
    gpu_compute_config = AmlCompute(
        name=f"{config['name']}-gpu-cluster",
        type="amlcompute",
        size=config['compute_target']['gpu_cluster']['vm_size'],
        min_instances=config['compute_target']['gpu_cluster']['min_nodes'],
        max_instances=config['compute_target']['gpu_cluster']['max_nodes'],
        idle_time_before_scale_down=config['compute_target']['gpu_cluster']['idle_seconds_before_scaledown']
    )
    ml_client.begin_create_or_update(gpu_compute_config).result()

def main():
    """Main setup function."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'azure_config.yml')
    config = load_config(config_path)
    
    # Create ML client
    ml_client = create_ml_client(config)
    
    # Setup compute targets
    setup_compute_target(ml_client, config)
    
    print("Azure ML workspace setup completed successfully!")

if __name__ == "__main__":
    main()
