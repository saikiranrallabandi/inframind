"""
InfraMind Data Module - Templates and Dataset Generation
"""

from .templates import (
    TERRAFORM_TEMPLATES,
    K8S_TEMPLATES,
    DOCKER_TEMPLATES,
    CICD_TEMPLATES,
    CLOUDFORMATION_TEMPLATES,
    PULUMI_TEMPLATES,
    ANSIBLE_TEMPLATES,
    HELM_TEMPLATES,
    MONITORING_TEMPLATES,
    ERROR_TEMPLATES,
    MLOPS_TEMPLATES,
    OUT_OF_DOMAIN_EXAMPLES,
    OUT_OF_DOMAIN_RESPONSE,
    PARAMS,
)

from .generators import (
    generate_tasks,
    create_dataset,
)

from .loaders import (
    load_training_data_from_json,
    EMBEDDED_TRAINING_DATA,
)

__all__ = [
    # Templates
    "TERRAFORM_TEMPLATES",
    "K8S_TEMPLATES",
    "DOCKER_TEMPLATES",
    "CICD_TEMPLATES",
    "CLOUDFORMATION_TEMPLATES",
    "PULUMI_TEMPLATES",
    "ANSIBLE_TEMPLATES",
    "HELM_TEMPLATES",
    "MONITORING_TEMPLATES",
    "ERROR_TEMPLATES",
    "MLOPS_TEMPLATES",
    "OUT_OF_DOMAIN_EXAMPLES",
    "OUT_OF_DOMAIN_RESPONSE",
    "PARAMS",
    # Generators
    "generate_tasks",
    "create_dataset",
    # Loaders
    "load_training_data_from_json",
    "EMBEDDED_TRAINING_DATA",
]
