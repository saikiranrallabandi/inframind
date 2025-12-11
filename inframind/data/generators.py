"""
InfraMind Dataset Generators - Generate training tasks from templates.
"""

import random
from typing import List, Dict, Optional

from .templates import (
    ALL_TEMPLATES,
    OUT_OF_DOMAIN_EXAMPLES,
    OUT_OF_DOMAIN_RESPONSE,
    PARAMS,
)


def generate_tasks(categories: Optional[List[str]] = None) -> List[Dict]:
    """
    Generate training tasks from all templates.

    Args:
        categories: Optional list of categories to include. If None, includes all.

    Returns:
        List of task dictionaries with instruction, input, category fields.
    """
    tasks = []
    task_id = 0

    templates_to_use = ALL_TEMPLATES
    if categories:
        templates_to_use = {k: v for k, v in ALL_TEMPLATES.items() if k in categories}

    for category, (templates, variations) in templates_to_use.items():
        for template, input_template in templates:
            for _ in range(variations):
                task_id += 1
                input_str = input_template
                instr = template

                # Fill in parameter placeholders
                for key, values in PARAMS.items():
                    if f"{{{key}}}" in input_str:
                        input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                    if f"{{{key}}}" in instr:
                        instr = instr.replace(f"{{{key}}}", random.choice(values))

                tasks.append({
                    "id": f"{category}-{task_id:03d}",
                    "instruction": instr,
                    "input": input_str,
                    "category": category,
                })

    # Add out-of-domain examples
    for instruction, input_str in OUT_OF_DOMAIN_EXAMPLES:
        task_id += 1
        tasks.append({
            "id": f"ood-{task_id:03d}",
            "instruction": instruction,
            "input": input_str,
            "output": OUT_OF_DOMAIN_RESPONSE,
            "category": "out-of-domain",
        })

    return tasks


def create_dataset(size: Optional[int] = None, categories: Optional[List[str]] = None) -> List[Dict]:
    """
    Create a dataset of IaC tasks.

    Args:
        size: Optional maximum number of tasks. If None, returns all.
        categories: Optional list of categories to include.

    Returns:
        List of task dictionaries.
    """
    tasks = generate_tasks(categories=categories)

    if size:
        tasks = tasks[:size]

    return tasks


def get_category_stats(tasks: List[Dict]) -> Dict[str, int]:
    """Get count of tasks per category."""
    stats = {}
    for task in tasks:
        cat = task.get("category", "unknown")
        stats[cat] = stats.get(cat, 0) + 1
    return stats
