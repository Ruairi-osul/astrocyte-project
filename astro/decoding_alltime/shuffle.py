from .config import ConfigTemplate
from typing import Callable
from trace_minder.surrogates import SurrogateGenerator
from copy import deepcopy


def make_shuffle_templates(
    template: ConfigTemplate,
    n: int,
    surrogate_factory: Callable[[], SurrogateGenerator],
) -> list[ConfigTemplate]:
    """
    Given a ConfigTemplate, create n surrogates using a provided surrogate factory.

    Args:
        template: The ConfigTemplate to use as a base for the shuffled templates.
        n: The number of shuffled templates to create.
        surrogate_factory: A function that returns a SurrogateGenerator object.

    Returns:
        A list of ConfigTemplates for shuffled data.
    """
    templates = []
    for i in range(n):
        new_template = deepcopy(template)
        new_template.session_name = template.session_name
        new_template.surrogate_factory = surrogate_factory
        templates.append(new_template)
    return templates


def replicate_template(
    template: ConfigTemplate,
    n: int,
) -> list[ConfigTemplate]:
    """
    Given a ConfigTemplate, replicate it n times.

    Args:
        template: The ConfigTemplate to replicate.
        n: The number of times to replicate the template.

    Returns:
        A list of ConfigTemplates.
    """
    templates = []
    for i in range(n):
        new_template = deepcopy(template)
        new_template.session_name = template.session_name
        templates.append(new_template)
    return templates
