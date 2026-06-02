"""Personality readout / measurement via the Pandora-trained text->OCEAN model.

This is the ONLY role of the Pandora model in MindForm: estimating EXPRESSED Big
Five traits from a piece of text (the agent's own output, or any text), for
display and for evaluating personality trajectories.

It is never used to encode an experience or to compute impact. Trait expression
answers "who tends to write this" -- an estimation signal, not a formation force.
Keeping it strictly read-only is what prevents the original estimation bug from
creeping back in.
"""

from trait_model import predict


def measure_expression(text):
    """Estimate expressed OCEAN traits from text. Returns {trait_name: [0, 1]}."""
    return predict(text)
