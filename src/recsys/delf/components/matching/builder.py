from .matching import NeuralCollaborativeFilteringLayer


def matching_fn_builder(**kwargs):
    cls = NeuralCollaborativeFilteringLayer
    return cls(**kwargs)