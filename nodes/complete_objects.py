"""Select the one authoritative asset for every instance, with or without generation."""
from candidates.pipeline import run_candidates


def make_complete_objects(service, config=None, sam=None, reviewer=None, runtime=None):
    if config is None:
        from app.config import PipelineConfig
        config = PipelineConfig(object_completion={'enabled': True})
    def complete_objects(state):
        return run_candidates(state, config, service, sam, reviewer, runtime)
    return complete_objects
