"""=============================================================================
Utility script for embedding the data if this fails in the main job.
============================================================================="""

from   data import loader
from   models import load_trained_model
from scripts import embedder

# ------------------------------------------------------------------------------

fname = 'data/gtex/model.pt'
cfg = loader.get_config('gtex')
model = load_trained_model(cfg, fname)
embedder.preembed_images('data/gtex', cfg, model)
