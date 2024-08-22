import torch

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain_tunemodel.model import ClassifierFullConnect

def load(ckp_path='./tuned_high_rate.pt'):
    # model.load_state_dict(torch.load(ckp_path))
    tuned_classifier = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": 'cuda'})
    tuned_classifier.mods.embedding_model.fc = ClassifierFullConnect(tuned_classifier)
    tuned_classifier.mods.embedding_model.load_state_dict(torch.load(ckp_path))
    return tuned_classifier.mods.embedding_model
    # return torch.load(ckp_path).to('cuda').eval().mods.embedding_model

