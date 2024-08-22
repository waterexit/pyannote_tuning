import torch
from speechbrain.inference.speaker import EncoderClassifier


class LastLiner(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # print(input.squeeze(2).shape)
        return super().forward(input.squeeze(2)).unsqueeze(2)


class ClassifierFullConnect(torch.nn.Module):
    def __init__(self, classifier: EncoderClassifier) -> None:
        super().__init__()
        last_liner = LastLiner(192, 2).to('cuda')
        self.fc = torch.nn.Sequential(classifier.mods.embedding_model.fc,
                                      torch.nn.ReLU().to('cuda'),
                                      last_liner)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.fc.forward(input)
