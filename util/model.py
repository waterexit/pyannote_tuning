import torch
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN,Conv1d


class LastLiner(torch.nn.Module):
    def __init__(self, encorder: ECAPA_TDNN, device='cuda'):
        super().__init__()
        self.encorder = encorder.to(device)
        self.encorder.fc = Conv1d(
            in_channels=6144,
            out_channels=2,
            kernel_size=1).train()

    def eval(self):
        self.trainig = False
        self.encorder.eval()
        self.dropout.eval()
        self.fc.eval()
        return self

    def forward(self, input, wav_lens):
        output = self.encorder(input, wav_lens)
        print('???')
        output = self.dropout(output)
        output = self.fc(output)
        return output
