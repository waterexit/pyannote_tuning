import pathlib
import os
import itertools
import math
import torch.nn.utils.rnn as rnn
from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf
import librosa
import torch
from audiomentations import AddGaussianNoise, PitchShift, Compose,  ClippingDistortion, Reverse, TanhDistortion, TimeMask
from torch.utils.data import Dataset


class AudioFeaturesDataset(Dataset):
    def __init__(self, classifier: EncoderClassifier, host_file_list, guest_file_list):
        self.classifier = classifier
        self.transform = Compose([AddGaussianNoise(min_amplitude=0.01, max_amplitude=5, p=0.75),
                                  PitchShift(min_semitones=-12, max_semitones=12, p=0.75),
                                  # PolarityInversion(p=0.75),
                                  Reverse(p=0.75),
                                  ClippingDistortion(),
                                  TanhDistortion(p=0.75),
                                  TimeMask(),
                                  # SevenBandParametricEQ(),
                                  ])
        self.data = [{'path': p, 'label': torch.zeros(1)} for p in host_file_list]
        self.data = self.data + [{'path': p, 'label': torch.ones(1)} for p in guest_file_list]

    def __getitem__(self, index):
        data = self.data[index]
        # y1 = self._transform(data['path']).squeeze(0)
        y1 = self._get_feature(data['path']).squeeze(0)
        y2 = self._transform(data['path']).squeeze(0)
        # return torch.cat((y1, y2), 0), data['label']
        return (y1, y2), data['label'], data['path']

    def __len__(self):
        return len(self.data)

    def _transform(self, path):
        signal, sr = librosa.load(path)
        signal = self.transform(samples=signal, sample_rate=sr)
        sf.write('tmp.wav', signal, sr)
        return self._get_feature('tmp.wav')

    def _get_feature(self, path):
        waveform = self.classifier.load_audio(path, '/tmp')
        # if self.classifier.training == True:
        return waveform
        # else:
        #     batch = waveform.unsqueeze(0)
        #     rel_length = torch.tensor([1.0])
        #     try:
        #         emb = self.classifier.encode_batch(batch, rel_length)
        #     except:
        #         emb = torch.zeros(1, 1, 192).to('cuda')
        #     return emb

    def collate_fn_for_supcon(self, batch):
        wavforms, labels, paths = list(zip(*batch))
        wavforms = list(itertools.chain.from_iterable(wavforms))
        pad_wavforms = rnn.pad_sequence(wavforms, batch_first=True)
        max_dim_length = max([wavform.shape[0] for wavform in wavforms])
        lr_tensor = torch.Tensor([wavform.shape[0]/max_dim_length for wavform in wavforms]).to('cuda')
        batch_for_supcon = self.classifier.encode_batch(pad_wavforms, lr_tensor).reshape(len(batch), 2, 192).to('cuda')
        return batch_for_supcon, torch.stack(labels), list(paths)


class SimpleFileDataset(Dataset):
    def __init__(self, host_file_list, guest_file_list):
        self.data = [{'path': p, 'label': torch.Tensor([1, 0]).to('cuda')} for p in host_file_list]
        self.data = self.data + [{'path': p, "label": torch.Tensor([0, 1]).to('cuda')} for p in guest_file_list]

    def __getitem__(self, index):
        data = self.data[index]
        return data['path'], data['label']

    def __len__(self):
        return len(self.data)

    # def _encode(self,path):
    #     waveform = self.classifier.load_audio(path, '/tmp')
    #     batch = waveform.unsqueeze(0)
    #     rel_length = torch.tensor([1.0])
    #     try:
    #         emb = self.classifier.encode_batch(batch, rel_length)
    #     except:
    #         emb = torch.zeros(1, 1, 192).to('cuda')
    #     return emb
    #


if __name__ == '__main__':
    def file_list(dir_path):
        return [os.path.join(dir_path, p) for p in os.listdir(dir_path) if pathlib.Path(os.path.join(dir_path, p)).is_file()]
    classifier = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": 'cuda'})
    classifier.train()
    dataset = AudioFeaturesDataset(classifier, file_list('./tmp/host'), file_list('./tmp/guest'))
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=25, collate_fn=dataset.collate_fn_for_supcon)
    for data, label in train_data_loader:
        print(data, label)
