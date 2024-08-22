from sortedcontainers.sortedlist import traceback
import torch
from speechbrain.inference.speaker import EncoderClassifier


class TuneRunner():
    def __init__(self, name: str, classifier: EncoderClassifier, optimizer, lr_scheduler):
        torch.device('cuda')
        self.name = name
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.valid_loss_acc_list = []
        self.correct_rate_list = []

    def _encode(self, paths):
        input = torch.zeros(0).to('cuda')
        for path in paths:
            waveform = self.classifier.load_audio(path, '/tmp')
            batch = waveform.unsqueeze(0)
            rel_length = torch.tensor([1.0])
            try:
                input = torch.cat((input, self.classifier.encode_batch(batch, rel_length).squeeze(0)), 0)
            except Exception as e:
                print('error', traceback.format_exc())
                input = torch.cat((input, torch.zeros(1, 2).to('cuda')), 0)
        return input

    def train(self, bsz, train_dataset):
        loss_value = 0
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, drop_last=True)
        for name, param in self.classifier.named_parameters():
            # if 'fc.weight' in name or 'fc.bias' in name:
            param.requires_grad = True

        for paths, labels in train_data_loader:
            input = self._encode(paths)
            certication = torch.nn.CrossEntropyLoss().to('cuda')
            loss = certication(input, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = loss_value + loss.item()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss_value

    def valid(self, bsz, val_dataset):
        for _, param in self.classifier.named_parameters():
            param.requires_grad = False
        valid_loss = 0
        correct_count = 0
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, drop_last=True)
        for paths, labels in val_data_loader:
            input = self._encode(paths)
            certication = torch.nn.CrossEntropyLoss()
            loss = certication(input, labels)
            valid_loss = valid_loss + loss.item()
            correct_count = correct_count + \
                torch.count_nonzero(torch.eq(torch.argmax(input, -1), torch.argmax(labels, -1))).item()
        return valid_loss, correct_count/len(val_dataset)

    def main(self, all_dataset, epoch_num, bsz, train_ratio=0.8):
        n_samples = len(all_dataset)
        train_size = int(len(all_dataset) * train_ratio)
        val_size = n_samples - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size])

        for epoch in range(epoch_num):
            train_loss = self.train(bsz, train_dataset)
            print(f'epoch{epoch + 1} {self.name} train_loss', train_loss, "acc", train_loss/len(train_dataset))

            valid_loss, correct_rate = self.valid(bsz, val_dataset)
            valid_loss_acc = valid_loss/len(val_dataset)
            print(f'epoch{epoch + 1} {self.name} valid_loss', valid_loss, 'acc', valid_loss_acc)
            print(f'epoch{epoch + 1} {self.name} correct_rate', correct_rate)
            self.valid_loss_acc_list.append(valid_loss_acc)
            self.correct_rate_list.append(correct_rate)

        return self.valid_loss_acc_list, self.correct_rate_list

    def save_model(self, save_name=None):
        if save_name == None:
            torch.save(self.classifier.mods.embedding_model.state_dict(), self.name)
        else:
            torch.save(self.classifier.mods.embedding_model.state_dict(), save_name)
