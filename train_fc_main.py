from speechbrain.inference.speaker import EncoderClassifier
import torch
from util import utility
from util.dataset import SimpleFileDataset
from tune_runner import TuneRunner
import sys
from speechbrain_tunemodel.model import ClassifierFullConnect

torch.device('cuda')
epoch_num = 10
bsz = 8
tuned_classifier = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": 'cuda'})
if tuned_classifier == None:
    sys.exit()

tuned_classifier.mods.embedding_model.fc = ClassifierFullConnect(tuned_classifier)
print(tuned_classifier.mods.embedding_model.fc.fc[2])
tuned_optimizer = torch.optim.SGD(tuned_classifier.mods.embedding_model.fc.fc[2].parameters(), lr=1.0, momentum=0.1)
cosine_annealing_scheduler_tuned = torch.optim.lr_scheduler.CosineAnnealingLR(
    tuned_optimizer, T_max=10, eta_min=0)

runner = TuneRunner('final_model', tuned_classifier, tuned_optimizer, cosine_annealing_scheduler_tuned)

host_dir = '/sample_data/host/'
guest_dir = '/sample_data/guest/'
host_file_list = utility.file_list(host_dir)
guest_file_list = utility.file_list(guest_dir)
all_dataset = SimpleFileDataset(host_file_list, guest_file_list)

valid_loss_acc_list, correct_rate_list = runner.main(all_dataset, epoch_num, bsz)
runner.save_model('./speechbrain_tunemodel/final_model_state')

