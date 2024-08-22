from pyannote.audio import Pipeline
import re
import glob
from util.custom_clustering import CustomCluster
import math
import torch
from pydub import AudioSegment
import os
import time
import json
import argparse
from dotenv import load_dotenv

load_dotenv()
parser = argparse.ArgumentParser('main')
parser.add_argument('target_path')
args = parser.parse_args()

pipeline = Pipeline.from_pretrained(checkpoint_path="./config.yaml",
                                    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
pipeline.to(torch.device("cuda"))

pipeline.clustering = CustomCluster(pipeline._embedding.classifier_)


def run(pipeline, origin_file_path, guest_name, result_path=None, split_minutes=5, export=False):
    start = time.time()
    # メモリが枯渇するので、ある程度元ファイルを分割する
    tmp_file_name = 'work.mp3'
    split_sec = (60 * split_minutes)
    file_name = os.path.basename(origin_file_path)
    chart = {'host': 0,  guest_name: 0}
    part_num = 0
    while True:
        part = AudioSegment.from_file(origin_file_path, start_second=split_sec*part_num, duration=split_sec)
        part.export(tmp_file_name)
        if len(part) == 0:
            break
        diarization = pipeline(tmp_file_name, num_speakers=2, return_embeddings=False)

        part_num = part_num + 1
        try:
            chart['host'] = chart['host'] + [x[1] for x in diarization.chart() if x[0] == 'SPEAKER_00'][0]
            chart[guest_name] = chart[guest_name] + [x[1] for x in diarization.chart() if x[0] == 'SPEAKER_01'][0]
        except:
            pass

        index = 0
        part_audio_seg = AudioSegment.from_file(tmp_file_name)
        if export:
            if not os.path.exists(f'{result_path}/{file_name}/host'):
                os.makedirs(f'{result_path}/{file_name}/host')
            if not os.path.exists(f'{result_path}/{file_name}/other'):
                os.makedirs(f'{result_path}/{file_name}/other')
            for seg, _, label in diarization.itertracks(yield_label=True):
                label = 'host' if label == 'SPEAKER_00' else 'other'
                part_audio_seg[math.floor(seg.start*1000):math.floor(seg.end*1000)].export(
                    f'{result_path}/{file_name}/{label}/{part_num}_{index}.mp3', "mp3")
                index = index + 1
    end = time.time()
    print(chart)
    with open(f'{file_name}_chart.json', mode='a') as f:
        json.dump(chart, f, ensure_ascii=False)
    os.remove(tmp_file_name)
    print(end-start)


start = time.time()
for p in glob.glob(f'{args.target_path}/*'):
    guest_name = re.match(r'\[(.*?)\]', os.path.basename(p)).group(1)
    run(pipeline, p, guest_name=guest_name, split_minutes=2, result_path='./result', export=True)
print(time.time() - start)
