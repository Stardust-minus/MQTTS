from tester import Wav2TTS_infer
import argparse
import soundfile as sf
import pyloudnorm as pyln
import os
from pathlib import Path
import json

from text import symbols
from text.cleaner import clean_text

parser = argparse.ArgumentParser()

#Path
parser.add_argument('--outputdir', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--spkr_embedding_path', type=str, default=None)

#Data
parser.add_argument('--sample_rate', type=int, default=32000)
parser.add_argument('--batch_size', type=int, default=32)

#Sampling
parser.add_argument('--use_repetition_gating', action='store_true')
parser.add_argument('--repetition_penalty', type=float, default=1.0)
parser.add_argument('--sampling_temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=-1)
parser.add_argument('--min_top_k', type=int, default=1)
parser.add_argument('--top_p', type=float, default=0.7)
parser.add_argument('--length_penalty_max_length', type=int, default=50)
parser.add_argument('--length_penalty_max_prob', type=float, default=0.8)
parser.add_argument('--max_output_length', type=int, default=100000)
parser.add_argument('--phone_context_window', type=int, default=4)

#Speech Prior
parser.add_argument('--clean_speech_prior', action='store_true')
parser.add_argument('--prior_noise_level', type=float, default=1e-5)
parser.add_argument('--prior_frame', type=int, default=3)

args = parser.parse_args()

args.phoneset = symbols

with open(args.config_path, 'r') as f:
    argdict = json.load(f)
    assert argdict['sample_rate'] == args.sample_rate, f"Sampling rate not consistent, stated {args.sample_rate}, but the model is trained on {argdict['sample_rate']}"
    argdict.update(args.__dict__)
    args.__dict__ = argdict

if __name__ == '__main__':
    Path(args.outputdir).mkdir(parents=True, exist_ok=True)
    meter = pyln.Meter(args.sample_rate)
    with open(args.input_path, 'r') as f:
        input_file = json.load(f)
    model = Wav2TTS_infer(args)
    model.cuda()
    model.vocoder.generator.remove_weight_norm()
    model.vocoder.encoder.remove_weight_norm()
    model.eval()
    i_wavs,sids, i_phones, written = [], [], [], 0
    for i, (speaker_path, sid, sentence) in enumerate(input_file):
        if args.spkr_embedding_path:
            i_wavs.append(os.path.join(args.spkr_embedding_path, os.path.basename(speaker_path)[:-4] + '.npy'))
        else:
            audio, sr = sf.read(speaker_path)
            assert sr == args.sample_rate
            loudness = meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, loudness, -20.0)
            i_wavs.append(audio)
        phones = clean_text(sentence)
        sids.append(int(sid))
        i_phones.append(phones)
        if len(i_wavs) == args.batch_size:
            print (f"Inferencing batch {written//args.batch_size+1}, total {len(input_file)//args.batch_size+1} baches.")
            synthetic = model(i_wavs,sids, i_phones)
            for s in synthetic:
                sf.write(os.path.join(args.outputdir, f'sentence-{written+1}-1.wav'), s, args.sample_rate)
                written += 1
            i_wavs,sids, i_phones = [],[], []
    if len(i_wavs) > 0:
        synthetic = model(i_wavs, i_phones)
        for s in synthetic:
            sf.write(os.path.join(args.outputdir, f'sentence-{written+1}-1.wav'), s, args.sample_rate)
            written += 1
