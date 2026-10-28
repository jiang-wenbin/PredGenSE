import os
import argparse
import glob
from platform import node
from utils import find_available_gpu
import pandas as pd
from tqdm import tqdm
import torch
from torchmetrics.text import WordErrorRate, CharErrorRate
import whisper

'''
python calculate_WER.py -c wav/clean/ -e wav/enh/
'''

def init():
    if "CUDA_VISIBLE_DEVICES" not in os.environ: # slrum will set it
        os.environ['CUDA_VISIBLE_DEVICES'] = find_available_gpu()
    print("Hostname: {}, CUDA_VISIBLE_DEVICES: [{}], device_count: {}".format(
        node(), os.environ.get('CUDA_VISIBLE_DEVICES', None), torch.cuda.device_count()))

def preprocess_text(text):
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace("-", "")
    text = text.replace(",", "")
    text = text.replace("!", "")
    text = text.lower()
    return text.strip()

def calculate_average_wer(clean_folder, enhanced_folder, model_name, device):
    clean_files = glob.glob(os.path.join(clean_folder, "*.wav"))
    clean_files.extend(glob.glob(os.path.join(clean_folder, "*.flac")))
    clean_files.extend(glob.glob(os.path.join(clean_folder, "*.mp3")))

    enhanced_files = glob.glob(os.path.join(enhanced_folder, "*.wav"))
    enhanced_files.extend(glob.glob(os.path.join(enhanced_folder, "*.flac")))
    enhanced_files.extend(glob.glob(os.path.join(enhanced_folder, "*.mp3")))

    wer_metric = WordErrorRate().to(device)
    cer_metric = CharErrorRate().to(device)
    model = whisper.load_model(model_name)
    model = model.to(device)

    clean_files_dict = {os.path.basename(f): f for f in clean_files}

    rows = []
    for enhanced_file in tqdm(enhanced_files, desc='WER calculating'):
        filename = os.path.basename(enhanced_file)
        if filename in clean_files_dict:
            clean_file = clean_files_dict[filename]

            result_gt = model.transcribe(clean_file, temperature=0)
            result_pred = model.transcribe(enhanced_file, temperature=0)
            if result_gt['language'] == 'zh':
                result_gt = model.transcribe(clean_file, initial_prompt="以下是普通话的句子")
            if result_pred['language'] == 'zh':
                result_pred = model.transcribe(enhanced_file, initial_prompt="以下是普通话的句子")

            content_gt = result_gt["text"]
            content_pred = result_pred["text"]

            content_gt = preprocess_text(content_gt)
            content_pred = preprocess_text(content_pred)

            if result_gt['language'] == 'zh' and result_pred['language'] == 'zh':
                wer_score = cer_metric(content_pred, content_gt).item() * 100
                wer_score = min(wer_score, 100.0)
                rows.append({"filename": filename, "WER": wer_score, "mode": "CER", "language": "zh"})
            else:
                wer_score = wer_metric(content_pred, content_gt).item() * 100
                wer_score = min(wer_score, 100.0)
                rows.append({"filename": filename, "WER": wer_score, "mode": "WER", "language": result_gt['language']})
        else:
            print(f"Warning: No matching clean file found for {filename}")

    if rows:
        df = pd.DataFrame(rows)
        average_wer = df['WER'].mean()
        print("Average WER:", round(average_wer, 2))
        if 'mode' in df.columns:
            for mode in df['mode'].unique():
                subset = df[df['mode'] == mode]
                print(f"  Average {mode}: {round(subset['WER'].mean(), 2)} ({len(subset)} files)")
    else:
        print("No WER values calculated. Check if the folders contain matching audio files.")

def main():
    init()
    parser = argparse.ArgumentParser(description="Calculate average WER for audio files in specified folders using Whisper ASR.")
    parser.add_argument('--clean_folder', '-c', type=str, required=True, help="Path to the folder containing clean audio files")
    parser.add_argument('--enhanced_folder', '-e', type=str, required=True, help="Path to the folder containing enhanced audio files")
    parser.add_argument('--model', '-m', type=str, default='turbo', choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'], help="Whisper model size (default: turbo)")
    parser.add_argument('--device', type=str, default='cuda', help="Device for running whisper ASR")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU", flush=True)
        device = "cpu"

    calculate_average_wer(args.clean_folder, args.enhanced_folder, args.model, device)

if __name__ == "__main__":
    main()
