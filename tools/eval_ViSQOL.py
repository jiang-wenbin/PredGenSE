import argparse
import csv
import glob
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from tqdm import tqdm

'''
python tools/eval_ViSQOL.py -c testset/clean -e testset/noisy
'''

TARGET_FS = 16000  # ViSQOL speech mode expects 16 kHz input
VISQOL_REPO = os.path.expanduser("~/workspace/agent/visqol")
DEFAULT_VISQOL_BIN = os.path.join(VISQOL_REPO, "bazel-bin", "visqol")
DEFAULT_LATTICE_MODEL = os.path.join(
    VISQOL_REPO, "model",
    "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite",
)


def load_mono_16k(path):
    audio, fs = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    if fs != TARGET_FS:
        audio = soxr.resample(audio, fs, TARGET_FS)
    return audio


def main(args):
    visqol_bin = os.path.abspath(args.visqol_bin)
    if not os.path.isfile(visqol_bin) or not os.access(visqol_bin, os.X_OK):
        raise FileNotFoundError(
            f"ViSQOL binary not found or not executable: {visqol_bin} "
            "(build it with `bazel build -c opt :visqol` in the ViSQOL repo)"
        )
    lattice_model = os.path.abspath(args.lattice_model)
    if not os.path.isfile(lattice_model):
        raise FileNotFoundError(
            f"ViSQOL lattice model not found: {lattice_model}"
        )

    ref_files = glob.glob(os.path.join(args.clean_folder, "*.wav"))
    ref_files.extend(glob.glob(os.path.join(args.clean_folder, "*.flac")))
    enhanced_files = glob.glob(os.path.join(args.enhanced_folder, "*.wav"))
    enhanced_files.extend(glob.glob(os.path.join(args.enhanced_folder, "*.flac")))

    enhanced_dict = {}
    for f in enhanced_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        enhanced_dict[stem] = f

    pairs = []
    for ref_file in tqdm(ref_files, desc="Matching pairs"):
        ref_stem = os.path.splitext(os.path.basename(ref_file))[0]
        for estem, epath in enhanced_dict.items():
            if estem == ref_stem or estem.startswith(ref_stem + "_"):
                pairs.append((ref_file, epath))
                break

    print(f"Processing {len(pairs)} pairs with ViSQOL (speech mode)...")
    scores = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_csv = os.path.join(tmpdir, "batch_input.csv")
        out_csv = os.path.join(tmpdir, "visqol_output.csv")

        with open(batch_csv, "w", newline="") as w:
            writer = csv.writer(w)
            writer.writerow(["reference", "degraded"])
            for ref_path, enh_path in pairs:
                ref_16k = os.path.join(tmpdir, "ref_" + os.path.basename(ref_path))
                enh_16k = os.path.join(tmpdir, "enh_" + os.path.basename(enh_path))
                sf.write(ref_16k, load_mono_16k(ref_path), TARGET_FS)
                sf.write(enh_16k, load_mono_16k(enh_path), TARGET_FS)
                writer.writerow([ref_16k, enh_16k])

        cmd = [
            visqol_bin,
            "--batch_input_csv", batch_csv,
            "--results_csv", out_csv,
            "--use_speech_mode",
            # ViSQOL resolves its default model relative to the CWD, so pass
            # the lattice model explicitly as an absolute path.
            "--similarity_to_quality_model", lattice_model,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with open(out_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row, (ref_path, enh_path) in zip(reader, pairs):
                uid = os.path.splitext(os.path.basename(enh_path))[0]
                scores[uid] = float(row["moslqo"])

    if not scores:
        print("No ViSQOL values calculated. Check if the folders contain matching audio files.")
        return

    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "ViSQOL.scp").open("w") as w:
            for uid, score in scores.items():
                w.write(f"{uid} {score}\n")
        with (outdir / "RESULTS.txt").open("w") as w:
            w.write(f"ViSQOL: {np.nanmean(list(scores.values())):.4f}\n")
        print(f"Results -> {outdir}", flush=True)

    print({"ViSQOL": round(float(np.nanmean(list(scores.values()))), 3)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate average ViSQOL (speech mode) for audio file pairs."
    )
    parser.add_argument('--clean_folder', '-c', type=str, required=True,
                        help="Path to the folder containing clean audio files")
    parser.add_argument('--enhanced_folder', '-e', type=str, required=True,
                        help="Path to the folder containing enhanced audio files")
    parser.add_argument('--visqol_bin', type=str, default=DEFAULT_VISQOL_BIN,
                        help="Path to the ViSQOL binary")
    parser.add_argument('--lattice_model', type=str, default=DEFAULT_LATTICE_MODEL,
                        help="Path to the ViSQOL speech-mode lattice model (.tflite)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Optional dir to write per-file scores and results")
    main(parser.parse_args())
