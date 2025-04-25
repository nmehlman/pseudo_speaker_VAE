import torch
import os
import certifi
from utils import load_freevc_model
from speech_synthesis import run_embedding_conditioned_vc, FREE_VC_OUTPUT_SR
import pandas as pd
from itertools import product
import torchaudio
import tqdm
import time

os.environ["SSL_CERT_FILE"] = certifi.where()  # Set SSL_CERT_FILE for TTS

DEMO = 'gender'

DEMOS_TO_TEST = ['male', 'female']

N_SRC_PER_DEMO = 5
N_TGT_PER_DEMO = 5

vc_ckpt_dir = "/home1/nmehlman/.local/share/tts/voice_conversion_models--multilingual--vctk--freevc24/"
data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
metadata_path = os.path.join(data_root, "train_embeds.tsv")
audio_dir = os.path.join(data_root, "clips")
emebd_dir = os.path.join(data_root, "embeds_vc/train")
save_dir = "/home1/nmehlman/arts/pseudo_speakers/samples/demo_consistency/"

# Load the FreeVC model
model = load_freevc_model(vc_ckpt_dir)

# Load CV metadata
start_time = time.time()
metadata = pd.read_csv(metadata_path, sep="\t")
print(f"Metadata loading took {time.time() - start_time:.2f} seconds")

for src_demo in DEMOS_TO_TEST: # Conversion goes source_audio -> target embedding
    for tgt_demo in DEMOS_TO_TEST:
        
        print(f"Testing {src_demo} --> {tgt_demo}...")
        subdir = os.path.join(save_dir, f"{src_demo}-->{tgt_demo}")
        os.makedirs(subdir, exist_ok=True)
        
        # Randomly select N_SRC_PER_DEMO source speakers and N_TGT_PER_DEMO target speakers
        src_candidates = metadata[metadata['gender'] == src_demo]
        tgt_candidates = metadata[metadata['gender'] == tgt_demo]
        scr_samples = src_candidates.sample(N_SRC_PER_DEMO)
        tgt_samples = tgt_candidates.sample(N_TGT_PER_DEMO)

        for i, ((_, src_row), (_, tgt_row)) in tqdm.tqdm(
            enumerate(
                product(scr_samples.iterrows(), 
                              tgt_samples.iterrows())),
                total=N_SRC_PER_DEMO * N_TGT_PER_DEMO
            ):
                                
            src_audio_path = os.path.join(src_row['path'])
            tgt_emebd_path = os.path.join(emebd_dir, tgt_row['path'].replace(".mp3", ".pth"))

            # Load target embedding
            tgt_embed = torch.load(tgt_emebd_path)

            converted_audio = run_embedding_conditioned_vc( # Run VC
                speaker_embedding = tgt_embed, 
                source_audio_path = os.path.join(audio_dir, src_audio_path),
                free_vc_model = model,
            )

            # Save audio
            torchaudio.save(
                os.path.join(subdir, f"sample_{i}.wav"), 
                converted_audio, 
                FREE_VC_OUTPUT_SR
            )