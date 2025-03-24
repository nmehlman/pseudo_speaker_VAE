import torch
import torchaudio
import os
import sys

from utils import load_freevc_model

FREE_VC_INPUT_SR = 16000
FREE_VC_OUTPUT_SR = 24000

def run_embedding_conditioned_vc(
        speaker_embedding: torch.Tensor, 
        source_audio_path: str,
        free_vc_model,
    ):
    
    # Load audio
    source_audio, fs = torchaudio.load(source_audio_path)
    if fs != FREE_VC_INPUT_SR:
        source_audio = torchaudio.transforms.Resample(fs, FREE_VC_INPUT_SR)(source_audio)
    
    # Run conversion
    converted_audio = free_vc_model.voice_conversion_embed(source_audio.squeeze(), speaker_embedding) # DEBUG
    
    return converted_audio.squeeze(0)

if __name__ == "__main__":
    
    import tqdm
    
    source_audio_path = "/home1/nmehlman/nick_codebase/misc/test_audio.wav"
    embedding_dir = "/project/shrikann_35/nmehlman/psg_data/vctk_sample"
    save_dir = "/home1/nmehlman/arts/pseudo_speakers/audio_samples/real_embeds"
    vc_ckpt_dir = "/home1/nmehlman/.local/share/tts/voice_conversion_models--multilingual--vctk--freevc24/"

    # Load the FreeVC model
    model = load_freevc_model(vc_ckpt_dir)

    for embed_file in tqdm.tqdm(os.listdir(embedding_dir), desc="Converting audio"):
        
        id = embed_file.split("_")[-1].replace(".pt", "")
        
        emedding = torch.load(os.path.join(embedding_dir, embed_file)).reshape(1,-1,1)
        converted_audio = run_embedding_conditioned_vc(emedding, source_audio_path, free_vc_model=model)
        torchaudio.save(os.path.join(save_dir, f"sample_embed_{id}.wav"), converted_audio, FREE_VC_OUTPUT_SR)
        
