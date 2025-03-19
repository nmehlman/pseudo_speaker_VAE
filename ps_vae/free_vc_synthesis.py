import torch
import torchaudio
import os
import sys

sys.path.append('/home1/nmehlman/arts/pseudo_speakers/TTS')

from TTS.vc.models.freevc import FreeVC
from TTS.config import load_config

FREE_VC_INPUT_SR = 16000
FREE_VC_OUTPUT_SR = 24000

def run_embedding_conditioned_vc(
        speaker_embedding: torch.Tensor, 
        source_audio_path: str,
        free_vc_model: FreeVC,
    ):
    
    # Load audio
    source_audio, fs = torchaudio.load(source_audio_path)
    if fs != FREE_VC_INPUT_SR:
        source_audio = torchaudio.transforms.Resample(fs, FREE_VC_INPUT_SR)(source_audio)
    
    # Run conversion
    converted_audio = free_vc_model.voice_conversion_embed(source_audio.squeeze(), speaker_embedding/speaker_embedding.norm()) # DEBUG
    
    return converted_audio.squeeze(0)

if __name__ == "__main__":
    
    import tqdm
    
    source_audio_path = "/home1/nmehlman/nick_codebase/misc/test_audio.wav"
    embedding_dir = "/home1/nmehlman/arts/pseudo_speakers/generated_embeddings"
    save_dir = "/home1/nmehlman/arts/pseudo_speakers/audio_samples"
    vc_ckpt_dir: str = "/home1/nmehlman/.local/share/tts/voice_conversion_models--multilingual--vctk--freevc24/"

    # Load the FreeVC model
    config = load_config(os.path.join(vc_ckpt_dir, "config.json"))
    model = FreeVC(config)
    model.load_checkpoint(config, os.path.join(vc_ckpt_dir, "model.pth"))

    for embed_file in tqdm.tqdm(os.listdir(embedding_dir), desc="Converting audio"):
        
        id = embed_file.split("_")[-1].replace(".pt", "")
        
        emedding = torch.load(os.path.join(embedding_dir, embed_file)).reshape(1,-1,1)
        converted_audio = run_embedding_conditioned_vc(emedding, source_audio_path, free_vc_model=model)
        torchaudio.save(os.path.join(save_dir, f"sample_embed_{id}.wav"), converted_audio, FREE_VC_OUTPUT_SR)
        
