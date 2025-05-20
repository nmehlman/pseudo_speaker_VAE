import torch
import torchaudio
import os
import certifi
from utils import load_freevc_model

os.environ["SSL_CERT_FILE"] = certifi.where()  # Set SSL_CERT_FILE for TTS

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
    converted_audio = free_vc_model.voice_conversion_embed(source_audio.squeeze(), speaker_embedding.reshape(1,-1,1)) # DEBUG
    
    return converted_audio.squeeze(0)

if __name__ == "__main__":
    
    import tqdm
    import os
    import argparse

    default_vc_ckpt_dir = "/home1/nmehlman/.local/share/tts/voice_conversion_models--multilingual--vctk--freevc24/" # Default
    
    parser = argparse.ArgumentParser(description="Run voice conversion using FreeVC.")
    parser.add_argument("--source_audio_path", type=str, required=True, help="Path to the source audio file.")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Directory containing speaker embeddings.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the converted audio files.")
    parser.add_argument("--vc_ckpt_dir", type=str, default=default_vc_ckpt_dir, help="Path to the FreeVC checkpoint directory.")
    
    args = parser.parse_args()
    
    source_audio_path = args.source_audio_path
    embedding_dir = args.embedding_dir
    save_dir = args.save_dir
    vc_ckpt_dir = args.vc_ckpt_dir
    
    os.makedirs(save_dir, exist_ok=True)

    # Load the FreeVC model
    model = load_freevc_model(vc_ckpt_dir)

    for embed_file in tqdm.tqdm(os.listdir(embedding_dir), desc="Converting audio"):
        
        if not embed_file.endswith(".pt"): 
            continue
        
        id = embed_file.replace(".pt", "") 
        
        emedding = torch.load(os.path.join(embedding_dir, embed_file))
        converted_audio = run_embedding_conditioned_vc(emedding, source_audio_path, free_vc_model=model)
        torchaudio.save(os.path.join(save_dir, f"audio_{source_audio_path.split('/')[-1].split('.')[0]}_embed_{id}.wav"), converted_audio, FREE_VC_OUTPUT_SR)
        
