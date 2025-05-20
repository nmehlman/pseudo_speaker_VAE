import json
import torch
import tqdm
from utils import load_freevc_model


if __name__ == "__main__":
    
    manifest_path = "manifests/cv_manifest.json"
    vc_checkpoint_dir = "/home1/nmehlman/.local/share/tts/voice_conversion_models--multilingual--vctk--freevc24/"

    model = load_freevc_model(vc_checkpoint_dir)

    with open(manifest_path, "r") as f: # Maifest is a list of dictionaries with keys 'audio_filepath' and 'save_filepath'
        manifest = json.load(f)

    for sample in tqdm.tqdm(manifest, desc="Exporting embeddings"):
        
        audio = sample['audio_filepath']
        save = sample['save_filepath']
        
        embed = model.get_spk_embedding(audio).squeeze().detach().cpu()
        
        torch.save(embed, save)
