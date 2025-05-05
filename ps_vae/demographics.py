import torch
import sys, os
import json
import torch.nn.functional as F
import torchaudio
import numpy as np

sys.path.append("/home1/nmehlman/arts/pseudo_speakers/vox-profile-example")
from src.model.age_sex.wavlm_demographics import WavLMWrapper
from src.model.age_sex.whisper_demographics import WhisperWrapper
from tqdm import tqdm
import argparse

age_unique_labels = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies"]
sex_unique_labels = ["female", "male"]

WAVLM_MODEL_PATH = "/project/shrikann_35/nmehlman/models/arts-opensource-weights/age_sex/wavlm_large/"
WHISPER_MODEL_PATH = "/project/shrikann_35/nmehlman/models/arts-opensource-weights/age_sex/whisper_large/"
AGE_SEX_MODEL_SR = 16000

def load_models(wavlm_model_path: str = WAVLM_MODEL_PATH, whisper_model_path: str = WHISPER_MODEL_PATH, device: str = 'cpu'):
    """
    Load WavLM and Whisper models with their respective weights.

    Args:
        wavlm_model_path (str): Path to the WavLM model weights.
        whisper_model_path (str): Path to the Whisper model weights.
        device (torch.device): Device to load the models onto.

    Returns:
        tuple: Loaded WavLM and Whisper models.
    """
    # Load WavLM model
    wavlm_model = WavLMWrapper(
        pretrain_model="wavlm_large", 
        finetune_method="lora",
        lora_rank=16, 
        output_class_num=2,
        freeze_params=True, 
        use_conv_output=True,
        apply_gradient_reversal=False, 
        num_dataset=3
    ).to(device)
    
    # Load Whisper model
    whisper_model = WhisperWrapper(
        pretrain_model="whisper_large", 
        finetune_method="lora",
        lora_rank=16, 
        output_class_num=2,
        freeze_params=True, 
        use_conv_output=True,
        apply_gradient_reversal=False, 
        num_dataset=3
    ).to(device)
    
    # Load weights for WavLM model
    wavlm_model.load_state_dict(torch.load(os.path.join(wavlm_model_path, f"fold_1.pt"), weights_only=True, map_location=device), strict=False)
    wavlm_model.load_state_dict(torch.load(os.path.join(wavlm_model_path, f"fold_lora_1.pt"), map_location=device), strict=False)
    
    # Load weights for Whisper model
    whisper_model.load_state_dict(torch.load(os.path.join(whisper_model_path, f"fold_1.pt"), weights_only=True, map_location=device), strict=False)
    whisper_model.load_state_dict(torch.load(os.path.join(whisper_model_path, f"fold_lora_1.pt"), map_location=device), strict=False)
    
    return wavlm_model, whisper_model


def get_age_sex_predictions(wavlm_model: WavLMWrapper, whisper_model: WhisperWrapper, audio: torch.Tensor, device='cpu'):
    """
    Get age and sex predictions using ensemble of WavLM and Whisper models.

    Args:
        wavlm_model: Loaded WavLM model.
        whisper_model: Loaded Whisper model.
        audio (torch.Tensor): Input audio tensor (must be 16k Hz).
        device (str): Device to perform inference on.

    Returns:
        tuple: Ensemble probabilities for age and sex predictions.
    """
    audio = audio.to(device)
    wavlm_age_logits, wavlm_sex_logits, _ = wavlm_model(audio, return_feature=True)
    whisper_age_logits, whisper_sex_logits, _ = whisper_model(audio, return_feature=True)

    ensemble_age_logits = (whisper_age_logits + wavlm_age_logits) / 2
    ensemble_age_prob = F.softmax(ensemble_age_logits, dim=1)

    ensemble_sex_logits = (whisper_sex_logits + wavlm_sex_logits) / 2
    ensemble_sex_prob = F.softmax(ensemble_sex_logits, dim=1)

    return ensemble_age_prob, ensemble_sex_prob

def score_generated_samples(wavlm_model: WavLMWrapper, whisper_model: WhisperWrapper, dir_path: str, device='cpu'):
    """
    Score generated samples using the ensemble of WavLM and Whisper models.

    Args:
        wavlm_model: Loaded WavLM model.
        whisper_model: Loaded Whisper model.
        dir_path (str): Directory path containing audio files.
        device (str): Device to perform inference on.

    Returns:
        None
    """
    file_results = {}
    wav_files = [file for file in os.listdir(dir_path) if file.endswith(".wav")]
    
    for file in tqdm(wav_files, desc="Processing audio files"):
        audio_path = os.path.join(dir_path, file)
        
        audio, sr = torchaudio.load(audio_path)
        if sr != AGE_SEX_MODEL_SR:
            audio = torchaudio.functional.resample(audio, sr, AGE_SEX_MODEL_SR)
        audio = audio.to(device)

        with torch.no_grad():
            ensemble_age_prob, ensemble_sex_prob = get_age_sex_predictions(wavlm_model, whisper_model, audio, device)
        
        ensemble_age_prob = ensemble_age_prob.cpu().squeeze().numpy()
        ensemble_sex_prob = ensemble_sex_prob.cpu().squeeze().numpy()
        
        file_results[file] = {
            "age_probs": dict(zip(age_unique_labels, ensemble_age_prob.tolist())),
            "sex_probs": dict(zip(sex_unique_labels, ensemble_sex_prob.tolist())),
            "pred_age": age_unique_labels[np.argmax(ensemble_age_prob)],
            "pred_sex": sex_unique_labels[np.argmax(ensemble_sex_prob)]
        }
    
    # Get aggregated counts for each demographic category
    age_counts = {label: 0 for label in age_unique_labels}
    sex_counts = {label: 0 for label in sex_unique_labels}
    for file, results in file_results.items():
        age_counts[results["pred_age"]] += 1
        sex_counts[results["pred_sex"]] += 1
        
    results_summary = {
        "age_counts": age_counts,
        "sex_counts": sex_counts,
        "files": file_results
    }
    
    json_path = os.path.join(dir_path, "demographic_results.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"Demographic results saved to {json_path}")
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Score generated samples for demographics.")
    parser.add_argument("dir_path", type=str, help="Directory path containing audio files.")
    args = parser.parse_args()

    dir_path = args.dir_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wavlm_model, whisper_model = load_models(device=device)
    
    score_generated_samples(wavlm_model, whisper_model, dir_path, device)
    