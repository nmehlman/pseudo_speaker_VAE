import os
import sys
sys.path.append('/home1/nmehlman/arts/pseudo_speakers/TTS')

from TTS.vc.models.freevc import FreeVC
from TTS.config import load_config

def load_freevc_model(vc_ckpt_dir: str, device: str = 'cpu') -> FreeVC:
    config = load_config(os.path.join(vc_ckpt_dir, "config.json"))
    model = FreeVC(config)
    model.load_checkpoint(config, os.path.join(vc_ckpt_dir, "model.pth"), device=device)
    return model