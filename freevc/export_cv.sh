#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu
#SBATCH --time=14:00:00
#SBATCH --account=shrikann_35
#SBATCH --mem=24G
#SBATCH --output=/home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/freevc/%j_output.log    
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir /home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/freevc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmehlman@usc.edu

module purge
eval "$(conda shell.bash hook)"

conda activate TTS
python export_embeddings.py