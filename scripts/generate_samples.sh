#!/bin/bash

# Define the base sample directory
SAMPLE_DIR="/home1/nmehlman/arts/pseudo_speakers/samples/synthetic_embeddings/cv_train/psg-19_cond_male"
VAE_CHECKPOINT_PATH="/project/shrikann_35/nmehlman/logs/ps_vae/cv_freevc_gender_classifier/version_2/checkpoints/epoch=199-step=59400.ckpt"
SOURCE_AUDIO_PATH="/home1/nmehlman/nick_codebase/misc/test_audio.wav"
SYNTHESIS_TYPE="conditional"
CLASSIFIER_TARGET=0
NUM_STEPS=5000
STEP_SIZE=0.02
NOISE_WEIGHT=1.0

# Save the configuration information to a text file in the sample directory
mkdir -p "$SAMPLE_DIR"
CONFIG_FILE="$SAMPLE_DIR/config.txt"
cat <<EOL > "$CONFIG_FILE"
SAMPLE_DIR=$SAMPLE_DIR
VAE_CHECKPOINT_PATH=$VAE_CHECKPOINT_PATH
SOURCE_AUDIO_PATH=$SOURCE_AUDIO_PATH
SYNTHESIS_TYPE=$SYNTHESIS_TYPE
CLASSIFIER_TARGET=$CLASSIFIER_TARGET
NUM_STEPS=$NUM_STEPS
STEP_SIZE=$STEP_SIZE
NOISE_WEIGHT=$NOISE_WEIGHT
EOL

# Define subdirectories for embeddings and audio
EMBEDDING_DIR="$SAMPLE_DIR/embeds"
AUDIO_DIR="$SAMPLE_DIR/audio"

# Define the full paths to the scripts
INFERENCE_SCRIPT="../ps_vae/inference.py" 
ANALYSIS_SCRIPT="../analysis/embedding_similarity_plot.py" 
SPEECH_SYNTHESIS_SCRIPT="../freevc/speech_synthesis.py"

# Create the necessary subdirectories
mkdir -p "$EMBEDDING_DIR"
mkdir -p "$AUDIO_DIR"

# Copy the source audio file to the audio directory for reference
cp -p "$SOURCE_AUDIO_PATH" "$AUDIO_DIR" 

# Activate the psg Conda environment
eval "$(conda shell.bash hook)"
conda activate psg

# Run inference.py to generate embeddings
echo "Running inference.py..."
python "$INFERENCE_SCRIPT" --save_dir "$EMBEDDING_DIR" --vae_ckpt_path "$VAE_CHECKPOINT_PATH" --synthesis_type "$SYNTHESIS_TYPE" --classifier_target "$CLASSIFIER_TARGET" --num_steps "$NUM_STEPS" --step_size "$STEP_SIZE" --noise_weight "$NOISE_WEIGHT"

# Make similarity plot
python "$ANALYSIS_SCRIPT" "$EMBEDDING_DIR"

# Deactivate the psg Conda environment
conda deactivate

# Activate the TTS Conda environment
conda activate TTS
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')" # Set SSL_CERT_FILE for TTS

# Run speech_synthesis.py to generate audio samples
echo "Running speech_synthesis.py..."
python "$SPEECH_SYNTHESIS_SCRIPT" --embedding_dir "$EMBEDDING_DIR"  --save_dir "$AUDIO_DIR" --source_audio_path "$SOURCE_AUDIO_PATH"

echo "Samples saved to $SAMPLE_DIR"