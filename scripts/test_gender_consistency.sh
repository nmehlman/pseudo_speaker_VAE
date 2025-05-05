VAE_CHECKPOINT_PATH="/project/shrikann_35/nmehlman/logs/ps_vae/cv_freevc_gender_classifier/version_6/checkpoints/epoch=199-step=59400.ckpt"
DATA_ROOT="/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
SAVE_DIR="/home1/nmehlman/arts/pseudo_speakers/samples/synthetic_embed_gender_consistency/test"
NUM_STEPS=500
STEP_SIZE=0.01
NOISE_WEIGHT=1.0

# Save the configuration information to a text file in the sample directory
mkdir -p "$SAVE_DIR"
CONFIG_FILE="$SAVE_DIR/config.txt"
cat <<EOL > "$CONFIG_FILE"
SAVE_DIR=$SAVE_DIR
VAE_CHECKPOINT_PATH=$VAE_CHECKPOINT_PATH
NUM_STEPS=$NUM_STEPS
STEP_SIZE=$STEP_SIZE
NOISE_WEIGHT=$NOISE_WEIGHT
EOL

eval "$(conda shell.bash hook)"
conda activate psg

python /home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/analysis/synthetic_embed_consistency.py \
    --demo "gender" \
    --demos_to_test "male" "female" \
    --n_samples_per_demo 5 \
    --vae_ckpt_path "$VAE_CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --save_dir "$SAVE_DIR" \
    --step_size "$STEP_SIZE" \
    --num_steps "$NUM_STEPS" \
    --noise_weight "$NOISE_WEIGHT" \

# Deactivate the psg Conda environment
conda deactivate

# Activate the TTS Conda environment
conda activate TTS
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')" # Set SSL_CERT_FILE for TTS

# List all demos in the SAVE_DIR/audio subdirectory
for demo in "$SAVE_DIR/speech"/*; do

    source_demo_name=$(basename "$demo")
    
    # For each file in SAVE_DIR/audio/<source-demo>, call the speech_synthesis script once per embedding directory
    for audio_file in "$SAVE_DIR/audio/$source_demo_name"/*; do
        for target_demo in "$SAVE_DIR/embeds"/*; do
            target_demo_name=$(basename "$target_demo")
            
            # Create the directory structure <source-demo>--><target-demo>
            output_dir="$SAVE_DIR/converted/${source_demo_name}-->${target_demo_name}"
            mkdir -p "$output_dir"
            
            python /home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/freevc/speech_synthesis.py \
                --source_audio_path "$audio_file" \
                --embedding_dir "$target_demo" \
                --save_dir "$output_dir" \
                --vc_ckpt_dir "/home1/nmehlman/.local/share/tts/voice_conversion_models--multilingual--vctk--freevc24/"
        done
    done
done