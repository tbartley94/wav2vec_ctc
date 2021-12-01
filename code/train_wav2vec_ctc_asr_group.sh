#!/bin/bash
#SBATCH -A swdl
#SBATCH -p luna                 # backfill / luna
#SBATCH -N 128                    # number of nodes
#SBATCH -t 4:00:00              # wall time  (8 for backfill, 4 for Luna)
#SBATCH -J "swdl-langspeech:Wav2Vec-CTC-ASR_layer"     # job name (<< CHANGE ! >>)
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node=8     # n tasks per machine (one task per gpu)
#SBATCH --overcommit            # Needed for pytorch
#SBATCH --signal=SIGUSR1@7200

set -x
SLURM_ACCOUNT='swdl/swdl-langspeech'
USERID='tbartley'

# << CHANGE THIS >>
CONTAINER="gitlab-master.nvidia.com/yangzhang/nemo_containers:1.2.0"

# Directories for manifests, data, etc.
# << CHANGE THIS >>
TOKENIZERS="/lustre/fsw/${SLURM_ACCOUNT}/${USERID}/tokenizers"
CODE_DIR="/lustre/fsw/${SLURM_ACCOUNT}/${USERID}/code"
RESULTS_DIR="/lustre/fsw/${SLURM_ACCOUNT}/${USERID}/results/wav2vec/ASR"

TRAIN_DATA="/lustre/fsw/swdl/swdl-langspeech/datasets/data/NeMo_ASR_SET/English/v2.0/train"
TRAIN_MANIFEST="tarred_audio_manifest.json"

# LS Evaluation
EVAL_MANIFEST="/lustre/fsw/${SLURM_ACCOUNT}/${USERID}/manifests/librispeech"
EVAL_DATA="/lustre/fsr/datasets/speech/jasper/LibriSpeech"

# << CHANGE THIS >>
FILTERS="1024"
NORM="group"
VOCAB="CHAR"
STEPS="600k"
EXP_NAME="Wav2VecCTC_${FILTERS}_${NORM}_ASR_${VOCAB}_s${STEPS}"

# WandB info
# << CHANGE THIS >>
WANDB='2640ba7ea01264a146c1d9f3f075ec53350dd2f1'
PROJECT="asr_fr"

# Config file
CONFIG_PATH='/code/configs/'
CONFIG_NAME="wav2vecCTC_${NORM}.yaml"

# Checkpoint file
CHECKPOINT_PATH="/code/checkpoints/"
CHECKPOINT_NAME="Wav2VecCTC_${FILTERS}_${NORM}_ASR_${VOCAB}.nemo"

# << CHANGE THIS >>

# Note: the following overwrites the original LibriSpeech dir in the nemo_asr directory (which only has manifests).
# $DATA_DIR:/data
MOUNTS="--container-mounts=$TRAIN_DATA:/data/train,$TOKENIZERS:/tokenizers,$CODE_DIR:/code,$RESULTS_DIR:/results,$EVAL_DATA:/datals/LibriSpeech,$EVAL_MANIFEST:/manifests/eval"

# Necessary Exports
export HYDRA_FULL_ERROR=1

# Make results dir
mkdir -p ${RESULTS_DIR}

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB} \
&& echo "Starting training" \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /code/speech_to_text_wav2vec.py \
	--config-path=${CONFIG_PATH} \
	--config-name=${CONFIG_NAME} \
	+nemo_checkpoint_path=${CHECKPOINT_PATH}/${CHECKPOINT_NAME} \
	model.tokenizer.dir="/tokenizers/tokenizer_spe_char_v50" \
	model.tokenizer.type="bpe" \
	model.train_ds.manifest_filepath=/data/train/${TRAIN_MANIFEST} \
	model.train_ds.is_tarred=true \
	model.train_ds.tarred_audio_filepaths="/data/train/audio__OP_0..4095_CL_.tar" \
	model.train_ds.batch_size=32 \
	+model.train_ds.num_workers=16 \
	+model.train_ds.pin_memory=true \
	model.validation_ds.manifest_filepath=[/manifests/eval/librivox-test-other.json,/manifests/eval/librivox-test-clean.json,/manifests/eval/librivox-dev-clean.json,/manifests/eval/librivox-dev-other.json] \
	model.validation_ds.batch_size=8 \
	+model.validation_ds.num_workers=8 \
	+model.validation_ds.pin_memory=true \
	model.final_dim=${FILTERS} \
	model.feature_grad_mult=.1 \
	model.encoder.cfg.encoder.embedding_dim=${FILTERS} \
	model.encoder.cfg.encoder.encoder_layers=24 \
	model.encoder.cfg.encoder.encoder_layerdrop=.2 \
	model.encoder.cfg.encoder.ffn_embedding_dim=4096 \
	model.encoder.cfg.encoder.num_attention_heads=16 \
	model.decoder.feat_in=${FILTERS} \
	model.optim.lr=0.0003 \
	model.optim.name='novograd' \
	model.optim.betas=[0.9,0.98] \
	model.optim.weight_decay=0.01 \
	model.optim.sched.warmup_ratio=.1 \
	model.optim.sched.min_lr=0.000001 \
	trainer.gpus=-1 \
	trainer.num_nodes=$SLURM_JOB_NUM_NODES  \
	trainer.max_steps=600000 \
	trainer.max_epochs=1000 \
	trainer.log_every_n_steps=100 \
	+trainer.progress_bar_refresh_rate=100 \
	trainer.check_val_every_n_epoch=1 \
	trainer.precision=32 \
	trainer.sync_batchnorm=false \
	trainer.benchmark=false \
	exp_manager.exp_dir=/results/checkpoints/${EXP_NAME} \
	exp_manager.create_wandb_logger=true \
	exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
	exp_manager.wandb_logger_kwargs.project=${PROJECT} \
	exp_manager.resume_if_exists=true \
	exp_manager.resume_ignore_no_checkpoint=true
EOF

OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
set +x
