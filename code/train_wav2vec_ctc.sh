#!/bin/bash
#SBATCH -A ent_aiapps_asr
#SBATCH -p batch                # batch / batch_short / backfill
#SBATCH -N 2                    # number of nodes
#SBATCH -t 8:00:00              # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH -J "citrinet_spe_1024_vox_mcv_mls_fr"     # job name (<< CHANGE ! >>)
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --gpus-per-node=16      # n gpus per machine <required>
#SBATCH --ntasks-per-node=16    # n tasks per machine (one task per gpu) <required>
#SBATCH --overcommit            # Needed for pytorch
#SBATCH --nv-meta=ml-model.citrinetasr            # Needed for pytorch
#SBATCH --gres=gpfs:circe       # Needed for Circe-Draco <required>

set -x
SLURM_ACCOUNT_DIR='ent_aiapps'  # <Make sure you dont override SLURM_ACCOUNT!>
USERID='tbartley'

# << CHANGE THIS >>
CONTAINER="gitlab-master.nvidia.com/yangzhang/nemo_containers:1.2.0" #"gitlab-master.nvidia.com/smajumdar/nemo_containers:1.3.0"

# Directories for manifests, data, etc.
# << CHANGE THIS >>
TOKENIZERS="../../asr_data/model_prep/tokenizer"
CODE_DIR="."
RESULTS_DIR="../results"

# << CHANGE THIS >>
FILTERS="1024"
LR="5e-3"
EPOCH="10"
EXP_NAME="wav2vec_${FILTERS}_${LR}_${EPOCH}"

# Config file
CONFIG_PATH='configs'
CONFIG_NAME="wav2vecCTC.yaml"

# Necessary Exports
export HYDRA_FULL_ERROR=1

# Make results dir
mkdir -p ${RESULTS_DIR}

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------------" \
&& wandb login ${WANDB} \
&& echo "Starting training" \
&& CUDA_VISIBLE_DEVICES=1 python /code/speech_to_text_bpe_finetune.py \
        --config-path=${CONFIG_PATH} \
        --config-name=${CONFIG_NAME} \
        model.tokenizer.dir="${TOKENIZERS}/tokenizer_spe_unigram_v1024" \
        model.tokenizer.type="bpe" \
        model.train_ds.manifest_filepath=../train.json \
        model.train_ds.is_tarred=false \
        model.train_ds.batch_size=16 \
        +model.train_ds.num_workers=16 \
        +model.train_ds.pin_memory=true \
        model.validation_ds.manifest_filepath=eval.json \
        model.validation_ds.batch_size=8 \
        +model.validation_ds.num_workers=8 \
        +model.validation_ds.pin_memory=true \
        model.optim.lr=${LR} \
        model.optim.name='novograd' \
        model.optim.betas=[0.8,0.25] \
        model.optim.weight_decay=0.001 \
        model.optim.sched.warmup_steps=10000 \
        model.optim.sched.min_lr=0.00001 \
        trainer.gpus=-1 \
        trainer.num_nodes=$SLURM_JOB_NUM_NODES  \
        trainer.max_epochs=${EPOCH} \
        trainer.log_every_n_steps=100 \
        +trainer.progress_bar_refresh_rate=1000 \
        trainer.check_val_every_n_epoch=1 \
        trainer.val_check_interval=1.0 \
        trainer.precision=16 \
        trainer.sync_batchnorm=false \
        trainer.benchmark=false \
        trainer.accumulate_grad_batches=2 \
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
