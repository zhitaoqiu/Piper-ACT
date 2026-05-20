# ACT Pilot 10-Demo Training

## Dataset

```
data/lerobot_dataset_piper_bottle_pilot_10demo/
```

| Property | Value |
|----------|-------|
| Episodes | 10 (6 center, 2 left, 2 right) |
| FPS | 30 |
| Cameras | wrist (RealSense) + global (USB SN0002) |
| State dim | 7 [J1..J6, gripper] |
| Action dim | 7 [J1..J6, gripper] |
| Task | Full bottle grasp pilot (approach -> close on bottle -> lift) |

## Model

| Parameter | Value |
|-----------|-------|
| Type | ACT (Action Chunking Transformer) |
| chunk_size | 10 (~1 second at 10 FPS) |
| n_action_steps | 10 |
| dim_model | 128 |
| dim_feedforward | 512 |
| n_heads | 4 |
| n_encoder_layers | 2 |
| n_decoder_layers | 2 |
| dropout | 0.0 |
| use_vae | false |
| Image transforms | enabled |
| Optimizer LR | 3e-4 |
| Backbone LR | 1e-4 |
| Batch size | 8 |
| Steps | 5000 |
| Save/eval freq | 1000 |

## Training command

Training is allowed only after the sanity checker reports at least 8 passing
episodes:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate piper_act
python3 scripts/check_pilot_dataset.py \
    --dataset data/lerobot_dataset_piper_bottle_pilot_10demo/
```

The training script runs the same check and exits before training if the
dataset fails. It also refuses to overwrite an existing
`outputs/train/act_pilot_10demo/checkpoints/` directory unless
`ALLOW_EXISTING_OUTPUT=1` is set intentionally.

```bash
bash scripts/train_act_pilot_10demo.sh
```

The script defaults to `DEVICE=cuda` and refuses silent CPU fallback. If the
GPU is busy, wait rather than training this pilot on CPU.

Or manually:

```bash
conda activate piper_act
PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=piper/pilot_10demo \
  --dataset.root=data/lerobot_dataset_piper_bottle_pilot_10demo/ \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.dim_model=128 \
  --policy.dim_feedforward=512 \
  --policy.n_heads=4 \
  --policy.n_encoder_layers=2 \
  --policy.n_decoder_layers=2 \
  --policy.dropout=0.0 \
  --policy.use_vae=false \
  --policy.kl_weight=1.0 \
  --policy.optimizer_lr=3e-4 \
  --policy.optimizer_lr_backbone=1e-4 \
  --policy.device=cuda \
  --policy.repo_id=piper/pilot_10demo \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --num_workers=0 \
  --persistent_workers=false \
  --steps=5000 \
  --save_freq=1000 \
  --eval_freq=1000 \
  --output_dir=outputs/train/act_pilot_10demo/ \
  --job_name=act_pilot_10demo
```

## Output

```
outputs/train/act_pilot_10demo/
├── checkpoints/
│   ├── 001000/
│   ├── 002000/
│   ├── 003000/
│   ├── 004000/
│   └── 005000/
│       └── pretrained_model/
├── training_metadata_YYYYMMDD_HHMMSS.json
└── logs/
```

Expected checkpoint paths:

```text
outputs/train/act_pilot_10demo/checkpoints/005000/pretrained_model/
outputs/train/act_pilot_10demo/checkpoints/last/pretrained_model/
```

## Next steps after training

1. Run offline diagnostic: `--debug-offline-policy-rollout-from-recorded-start`
2. Run staged evaluation following `docs/ACT_PILOT_10DEMO_EVAL_PROTOCOL.md`
3. Do NOT jump directly to full e2e
