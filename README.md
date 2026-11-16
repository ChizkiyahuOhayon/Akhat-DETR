# Akhat-DETR

Code for the paper "Akhat-DETR: End-to-End Object Detection Model on Hazy Scenarios in Autonomous
Driving" (Zhao Liu, Zhiwei Liu). Akhat-DETR is DDQ-DETR with two additions: a lightweight
AOD-Net-style dehazing module in front of the backbone, and the FFTA neck. It is implemented as
custom modules for MMDetection 3.3.0.

The trained weights and the complete training logs of every run in the paper are archived on
Zenodo: https://doi.org/10.5281/zenodo.24282347

## Contents

| Path | What it is |
| --- | --- |
| `akhat/` | The three custom modules, registered in MMDetection: `Dehazing` (AOD-Net-style module), `FFTA` (neck) and `AkhatDETR` (DDQ-DETR with the dehazing module switched by one flag). |
| `configs/_base_rtts.py` | Shared settings: DDQ-DETR, 800 queries, 3 feature levels, 5 classes, 30 epochs, effective batch 16. Set `data_root` to your copy of the data. |
| `configs/c2_swinl_ffta_v2.py`, `c3_swinl_fpn_dehaze_v2.py`, `c4_swinl_ffta_dehaze_v2.py` | Swin-L rows a to c of Table 1: no dehazing module; dehazing module and standard FPN; full model. |
| `configs/r50_a_v2.py`, `r50_b_v2.py`, `r50_c_v2.py` | Rows d to f: the same three models with a ResNet-50 backbone. |
| `configs/s0_nodh_ffta.py` to `s4_nodh_fftaattn.py` | Design screening on a validation split (Swin-L, 12 epochs): no dehazing module with FFTA; pretrained dehazing module, fine-tuned; pretrained dehazing module, fixed; no dehazing module with FPN; FFTA with the attention block switched on. |
| `weights/dehaze_ots_pretrained.pth` | The dehazing module pretrained on RESIDE OTS (11 KB), used by screening runs s1 and s2. |
| `splits/` | File names of the 3,889 training and 433 test images of our RTTS split, a per-image manifest, md5 sums of the annotation files we used, and the 3,500/389 division of the training images used for screening. |
| `tools/rtts_voc_to_coco.py` | Rebuilds our COCO-format annotation files from the RTTS download. |
| `tools/verify_rtts_rebuild.py` | Compares two rebuilt copies image by image and box by box. |
| `tools/make_val_split.py` | Writes the screening split from the training annotations. |
| `tools/pretrain_dehaze.py` | Pretrains the dehazing module on paired synthetic images, as AOD-Net is trained. |
| `tools/collect_results.py` | Reads the training logs and prints the rows of Table 1 and the screening runs. |
| `tools/make_figs.py`, `tools/compose_figs.py` | Produce Figs. 1 and 5 from the trained weights. |
| `tools/smoke_model.py`, `tools/smoke_v2.py` | Build each model and run one step on fake data. |
| `tools/check_dead_init.py` | Measures how often the default initialisation of the dehazing module outputs a blank image. |
| `scripts/` | The Slurm scripts we used on the UCD Sonic cluster. |
| `original_release/` | `dehazing.py` and `ffta.py` exactly as first published in this repository. |
| `inference/` | Example detections from the first release. |

## Notes on the code

- `FFTA.forward` runs a 3×3 projection per level and then the cross-scale fusion module (CCFM) of
  RT-DETR. The source also defines a Dynamic-Head-style attention block, but the forward pass
  calls it only with `use_attention=True`; all reported runs use `use_attention=False`, as the
  first release did. Only screening run s4 switches it on (six blocks after CCFM).
- The code differs from the first release in two places, both switched by arguments whose
  defaults keep the old behaviour, and both turned on in the configs above:
  `dehazing_init='identity'` sets the last convolution of the dehazing module to zero weights and
  bias one, so that it starts as the identity (J = I); `ccfm_shared=False` gives each of the four
  fusion steps of CCFM its own weights (the first release shared one fusion block, one 1×1
  convolution and one down-sampling convolution across steps).
- `dehazing_init='pretrained'` with `dehazing_checkpoint` loads the pretrained module, and
  `dehazing_frozen=True` keeps it fixed during detector training.
- The dehazing module works on images in [0, 1]. `AkhatDETR.extract_feat` undoes the ImageNet
  normalisation, applies the module and normalises again.
- `pad_size_divisor=32` is needed because CCFM upsamples by exactly 2.

## Environment

Python 3.10, PyTorch 2.1.0 (CUDA 12.1), mmengine 0.10.4, mmcv 2.1.0 (prebuilt wheel),
MMDetection 3.3.0, numpy < 2, opencv-python 4.10.0.84, yapf 0.40.1.

```bash
conda create -n akhat python=3.10 -y && conda activate akhat
pip install torch==2.1.0 torchvision==0.16.0
pip install "numpy<2" "opencv-python==4.10.0.84"
pip install mmengine==0.10.4 yapf==0.40.1
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
git clone -b v3.3.0 https://github.com/open-mmlab/mmdetection.git
pip install -e mmdetection --no-build-isolation
pip install "numpy<2" "opencv-python==4.10.0.84"
```

The prebuilt mmcv 2.1.0 wheel has no kernels for H100 (sm_90); on that GPU training stops at the
first deformable-attention call. We trained on A100 and L40S GPUs.

## Data

RTTS is part of RESIDE and is distributed by its authors
(https://sites.google.com/view/reside-dehaze-datasets). We do not redistribute it. After
downloading RTTS:

```bash
python tools/rtts_voc_to_coco.py /path/to/RTTS splits/ /path/to/rtts_coco
```

This writes `annotations/instances_{train,val}2017.json` and links the images into
`images/{train2017,val2017}/`. The test split is stored under the name `val2017` because that is
what the configs read. Compare the result with `splits/rtts_split_manifest.csv`. Four RTTS
annotation files give a wrong image size; the script reads the size from the image itself.

For the screening runs, `python tools/make_val_split.py /path/to/rtts_coco` holds out 389
training images (seed 0) and writes `annotations_valsplit/`; compare with `splits/rtts_valsub.txt`.
The pretraining of the dehazing module uses `clear.zip` and `part2.zip` of the RESIDE OTS set:

```bash
python tools/pretrain_dehaze.py /path/to/OTS/clear /path/to/OTS/part2 weights/dehaze_ots_pretrained.pth --iters 20000
```

## Training and evaluation

Every run starts from the COCO-pretrained DDQ-DETR weights released with MMDetection
([Swin-L](https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth),
[ResNet-50](https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq-detr-4scale_r50_8xb2-12e_coco/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth))
with seed 0, on one GPU.

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python mmdetection/tools/train.py configs/c4_swinl_ffta_dehaze_v2.py --work-dir work_dirs/c4_swinl_ffta_dehaze_v2_seed0 \
    --cfg-options randomness.seed=0 load_from=/path/to/ddq_detr_swinl_30e.pth
python tools/collect_results.py work_dirs --epoch 30
```

The reported number for each run of Table 1 is the evaluation after epoch 30, fixed before
training. Screening runs are read after epoch 12 with `--epoch 12`.

## Results

COCO metrics on the 433 test images after epoch 30:

| Row | Backbone | Model | AP | AP50 | AP75 | APs | APm | APl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | Swin-L | no dehazing module | 0.642 | 0.875 | 0.738 | 0.399 | 0.690 | 0.820 |
| b | Swin-L | dehazing module + standard FPN | 0.636 | 0.877 | 0.724 | 0.403 | 0.681 | 0.812 |
| c | Swin-L | full model (dehazing module + FFTA) | 0.639 | 0.870 | 0.733 | 0.395 | 0.688 | 0.811 |
| d | ResNet-50 | no dehazing module | 0.595 | 0.836 | 0.673 | 0.388 | 0.642 | 0.742 |
| e | ResNet-50 | dehazing module + standard FPN | 0.598 | 0.835 | 0.680 | 0.393 | 0.637 | 0.759 |
| f | ResNet-50 | full model (dehazing module + FFTA) | 0.600 | 0.837 | 0.686 | 0.398 | 0.646 | 0.750 |

Design screening, COCO metrics on the 389 validation images after epoch 12 (Swin-L; the test
images were not used):

| Run | Dehazing module | Neck | AP | AP50 | AP75 | APs | APm | APl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0 | none | FFTA | 0.653 | 0.886 | 0.740 | 0.434 | 0.673 | 0.799 |
| s1 | pretrained, fine-tuned | FFTA | 0.655 | 0.887 | 0.736 | 0.433 | 0.678 | 0.801 |
| s2 | pretrained, fixed | FFTA | 0.642 | 0.875 | 0.723 | 0.404 | 0.660 | 0.795 |
| s3 | none | FPN | 0.657 | 0.891 | 0.741 | 0.433 | 0.679 | 0.801 |
| s4 | none | FFTA with attention | 0.604 | 0.847 | 0.693 | 0.370 | 0.632 | 0.768 |

The pretrained dehazing module raised PSNR on 910 held-out OTS pairs from 16.54 dB (hazy input)
to 23.30 dB.

## Licence

Apache License 2.0 (see `LICENSE`). `akhat/ffta.py` adapts MMDetection code and the CCFM design of
RT-DETR, both released under Apache 2.0.
