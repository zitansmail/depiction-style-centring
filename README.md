# Depiction Style Centring (DSC)

Code of the paper *Depiction Style Centring: Training-Free Reduction of Depiction Bias in Frozen Vision Embeddings
for Image Retrieval* (S. Zitane, I. Zeroual, S. Agoujil).

Frozen foundation models (DINOv2, CLIP, SigLIP 2) retrieve images **of the same depiction** rather than of the same
category when a collection mixes photographs, sketches, clip art and paintings. DSC removes most of this bias
**without training and without labels**:

1. SigLIP 2 names the depiction of each image with a fixed list of 13 generic words (photo, sketch, clipart, ...);
2. each named group is centred on its own mean (groups under 20 images are merged);
3. a gate leaves the collection unchanged when one group holds 2/3 of it or more (single-depiction collections).

It works on the frozen embedding of **any** descriptor, costs one subtraction per image, and leaves search time
unchanged.

## Use it on your images

```python
from dsc import DSC, load_siglip, word_embeddings
model, proc = load_siglip()
d = DSC().fit(x_db, siglip_db, word_embeddings(model, proc))   # x_db: your descriptors, siglip_db: SigLIP 2 embeddings
z_db = d.transform(x_db)                                       # search with the cosine of z_db
z_q = d.transform_queries(x_q, siglip_q)                       # new queries
```

Whole folder: `python examples/search_folder.py --images my_folder/ --query my_folder/some_sketch.png`

## Install

```bash
pip install -r requirements.txt        # Python 3.10+
python -m pytest -q tests              # CPU tests: metrics against brute force, properties of DSC
```

Docker: `make build-cpu && make test-cpu` (any machine, including Apple Silicon) or `make build` (GPU image,
Linux x86-64 with an NVIDIA driver and nvidia-container-toolkit).

## Reproduce the paper

```bash
python scripts/reproduce.py all --cap 2    # quick check of the whole pipeline (numbers meaningless)
python scripts/reproduce.py all            # every table (GPU; downloads about 25 GB of public datasets)
# with Docker: make build && make reproduce
```

| Stage | Output (in `results/`) | Paper |
|---|---|---|
| `main` | `T1_flaw.csv`, `T2_main.csv`, `T3_no_cost.csv` | depiction bias, main results, single-depiction collections |
| `literature` | `T4_literature.csv`, `T4_per_task.csv` | comparison with published methods, their protocol |
| `ablations` | `T5_ablations.csv`, `T5_renditions.csv` | k-means groups, minimum group size, gate, ImageNet-R |
| `cost` | `T6_cost.csv` | timings |
| `typical` | `typical_queries.csv`, `typical_queries.png` | typical Office-Home queries |

Collections: PACS, Office-Home, DomainNet (test split, 6 images per class and depiction), ImageNet-R (mixed);
Corel-1K, Corel-10K, Caltech-101 (no background), Caltech-256 (no clutter) (single depiction). All are downloaded
automatically from public sources. Protocol: leave-one-out on the whole collection, cosine, per-class averaging.

## Results of the paper

The exact files used in the paper are in `results/paper/`. Extract (from `results/paper/T2_main.csv`):

| Collection | Descriptor | mAP raw → DSC | cross-depiction mAP raw → DSC |
|---|---|---|---|
| PACS | DINOv2 | 64.62 → 78.83 | 44.08 → 69.92 |
| PACS | SigLIP 2 | 59.88 → 88.75 | 33.65 → 84.18 |
| PACS | Fusion | 66.75 → 87.84 | 43.59 → 82.39 |
| Office-Home | DINOv2 | 54.05 → 58.13 | 44.93 → 50.57 |
| Office-Home | SigLIP 2 | 44.32 → 66.07 | 29.88 → 56.77 |
| Office-Home | Fusion | 54.28 → 64.57 | 42.61 → 56.22 |
| DomainNet | DINOv2 | 18.37 → 22.41 | 14.51 → 18.74 |
| DomainNet | SigLIP 2 | 15.16 → 27.43 | 10.33 → 23.03 |
| DomainNet | Fusion | 19.59 → 27.55 | 14.68 → 23.06 |
| ImageNet-R | DINOv2 | 31.30 → 35.77 | 26.21 → 30.95 |
| ImageNet-R | SigLIP 2 | 26.55 → 42.68 | 18.69 → 35.45 |
| ImageNet-R | Fusion | 31.78 → 41.12 | 24.82 → 34.99 |

Published methods under their own protocol (from `results/paper/T4_literature.csv`):

| Method | Office-Home P@1 | DomainNet P@50 |
|---|---|---|
| ProtoOT (trained, ResNet-50) | 43.85 | 64.37 |
| Caption-Matching (CLIP ViT-H + BLIP-2) | 80.30 | 80.70 |
| SigLIP 2, frozen | 78.25 | 74.75 |
| **SigLIP 2 + DSC** | 84.88 | 84.17 |

## Run on Colab

Open `notebooks/colab_reproduce.ipynb` in Colab (T4 GPU): tests, quick check, full reproduction, comparison with
`results/paper/`.

## Datasets

All collections are public and downloaded automatically (no account needed). This repository contains no image.

| Collection | Source used by the code | Original reference |
|---|---|---|
| PACS | Hugging Face `flwrlabs/pacs` | Li et al., ICCV 2017 |
| Office-Home | Hugging Face `flwrlabs/office-home` | Venkateswara et al., CVPR 2017 |
| DomainNet | Hugging Face `wltjr1007/DomainNet` (test split, 6 images per class and depiction) | Peng et al., ICCV 2019 |
| ImageNet-R | Hugging Face `koorye/ImageNet-Rendition` | Hendrycks et al., ICCV 2021 |
| Corel-1K | GitHub `zitansmail/CBIR-New-Search-Engine-Approach` | Wang et al., TPAMI 2001 |
| Corel-10K | Kaggle `michelwilson/corel10k` | Liu and Yang, Pattern Recognition 2013 |
| Caltech-101 | Hugging Face `flwrlabs/caltech101` (no background class) | Fei-Fei et al., CVPRW 2004 |
| Caltech-256 | Hugging Face `ilee0022/Caltech-256` (no clutter class) | Griffin et al., 2007 |

The images keep the licences and copyrights of their owners; several web-collected images (for example PACS
cartoons) carry stock-photo watermarks. Use them for research only and do not redistribute them.

## Citation

See `CITATION.cff`. Paper: S. Zitane, I. Zeroual, S. Agoujil, *Depiction Style Centring: Training-Free Reduction
of Depiction Bias in Frozen Vision Embeddings for Image Retrieval* (submitted).

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences.
