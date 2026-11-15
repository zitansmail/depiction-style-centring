"""Search your own image folder with a frozen model, with and without DSC.

    python examples/search_folder.py --images my_photos_and_drawings/ --query my_photos_and_drawings/sketch.png
Prints the top results of the query with the raw descriptor and with DSC. Nothing is trained.
"""
import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsc import DSC, load_siglip, word_embeddings                      # noqa: E402
from dsc.data import PathSource                                         # noqa: E402
from dsc.features import DEVICE, siglip_embeddings, vit_descriptor      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="folder of images (the collection)")
ap.add_argument("--query", required=True, help="path of one image of the collection")
ap.add_argument("--descriptor", default="dinov2", choices=["dinov2", "clip", "siglip2"])
ap.add_argument("-k", type=int, default=10)
a = ap.parse_args()

paths = sorted(p for e in ("jpg", "jpeg", "png", "bmp", "webp") for p in glob.glob(os.path.join(a.images, f"**/*.{e}"),
                                                                                    recursive=True))
src = PathSource(paths)
sig = siglip_embeddings(src)                                            # names the depiction of each image
x = sig if a.descriptor == "siglip2" else vit_descriptor(src, a.descriptor)
d = DSC().fit(x, sig, word_embeddings(*load_siglip(device=DEVICE), device=DEVICE))
print(f"{len(paths)} images; DSC {'applied' if d.active else 'not applied (one depiction dominates)'}")
q = paths.index(os.path.abspath(a.query)) if os.path.abspath(a.query) in paths else paths.index(a.query)
for tag, z in (("raw", F.normalize(x, dim=1)), ("DSC", d.transform(x))):
    sim = z @ z[q]; sim[q] = float("-inf")
    print(f"\n{tag}:"); [print("  ", paths[i]) for i in sim.topk(a.k).indices.tolist()]
