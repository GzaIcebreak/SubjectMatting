#!/usr/bin/env bash
# Downloads the 10 benchmark images into test_images/ (they are not committed: COCO images carry mixed
# Flickr licences). 7 from COCO val2017, 3 sample images from the EfficientSAM3 repository.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p test_images
coco() { curl -fsSL -o "test_images/$1.jpg" "http://images.cocodataset.org/val2017/$2.jpg"; echo "test_images/$1.jpg"; }
es3() { curl -fsSL -o "test_images/$1" "https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/sam3/assets/$2"; echo "test_images/$1"; }
coco bear 000000000285
coco stop_sign 000000000724
coco teddy_bears 000000000776
coco cat_whiskers 000000004795
coco two_cats 000000039769
coco man_bw 000000003156
coco baseball 000000006471
es3 dog_person.jpg dog_person.jpeg
es3 truck.jpg images/truck.jpg
es3 groceries.jpg images/groceries.jpg
