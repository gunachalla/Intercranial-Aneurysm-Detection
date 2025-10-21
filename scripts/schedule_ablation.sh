#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

fold=0

python src/train.py experiment=251016-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e30-w01_005_1-dice data.fold=${fold} # w/ dice loss
python src/train.py experiment=251016-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e30-w01_005_1-heatmap data.fold=${fold} # w/ heatmap
python src/train.py experiment=251016-seg-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e30-w01_005_1 data.fold=${fold}    # w/o transformer
python src/train.py experiment=251016-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e30-w1_1_1 data.fold=${fold}    # loss weights w1_1_1
python src/train.py experiment=251016-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e30-w01_005_1-scratch data.fold=${fold} # scratch
python src/train.py experiment=251016-seg-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e30-w1_1_1 data.fold=${fold}    # loss weights w1_1_1 & w/o transformer
python src/train.py experiment=251016-seg_tf-v4-nnunet_truncate1_preV6_1-m32g64-e30-w01_005_1  data.fold=${fold}    # w/o extra mask branches
python src/train.py experiment=251016-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e30-w01_005_0 data.fold=${fold}    # w/o aux seg head