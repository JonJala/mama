#!/usr/bin/env bash

# Short example call to mama.py.

python3 ../mama.py --sumstats "./EAS_BMI.txt.gz,EAS,BMI" "./EUR_BMI.txt.gz,EUR,BMI" \
                   --ld-scores "./chr22_mind02_geno02_maf01_EAS_EUR.l2.ldscore.gz" \
                   --out "./BMI_MAMA" \
                   --add-a1-col-match "EA" \
                   --add-a2-col-match "OA"

#deactivate
