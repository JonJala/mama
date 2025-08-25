#!/usr/bin/env bash

echo "Make sure you initialize your mama virtualenv!!"
#source /path/to/mama_env/activate
sleep 2

python3 ../mama_ldscores.py --gendata ./LDSC_input/chr22_mind02_geno02_maf01_EAS,EAS \
                                      ./LDSC_input/chr22_mind02_geno02_maf01_EUR,EUR \
                            --window-cm 1 \
                            --out ./chr22_mind02_geno02_maf01_EAS_EUR


