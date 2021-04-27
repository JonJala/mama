#!/usr/bin/env bash

echo "Make sure you initialize your mama virtualenv!!"
#source /path/to/mama_env/activate
sleep 5

python3 ../mama_ldscores.py --ances-path "./LDSC_input/iid_ances_file" \
                            --snp-ances "./LDSC_input/snp_ances_file" \
                            --ld-wind-cm 1 \
                            --stream-stdout \
                            --bfile-merged-path "./LDSC_input/chr22_mind02_geno02_maf01_EAS_EUR" \
                            --out "chr22_mind02_geno02_maf01_EAS_EUR"

# deactivate