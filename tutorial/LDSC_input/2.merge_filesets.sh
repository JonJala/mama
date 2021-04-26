#!/usr/bin/env bash

# Now that we have constructed our definition files, we can merge the EAS and EUR filesets together.

plink1 --bfile "chr22_mind02_geno02_maf01_EAS" \
       --bmerge "chr22_mind02_geno02_maf01_EUR" \
       --make-bed \
       --out "chr22_mind02_geno02_maf01_EAS_EUR"