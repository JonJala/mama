## Introduction 

This brief, self-contained tutorial will walk the user through an end-to-end run of `mama`. We encourage new users to consult this document prior to using the package on their own data. We provide individual-level data of 10 people from 1000 Genomes (5 from EUR sample, 5 from EAS sample) and ~800 randomly selected
SNPs on chromosome 22. We also give a sample EAS GWAS of BMI (CKB + BBJ) and EUR GWAS of BMI (UKB) restricted to chromosome 22.

## MAMA LDSC

Prior to running the meta-analysis, we must construct a within- and cross-ancestry LD score reference panel. This script works similarly to Bulik-Sullivan et al. (2015)  ([paper](https://www.nature.com/articles/ng.3211), [github](https://github.com/bulik/ldsc)). However, we require some additional preprocessing to calculate cross-ancestry LD, a notion developed in this paper. 

In particular, to run `mama_ldscores.py` you must first append all your genotype data together across ancestries (details below). However, two additional definition files must be passed to the script, as described below, so it is clear which people and which SNPs belong to which ancestry (or ancestries). These definition files should be constructed before you append your genotype data together.

A full walkthrough of this process is documented with data and scripts in `./LDSC_input/`. 

### `--ances-path` 

This file tells `mama_ldscores` which IID's in the appended genotype data correspond to which ancestries. This file consists of two headerless, tab-separated columns. The first column contains all the IID's in the merged `.fam` file, and the second column contains a string mapping each IID to an ancestry. For example, from the `iid_ances_file` we provide:
```
$ cat iid_ances_file
HG00096 EUR
HG00097 EUR
HG00099 EUR
HG00103 EUR
HG00101 EUR
HG00403 EAS
HG00404 EAS
HG00406 EAS
HG00407 EAS
```
This file can be constructed by simply appending all of your `.fam` files together across ancestries and keeping track of which ancestry each IID came from.

### `--snp-ances`

Similarly to the above `--ances-path` file, this `--snp-ances` file is used to map each SNP to an ancestry. Unlike the above, however, a SNP may belong to multiple ancestries (i.e., the given SNP appears in more than one `.bim` file across the ancestries.) This file will have P+1 tab-separated columns (with a header), where P is the number of populations/ancestries being considered. The first column, `SNP` will contain rsID's for every SNP in the union of the `.bim` files. The next P columns will correspond to each ancestry an will contain a 1 if that SNP was in that ancestry's original `.bim` file, 0 otherwise. **Note**: These column names **must** match the ancestry strings used in `--ances-path`. For example,
```
$ head -n 4 snp_ances_file
SNP     EAS     EUR
rs587616822     0.0     1.0
rs367963583     1.0     0.0
rs62224609      1.0     1.0
```
This file states that the first SNP, rs587616822, appeared only in the original EUR `.bim` file, not in the EAS `.bim` file. The second SNP appeared only in the EAS `.bim` file. The third SNP appeared in both `.bim` files.

This file can be constructed by taking the universe of rsID's across ancestries, then left-joining this master list to each ancestry's corresponding `.bim` file, marking which SNPs in the master list also appear in the `.bim` file.

### `--bfile-merged-path`

Once the above two files have been created, you can now merge your genotype `.bed/.bim/.fam` files together across ancestries. This will be necessary as `mama_ldscores` only accepts a single genotype fileset (note: if you're storing your data at the chromosome-level, you can simply loop through each chromosome and call the script 22 times.) One will need to merge multiple Plink filesets together with [`--[b]merge`](https://www.cog-genomics.org/plink/1.9/data#merge).

### Running `mama_ldscores`

The script `./run_mama_ldscores.sh` provides an example of how to call `mama_ldscores.py`. Note that there are flags this script does not invoke. To see all the options, type `python3 mama_ldscores.py -h`. 

## Meta-Analysis

Now that the LDSC reference panel has been constructed, we can run the meta analysis. The only required inputs to MAMA are one or more GWAS summary statistics and corresponding LD score reference panels, but many more options are provided. To see all flags and their descriptions, type `python3 mama.py -h`. We provide an example call to `mama.py` in `2.run_mama_meta.sh`. 


### GWAS Delimiter

Our package automatically infers the delimiter of the GWAS by using the [python sniffer tool](https://docs.python.org/3/library/csv.html). File IO can be very slow with the python engine, so we have provided an (optional) `--input-sep` flag to speed up the `read_csv` operation. Simply provide the delimiter with this flag --  a tab delimited file could be specified with `--input-sep \t` for example. A compressed (ie gzip) GWAS can also be supplied.

### GWAS Column Names 

`mama` requires the following columns to exist in your GWAS:

   * Base Pair
   * Chromosome  
   * rsID
   * A1 
   * A2 
   * Frequency 
   * Beta
   * P-value
   * Standard error
   
Some columns (INFO score, sample size) can be optionally specified. For more detail, see the **Summary Statistics Column Options** in the help menu generated by typing `$ python mama.py -h` at the command line. 
   
`mama` uses the python regex ([re](https://docs.python.org/3/library/re.html)) module to flexibly identify columns in the GWAS. For each column, the user can add to the default regex by invoking the `--add-[XXX]-col-match` flag, or replace by the default regex by invoking the `--replace-[XXX]-col-match` flag. This will be necessary if a required column cannot be found with the default regex (either add or replace), or more than one column matches the default regex (replace).

## MAMA Specification 

In the within-ancestry case, `mama` estimates the coefficients of the following equation:

<img src="https://latex.codecogs.com/svg.image?\hat{\beta}^2_{p,j}&space;=&space;\gamma_{\ell,&space;pp}\ell_{pp,j}&space;&plus;&space;\gamma_{f,pp}&space;&plus;&space;\gamma_{\xi,p}S_{p,j}" title="\hat{\beta}^2_{p,j} = \gamma_{\ell, pp}\ell_{pp,j} + \gamma_{f,pp} + \gamma_{\xi,p}S_{p,j}" />


In the cross-ancestry case, `mama` estimates the coefficients of the following equation:

<img src="https://latex.codecogs.com/svg.image?\hat{\beta}{p,j}\hat{\beta}{q,j}&space;=&space;\gamma_{\ell,pq}\ell_{pq,j}&space;&plus;&space;\gamma_{f,pq}" title="\hat{\beta}{p,j}\hat{\beta}{q,j} = \gamma_{\ell,pq}\ell_{pq,j} + \gamma_{f,pq}" />


See the Supplementary Note for more details. Some aspects of this specification can be modified from the command line. Commands to modify the coefficients on the LD scores are of the form `--reg-ld-XXX`. Commands to modify the coefficients on the standard errors are of the form `--reg-se2-XXX`. Commands to modify the coefficients on the intercept are of the form `--reg-int-XXX`. For example, in the paper, we set the genetic correlation between ancestries to 1 by calling `--reg-ld-set-corr 1`. Because the standard errors across SNPs will oftentimes be approximately collinear with the intercept, we recommend setting either the standard error coefficient or the intercept to 0. In the paper, we set the  intercept to zero with `--reg-int-zero`. Note that the regressions can be estimated in standardized genotype units with `--use-standardized-units` (input and output will always be in allele counts, but the meta-analysis itself can be done in standardized genotype units.)



