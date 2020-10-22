#!/usr/bin/env python

'''
(c) 2018 Hui Li and Raymond Walters

mama_ldscore is a command line tool for estimating 
cross-population LD scores.

'''

# import necessary packages
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object
import warnings
warnings.filterwarnings("ignore")

import os, sys, re
import logging, time, traceback
import argparse

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import t as tdist
from scipy.stats import chi2
#import scipy.optimize
#import scipy.linalg

import itertools as it
from functools import reduce
import subprocess
import copy
import argparse

# these imports don't work as written
# from legacy import mama_ldcalc as ld
# from legacy import mama_parse as ps
import mama_ldcalc as ld
import mama_parse as ps

__version__ = '1.0.0'

borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
short_border = "<><><<>><><><><><><><><><><><><><><><>"

header ="\n"
header += borderline +"\n"
header += "<>\n"
header += "<> MAMA: Multi-Ancestry Meta-Analysis \n"
header += "<> Version: {}\n".format(str(__version__))
header += "<> (C) 2018 Hui Li, Alicia Martin, Patrick Turley and Raymond Walters\n"
header += "<> Harvard University Department of Economics / Broad Institute of MIT and Harvard\n"
header += "<> GNU General Public License v3\n"
header += borderline + "\n"
header += "<> Software-related correspondence: hli@broadinstitute.org \n"
header += "<> All other correspondence: paturley@broadinstitute.org \n"
header += borderline +"\n"
header += "\n\n"

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 800)
pd.set_option('precision', 12)
pd.set_option('max_colwidth', 800)
pd.set_option('colheader_justify', 'left')

np.set_printoptions(linewidth=800)
np.set_printoptions(precision=3)

## Helper functions

class Logger_to_Logging(object):
    """
    Logger class that write uses logging module and is needed to use munge_sumstats or ldsc from the LD score package.
    """
    def __init__(self):
        logging.info('created Logger instance to pass through ldsc.')
        super(Logger_to_Logging, self).__init__()

    def log(self,x):
        logging.info(x)

class DisableLogger(object):
    '''
    For disabling the logging module when calling munge_sumstats
    '''
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)

def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f

def safely_create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError:
        if not os.path.isdir(folder_path):
            raise

def matrix_fillin(on_diag, off_diag, n, fill=None):
    x = np.full((n, n), fill, dtype=object)
    x[np.diag_indices(n)] = on_diag
    x[np.triu_indices(n,k=1)] = off_diag
    x[np.tril_indices(x.shape[0], k=-1)]=x[np.triu_indices(x.shape[0],k=1)]
    return x

## Major functions

def multi_ldScoreVarBlocks(args, ances_ind, ances_flag, ances_n, snp_index, ind_index, array_file, array_obj, array_snps):
    '''
    Calculate the paired-ancestry LD scores.

    array_obj: PlinkBEDFile class object defined in <mama_ldcalc.py>
    array_file: Path of the .bed file merged across all reference ancestries
    array_snps: mx1 array of SNP IDs read in from .bim file merged across all reference ancestries
    
    snp_index: dictionary of snp indices for each ancestry
    ind_index: dictionary of individual indices for each ancestry
    
    ances_ind: |T| list of distinct ancestry groups found in --snp-ances ('s header)
    ances_flag: nx1 array of integers indexing ancestry of each individual
    ances_n: Tx1 array of sample sizes for each ancestry

    '''

    # LD blocks change for each different LD estimation
    c = args.chunk_size
    n = ances_flag.shape[0]
    T = len(ances_ind)
    if args.bootstrap: # for embarrasingly parallelization
        bootstrap = 1
    else:
        bootstrap = None

    #bootstrap = int(args.bootstrap)

    # Build the cross-ancestry LD score matrix
    score_tags = [x for x in it.combinations(ances_ind,2)]
    col_list = ['_'.join(x) for x in score_tags] + ['{P}_{P}'.format(P=x) for x in ances_ind]

    mama_ld_df = pd.DataFrame(index=list(range(len(array_snps.IDList))), columns=col_list)
    mama_ld_flat=list()
    #mama_ld_dict=dict()
    M = dict()
    M_5_50 = dict()

    for t in range(T): # columns
        for j in range(t, T): # rows

            # extract subset of SNPs used for LD estimation
            snplist = np.intersect1d(snp_index[ances_ind[t]],snp_index[ances_ind[j]])

            # extract individuals used for standardizing genotypes
            if t==j:
                indlist = ind_index[ances_ind[t]]
            else:
                indlist = np.union1d(ind_index[ances_ind[t]],ind_index[ances_ind[j]])
                #indlist = None

            # read genotype array
            logging.info('Reading genotypes from {fname} for LD estimation based on {P1}-{P2}'.format(fname=array_file,P1=ances_ind[t],P2=ances_ind[j]))
            geno_array = array_obj(array_file, n, array_snps, keep_snps=snplist, keep_indivs=indlist, mafMin=args.maf)

            # log number of SNPs used for each set of scores (for regression)
            M[(ances_ind[t], ances_ind[j])] = geno_array.m
            M_5_50[(ances_ind[t], ances_ind[j])] = np.sum(geno_array.maf > 0.05)

            # determine block widths
            max_dist, coords = ld.block_width(args, geno_array, array_snps)
            block_left = ld.getBlockLefts(coords, max_dist) # coordinate of the leftmost SNPs included in blocks

            if args.ldBlockSize:
                block_size = ld.getBlockM(coords, max_dist) # log number of SNPs in the LD block (score set specific)
                mama_ld_df.loc[geno_array.kept_snps, '{P1}_{P2}_ldBlockSize'.format(P1=ances_ind[t],P2=ances_ind[j])] = block_size

            # pairwise LD calculation
            logging.info('Begin calculating LD scores based on {P1}-{P2}'.format(P1=ances_ind[t], P2=ances_ind[j]))
            eff_ances_flag = ances_flag.loc[indlist,:].reset_index(drop=True) # added 9/28 for T>2 ancestry
            #pair_ldscore = pair_ldScoreVarBlocks_OLD(args, t, j, ances_ind, eff_ances_flag, ances_n, c, block_left, geno_array)
            pair_ldscore = pair_ldScoreVarBlocks(args, t, j, ances_ind, eff_ances_flag, ances_n, c, block_left, array_obj, array_file, n, array_snps, snplist, indlist, bootstrap=bootstrap)

            logging.info('Recording {M} scores...'.format(M=pair_ldscore.shape[0]))
            mama_ld_df.loc[geno_array.kept_snps, '{P1}_{P2}'.format(P1=ances_ind[t],P2=ances_ind[j])] = pair_ldscore[:,0].reshape(-1,)

            if args.bootstrap:
                #mama_ld_df.loc[geno_array.kept_snps, '{P1}_{P2}_SE'.format(P1=ances_ind[t],P2=ances_ind[j])] = np.nanstd(pair_ldscore[:,1:], axis=1)
                #mama_ld_df.loc[geno_array.kept_snps, '{P1}_{P2}_lower'.format(P1=ances_ind[t],P2=ances_ind[j])] = np.nanpercentile(pair_ldscore[:,1:], 2.5, axis=1)
                #mama_ld_df.loc[geno_array.kept_snps, '{P1}_{P2}_upper'.format(P1=ances_ind[t],P2=ances_ind[j])] = np.nanpercentile(pair_ldscore[:,1:], 97.5, axis=1)
                mama_ld_df.loc[geno_array.kept_snps, '{P1}_{P2}_bootstrap'.format(P1=ances_ind[t],P2=ances_ind[j])] = pair_ldscore[:,1].reshape(-1,)

            mama_ld_flat.append(pair_ldscore)
            #mama_ld_dict['{p1}_{p2}'.format(p1=ances_ind[t], p2=ances_ind[j])] = pair_ldscore
            logging.info(short_border+"\n")

    # construct matrices
    mama_ld_mat = np.empty(shape=[T,T], dtype=object)
    mama_ld_mat.fill(np.nan)
    mama_ld_mat[np.triu_indices(T)] = mama_ld_flat
    #mama_ld_mat[(np.triu_indices(T)[1],np.triu_indices(T)[0])] = mama_ld_flat

    # monomorphic variants
    mama_mono_log = pd.DataFrame(index=col_list)
    for score in col_list:
        mama_mono_log.loc[score, 'monomorphic in LD scores'] = np.sum(mama_ld_df[score]==0, axis=0)
    
        if args.bootstrap: 
            mama_mono_log.loc[score, 'monomorphic in bootstrap'] = np.sum(mama_ld_df[score+"_bootstrap"]==0, axis=0)
    
    if mama_mono_log.sum().sum() > 0: 
        logging.info("Warning: There are monomorphic variants...LD scores will be assigned NaN. \n")
        logging.info(mama_mono_log)

    mama_ld_df = mama_ld_df.replace(0, np.NaN)
    
    #on_diag = mama_ld_dict[['{P}_{P}'.format(P=x) for x in ances_ind]]
    #off_diag = mama_ld_dict[['_'.join(x) for x in score_tags]]
    #mama_ld_mat = matrix_fillin(on_diag, off_diag, len(ances_ind))

    return mama_ld_mat, mama_ld_df, M, M_5_50

def pair_ldScoreVarBlocks(args, t, j, ances_ind, eff_ances_flag, ances_n, c_size, block_left, array_obj, array_file, array_n, array_snps, snplist, indlist, bootstrap=None):
    '''
    Estimate the single-/cross- ancestry LD scores for each pair.

    1/2 represent ance_1 and ances_2
    A/B represent genotype blocks, B is nested in A

    '''

    m = block_left.shape[0]
    if t==j:
        n = ances_n[t]
    else:
        n = ances_n[t] + ances_n[j]
    #n = ances_n[t] if t==j else eff_ances_flag.shape[0]

    # initializing blocks
    b = np.nonzero(block_left > 0)
    if np.any(b):
        b = b[0][0]
    else:
        b = m
    b = int(np.ceil(b/c_size)*c_size)  # round up to a multiple of c; b may increase
    if b > m:
        c_size = 1
        b = m

    # calculate block sizes
    block_sizes = np.array(np.arange(m) - block_left)
    block_sizes = np.ceil(block_sizes/c_size)*c_size # round up to a multiple of c; b may increase block_sizes
    #annot = np.ones((m,1))

    # indexes for ancestries
    flag_1 = np.where(eff_ances_flag.ancestry==ances_ind[t])
    flag_2 = np.where(eff_ances_flag.ancestry==ances_ind[j])

    exp = float(args.pq_exp) if args.pq_exp else 0

    # if bootstrap: # ADDED 11/5: record bootstrap indexes
    #     bs_ind_1 = np.zeros((ances_n[0], bootstrap), dtype=np.int32)
    #     bs_ind_2 = np.zeros((ances_n[1], bootstrap), dtype=np.int32)
    #     cor_sum = np.zeros((m, int(bootstrap)+1))
    #
    #     for i in range(bootstrap):
    #         bs_geno_array = array_obj(array_file, array_n, array_snps, keep_snps=snplist, keep_indivs=indlist, mafMin=args.maf) # static
    #         bs_A = bs_geno_array.nextSNPs(b) # static
    #         bs_l_A = 0
    #         bs_ind_1[:, i] = np.random.choice(flag_1[0], size=ances_n[0], replace=True) # bootstrap
    #         bs_ind_2[:, i] = np.random.choice(flag_2[0], size=ances_n[1], replace=True) # bootstrap
    #         c = c_size
    #         [rfuncAB, rfuncAB_1, rfuncAB_2] = [np.zeros((b, c))] * 3
    #         [rfuncBB, rfuncBB_1, rfuncBB_2] = [np.zeros((c, c))] * 3
    #
    #         # chunk inside the first block
    #         for bs_l_B in range(0, b, c):  # bs_l_B := index of leftmost SNP in matrix B
    #             bs_B = bs_A[:, bs_l_B:bs_l_B+c]
    #
    #             if t==j: # ances_1 = ances_2
    #                 (A_trans, B_trans) = ld.scale_trans(bs_A, bs_B, bs_ind_1[:,i], exp)
    #                 rfuncAB_1 = np.dot(A_trans[:].T, B_trans[:] / ances_n[t])
    #                 rfuncAB_2 = np.dot(A_trans[:].T, B_trans[:] / ances_n[j])
    #
    #                 #assert np.allclose(rfuncAB_1,rfuncAB_2, atol=1e-08), "The SNP correlations for the same ancestry group do not match!"
    #                 # absolute(a - b) <= (atol + rtol * absolute(b))
    #
    #                 if args.no_single_correct:
    #                     rfuncAB = np.multiply(rfuncAB_1, rfuncAB_2)
    #                 else:
    #                     rfuncAB = ld.l2_unbiased(rfuncAB_1, ances_n[t])
    #
    #
    #
    #             else: # ances_1 != ances_2
    #
    #                 (A1_trans, B1_trans) = ld.scale_trans(bs_A, bs_B, bs_ind_1[:,i], exp)
    #                 rfuncAB_1 = np.dot(A1_trans[:].T, B1_trans[:] / ances_n[t])
    #
    #                 (A2_trans, B2_trans) = ld.scale_trans(bs_A, bs_B, bs_ind_2[:,i], exp)
    #                 rfuncAB_2 = np.dot(A2_trans[:].T, B2_trans[:] / ances_n[j])
    #
    #                 rfuncAB = np.multiply(rfuncAB_1, rfuncAB_2)
    #
    #             cor_sum[bs_l_A:bs_l_A+b, i+1] += np.nansum(rfuncAB, axis=1).reshape(-1,)
    #
    #         # move on to the next block
    #         b0 = b
    #         md = int(c*np.floor(m/c)) # md <= m, but is multiple of c
    #         end = md + 1 if md != m else md
    #         bs_b = b
    #
    #         # chunk inside the next block
    #         for bs_l_B in range(b0, end, c):
    #             bs_old_b = bs_b
    #             bs_b = int(block_sizes[bs_l_B]) # block_size is an array of floats
    #             if bs_l_B > b0 and bs_b > 0:
    #                 # block_size can't increase more than c
    #                 # block_size can't be less than c unless it is zero
    #                 # both of these things make sense
    #                 bs_A = np.hstack((bs_A[:, bs_old_b-bs_b+c:bs_old_b], bs_B))
    #                 bs_l_A += bs_old_b-bs_b+c
    #             elif bs_l_B == b0 and bs_b > 0:
    #                 bs_A = bs_A[:, b0-bs_b:b0]
    #                 bs_l_A = b0-bs_b
    #             elif bs_b == 0:  # no SNPs to left in window, e.g., after a sequence gap
    #                 bs_A = np.array(()).reshape((n, 0))
    #                 bs_l_A = bs_l_B
    #             if bs_l_B == md:
    #                 c = m - md # need to re-initialize the matrix
    #                 [rfuncAB, rfuncAB_1, rfuncAB_2] = [np.zeros((bs_b, c))] * 3
    #                 [rfuncBB, rfuncBB_1, rfuncBB_2] = [np.zeros((c, c))] * 3
    #             if bs_b != bs_old_b:
    #                 [rfuncAB, rfuncAB_1, rfuncAB_2] = [np.zeros((bs_b, c))] * 3
    #
    #             bs_B = bs_geno_array.nextSNPs(c)
    #
    #             if t==j:
    #                 (A_trans, B_trans) = ld.scale_trans(bs_A, bs_B, bs_ind_1[:,i], exp)
    #                 rfuncAB_1 = np.dot(A_trans[:].T, B_trans[:] / ances_n[t])
    #                 rfuncAB_2 = np.dot(A_trans[:].T, B_trans[:] / ances_n[j])
    #
    #                 #assert np.allclose(rfuncAB_1,rfuncAB_2, atol=1e-08), "This error should be catched earlier in the codes!"
    #
    #                 if args.no_single_correct:
    #                     rfuncAB = np.multiply(rfuncAB_1, rfuncAB_2)
    #                 else:
    #                     rfuncAB = ld.l2_unbiased(rfuncAB_1, ances_n[t])
    #             else:
    #
    #                 (A1_trans, B1_trans) = ld.scale_trans(bs_A, bs_B, bs_ind_1[:,i], exp)
    #                 rfuncAB_1 = np.dot(A1_trans[:].T, B1_trans[:] / ances_n[t])
    #
    #                 (A2_trans, B2_trans) = ld.scale_trans(bs_A, bs_B, bs_ind_2[:,i], exp)
    #                 rfuncAB_2 = np.dot(A2_trans[:].T, B2_trans[:] / ances_n[j])
    #
    #                 rfuncAB = np.multiply(rfuncAB_1, rfuncAB_2)
    #
    #             cor_sum[bs_l_A:bs_l_A+bs_b, i+1] += np.nansum(rfuncAB, axis=1).reshape(-1,)
    #             cor_sum[bs_l_B:bs_l_B+c, i+1] += np.nansum(rfuncAB, axis=0).reshape(-1,)
    #
    #             if t==j:
    #                 sing_ind = np.arange(bs_B.shape[0])
    #                 (B_trans, _) = ld.scale_trans(bs_B, bs_B, sing_ind, exp)
    #                 rfuncBB_1 = np.dot(B_trans[:].T, B_trans[:] / ances_n[t])
    #                 rfuncBB_2 = np.dot(B_trans[:].T, B_trans[:] / ances_n[j])
    #
    #                 #assert np.allclose(rfuncBB_1,rfuncBB_2, atol=1e-08), "This error should be catched earlier in the codes!"
    #                 if args.no_single_correct:
    #                     rfuncBB = np.multiply(rfuncBB_1, rfuncBB_2)
    #                 else:
    #                     rfuncBB = ld.l2_unbiased(rfuncBB_1, ances_n[t])
    #             else:
    #
    #                 (B1_trans, _) = ld.scale_trans(bs_B, bs_B, bs_ind_1[:,i], exp)
    #                 rfuncBB_1 = np.dot(B1_trans[:].T, B1_trans[:] / ances_n[t])
    #
    #                 (B2_trans, _) = ld.scale_trans(bs_B, bs_B, bs_ind_2[:,i], exp)
    #                 rfuncBB_2 = np.dot(B2_trans[:].T, B2_trans[:] / ances_n[j])
    #
    #                 rfuncBB = np.multiply(rfuncBB_1, rfuncBB_2)
    #
    #             cor_sum[bs_l_B:bs_l_B+c, i+1] += np.nansum(rfuncBB, axis=1).reshape(-1,)
    #
    # else:
    cor_sum = np.zeros((m, 1))

    # Calculation based on real data (static)
    l_A = 0  # initial index of leftmost SNP in matrix A
    geno_array = array_obj(array_file, array_n, array_snps, keep_snps=snplist, keep_indivs=indlist, mafMin=args.maf) # avoid joint use of array_obj when bootstrapping
    # allele counts stored in A
    A = geno_array.nextSNPs(b)
    assert A.shape[0] == n, "The reading of nextSNPs does not match with the ancestry flag indicator."

    c = c_size
    [rfuncAB, rfuncAB_1, rfuncAB_2] = [np.zeros((b, c))] * 3
    [rfuncBB, rfuncBB_1, rfuncBB_2] = [np.zeros((c, c))] * 3

    for l_B in range(0, b, c):  # l_B := index of leftmost SNP in matrix B
        B = A[:, l_B:l_B+c]
        # DEFINITIONS:
            # A, B: Raw allele counts
            # A_trans, B_trans: Standardized allele counts
            # rfuncAB_1[i,j], rfuncAB_2[i,j]: covariance of SNP i and j in pops 1 and 2
            # rfuncAB: squared correlation (from standardized geno units)
        if t==j: # ances_1 = ances_2
            sing_ind = np.arange(A.shape[0])
            (A_trans, B_trans) = ld.scale_trans(A, B, sing_ind, exp)
            rfuncAB_1 = np.dot(A_trans[:].T, B_trans[:] / ances_n[t])
            rfuncAB_2 = np.dot(A_trans[:].T, B_trans[:] / ances_n[j])

            assert np.allclose(rfuncAB_1,rfuncAB_2, atol=1e-08), "The SNP correlations for the same ancestry group do not match!"
            # absolute(a - b) <= (atol + rtol * absolute(b))

            if args.no_single_correct:
                rfuncAB = np.multiply(rfuncAB_1, rfuncAB_2)
            else:
                rfuncAB = ld.l2_unbiased(rfuncAB_1, ances_n[t])
                
        else: # ances_1 != ances_2

            (A1_trans, B1_trans) = ld.scale_trans(A, B, flag_1[0], exp)
            rfuncAB_1 = np.dot(A1_trans[:].T, B1_trans[:] / ances_n[t])

            (A2_trans, B2_trans) = ld.scale_trans(A, B, flag_2[0], exp)
            rfuncAB_2 = np.dot(A2_trans[:].T, B2_trans[:] / ances_n[j])

            rfuncAB = np.multiply(rfuncAB_1, rfuncAB_2)

        # if allele count:
            # multiply each sq correlation with ratio of variances
            # variance of x_k = 2p_k(1-p_k)
        if not args.std_geno_ldsc:
            # frequency is total allele count over 2*number of people (double for two chromosomes)
            pop1_freq_A = A[flag_1[0],:].sum(axis=0) / (2*A[flag_1[0],:].shape[0])
            pop2_freq_A = A[flag_2[0],:].sum(axis=0) / (2*A[flag_2[0],:].shape[0])
            pop1_freq_B = B[flag_1[0],:].sum(axis=0) / (2*B[flag_1[0],:].shape[0])
            pop2_freq_B = B[flag_2[0],:].sum(axis=0) / (2*B[flag_2[0],:].shape[0])

            # snp variance is 2p(1-p)
            pop1_var_A = 2*pop1_freq_A*(1-pop1_freq_A)
            pop2_var_A = 2*pop2_freq_A*(1-pop2_freq_A)
            pop1_var_B = 2*pop1_freq_B*(1-pop1_freq_B)
            pop2_var_B = 2*pop2_freq_B*(1-pop2_freq_B)

            # construct fraction
            numerator = np.sqrt(pop1_var_A*pop2_var_A)
            denominator = np.sqrt(np.reciprocal(pop1_var_B*pop2_var_B))
            scale_by = np.outer(numerator, denominator)

            # convert to ale count ldsc
            cor_sum[l_A:l_A+b, 0] += np.nansum(np.multiply(rfuncAB,scale_by), axis=1).reshape(-1,)
        else:
            cor_sum[l_A:l_A + b, 0] += np.nansum(rfuncAB, axis=1).reshape(-1, )
 
    # move on to the next block
    b0 = b
    md = int(c*np.floor(m/c)) # md <= m, but is multiple of c
    end = md + 1 if md != m else md

    # chunk inside the next block
    for l_B in range(b0, end, c):
        old_b = b
        b = int(block_sizes[l_B]) # block_size is an array of floats
        if l_B > b0 and b > 0:
            # block_size can't increase more than c
            # block_size can't be less than c unless it is zero
            # both of these things make sense
            A = np.hstack((A[:, old_b-b+c:old_b], B))
            l_A += old_b-b+c
        elif l_B == b0 and b > 0:
            A = A[:, b0-b:b0]
            l_A = b0-b
        elif b == 0:  # no SNPs to left in window, e.g., after a sequence gap
            A = np.array(()).reshape((n, 0))
            l_A = l_B
        if l_B == md:
            c = m - md # need to re-initialize the matrix
            [rfuncAB, rfuncAB_1, rfuncAB_2] = [np.zeros((b, c))] * 3
            [rfuncBB, rfuncBB_1, rfuncBB_2] = [np.zeros((c, c))] * 3
        if b != old_b:
            [rfuncAB, rfuncAB_1, rfuncAB_2] = [np.zeros((b, c))] * 3

        B = geno_array.nextSNPs(c)

        if t==j:

            sing_ind = np.arange(A.shape[0])
            (A_trans, B_trans) = ld.scale_trans(A, B, sing_ind, exp)
            rfuncAB_1 = np.dot(A_trans[:].T, B_trans[:] / ances_n[t])
            rfuncAB_2 = np.dot(A_trans[:].T, B_trans[:] / ances_n[j])

            assert np.allclose(rfuncAB_1,rfuncAB_2, atol=1e-08), "This error should be catched earlier in the codes!"
            if args.no_single_correct:
                rfuncAB = np.multiply(rfuncAB_1, rfuncAB_2)
            else:
                rfuncAB = ld.l2_unbiased(rfuncAB_1, ances_n[t])
        else:

            (A1_trans, B1_trans) = ld.scale_trans(A, B, flag_1[0], exp)
            rfuncAB_1 = np.dot(A1_trans[:].T, B1_trans[:] / ances_n[t])

            (A2_trans, B2_trans) = ld.scale_trans(A, B, flag_2[0], exp)
            rfuncAB_2 = np.dot(A2_trans[:].T, B2_trans[:] / ances_n[j])

            rfuncAB = np.multiply(rfuncAB_1, rfuncAB_2)
        if not args.std_geno_ldsc:
            # frequency is total allele count over 2*number of people (double for two chromosomes)
            pop1_freq_A = A[flag_1[0],:].sum(axis=0) / (2*A[flag_1[0],:].shape[0])
            pop2_freq_A = A[flag_2[0],:].sum(axis=0) / (2*A[flag_2[0],:].shape[0])
            pop1_freq_B = B[flag_1[0],:].sum(axis=0) / (2*B[flag_1[0],:].shape[0])
            pop2_freq_B = B[flag_2[0],:].sum(axis=0) / (2*B[flag_2[0],:].shape[0])

            # snp variance is 2p(1-p)
            pop1_var_A = 2*pop1_freq_A*(1-pop1_freq_A)
            pop2_var_A = 2*pop2_freq_A*(1-pop2_freq_A)
            pop1_var_B = 2*pop1_freq_B*(1-pop1_freq_B)
            pop2_var_B = 2*pop2_freq_B*(1-pop2_freq_B)

            # construct fraction
            numerator = np.sqrt(pop1_var_A*pop2_var_A)
            denominator = np.sqrt(np.reciprocal(pop1_var_B*pop2_var_B))
            scale_by = np.outer(numerator, denominator)
            # numerator_B = np.sqrt(pop1_var_B*pop2_var_B)
            # denominator_B = np.sqrt(np.reciprocal(pop1_var_B * pop2_var_B))
            # scale_by_B = np.outer(numerator_B, denominator_B)

            cor_sum[l_A:l_A + b, 0] += np.nansum(np.multiply(rfuncAB, scale_by), axis=1).reshape(-1,)
            cor_sum[l_B:l_B + c, 0] += np.nansum(np.multiply(rfuncAB, scale_by), axis=0).reshape(-1,)
        else:

            cor_sum[l_A:l_A+b, 0] += np.nansum(rfuncAB, axis=1).reshape(-1,)
            cor_sum[l_B:l_B+c, 0] += np.nansum(rfuncAB, axis=0).reshape(-1,)
        
        if t==j:
            sing_ind = np.arange(B.shape[0])
            (B_trans, _) = ld.scale_trans(B, B, sing_ind, exp)
            rfuncBB_1 = np.dot(B_trans[:].T, B_trans[:] / ances_n[t])
            rfuncBB_2 = np.dot(B_trans[:].T, B_trans[:] / ances_n[j])

            #assert np.allclose(rfuncBB_1,rfuncBB_2, atol=1e-08), "This error should be catched earlier in the codes!"
            if args.no_single_correct:
                rfuncBB = np.multiply(rfuncBB_1, rfuncBB_2)
            else:
                rfuncBB = ld.l2_unbiased(rfuncBB_1, ances_n[t])
        else:

            (B1_trans, _) = ld.scale_trans(B, B, flag_1[0], exp)
            rfuncBB_1 = np.dot(B1_trans[:].T, B1_trans[:] / ances_n[t])

            (B2_trans, _) = ld.scale_trans(B, B, flag_2[0], exp)
            rfuncBB_2 = np.dot(B2_trans[:].T, B2_trans[:] / ances_n[j])

            rfuncBB = np.multiply(rfuncBB_1, rfuncBB_2)

        if not args.std_geno_ldsc:
            # frequency is total allele count over 2*number of people (double for two chromosomes)
            pop1_freq = B[flag_1[0],:].sum(axis=0) / (2*B[flag_1[0],:].shape[0])
            pop2_freq = B[flag_2[0],:].sum(axis=0) / (2*B[flag_2[0],:].shape[0])

            # snp variance is 2p(1-p)
            pop1_var = 2*pop1_freq*(1-pop1_freq)
            pop2_var = 2*pop2_freq*(1-pop2_freq)

            # construct fraction
            numerator = np.sqrt(pop1_var*pop2_var)
            denominator = np.sqrt(np.reciprocal(pop1_var*pop2_var))
            scale_by = np.outer(numerator, denominator)

            # convert to ale count ldsc
            cor_sum[l_B:l_B+c, 0] += np.nansum(np.multiply(rfuncBB, scale_by), axis=1).reshape(-1,)
        else:
            cor_sum[l_B:l_B+c, 0] += np.nansum(rfuncBB, axis=1).reshape(-1,)

    return cor_sum

def estimate_LD_score_MAMA(args):
    '''
    Wrapper function for estimationg single ancestry, cross-ancestry LD scores from reference panel genotypes.

    '''

    #-------------------------
    # 1. input checker
    #-------------------------

    if args.bfile_merged_path:
        bfile_merged_path = str(args.bfile_merged_path)
    else:
        raise IOError('Please specify filepath to merged bfiles.')

    if args.ances_path:
        df_ances = pd.read_csv(args.ances_path, delim_whitespace=True, index_col=None,names=['IID','ancestry'],dtype={'IID':object,'ancestry':object})
    else:
        raise IOError('Please specify the filepath to ancestry indicator.')

    ances_list = list(sorted(df_ances.ancestry.unique()))

    snp_file, snp_obj = bfile_merged_path +'.bim', ps.PlinkBIMFile
    ind_file, ind_obj = bfile_merged_path +'.fam', ps.PlinkFAMFile
    array_file, array_obj = bfile_merged_path +'.bed', ld.PlinkBEDFile

    #-------------------------
    # 2. Read input files
    #-------------------------

    # read fam
    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)
    logging.info('Read list of {n} individuals from {f}'.format(n=n, f=ind_file))

    # clean the ancestry file
    ances_flag = array_indivs.IDList.merge(df_ances, how='inner', on='IID')

    if len(array_indivs.IDList['IID']) > len(ances_flag):
        raise IOError('There are {N} individuals in the merged .fam file that cannot be identified using the ancestry source file.'.format(N=len(array_indivs.IDList['IID'])-len(ances_flag)))
    elif len(df_ances) > len(ances_flag):
        logging.info('There are {N} individuals in the ancestry file with no genotypic data present.'.format(N=len(df_ances)-len(ances_flag)))

    # read bim/snp
    array_snps = snp_obj(snp_file)
    m = len(array_snps.IDList)
    logging.info('Read list of {m} SNPs from {f}'.format(m=m, f=snp_file))

    # snp_ances checkers
    if args.snp_ances:
        df_snp_ances = pd.read_csv(args.snp_ances, delim_whitespace=True, index_col=None)
        df_snp_ances['SNP']=df_snp_ances['SNP'].astype('object')
    else:
        raise IOError('Please specify filepath to snp_ances, a text file with SNP ID and indicator columns for each ancestry group. Note that the header row is required.')

    # check name consistency
    ances_ind=sorted([ances for ances in df_snp_ances.columns if ances not in ['SNP']])
    if not set(ances_ind).issubset(set(ances_list)):
        raise IOError('Please make sure that the naming convention of ancestry groups provided in --ances-path and --snp-ances is consistent.')

    # when inconsistent, use snp-ances
    ances_n={x:len(ances_flag[ances_flag.ancestry==a])for x,a in enumerate(ances_ind)}
    T = len(ances_ind)

    # exclude SNP with unspecified ancestry sources
    df_check = array_snps.IDList.merge(df_snp_ances, how='left', on='SNP', indicator=True)

    logging.info('There are {M} SNPs in the merged .bim file without identified source of ancestry groups. These variants are dropped in the LD score estimation.'.format(M=np.sum(df_check._merge=="left_only")))

    if np.sum(df_check._merge=="right_only")!=0:
        logging.info('There are {M} SNPs in --snp-ances but are not in the merged .bim file.'.fommat(M=np.sum(df_check._merge=="right_only")))

    logging.info(short_border)

    #-------------------------
    # 3. Apply filters
    #-------------------------

    # filter based on the provided snp_ances list
    snps_extract = df_check.loc[df_check._merge=="both","SNP"]
    snps_extract.to_csv(args.out+".snplist", index=False, sep='\t')
    good_snps = ps.__filter__(args.out+".snplist", 'SNPs', 'include', array_snps) # index
    os.remove(args.out+".snplist")

    snp_index = dict.fromkeys(ances_ind)
    for i,ances in enumerate(ances_ind):
        snp_list = df_check.loc[np.logical_and(df_check._merge=="both",df_check[ances]==1),"SNP"]
        snp_list.to_csv(args.out+".snplist."+ances, index=False, sep='\t')
        snp_index[ances] = ps.__filter__(args.out+".snplist."+ances, 'SNPs', 'include', array_snps)
        os.remove(args.out+".snplist."+ances)

    # filter based on the provided ances_path list
    ind_index = dict.fromkeys(ances_ind)
    for i,ances in enumerate(ances_ind):
        ind_list = ances_flag.loc[ances_flag.ancestry==ances, "IID"]
        ind_list.to_csv(args.out+".indlist."+ances, index=False, sep='\t')
        ind_index[ances] = ps.__filter__(args.out+".indlist."+ances, 'individuals', 'include', array_indivs)
        os.remove(args.out+".indlist."+ances)

    #-------------------------
    # 4. Estimate Scores
    #-------------------------
    logging.info("\n" + borderline)
    if args.bootstrap:
        logging.info("Estimating LD Score with {} bootstraps.".format(int(args.bootstrap)))
    else:
        logging.info("Estimating LD Score.")

    lN_mat, lN_df, M, M_5_50 = multi_ldScoreVarBlocks(args, ances_ind, ances_flag, ances_n, snp_index, ind_index, array_file, array_obj, array_snps)
    col_prefix = "L2"
    file_suffix = "l2"

    # restrict the scores to those that pass built-in filters
    logging.info("Organizing all sets of LD scores based on merged .bim file")
    geno_array = array_obj(array_file, n, array_snps, keep_snps=None, keep_indivs=None, mafMin=args.maf)
    lN_df = lN_df.loc[geno_array.kept_snps,:] 
    # change of index to pd.indexes.numeric.Int64Index
    # bim_df = pd.DataFrame(data=geno_array.df, columns=geno_array.colnames)
    # df = pd.concat([bim_df, lN_df], axis=1)
    df = pd.DataFrame.from_records(np.c_[geno_array.df, lN_df])
    out_fname = args.out + '.' + file_suffix + '.ldscore'
    new_colnames = geno_array.colnames + list(lN_df.columns)
    df.columns = new_colnames
    
    #-------------------------
    # 5. Output Scores
    #-------------------------
    if args.print_snps:
        if args.print_snps.endswith('gz'):
            print_snps = pd.read_csv(args.print_snps, header=None, compression='gzip')
        elif args.print_snps.endswith('bz2'):
            print_snps = pd.read_csv(args.print_snps, header=None, compression='bz2')
        else:
            print_snps = pd.read_csv(args.print_snps, header=None)
        if len(print_snps.columns) > 1:
            raise ValueError('--print-snps must refer to a file with a one column of SNP IDs.')
        logging.info('Reading list of {N} SNPs for which to print LD Scores from {F}'.format(\
                        F=args.print_snps, N=len(print_snps)))

        print_snps.columns=['SNP']
        df = df.ix[df.SNP.isin(print_snps.SNP),:]
        if len(df) == 0:
            raise ValueError('After merging with --print-snps, no SNPs remain.')
        else:
            msg = 'After merging with --print-snps, LD Scores for {N} SNPs will be printed.'
            logging.info(msg.format(N=len(df)))

    # save scores for SNPs with cross-ancestry only
    if args.print_cross_only:
        df.dropna(how='all',inplace=True)

    l2_suffix = '.gz'

    logging.info("Writing LD Scores for {N} SNPs to {f}.gz".format(f=out_fname, N=len(df)))
    df.drop(['CM','MAF'], axis=1).to_csv(out_fname, sep="\t", header=True, index=False, float_format='%.10f', na_rep="NA")
    subprocess.call(['gzip', '-f', out_fname])

    # print .M
    np.save(args.out + '.'+ file_suffix +'.M', M)

    # print .M_5_50
    np.save(args.out + '.'+ file_suffix +'.M_5_50', M_5_50)

    # start_time isn't defined in this method
    # logging.info('MAMA score estimation completed. Time elapsed: {}'.format(sec_to_str(time.time()-start_time)))

    return df

## Argument parsers
parser = argparse.ArgumentParser(description="\n LD Score Calculation for MAMA")

## output file paths
ofile = parser.add_argument_group(title="Output Options", description="Output directory and options to write to files.")
ofile.add_argument('--out', default='mama_ldscore', type=str, 
    help='Output filename prefix. If --out is not set, LDSC will use ldsc as the default output filename prefix.')
ofile.add_argument('--stream-stdout', default=False, action="store_true", help='Stream log information on console in addition to writing to log file.')

# Basic LD Score Estimation Flags
basic_ldsc = parser.add_argument_group(title="LD Score Estimation Flags", 
    description="File paths to Plink format genetic files and l2 scores.")
basic_ldsc.add_argument('--snp-ances', default=None, type=str,
    help='Filepath to tab delimited file with 1) variant IDs; 2) an indicator column for each of the ancestry groups (e.g. AMR, AFR, EAS, EUR, SAS), with values 0 and 1.')
basic_ldsc.add_argument('--ances-path', default=None, type=str,
    help='Filepath to tab delimited file with two columns: 1) IID that correspond to the .fam file; 2) Indicator of ancestry groups (e.g. AMR, AFR, EAS, EUR, SAS).')
basic_ldsc.add_argument('--no-single-correct', default=False, action='store_true', 
    help='By default, MAMA corrects the biasedness in single-ancestry score estimation. Turning on this flag will disable this calculation.')
basic_ldsc.add_argument('--bfile-merged-path', default=None, type=str, 
    help='Prefix for merged genotypic files across ancestry groups. Please have the files in plink binary format (.bed/.bim/.fam)')

# Filtering / Data Management for LD Score
data_filter = parser.add_argument_group(title="LD Score Estimation Data Filters", description="Data management options for LD scores.")
data_filter.add_argument('--std-geno-ldsc', default=False, action='store_true',
      help='Generate LD scores from standardized genotypes (default is allele counts).')
data_filter.add_argument('--ld-extract', default=None, type=str,
    help='File with SNPs to include in LD Score estimation. The file should contain one SNP ID per row.')
data_filter.add_argument('--ld-keep', default=None, type=str, 
    help='Single file with individuals to include in LD Score estimation. The file should contain one individual ID per row.')
data_filter.add_argument('--ld-wind-snps', default=None, type=int, 
    help='Specify the window size to be used for estimating LD Scores in units of # of SNPs. You can only specify one --ld-wind-* option.')
data_filter.add_argument('--ld-wind-kb', default=None, type=float, 
    help='Specify the window size to be used for estimating LD Scores in units of kilobase-pairs (kb). You can only specify one --ld-wind-* option.')
data_filter.add_argument('--ld-wind-cm', default=None, type=float, 
    help='Specify the window size to be used for estimating LD Scores in units of centiMorgans (cM). You can only specify one --ld-wind-* option.')
data_filter.add_argument('--print-snps', default=None, type=str, 
    help='This flag tells LDSC to only print LD Scores for the SNPs listed (one ID per row) in PRINT_SNPS. The calculation will still include SNPs not in PRINT_SNPs.' )
data_filter.add_argument('--print-cross-only', default=False, action='store_true',
    help='This flag tells LDSC to only print LD Scores for SNPs with valid cross-ancestry scores.')

# Flags you should almost never use
advanced_opt = parser.add_argument_group(title="Advanced Options")
advanced_opt.add_argument('--chunk-size', default=50, type=int,
    help='Chunk size for LD Score calculation. Use the default 50.')
advanced_opt.add_argument('--pq-exp', default=0, type=float, help='Setting this flag causes LDSC to compute LD Scores with the given scale factor, i.e., \ell_j := \sum_k 2(p_k(1-p_k))^a r^2_{jk}, where p_k denotes the MAF of SNP j and a is the argument to --pq-exp. ')
advanced_opt.add_argument('--maf', default=None, type=float, help='Minor allele frequency lower bound. Default is MAF > 0.')
advanced_opt.add_argument('--bootstrap', default=None, type=int, help='Number of bootstraps used for LD scores estimation. Increasing the value can lead to significant increase in computing runtime.')
advanced_opt.add_argument('--ldBlockSize', default=False, action='store_true', 
    help='Report the LD block size (# of SNPs) for each score set. Use with caution as the flag might substantially slow down the computing runtime. Recommend using the flag for testing/debugging, and restricting to less than 10000 SNPs.')


def main_func(argv):
    # I feel like this came up in PGI, I forget what the solution is but passing everything in sys.argv
    # also includes the path to the script itself, so you get an error that mama_ldscore.py isn't an accepted flag.
    args = parser.parse_args(argv[1:])

    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.out + '.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d %I:%M:%S %p')
    if args.stream_stdout:
        logging.getLogger().addHandler(logging.StreamHandler()) # prints to console

    start_time = time.time()

    try:
        header_sub = header
        header_sub += "Calling ./mama_ldscore.py \\\n"
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in list(opts.keys()) if opts[x] != defaults[x]]
        options = ['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults]
        header_sub += '\n'.join(options).replace('True','').replace('False','')
        header_sub = header_sub[0:-1] + '\n'

        logging.info(header_sub)
        logging.info("Beginning to estimate cross-ancestry LD scores...") 

        mama_LD_scores = estimate_LD_score_MAMA(args)

    except Exception as e:
        logging.info(borderline+"\n")
        logging.error(e,exc_info=True)
        logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

    logging.info('Cross-ancestry LD score estimation complete. Time elapsed: {}'.format(sec_to_str(time.time()-start_time)))
    logging.info(borderline)    


## Operators
if __name__ == '__main__':
    main_func(sys.argv)