import collections
import itertools as it
import math
from typing import Optional

import numpy as np
# TODO(jonbjala) Comment code / include function descriptions

# General constants ==============================

_BITS_PER_BYTE = 8

# ================================================


# Bed-specific Constants =========================

# The first two bytes of a .bed file should be this
_BED_FILE_PREFIX_MAGIC_HEX = '6c1b'
_BED_FILE_PREFIX_MAGIC_BYTEARRAY = bytearray.fromhex(_BED_FILE_PREFIX_MAGIC_HEX)

# The third byte of a .bed file indicating SNP- or individual-major (should be SNP)
_BED_FILE_PREFIX_SNP_MAJOR_MAGIC_HEX = '01'
_BED_FILE_PREFIX_SNP_MAJOR_MAGIC_BYTEARRAY = bytearray.fromhex(_BED_FILE_PREFIX_SNP_MAJOR_MAGIC_HEX)

# The number of bits used for each sample in a .bed file
_BED_BITS_PER_SAMPLE = 2

# BED file suffix
BED_SUFFIX = ".bed"

# ================================================


# Derived constants ==============================

# The number of .bed samples contained in one byte of data
_BED_SAMPLES_PER_BYTE = _BITS_PER_BYTE // _BED_BITS_PER_SAMPLE

# ================================================


# TODO(jonbjala) Allow for float64?
# TODO(jonbjala) Add indices functionality
def read_bed_file(bed_filename: str, M: int, N: int,
                  indices: Optional[collections.abc.Sequence] = None):

    # Amount of bytes to read in for each SNP
    bed_block_size_in_bytes = math.ceil(N / _BED_SAMPLES_PER_BYTE)

    G = np.zeros((M, N), dtype=np.float32)
    
    with open(bed_filename, 'rb') as bed_file:

        # Read in the first 3 bytes and check against expected .bed file prefix
        initial_bytes = bed_file.read(3)        
        if not(initial_bytes[0:2] == _BED_FILE_PREFIX_MAGIC_BYTEARRAY):
            raise RuntimeError("Error: Initial bytes of bed file [0x%s] are not expected [%s].",
                initial_bytes[0:2].hex(), BED_FILE_PREFIX_MAGIC_HEX)

        if not(initial_bytes[2] == _BED_FILE_PREFIX_SNP_MAJOR_MAGIC_BYTEARRAY[0]):
            raise RuntimeError("Error: BED file not in SNP major order, third byte = %s" % 
                hex(initial_bytes[2]))

        # Read in the actual genetic data and check that it's the expected length
        raw_bed_file_contents = bed_file.read()

        expected_data_size_in_bytes = M * bed_block_size_in_bytes
        actual_data_size_in_bytes = len(raw_bed_file_contents)
        if actual_data_size_in_bytes != expected_data_size_in_bytes:
            # TODO(jonbjala) Log warning and throw exception?
            print("Bed file %s, which is supposed to have %s SNPs for %s individuals should "
                  "contain %s bytes of information, but contains %s." %
                  (bed_filename, M, N, expected_data_size_in_bytes, actual_data_size_in_bytes))


        # Convert the raw data into a matrix of floats (potentially with NaNs)
        for i in range(0, M):
            start_byte_pos = i * bed_block_size_in_bytes
            G[i] = read_bed_file._BED_MAP_ARRAY[list(raw_bed_file_contents[
                start_byte_pos : start_byte_pos + bed_block_size_in_bytes])].ravel()[0:N]

    return G


# See https://www.cog-genomics.org/plink/1.9/formats#bed
read_bed_file._BED_BINARY_TO_VALUE_MAP = {
    '00' : 0.0,
    '01' : np.nan,
    '10' : 1.0,
    '11' : 2.0
}

# A bit complicated, but this is to map a byte's worth of data from a .bed file to a Numpy array
# that contains the genotype information for the entries that comprise the byte
read_bed_file._BED_BYTE_TO_VALARR_MAP = {
    int("%s%s%s%s" % tup, base=2) :
        np.array([read_bed_file._BED_BINARY_TO_VALUE_MAP[element] for element in reversed(tup)],
                 dtype=np.float32) for tup in it.product(
                     read_bed_file._BED_BINARY_TO_VALUE_MAP.keys(),
                     repeat=_BED_SAMPLES_PER_BYTE)
}

read_bed_file._BED_MAP_ARRAY = np.array([read_bed_file._BED_BYTE_TO_VALARR_MAP[x]
                                        for x in sorted(
                                            read_bed_file._BED_BYTE_TO_VALARR_MAP.keys())])



# TODO(jonbjala) This needs to be written better
def write_bed_file(bed_filename: str, G: np.ndarray):
    # G is shape (M, N)
    M, N = G.shape

    # Amount of bytes to associate each SNP
    bed_block_size_in_bytes = math.ceil(N / _BED_SAMPLES_PER_BYTE)
    pad_amount = bed_block_size_in_bytes * _BED_SAMPLES_PER_BYTE - N

    padded_G = np.pad(np.nan_to_num(G, nan=write_bed_file._NAN_REPLACEMENT),
        pad_width=((0,0), (0, pad_amount)))

    temp_bytes = bytearray(bed_block_size_in_bytes)

    with open(bed_filename, 'wb') as bed_file:
        bed_file.write(_BED_FILE_PREFIX_MAGIC_BYTEARRAY)
        bed_file.write(_BED_FILE_PREFIX_SNP_MAJOR_MAGIC_BYTEARRAY)

        for snp in range(0, M):
            count = 0
            for cluster_start in range(0, N, _BED_SAMPLES_PER_BYTE):
                map_key = tuple(padded_G[snp, cluster_start:cluster_start+_BED_SAMPLES_PER_BYTE])
                temp_bytes[count] = write_bed_file._BED_VALARR_TO_BYTE_MAP[map_key]
                count += 1

            bed_file.write(temp_bytes)

# See https://www.cog-genomics.org/plink/1.9/formats#bed
write_bed_file._NAN_REPLACEMENT = -1.0
write_bed_file._BED_VALUE_TO_BINARY_MAP = {
    0.0 : '00',
    write_bed_file._NAN_REPLACEMENT : '01',
    1.0 : '10',
    2.0 : '11'
}
write_bed_file._BED_VALARR_TO_BYTE_MAP = {
    tup : int("%s%s%s%s" % tuple(write_bed_file._BED_VALUE_TO_BINARY_MAP[element] 
        for element in reversed(tup)), base=2)
            for tup in it.product(write_bed_file._BED_VALUE_TO_BINARY_MAP.keys(),
                                  repeat=_BED_SAMPLES_PER_BYTE)}
