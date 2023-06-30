import pandas as pd

# TODO(jonbjala) Comment code / include function descriptions

# Bim-specific Constants =========================

# BIM columns
BIM_CHR_COL = 'CHR'
BIM_RSID_COL = 'RSID'
BIM_CM_COL = 'CM'
BIM_BP_COL = 'BP'
BIM_A1_COL = 'A1'
BIM_A2_COL = 'A2'
BIM_COL_TYPES = {
    BIM_CHR_COL : int,
    BIM_RSID_COL : str,
    BIM_CM_COL : float,
    BIM_BP_COL : int,
    BIM_A1_COL : str,
    BIM_A2_COL : str
    }
BIM_COLUMNS = tuple(BIM_COL_TYPES.keys()) #(BIM_CHR_COL, BIM_RSID_COL, BIM_CM_COL, BIM_BP_COL, BIM_A1_COL, BIM_A2_COL)  TODO(jonbjala) Delete this commented out part?

# BIM separator
BIM_SEPARATOR = "\t"

# BIM file suffix
BIM_SUFFIX = ".bim"

# ================================================

def read_bim_file(bim_filename: str):
    bim_df = pd.read_csv(bim_filename, sep=BIM_SEPARATOR, names=BIM_COLUMNS, dtype=BIM_COL_TYPES)
    return bim_df


# TODO(jonbjala) Add write_bim_file?
