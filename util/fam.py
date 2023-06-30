# TODO(jonbjala) Comment code / include function descriptions

# FAM file suffix
FAM_SUFFIX = ".fam"


def get_sample_size_from_fam_file(fam_filename: str):
    with open(fam_filename) as f:
        N = sum(1 for line in f)    

    return N