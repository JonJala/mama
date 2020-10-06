#!/usr/bin/env python3

"""
Wrapper / proxy to call old legacy code to create LD Scores file for MAMA.  This will
eventually be replaced with newly written code.
"""
from legacy.mama_ldscore import main_func

if __name__ == '__main__':
    main_func(sys.argv)