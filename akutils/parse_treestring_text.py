# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Parse treestrings from text file
# Author: Timm Nawrocki, Matt Macander
# Last Updated: 2026-02-26
# Usage: Must be executed in a Python 3.12+ installation.
# Description: "Parse treestrings from text file" provides a function to parse the tree strings from a text file (exported during the model training process) and return the formatted treestrings in memory.
# ---------------------------------------------------------------------------

def parse_treestring_text(text_input):
    """
    Description: loads and parses the treestrings from an exported text file so that trees are processed individually
    Inputs: 'text_input' -- a file path for a text file storing the exported tree strings
    Returned Value: returns string representations of each tree
    Preconditions: requires a text file exported during the model training process
    """

    # Import packages
    import re

    # Read tree string input file
    with open(text_input, 'r') as f:
        content = f.read()

    # Split by the root node indicator to get individual trees
    raw_trees = re.split(r'(?=1\) root)', content)

    # Filter out empty strings and clean up whitespace
    trees = [t.strip() for t in raw_trees if t.strip()]

    return trees
