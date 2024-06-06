# FT_TextChemT5
Term Project for Machine Learning in Bioinformatics

## Description - Motivation

I focus on the task of Molecule Captioning: Given a molecule SMILES, the Language Model (LM) returns a description of the given molecule.
To my knowledge, current top performing LMs in this domain are MolT5 (https://arxiv.org/abs/2204.11817 - Edwards et al, 2022) and TextChemT5 (https://proceedings.mlr.press/v202/christofidellis23a.html - Christofidellis et al, 2023), both based upon the T5 architecture.

TextChemT5 exhibits multi-task and multi-domain capabilities (text2text, text2mol, mol2mol, mol2text) but for the sake of my term project, I solely focus on the task of Molecule Captioning (i.e. mol2text).

In the context of the "Language + Molecules @ ACL 2024" Workshop, a new dataset of 'molecule-description pairs' was publicly released.
I therefore attempt to fine-tune pretrained TextChemT5 models on this dataset to achieve better performance on Molecule Captioning.
