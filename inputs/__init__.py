"""Read-only input layer for the external BMsemi_Model repository (model corpus + compiler reference)."""
import sys

# Importing BMsemi_Model code must never write __pycache__ files into that (read-only) repo.
sys.dont_write_bytecode = True
