# Thanks to http://blog.ianpreston.ca/2020/05/13/conda_envs.html for working some of this out!

# Oneshell means all lines in a recipe run in the same shell
.ONESHELL:

# Need to specify bash in order for conda activate to work
SHELL=/bin/bash

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Define the name of the conda environment
CONDA_ENV=adl-hw1-2024

all: conda-env-update pip-compile pip-sync

# Create or update conda environment
conda-env-update:
	conda env update --prune

# Compile exact pip packages from requirements.in
pip-compile:
	$(CONDA_ACTIVATE) $(CONDA_ENV)
	pip-compile -v requirements.in

# Install pip packages from requirements.txt
pip-sync:
	$(CONDA_ACTIVATE) $(CONDA_ENV)
	pip-sync requirements.txt
