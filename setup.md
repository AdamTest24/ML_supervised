---
title: Setup
---

## Setting up virtual environment
In Python, the use of virtual environments allows you to avoid installing Python packages globally which could break system tools or other projects.  Each virtual environment has its own Python binary (which matches the version of the binary that was used to create this environment) and can have its own independent set of installed Python packages in its site directories.

A virtual environment can be created by executing the command `venv` in your Terminal (Mac OS and Unix) or at the command line prompt (Windows):

```
python3 -m venv pyML
```

By running this command a new environment will be installed at your home directory.

The environment can be activated as:

```
source pyML/bin/activate 
```

Now the packages required for machine learning can be installed as:

```
pip3 install pandas scikit-learn matplotlib nibabel
```

This environment kernel needs to be added to your Jupyter notebook. This can be done as:

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=firstEnv
```

After running these 2 commands, you will be able to select your virtual environment from the `Kernel` tab of your Jupyter notebook. More information can be accessed at this [link](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084).

## Dataset
Dataset for this lesson includes:

- [patients_data.csv](data/patients_data.csv)
- [breast_cancer.csv](data/breast_cancer.csv)
