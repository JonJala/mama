# `mama`
[`mama`](https://www.biorxiv.org/content/10.1101/2021.04.23.441003v1) is a Python-based command line tool that meta-analyzes GWAS summary statistics generated from distinct ancestry groups. For more details please see Turley et al. (2021) and [Turley et al. (2018)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5805593/).


### Getting started
You can clone the repository with 
```
$ git clone git@github.com:JonJala/mama.git
$ cd mama
```
The easiest way to ensure your libraries will be compatible with the dependencies in the software is to instantiate a virtual environment with [`virtualenv`](https://virtualenv.pypa.io/en/latest/). Once `virtualenv` is installed on your machine, you can type the following:
```
$ virtualenv -p $(which python3) mama_env
$ source mama_env/bin/activate 
$ pip install -r /path/to/mama/requirements.txt
```
To test proper installation, ensure that typing 
```
$ python3 ./mama.py -h
```
gives a description of the software and accepted command line flags. If an error is thrown then the installation was unsuccessful. 

### Updating `mama`
You should keep your local instance of this software up to date with updates that are made on the github repository. To do that, type 
```
$ git pull
```
in the `mama` directory. If your local instance is outdated, `git` will retrieve all changes and update the code. Otherwise, you will be told that your local instance is already up to date. 

### Support
We are happy to answer any questions you may have about using the software. Before [opening an issue](https://github.com/JonJala/mama/issues), please be sure to read the wiki, description of the method in the papers linked above, and the description of the input flags and their proper usage. If your problem persists, **please do the following:**

  1. Rerun the specification that is causing the error, being sure to specify `--verbose` to generate a descriptive logfile. 
  2. Attach your log file in the issue. 
  
You may also contact us via email, although we encourage github issues so others can benefit from your question as well!    

### Citation

If you are using the `mama` method or software, please cite Turley, Patrick et al. "Multi-Ancestry Meta-Analysis yields novel genetic discoveries and ancestry-specific associations". *bioRxiv*. (2021).
