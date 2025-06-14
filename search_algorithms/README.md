The search algorithms require you to have conda installed on your machine. 
The reason is that the search algorithms are using graphviz for visualization of decision trees, which in turn requires dot executable to be installed on your machine.
To avoid installing graphviz manually, we are using conda to install it for you in conda virtual environment.

Installing conda on your machine is easy, you can follow the instructions from the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### To run the search algorithms, you can use the following commands:

```commandline
conda env create <your_environment_name>
conda activate <your_environment_name>
conda install graphviz
python -m search_algorithms.<path>.<script_name>
```

# Example for running the MinMax algorithm:
```commandline
conda env create search_algorithms
conda activate search_algorithms
conda install graphviz
python -m search_algorithms.minmax.main
```

# Example for running Markov Decision Process:
```commandline
conda activate search_algorithms
conda install numpy
python -m search_algorithms.markov_decision_processes.main
```


