# Approximate Vector Set Search: A Bio-Inspired Approach for High-Dimensional Spaces

## Overview

This project, "Approximate Vector Set Search: A Bio-Inspired Approach for High-Dimensional Spaces," presents a novel method inspired by biological processes to address the challenges of searching in high-dimensional vector spaces. The approach leverages advanced computational techniques to improve the efficiency and accuracy of approximate vector set searches, a critical task in various domains such as bioinformatics, machine learning, and data mining.

CS vector sample dataset ([Link](https://pan.quark.cn/s/eae45745d099), Access Code: Wzsb). Additional data available from public sources.

## Repository Structure

The repository is organized into several key directories, each containing specific components of the project:

- `main.py` and `utils.py`: Core scripts for running the main algorithm and utility functions.
- `ComparativeExperiment`: Contains scripts and modules for comparative experiments to evaluate the proposed method against existing approaches.
- `_NaiveBioVSS`: Includes a naive implementation of the BioVSS algorithm and related scripts.
- `C++code`: Houses C++ implementations and CUDA code for performance-critical components.
- `data`: Contains benchmark datasets used for testing and evaluation.
- `logs`: Logging utilities for tracking experiments and results.
- `src`: Core source code including data loading, filtering, and refinement modules.

## Key Components

### Main Algorithm
- **`main.py`**: Entry point for executing the approximate vector set search algorithm.
- **`utils.py`**: Utility functions supporting the main algorithm.

### Comparative Experiments
- **`ComparativeExperiment/CompareBase.py`**: Base class for comparative experiments.
- **`ComparativeExperiment/main_c.py`**: Script to execute comparative experiments.
- **`ComparativeExperiment/CompareMethod`**: Directory containing various comparison methods such as `BruceExactHausdorff`, `IndexHNSWMean`, `IndexIVFFlatMean`, etc.
- **`ComparativeExperiment/DataLoader`**: Data loading utilities for comparative experiments.
- **`ComparativeExperiment/logs/Logger.py`**: Logging utilities for experiments.

### Naive Implementation
- **`_NaiveBioVSS/hausdorff_distance_naive_lsh.cpp`**: C++ implementation of naive Hausdorff distance calculation using LSH.
- **`_NaiveBioVSS/main.py`**: Main script for running the naive implementation.
- **`_NaiveBioVSS/setup.py`**: Setup script for the naive implementation.
- **`_NaiveBioVSS/test.py`**: Testing script for the naive implementation.

### C++ and CUDA Code
- **`C++code/cudaDenseHausdorff`**: Directory containing CUDA implementations for dense Hausdorff distance calculations.
- **`C++code/cudaDenseHausdorff/haudorff`**: Subdirectory for basic Hausdorff distance calculations.
- **`C++code/cudaDenseHausdorff/haudorff_lsh`**: Subdirectory for LSH-based Hausdorff distance calculations.
- **`C++code/cudaDenseHausdorff/src`**: Source code for additional implementations.

### Data and Logging
- **`data/benchmark/CS/test`**: Test datasets for benchmarking.
- **`logs/Logger.py`**: Logging utilities for the main algorithm.

### Source Code
- **`src/Pipeline.py`**: Pipeline for executing the vector set search.
- **`src/DataLoader`**: Data loading modules for the main algorithm.
- **`src/Filter`**: Filtering modules including `BloomGraph`, `IndexCount`, and `OverlapVector`.
- **`src/Refinement`**: Refinement modules including `ExactHausdorff` and `ParallelExactHausdorff`.

## Getting Started

### Prerequisites
- Python 3.10
- Required Python libraries (specified in `requirements.txt`)

### Installation
First download all the data and compile the C++ code.
second, complete all data preparation using BioHash.
1. Clone the repository:
   ```bash
   git clone https://github.com/CS-BruceChen/biovss.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BioVSS
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Algorithm
To execute the main algorithm, run:
```bash
python main.py
```

## Experimental Evaluation
Detailed instructions for running comparative experiments can be found in the `ComparativeExperiment` directory. 

## Contributing
Contributions are welcome. Please submit a pull request or contact the authors for major changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
This work was supported by [Your Funding Source]. We thank our collaborators and contributors for their valuable input.
