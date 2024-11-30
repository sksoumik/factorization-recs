# Matrix Factorization Based Recommender System

A recommender system implementation using LightFM, designed for experimentation and evaluation of different recommendation strategies. This provides a pipeline for generating synthetic data, training models, and evaluating performance with various configurations.

## Features

- **Hybrid Recommendation Model**
  - Matrix factorization with LightFM
  - Support for user and item features
  - Multiple loss functions (WARP, BPR, etc.)

- **Synthetic Data Generation**
  - Realistic user profiles and demographics
  - Item features and characteristics
  - Configurable interaction patterns

- **Comprehensive Evaluation**
  - Cross-validation support
  - Multiple evaluation metrics
  - Statistical analysis of results
  - MLflow experiment tracking

- **Experiment Management**
  - Configuration-based experiments
  - Result persistence and analysis
  - Automated reporting
  - Experiment comparison and visualization

## Project Structure

```
.
├── data/                   # Data generation and processing
├── evaluation/             # Evaluation metrics and tools
├── examples/               # Example scripts and notebooks
├── models/                 # Model implementations
├── schemas/                # Data schemas and validation
├── utils/                  # Utility functions and logging
├── mlruns/                 # MLflow tracking directory
└── config.yaml             # Default configuration
```

## Installation

### Prerequisites

- Python 3.11+
- Conda (recommended for environment management)
- Make (for using Makefile commands)

### Setup Using Conda (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd factorization-recs
   ```

2. Create and setup conda environment:
   ```bash
   make setup-conda
   ```

3. Activate the environment:
   ```bash
   conda activate recs
   ```

4. Install pre-commit hooks (for development):
   ```bash
   make setup-pre-commit
   ```

### Quick Setup

For a complete setup including conda environment and pre-commit hooks:
```bash
make setup
```

## Running Experiments

### Configuration

1. Review and modify `config.yaml` for experiment parameters:
   ```yaml
   model:
     learning_rate: 0.05
     loss: "warp"
     no_components: 64
     ...

   training:
     num_epochs: 10
     num_threads: 4
     ...
   ```

2. Run experiments:
   ```python
   python examples/run_experiments.py
   ```

### Experiment Output

Results are saved in `experiment_results/` with the following structure:
```
experiment_results/
├── default/
│   ├── config.yaml           # Experiment configuration
│   ├── results.json          # Detailed results
│   └── synthetic_data/       # Generated datasets
│       ├── users.csv
│       ├── items.csv
│       └── interactions.csv
├── high_lr/
│   └── ...
└── summary_results.csv       # Overall experiment summary
```

### Experiment Tracking

The project uses MLflow for experiment tracking and visualization:

1. Start the MLflow UI:
   ```bash
   make mlflow-ui
   ```

2. View experiments at [http://localhost:5001](http://localhost:5001)

MLflow tracks:
- Model parameters
- Training metrics
- Dataset statistics
- Performance metrics
- Artifacts (configs, results)

Compare experiments by:
- Parameter values
- Metric performance
- Cross-validation results
- Dataset characteristics

## Development

### Code Formatting

Format code using isort and black:
```bash
make format
```

Check formatting without making changes:
```bash
make check-format
```

### Linting

Run all linting checks:
```bash
make check-lint
```

Individual linting tools:
```bash
make lint-flake8     # Run flake8
make lint-pylint     # Run pylint
```

### Type Checking

Run mypy type checks:
```bash
make test-mypy
```

Check for missing type hints:
```bash
make check-missing-type
```

### Pre-commit Hooks

Run pre-commit checks:
```bash
make test-pre-commit     # Test pre-commit hooks
make test-pre-push      # Test pre-push hooks
```

## Cleaning Up

Remove generated files and caches:
```bash
make clean              # Clean all temporary files and results
make clean-conda        # Remove conda environment
```
