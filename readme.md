# HallAgent4Rec Documentation

## Table of Contents
1. [hallagent4rec.py](#hallagent4recpy)
2. [experimental_framework.py](#experimental_frameworkpy)
3. [utilities.py](#utilitiespy)
4. [baseline_integration.py](#baseline_integrationpy)
5. [comparative_experiment.py](#comparative_experimentpy)
6. [main_experiment.py](#main_experimentpy)
7. [MovieLens Extensions](#movielens-extensions)

---

<a id="hallagent4recpy"></a>
## 1. hallagent4rec.py

### Purpose
Core implementation of the HallAgent4Rec model, which enhances generative agents for recommendation systems through hallucination mitigation and computational efficiency.

### Key Components

#### HallAgent4Rec Class
The main class that implements the hallucination-aware recommendation framework combining generative agents, probabilistic matrix factorization, and retrieval-augmented generation.

```python
def __init__(
    self,
    num_clusters: int = 10,
    latent_dim: int = 20,
    lambda_u: float = 0.1,
    lambda_v: float = 0.1,
    lambda_h: float = 1.0,
    learning_rate: float = 0.01,
    decay_rate: float = 0.0001,
    max_iterations: int = 100,
    similarity_threshold: float = 0.5,
    relevance_threshold: float = 0.1,
):
    """
    Initialize the HallAgent4Rec system.
    
    Args:
        num_clusters: Number of clusters for item grouping
        latent_dim: Dimensionality of user and item embeddings
        lambda_u: User regularization coefficient
        lambda_v: Item regularization coefficient
        lambda_h: Hallucination penalty coefficient
        learning_rate: Initial learning rate for optimization
        decay_rate: Learning rate decay parameter
        max_iterations: Maximum number of iterations for matrix factorization
        similarity_threshold: Threshold for item similarity in retrieval
        relevance_threshold: Threshold for item relevance in knowledge base
    """
```

### Core Methods

#### Data Loading and Preprocessing
```python
def load_data(self, user_data: pd.DataFrame, item_data: pd.DataFrame, interactions: pd.DataFrame)
def create_interaction_matrix(self)
```
Loads and formats data for the model, converting user-item interactions into an interaction matrix.

#### Item Clustering
```python
def cluster_items(self)
def create_user_cluster_matrix(self)
```
Implements item clustering to group similar items, reducing the recommendation space while preserving relevance.

#### Matrix Factorization
```python
def matrix_factorization(self)
def compute_training_error(self)
```
Performs hallucination-aware matrix factorization to learn user and item latent factors, incorporating a hallucination penalty term in the optimization objective.

#### Agent Management
```python
def initialize_agents(self)
def add_selective_memories(self, agent, user_id, max_memories=10, strategy='diverse')
```
Initializes generative agents for each user and manages agent memory with various strategies to reduce API calls.

#### RAG Mechanism
```python
def construct_rag_query(self, user_id)
def construct_knowledge_base(self, user_id)
def retrieve_items(self, user_id, query_embedding, knowledge_base)
```
Implements the retrieval-augmented generation pipeline for hallucination mitigation, ensuring recommendations are grounded in factual data.

#### Recommendation Generation
```python
def generate_recommendations(self, user_id, num_recommendations=5)
```
Generates recommendations for a specific user using the full HallAgent4Rec methodology.

#### Training and Evaluation
```python
def train(self)
def evaluate(self, test_interactions, metrics=['precision', 'recall', 'hallucination_rate'])
```
Trains the complete model and evaluates its performance on test data.

### Dependencies
- numpy, pandas, faiss
- sklearn (for clustering and similarity metrics)
- langchain (for generative agent components)
- langchain_google_genai (for LLM)

### Usage Example
```python
# Initialize HallAgent4Rec
hall_agent = HallAgent4Rec(num_clusters=15, latent_dim=20)

# Load data
hall_agent.load_data(user_data, item_data, interactions)

# Train the system
hall_agent.train()

# Generate recommendations for a user
recommendations = hall_agent.generate_recommendations(user_id=1, num_recommendations=5)
```

---

<a id="experimental_frameworkpy"></a>
## 2. experimental_framework.py

### Purpose
Implements the experimental evaluation framework for HallAgent4Rec, providing methodologies for sampling, surrogate metrics, statistical validation, and ablation studies.

### Key Components

#### ExperimentalFramework Class
Main class that encapsulates all experimental methodologies for evaluating HallAgent4Rec.

```python
def __init__(self, hall_agent):
    """
    Initialize the experimental framework with a trained HallAgent4Rec instance.
    
    Args:
        hall_agent: A trained HallAgent4Rec instance
    """
```

### Core Methods

#### Diversity Measurement
```python
def compute_icrd_score(self, cluster_id: int, recommendations_by_user: Dict[int, List[Dict]]) -> float
```
Computes the Intra-Cluster Recommendation Diversity (ICRD) score for a cluster as defined in equation 28 of the paper.

#### Sampling Methodology
```python
def stratified_sampling(self, n_samples: int = 50, random_state: int = 42) -> List[int]
```
Implements stratified sampling to select a representative subset of users based on their activity level and preference diversity.

#### Surrogate Metrics
```python
def compute_surrogate_metrics(self, user_id: int, recommendations: List[Dict]) -> Dict[str, float]
```
Computes surrogate metrics (Grounding Violation Rate, Retrieval-Prediction Alignment, Cluster Consistency) for a user's recommendations as defined in Section 4.2 of the paper.

#### Scaling Analysis
```python
def progressive_scaling_analysis(self, sample_sizes: List[int], num_recommendations: int = 10, random_state: int = 42)
```
Performs progressive scaling analysis to establish statistical stability by evaluating metrics on increasingly larger sample sizes.

#### Statistical Validation
```python
def compute_effect_size(self, metric: str, baseline: str, experiment: str, sample_size: int) -> float
def statistical_power_analysis(self, sample_sizes: List[int], baseline: str, experiment: str, metric: str, alpha: float = 0.05)
def correlation_analysis(self, sampled_users: List[int], full_users: List[int], metrics: List[str])
```
Provides statistical validation tools including effect size calculation, power analysis, and correlation analysis between sample and full dataset metrics.

#### Bootstrap Validation
```python
def bootstrap_validation(self, sampled_users: List[int], metric: str, n_iterations: int = 1000)
```
Performs bootstrap resampling to estimate confidence intervals for metrics on the full dataset based on a sample.

#### Ablation Studies
```python
def perform_ablation_studies(self, sampled_users: List[int], num_recommendations: int = 10)
def _create_no_clustering_variant(self)
def _create_no_rag_variant(self)
def _create_no_hallucination_reg_variant(self)
```
Implements ablation studies by removing key components of HallAgent4Rec and evaluating the impact on performance.

#### Visualization
```python
def plot_progressive_scaling_results(self)
def plot_ablation_results(self)
def plot_bootstrap_results(self)
```
Creates visualizations of experimental results for progressive scaling, ablation studies, and bootstrap validation.

#### Complete Evaluation
```python
def run_full_evaluation(self, n_samples: int = 50, random_state: int = 42)
```
Runs the complete experimental evaluation pipeline as described in the paper.

### Dependencies
- numpy, pandas
- scipy.stats (for statistical analysis)
- matplotlib, seaborn (for visualization)
- sklearn.metrics (for evaluation metrics)

### Usage Example
```python
# Create experimental framework from trained HallAgent4Rec instance
exp_framework = ExperimentalFramework(hall_agent)

# Perform stratified sampling
sampled_users = exp_framework.stratified_sampling(n_samples=50)

# Run complete evaluation
evaluation_results = exp_framework.run_full_evaluation()

# Visualize results
exp_framework.plot_progressive_scaling_results()
exp_framework.plot_ablation_results()
```

---

<a id="utilitiespy"></a>
## 3. utilities.py

### Purpose
Provides utility functions for data generation, analysis, and result handling in the experimental framework.

### Key Functions

#### Sample Data Generation
```python
def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
def load_or_create_frappe_sample() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
def load_or_create_musicincar_sample() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
```
Creates synthetic datasets for testing and experimentation, mimicking the structure of Frappe and MusicInCar datasets.

#### Dataset Generation
```python
def generate_synthetic_dataset(dataset_name, num_users, num_items, num_interactions)
def generate_synthetic_frappe(num_users, num_items, num_interactions)
def generate_synthetic_musicincar(num_users, num_items, num_interactions)
```
Generates scalable synthetic datasets with specified numbers of users, items, and interactions.

#### Result Management
```python
def save_results(results, filename)
def load_results(filename)
```
Saves and loads experimental results in both pickle and JSON formats for future analysis.

#### Analysis and Visualization
```python
def plot_convergence_analysis(metrics_by_sample_size, title="Convergence of Metrics with Sample Size")
def analyze_user_distributions(user_data, item_data, interactions)
def plot_user_distributions(distribution_stats, title="User Characteristic Distributions")
```
Analyzes and visualizes data distributions and experimental results.

### Dependencies
- numpy, pandas
- matplotlib, seaborn (for visualization)
- datetime, random (for data generation)
- pickle, json (for result serialization)

### Usage Example
```python
# Generate synthetic data
user_data, item_data, interactions, test_interactions = load_or_create_frappe_sample()

# Analyze user distributions
dist_stats = analyze_user_distributions(user_data, item_data, interactions)
plot_user_distributions(dist_stats)

# Save experimental results
save_results(experiment_results, "frappe_experiment_results")

# Load previously saved results
previous_results = load_results("frappe_experiment_results")
```

---

<a id="baseline_integrationpy"></a>
## 4. baseline_integration.py

### Purpose
Integrates RecBole baseline recommendation models for comparative evaluation with HallAgent4Rec, ensuring fair comparison between traditional and generative approaches.

### Key Components

#### BaselineModels Class
Main class that provides a wrapper for RecBole models to make them compatible with the HallAgent4Rec experimental framework.

```python
def __init__(self, data_path: str = './data'):
    """
    Initialize the BaselineModels wrapper.
    
    Args:
        data_path: Path to store RecBole formatted data
    """
```

### Core Methods

#### Data Conversion
```python
def convert_data_to_recbole_format(self, user_data: pd.DataFrame, item_data: pd.DataFrame, 
                                 interactions: pd.DataFrame, test_interactions: pd.DataFrame = None) -> str
```
Converts data from HallAgent4Rec format to RecBole's required format.

#### Data Preparation
```python
def load_and_prepare_data(self, dataset_name: str, config_dict: Dict = None)
```
Loads and prepares the RecBole dataset with appropriate configuration.

#### Model Training
```python
def train_model(self, model_name: str, param_dict: Dict = None)
```
Trains a RecBole model with specified parameters. Supports models like BPR (PMF), NeuMF (NMF), and LightGCN.

#### Evaluation
```python
def evaluate_model(self, model_name: str, metrics: List[str] = None, topk: List[int] = None)
def compare_models(self, model_names: List[str], metrics: List[str] = None, topk: List[int] = None)
```
Evaluates trained models and compares multiple models using consistent metrics.

#### Recommendation Generation
```python
def get_recommendations(self, model_name: str, user_ids: List, top_k: int = 10) -> Dict[int, List[int]]
```
Generates recommendations for specific users from a trained model.

#### Hallucination Detection
```python
def detect_hallucination(self, model_name: str, recommendations: Dict[int, List[int]], 
                      item_data: pd.DataFrame) -> Dict[int, List[bool]]
```
Detects hallucinations in recommendations, providing a consistent methodology for comparison with HallAgent4Rec.

#### Comparative Evaluation
```python
def compare_with_hallagent4rec(self, hall_agent, sampled_users: List[int], 
                              baseline_models: List[str], top_k: int = 10)
```
Compares traditional RecBole models with HallAgent4Rec using the same evaluation protocol.

#### Visualization
```python
def visualize_comparison(self, comparison_df: pd.DataFrame, metric_filter: List[str] = None)
def visualize_comparison_with_hallagent4rec(self, comparison_df: pd.DataFrame)
```
Creates visualizations to compare models across various metrics.

### Dependencies
- RecBole (for baseline models)
- torch (for PyTorch-based models)
- pandas, numpy
- matplotlib, seaborn (for visualization)

### Usage Example
```python
# Initialize BaselineModels
baseline = BaselineModels()

# Convert data to RecBole format
dataset_name = baseline.convert_data_to_recbole_format(user_data, item_data, interactions, test_interactions)

# Load and prepare data
baseline.load_and_prepare_data(dataset_name)

# Train models
baseline.train_model('BPR')  # PMF equivalent
baseline.train_model('NeuMF')  # NMF equivalent
baseline.train_model('LightGCN')

# Compare with HallAgent4Rec
comparison_df = baseline.compare_with_hallagent4rec(
    hall_agent, 
    sampled_users, 
    ['BPR', 'NeuMF', 'LightGCN']
)

# Visualize comparison
baseline.visualize_comparison_with_hallagent4rec(comparison_df)
```

---

<a id="comparative_experimentpy"></a>
## 5. comparative_experiment.py

### Purpose
Provides a dedicated script to run comparative experiments between HallAgent4Rec and traditional baseline models from RecBole.

### Key Functions

#### Main Experiment
```python
def run_comparative_experiments(dataset_name, num_users=50, random_state=42, top_k=10)
```
Runs comprehensive comparative experiments between HallAgent4Rec and baseline models (PMF, NMF, LightGCN) on the specified dataset.

#### Ablation Comparison
```python
def run_ablation_comparison(dataset_name, num_users=20, random_state=42, top_k=10)
```
Runs ablation studies to compare different variants of HallAgent4Rec with a baseline model.

#### Visualization
```python
def plot_comparative_results(results, dataset_name)
def plot_ablation_comparison(results_df, dataset_name)
```
Creates visualizations for comparative experiments and ablation studies.

### Command-Line Interface
```python
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run comparative experiments for HallAgent4Rec')
    parser.add_argument('--dataset', type=str, default='frappe', choices=['frappe', 'musicincar'],
                        help='Dataset to use (frappe or musicincar)')
    parser.add_argument('--mode', type=str, default='comparative', 
                       choices=['comparative', 'ablation'],
                       help='Experiment mode')
    parser.add_argument('--users', type=int, default=50, 
                       help='Number of users to sample')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--topk', type=int, default=10, 
                       help='Number of recommendations to generate')
```
Provides a flexible command-line interface for running experiments with different configurations.

### Dependencies
- hallagent4rec (for the main model)
- experimental_framework (for evaluation methodologies)
- baseline_integration (for baseline models)
- utilities (for utility functions)
- pandas, numpy, torch
- matplotlib, seaborn (for visualization)

### Usage Example
```bash
# Run comparative experiments on Frappe dataset
python comparative_experiment.py --dataset frappe --mode comparative --users 50 --topk 10

# Run ablation comparison on MusicInCar dataset
python comparative_experiment.py --dataset musicincar --mode ablation --users 20 --topk 10
```

---

<a id="main_experimentpy"></a>
## 6. main_experiment.py

### Purpose
Provides the main entry point for running the complete experimental pipeline for HallAgent4Rec as described in the paper.

### Key Functions

#### Full Experiment
```python
def run_experiments(dataset_name, num_users=50, random_state=42, ablation=True, 
                   progressive_scaling=True, bootstrap=True)
```
Runs the complete experimental pipeline including clustering, matrix factorization, agent initialization, sampling, surrogate metrics, progressive scaling, bootstrap validation, and ablation studies.

#### Scaling Experiments
```python
def run_scaling_experiments(dataset_name, sample_sizes=[10, 20, 30, 40, 50], random_state=42)
```
Runs experiments with progressively increasing sample sizes to analyze performance stability.

#### Ablation Studies
```python
def run_ablation_experiments(dataset_name, num_users=20, random_state=42)
```
Runs ablation studies to evaluate the contribution of different components of HallAgent4Rec.

#### Bootstrap Validation
```python
def run_bootstrap_validation(dataset_name, num_users=50, random_state=42, n_iterations=1000)
```
Runs bootstrap validation to estimate confidence intervals for metrics on the full dataset.

### Command-Line Interface
```python
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run experiments for HallAgent4Rec')
    parser.add_argument('--dataset', type=str, default='frappe', choices=['frappe', 'musicincar'],
                        help='Dataset to use (frappe or musicincar)')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'scaling', 'ablation', 'bootstrap'],
                       help='Experiment mode')
    parser.add_argument('--users', type=int, default=50, 
                       help='Number of users to sample')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--iterations', type=int, default=1000, 
                       help='Number of bootstrap iterations')
```
Provides a flexible command-line interface for running different experiments.

### Dependencies
- hallagent4rec (for the main model)
- experimental_framework (for evaluation methodologies)
- baseline_integration (for baseline models)
- utilities (for utility functions)
- pandas, numpy, torch
- matplotlib, seaborn (for visualization)

### Usage Example
```bash
# Full experiment suite on the Frappe dataset
python main_experiment.py --dataset frappe --mode full --users 50

# Progressive scaling analysis on MusicInCar dataset
python main_experiment.py --dataset musicincar --mode scaling --users 50

# Ablation studies with 20 users
python main_experiment.py --dataset frappe --mode ablation --users 20

# Bootstrap validation with 1000 iterations
python main_experiment.py --dataset frappe --mode bootstrap --users 50 --iterations 1000
```

---

<a id="movielens-extensions"></a>
## 7. MovieLens Extensions

### 7.1 movielens_loader.py

#### Purpose
Handles loading and preprocessing the MovieLens-100K dataset for use with HallAgent4Rec.

#### Key Functions
- `download_movielens_100k(data_path)`: Downloads the dataset if not available
- `load_movielens_100k(data_path)`: Loads and formats the dataset
- `split_train_test(interactions, test_ratio)`: Splits interactions into training and testing sets
- `create_movie_prompts(movie_data)`: Creates detailed prompts for each movie
- `create_user_profile_for_llm(user_id, user_data, interactions, item_data)`: Creates comprehensive user profiles

### 7.2 movielens_adapter.py

#### Purpose
Extends HallAgent4Rec with MovieLens-specific functionality.

#### Key Components
- `MovieLensAgent4Rec` class: Extends HallAgent4Rec with movie-specific functionality
- Movie-specific recommendation generation
- Enhanced evaluation metrics for movie recommendations
- Personalized explanations for recommended movies

### 7.3 movielens_experiment.py

#### Purpose
Provides scripts for running experiments on the MovieLens dataset.

#### Key Functions
- `run_movielens_experiments(num_users, random_state, top_k)`: Runs comprehensive experiments
- `run_progressive_scaling_analysis(max_users, step, random_state)`: Analyzes metric stability
- `run_memory_strategy_comparison(num_users, random_state)`: Compares memory selection strategies
- Visualization functions for MovieLens-specific experiments

### Usage Example
```bash
# Full experimental evaluation with 50 users
python movielens_experiment.py --mode full --users 50

# Progressive scaling analysis
python movielens_experiment.py --mode scaling --users 50

# Memory strategy comparison
python movielens_experiment.py --mode memory --users 30
```