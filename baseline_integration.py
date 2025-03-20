import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# RecBole imports
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, NeuMF, LightGCN, GCMC, NGCF, SimpleX
from recbole.model.context_aware_recommender import NFM, DeepFM, WideDeep
from recbole.model.knowledge_aware_recommender import KGAT, KTUP, CFKG
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model, get_trainer, set_color

class BaselineModels:
    """
    A wrapper class to integrate RecBole baseline models for comparison with HallAgent4Rec.
    """
    
    def __init__(self, data_path: str = './data'):
        """
        Initialize the BaselineModels wrapper.
        
        Args:
            data_path: Path to store RecBole formatted data
        """
        self.data_path = data_path
        self.models = {}
        self.config = None
        self.dataset = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.results = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
    
    def convert_data_to_recbole_format(self, user_data: pd.DataFrame, item_data: pd.DataFrame, 
                                     interactions: pd.DataFrame, test_interactions: pd.DataFrame = None) -> str:
        """
        Convert data from HallAgent4Rec format to RecBole format.
        
        Args:
            user_data: DataFrame with user information
            item_data: DataFrame with item information
            interactions: DataFrame with user-item interactions
            test_interactions: Optional DataFrame with test interactions
            
        Returns:
            Path to the RecBole dataset folder
        """
        print("Converting data to RecBole format...")
        
        # Create dataset specific folder
        dataset_name = 'custom_dataset'
        dataset_path = os.path.join(self.data_path, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # 1. Create user.inter file (interactions)
        # RecBole requires user-item interactions with specific format
        inter_data = interactions.copy()
        inter_data.rename(columns={
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'timestamp': 'timestamp:float'
        }, inplace=True)
        
        # Add a rating column if not present (implicit feedback as 1.0)
        if 'rating' not in inter_data.columns:
            inter_data['rating:float'] = 1.0
        else:
            inter_data.rename(columns={'rating': 'rating:float'}, inplace=True)
            
        # Save interaction file
        inter_data.to_csv(os.path.join(dataset_path, f'{dataset_name}.inter'), sep='\t', index=False)
        
        # 2. Create user.user file (user features)
        if user_data is not None:
            user_feature_data = user_data.copy()
            # Rename ID column and add type indicators to column names
            user_feature_columns = {}
            for col in user_feature_data.columns:
                if col == 'user_id':
                    user_feature_columns[col] = 'user_id:token'
                # Detect column type and add appropriate suffix
                elif user_feature_data[col].dtype == np.float64 or user_feature_data[col].dtype == np.float32:
                    user_feature_columns[col] = f'{col}:float'
                elif user_feature_data[col].dtype == np.int64 or user_feature_data[col].dtype == np.int32:
                    user_feature_columns[col] = f'{col}:token'
                else:
                    user_feature_columns[col] = f'{col}:token'
            
            user_feature_data.rename(columns=user_feature_columns, inplace=True)
            user_feature_data.to_csv(os.path.join(dataset_path, f'{dataset_name}.user'), sep='\t', index=False)
        
        # 3. Create item.item file (item features)
        if item_data is not None:
            item_feature_data = item_data.copy()
            # Rename ID column and add type indicators to column names
            item_feature_columns = {}
            for col in item_feature_data.columns:
                if col == 'item_id':
                    item_feature_columns[col] = 'item_id:token'
                # Detect column type and add appropriate suffix
                elif item_feature_data[col].dtype == np.float64 or item_feature_data[col].dtype == np.float32:
                    item_feature_columns[col] = f'{col}:float'
                elif item_feature_data[col].dtype == np.int64 or item_feature_data[col].dtype == np.int32:
                    item_feature_columns[col] = f'{col}:token'
                else:
                    item_feature_columns[col] = f'{col}:token'
            
            item_feature_data.rename(columns=item_feature_columns, inplace=True)
            item_feature_data.to_csv(os.path.join(dataset_path, f'{dataset_name}.item'), sep='\t', index=False)
        
        # 4. Split train/valid/test if test interactions are provided
        if test_interactions is not None:
            # Create a copy of the test interactions file
            test_data = test_interactions.copy()
            test_data.rename(columns={
                'user_id': 'user_id:token',
                'item_id': 'item_id:token'
            }, inplace=True)
            
            # Add a rating column if not present
            if 'rating' not in test_data.columns:
                test_data['rating:float'] = 1.0
            else:
                test_data.rename(columns={'rating': 'rating:float'}, inplace=True)
            
            # Add timestamp if not present
            if 'timestamp' not in test_data.columns:
                test_data['timestamp:float'] = interactions['timestamp'].max() + 1
            else:
                test_data.rename(columns={'timestamp': 'timestamp:float'}, inplace=True)
            
            # Save to test file
            test_data.to_csv(os.path.join(dataset_path, f'{dataset_name}.test'), sep='\t', index=False)
        
        print(f"Data converted to RecBole format and saved to {dataset_path}")
        return dataset_name
    
    def load_and_prepare_data(self, dataset_name: str, config_dict: Dict = None):
        """
        Load and prepare the RecBole dataset.
        
        Args:
            dataset_name: Name of the dataset in RecBole format
            config_dict: Additional configuration parameters for RecBole
        """
        print("Loading and preparing data in RecBole...")
        
        # Default configuration
        default_config = {
            'data_path': self.data_path,
            'dataset': dataset_name,
            'load_col': None,
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'RATING_FIELD': 'rating',
            'TIME_FIELD': 'timestamp',
            # Training settings
            'epochs': 20,
            'train_batch_size': 2048,
            'eval_batch_size': 2048,
            'learning_rate': 0.001,
            'eval_step': 1,
            'stopping_step': 3,
            # Evaluation settings
            'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
            'topk': [5, 10, 20],
            'valid_metric': 'MRR@10',
            # Splitting
            'eval_args': {
                'split': {'RS': [0.8, 0.1, 0.1]},
                'order': 'RO',
                'mode': 'full'
            }
        }
        
        # Update with custom config if provided
        if config_dict:
            default_config.update(config_dict)
        
        # Initialize RecBole config
        self.config = Config(model='BPR', dataset=dataset_name, config_dict=default_config)
        
        # Create dataset
        self.dataset = create_dataset(self.config)
        
        # Data preparation
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)
        
        print(f"Dataset loaded: {self.dataset.dataset_name}")
        print(f"Number of users: {self.dataset.user_num}")
        print(f"Number of items: {self.dataset.item_num}")
        print(f"Number of interactions: {self.dataset.inter_num}")
    
    def train_model(self, model_name: str, param_dict: Dict = None):
        """
        Train a RecBole model.
        
        Args:
            model_name: Name of the RecBole model to train
            param_dict: Optional model-specific parameters
            
        Returns:
            Trained model
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_and_prepare_data first.")
        
        print(f"Training {model_name} model...")
        
        # Update config with model-specific parameters
        if param_dict:
            for key, value in param_dict.items():
                self.config[key] = value
        
        # Initialize model
        if model_name == 'PMF':
            # PMF is implemented as BPR with specific parameters
            model = BPR(self.config, self.train_data.dataset)
            self.config['negative_sampling'] = None
        elif model_name == 'NMF':
            # Use NeuMF with specific parameters for NMF behavior
            model = NeuMF(self.config, self.train_data.dataset)
        elif model_name == 'LightGCN':
            model = LightGCN(self.config, self.train_data.dataset)
        else:
            # Get the model class dynamically for other models
            model_class = get_model(model_name)
            model = model_class(self.config, self.train_data.dataset)
        
        # Train the model
        trainer = Trainer(self.config, model)
        trainer.fit(self.train_data, self.valid_data)
        
        # Store the trained model
        self.models[model_name] = model
        
        return model
    
    def evaluate_model(self, model_name: str, metrics: List[str] = None, topk: List[int] = None):
        """
        Evaluate a trained model.
        
        Args:
            model_name: Name of the model to evaluate
            metrics: List of metrics to compute
            topk: List of top-k values for evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained. Call train_model first.")
        
        print(f"Evaluating {model_name} model...")
        
        # Set evaluation metrics if provided
        if metrics:
            self.config['metrics'] = metrics
        if topk:
            self.config['topk'] = topk
        
        # Get the trainer
        model = self.models[model_name]
        trainer = Trainer(self.config, model)
        
        # Evaluate on test set
        test_result = trainer.evaluate(self.test_data)
        
        # Store results
        self.results[model_name] = test_result
        
        return test_result
    
    def compare_models(self, model_names: List[str], metrics: List[str] = None, topk: List[int] = None):
        """
        Compare multiple trained models.
        
        Args:
            model_names: List of model names to compare
            metrics: List of metrics to compare
            topk: List of top-k values to compare
            
        Returns:
            DataFrame of comparison results
        """
        if not metrics:
            metrics = ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
        if not topk:
            topk = [5, 10, 20]
        
        print("Comparing models...")
        
        # Evaluate all models if not already evaluated
        for model_name in model_names:
            if model_name not in self.results:
                self.evaluate_model(model_name, metrics, topk)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name in model_names:
            results = self.results[model_name]
            for metric in metrics:
                for k in topk:
                    key = f"{metric}@{k}"
                    if key in results:
                        comparison_data.append({
                            'Model': model_name,
                            'Metric': key,
                            'Value': results[key]
                        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def visualize_comparison(self, comparison_df: pd.DataFrame, metric_filter: List[str] = None):
        """
        Visualize model comparison results.
        
        Args:
            comparison_df: DataFrame from compare_models
            metric_filter: Optional list of metrics to include in visualization
            
        Returns:
            Matplotlib figure
        """
        # Filter metrics if specified
        if metric_filter:
            comparison_df = comparison_df[comparison_df['Metric'].isin(metric_filter)]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        sns.barplot(x='Metric', y='Value', hue='Model', data=comparison_df)
        
        plt.title('Model Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        plt.tight_layout()
        
        # Save figure
        plt.savefig('results/model_comparison.png', dpi=300)
        
        return plt.gcf()
    
    def get_recommendations(self, model_name: str, user_ids: List, top_k: int = 10) -> Dict[int, List[int]]:
        """
        Get recommendations for specific users from a trained model.
        
        Args:
            model_name: Name of the trained model
            user_ids: List of user IDs to get recommendations for
            top_k: Number of recommendations to generate per user
            
        Returns:
            Dictionary mapping user IDs to lists of recommended item IDs
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained. Call train_model first.")
        
        model = self.models[model_name]
        
        # Convert external user IDs to internal IDs
        internal_user_ids = []
        external_to_internal = {}
        
        for user_id in user_ids:
            try:
                internal_id = self.dataset.token2id(self.config['USER_ID_FIELD'], str(user_id))
                internal_user_ids.append(internal_id)
                external_to_internal[user_id] = internal_id
            except ValueError:
                print(f"User ID {user_id} not found in the dataset")
        
        # Get recommendations
        model.eval()
        with torch.no_grad():
            # Convert internal user IDs to tensor
            user_tensor = torch.tensor(internal_user_ids, dtype=torch.int64).to(model.device)
            
            # Get scores for all items
            scores = model.forward(user_tensor)
            
            # Get top-k item indices
            _, topk_indices = torch.topk(scores, k=top_k, dim=1)
            
            # Convert internal item IDs back to external IDs
            recommendations = {}
            for i, user_id in enumerate(user_ids):
                if user_id in external_to_internal:
                    internal_items = topk_indices[i].cpu().numpy()
                    external_items = [int(self.dataset.id2token(self.config['ITEM_ID_FIELD'], item_id)) 
                                     for item_id in internal_items]
                    recommendations[user_id] = external_items
        
        return recommendations
    
    def detect_hallucination(self, model_name: str, recommendations: Dict[int, List[int]], 
                          item_data: pd.DataFrame) -> Dict[int, List[bool]]:
        """
        Detect hallucinations in recommendations.
        
        Args:
            model_name: Name of the model
            recommendations: Dictionary of user recommendations
            item_data: DataFrame with valid item information
            
        Returns:
            Dictionary mapping user IDs to lists of boolean hallucination flags
        """
        valid_items = set(item_data['item_id'])
        hallucinations = {}
        
        for user_id, items in recommendations.items():
            hallucinations[user_id] = [item not in valid_items for item in items]
        
        return hallucinations
    
    def compare_with_hallagent4rec(self, hall_agent, sampled_users: List[int], 
                                  baseline_models: List[str], top_k: int = 10):
        """
        Compare traditional RecBole models with HallAgent4Rec.
        
        Args:
            hall_agent: Trained HallAgent4Rec instance
            sampled_users: List of user IDs to compare on
            baseline_models: List of baseline model names to compare
            top_k: Number of recommendations to generate
            
        Returns:
            DataFrame of comparison results
        """
        print(f"Comparing HallAgent4Rec with {', '.join(baseline_models)} on {len(sampled_users)} sampled users...")
        
        # Results storage
        comparison_results = []
        
        # Get HallAgent4Rec recommendations
        hall_recommendations = {}
        hall_hallucinations = {}
        hall_precision = []
        hall_recall = []
        
        # Items with valid interactions in the dataset
        valid_interactions = set(zip(
            hall_agent.interactions['user_id'],
            hall_agent.interactions['item_id']
        ))
        
        # Get ground truth for each user (items in test set)
        ground_truth = {}
        for user_id in sampled_users:
            user_test = hall_agent.interactions[hall_agent.interactions['user_id'] == user_id]['item_id'].tolist()
            ground_truth[user_id] = set(user_test)
        
        print("Generating HallAgent4Rec recommendations...")
        for user_id in tqdm(sampled_users):
            # Generate HallAgent4Rec recommendations
            recommendations = hall_agent.generate_recommendations(user_id, num_recommendations=top_k)
            if not recommendations:
                continue
                
            # Extract recommended item IDs
            rec_items = [item['item_id'] for item in recommendations]
            hall_recommendations[user_id] = rec_items
            
            # Detect hallucinations
            query, query_embedding = hall_agent.construct_rag_query(user_id)
            knowledge_base, _ = hall_agent.construct_knowledge_base(user_id)
            retrieved_items = hall_agent.retrieve_items(user_id, query_embedding, knowledge_base)
            
            retrieved_item_ids = [item['item_id'] for item in retrieved_items]
            hallucinations = [item_id not in retrieved_item_ids for item_id in rec_items]
            hall_hallucinations[user_id] = hallucinations
            
            # Calculate precision and recall
            if user_id in ground_truth:
                hits = len(set(rec_items) & ground_truth[user_id])
                precision = hits / len(rec_items) if rec_items else 0
                recall = hits / len(ground_truth[user_id]) if ground_truth[user_id] else 0
                
                hall_precision.append(precision)
                hall_recall.append(recall)
        
        # Calculate average hallucination rate for HallAgent4Rec
        hall_hallucination_rate = sum(sum(h) for h in hall_hallucinations.values()) / sum(len(h) for h in hall_hallucinations.values()) if hall_hallucinations else 0
        
        # Add HallAgent4Rec results to comparison
        comparison_results.append({
            'Model': 'HallAgent4Rec',
            'Precision': np.mean(hall_precision) if hall_precision else 0,
            'Recall': np.mean(hall_recall) if hall_recall else 0,
            'Hallucination_Rate': hall_hallucination_rate
        })
        
        # Get baseline model recommendations
        for model_name in baseline_models:
            if model_name not in self.models:
                print(f"Model {model_name} not trained. Skipping.")
                continue
                
            print(f"Generating {model_name} recommendations...")
            recommendations = self.get_recommendations(model_name, sampled_users, top_k)
            
            # Detect hallucinations
            hallucinations = self.detect_hallucination(model_name, recommendations, hall_agent.item_data)
            
            # Calculate metrics
            model_precision = []
            model_recall = []
            
            for user_id, items in recommendations.items():
                if user_id in ground_truth:
                    hits = len(set(items) & ground_truth[user_id])
                    precision = hits / len(items) if items else 0
                    recall = hits / len(ground_truth[user_id]) if ground_truth[user_id] else 0
                    
                    model_precision.append(precision)
                    model_recall.append(recall)
            
            # Calculate average hallucination rate
            hallucination_rate = sum(sum(h) for h in hallucinations.values()) / sum(len(h) for h in hallucinations.values()) if hallucinations else 0
            
            # Add to comparison results
            comparison_results.append({
                'Model': model_name,
                'Precision': np.mean(model_precision) if model_precision else 0,
                'Recall': np.mean(model_recall) if model_recall else 0,
                'Hallucination_Rate': hallucination_rate
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        return comparison_df
    
    def visualize_comparison_with_hallagent4rec(self, comparison_df: pd.DataFrame):
        """
        Visualize comparison between HallAgent4Rec and baseline models.
        
        Args:
            comparison_df: DataFrame from compare_with_hallagent4rec
            
        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot precision
        sns.barplot(x='Model', y='Precision', data=comparison_df, ax=axes[0])
        axes[0].set_title('Precision Comparison')
        axes[0].set_ylim(0, max(comparison_df['Precision']) * 1.2)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot recall
        sns.barplot(x='Model', y='Recall', data=comparison_df, ax=axes[1])
        axes[1].set_title('Recall Comparison')
        axes[1].set_ylim(0, max(comparison_df['Recall']) * 1.2)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot hallucination rate
        sns.barplot(x='Model', y='Hallucination_Rate', data=comparison_df, ax=axes[2])
        axes[2].set_title('Hallucination Rate Comparison')
        axes[2].set_ylim(0, max(comparison_df['Hallucination_Rate']) * 1.2)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig('results/hallagent_comparison.png', dpi=300)
        
        return fig


# Example usage
def example_usage():
    """Example of how to use the BaselineModels class."""
    from utilities import load_or_create_frappe_sample
    from hallagent4rec import HallAgent4Rec
    
    # Load sample data
    user_data, item_data, interactions, test_interactions = load_or_create_frappe_sample()
    
    # Initialize HallAgent4Rec
    hall_agent = HallAgent4Rec(num_clusters=3, latent_dim=10)
    hall_agent.load_data(user_data, item_data, interactions)
    hall_agent.train()
    
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
    
    # Compare models with HallAgent4Rec
    sampled_users = list(user_data['user_id'].sample(10))
    comparison_df = baseline.compare_with_hallagent4rec(
        hall_agent, 
        sampled_users, 
        ['BPR', 'NeuMF', 'LightGCN']
    )
    
    # Visualize comparison
    baseline.visualize_comparison_with_hallagent4rec(comparison_df)
    
    return comparison_df


if __name__ == "__main__":
    # Run example usage
    result = example_usage()
    print(result)