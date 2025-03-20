import os
import sys
import time
import datetime
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import torch

# Import the HallAgent4Rec class and experimental framework
from hallagent4rec import HallAgent4Rec
from experimental_framework import ExperimentalFramework
from baseline_integration import BaselineModels
from utilities import (
    load_or_create_frappe_sample, 
    load_or_create_musicincar_sample,
    save_results, 
    load_results,
    analyze_user_distributions
)

def run_comparative_experiments(dataset_name, num_users=50, random_state=42, top_k=10):
    """
    Run comparative experiments between HallAgent4Rec and RecBole baseline models.
    
    Args:
        dataset_name: Name of the dataset to use ('frappe' or 'musicincar')
        num_users: Number of users to sample for evaluation
        random_state: Random seed for reproducibility
        top_k: Number of recommendations to generate
    
    Returns:
        Dictionary of experimental results
    """
    print(f"=== Starting comparative experiments on {dataset_name.upper()} dataset ===")
    print(f"Sample size: {num_users} users, Random seed: {random_state}")
    start_time = time.time()
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    if dataset_name.lower() == 'frappe':
        user_data, item_data, interactions, test_interactions = load_or_create_frappe_sample()
    elif dataset_name.lower() == 'musicincar':
        user_data, item_data, interactions, test_interactions = load_or_create_musicincar_sample()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Dataset loaded: {len(user_data)} users, {len(item_data)} items, {len(interactions)} interactions")
    
    # 2. Initialize and train HallAgent4Rec
    print("\n2. Initializing and training HallAgent4Rec...")
    # Set parameters based on dataset
    if dataset_name.lower() == 'frappe':
        num_clusters = 15
    else:  # MusicInCar
        num_clusters = 10
    
    hall_agent = HallAgent4Rec(
        num_clusters=num_clusters,
        latent_dim=20,
        lambda_u=0.1,
        lambda_v=0.1,
        lambda_h=1.0,
        learning_rate=0.01,
        decay_rate=0.0001,
        max_iterations=50,  # Reduced for faster experiments
        similarity_threshold=0.5,
        relevance_threshold=0.1
    )
    
    # Load data
    hall_agent.load_data(user_data, item_data, interactions)
    
    # Train model
    print("Training HallAgent4Rec...")
    hall_agent.train()
    print("HallAgent4Rec training complete!")
    
    # 3. Create experimental framework
    print("\n3. Creating experimental framework...")
    exp_framework = ExperimentalFramework(hall_agent)
    
    # 4. Perform stratified sampling
    print("\n4. Performing stratified sampling...")
    sampled_users = exp_framework.stratified_sampling(n_samples=num_users, random_state=random_state)
    print(f"Sampled {len(sampled_users)} users for evaluation")
    
    # 5. Initialize baseline models
    print("\n5. Initializing baseline models...")
    baseline = BaselineModels()
    
    # Convert data to RecBole format
    dataset_name_recbole = baseline.convert_data_to_recbole_format(
        user_data, item_data, interactions, test_interactions)
    
    # Load and prepare data
    print("Loading and preparing data for RecBole...")
    baseline.load_and_prepare_data(dataset_name_recbole)
    
    # 6. Train baseline models
    print("\n6. Training baseline models...")
    baseline_models = []
    
    # Train PMF (Probabilistic Matrix Factorization)
    print("Training PMF model...")
    try:
        pmf_config = {
            'loss_type': 'square',  # Use square loss for PMF
            'embedding_size': 20,   # Match latent_dim in HallAgent4Rec
            'negative_sampling': None
        }
        baseline.train_model('BPR', pmf_config)  # Use BPR with square loss as PMF
        baseline_models.append('BPR')
    except Exception as e:
        print(f"Error training PMF model: {e}")
    
    # Train NMF (Non-negative Matrix Factorization)
    print("Training NMF model...")
    try:
        nmf_config = {
            'embedding_size': 20,
            'dropout_prob': 0.0
        }
        baseline.train_model('NeuMF', nmf_config)  # Use NeuMF as approximation of NMF
        baseline_models.append('NeuMF')
    except Exception as e:
        print(f"Error training NMF model: {e}")
    
    # Train LightGCN
    print("Training LightGCN model...")
    try:
        lightgcn_config = {
            'embedding_size': 20,
            'n_layers': 3
        }
        baseline.train_model('LightGCN', lightgcn_config)
        baseline_models.append('LightGCN')
    except Exception as e:
        print(f"Error training LightGCN model: {e}")
    
    # 7. Compare models with HallAgent4Rec
    print("\n7. Comparing models with HallAgent4Rec...")
    comparison_df = baseline.compare_with_hallagent4rec(
        hall_agent, 
        sampled_users, 
        baseline_models,
        top_k=top_k
    )
    
    # 8. Visualize comparison results
    baseline.visualize_comparison_with_hallagent4rec(comparison_df)
    
    # 9. Calculate hallucination rates
    print("\n8. Calculating hallucination rates...")
    
    # HallAgent4Rec hallucination rate
    hall_hallucination_count = 0
    hall_total_recommendations = 0
    
    for user_id in tqdm(sampled_users, desc="Evaluating HallAgent4Rec hallucination"):
        recommendations = hall_agent.generate_recommendations(user_id, num_recommendations=top_k)
        if not recommendations:
            continue
            
        # Detect hallucinations
        query, query_embedding = hall_agent.construct_rag_query(user_id)
        knowledge_base, _ = hall_agent.construct_knowledge_base(user_id)
        retrieved_items = hall_agent.retrieve_items(user_id, query_embedding, knowledge_base)
        
        retrieved_item_ids = set(item['item_id'] for item in retrieved_items)
        
        for item in recommendations:
            hall_total_recommendations += 1
            if item['item_id'] not in retrieved_item_ids:
                hall_hallucination_count += 1
    
    hall_hallucination_rate = hall_hallucination_count / hall_total_recommendations if hall_total_recommendations > 0 else 0
    
    # Baseline models hallucination rates
    baseline_hallucination_rates = {}
    
    for model_name in baseline_models:
        print(f"Evaluating {model_name} hallucination rate...")
        recommendations = baseline.get_recommendations(model_name, sampled_users, top_k)
        hallucinations = baseline.detect_hallucination(model_name, recommendations, item_data)
        
        hallucination_count = sum(sum(h) for h in hallucinations.values())
        total_count = sum(len(h) for h in hallucinations.values())
        
        baseline_hallucination_rates[model_name] = hallucination_count / total_count if total_count > 0 else 0
    
    # 10. Create hallucination comparison table
    hallucination_data = [{'Model': 'HallAgent4Rec', 'Hallucination Rate': hall_hallucination_rate}]
    for model, rate in baseline_hallucination_rates.items():
        hallucination_data.append({'Model': model, 'Hallucination Rate': rate})
    
    hallucination_df = pd.DataFrame(hallucination_data)
    
    # 11. Create precision and recall comparison table
    precision_recalls = []
    
    # HallAgent4Rec precision and recall
    hall_precision = []
    hall_recall = []
    
    # Get ground truth from test interactions
    ground_truth = {}
    for user_id in sampled_users:
        user_test_items = test_interactions[test_interactions['user_id'] == user_id]['item_id'].tolist()
        if user_test_items:
            ground_truth[user_id] = set(user_test_items)
    
    for user_id in sampled_users:
        if user_id not in ground_truth:
            continue
            
        recommendations = hall_agent.generate_recommendations(user_id, num_recommendations=top_k)
        if not recommendations:
            continue
            
        rec_items = set(item['item_id'] for item in recommendations)
        hits = len(rec_items & ground_truth[user_id])
        
        precision = hits / len(rec_items) if rec_items else 0
        recall = hits / len(ground_truth[user_id]) if ground_truth[user_id] else 0
        
        hall_precision.append(precision)
        hall_recall.append(recall)
    
    precision_recalls.append({
        'Model': 'HallAgent4Rec',
        'Precision': np.mean(hall_precision) if hall_precision else 0,
        'Recall': np.mean(hall_recall) if hall_recall else 0
    })
    
    # Baseline models precision and recall
    for model_name in baseline_models:
        recommendations = baseline.get_recommendations(model_name, list(ground_truth.keys()), top_k)
        
        model_precision = []
        model_recall = []
        
        for user_id, items in recommendations.items():
            if user_id not in ground_truth:
                continue
                
            rec_items = set(items)
            hits = len(rec_items & ground_truth[user_id])
            
            precision = hits / len(rec_items) if rec_items else 0
            recall = hits / len(ground_truth[user_id]) if ground_truth[user_id] else 0
            
            model_precision.append(precision)
            model_recall.append(recall)
        
        precision_recalls.append({
            'Model': model_name,
            'Precision': np.mean(model_precision) if model_precision else 0,
            'Recall': np.mean(model_recall) if model_recall else 0
        })
    
    precision_recall_df = pd.DataFrame(precision_recalls)
    
    # 12. Save results
    results = {
        'comparison': comparison_df,
        'hallucination': hallucination_df,
        'precision_recall': precision_recall_df
    }
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(results, f"{dataset_name}_comparative_{timestamp}")
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExperiments completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # 13. Plot comparison results
    plot_comparative_results(results, dataset_name)
    
    return results

def plot_comparative_results(results, dataset_name):
    """
    Plot comparative results between HallAgent4Rec and baseline models.
    
    Args:
        results: Dictionary of comparison results
        dataset_name: Name of the dataset used
    """
    # 1. Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 2. Plot precision comparison
    sns.barplot(x='Model', y='Precision', data=results['precision_recall'], ax=axes[0])
    axes[0].set_title(f'Precision Comparison ({dataset_name.upper()})')
    axes[0].set_ylim(0, max(results['precision_recall']['Precision']) * 1.2)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Annotate with values
    for i, v in enumerate(results['precision_recall']['Precision']):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    # 3. Plot recall comparison
    sns.barplot(x='Model', y='Recall', data=results['precision_recall'], ax=axes[1])
    axes[1].set_title(f'Recall Comparison ({dataset_name.upper()})')
    axes[1].set_ylim(0, max(results['precision_recall']['Recall']) * 1.2)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Annotate with values
    for i, v in enumerate(results['precision_recall']['Recall']):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    # 4. Plot hallucination rate comparison
    sns.barplot(x='Model', y='Hallucination Rate', data=results['hallucination'], ax=axes[2])
    axes[2].set_title(f'Hallucination Rate Comparison ({dataset_name.upper()})')
    axes[2].set_ylim(0, max(results['hallucination']['Hallucination Rate']) * 1.2)
    axes[2].tick_params(axis='x', rotation=45)
    
    # Annotate with values
    for i, v in enumerate(results['hallucination']['Hallucination Rate']):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{dataset_name}_comparative_results.png', dpi=300)
    plt.show()

def run_ablation_comparison(dataset_name, num_users=20, random_state=42, top_k=10):
    """
    Run ablation studies to compare different variants of HallAgent4Rec with baseline models.
    
    Args:
        dataset_name: Name of the dataset to use ('frappe' or 'musicincar')
        num_users: Number of users to sample for evaluation
        random_state: Random seed for reproducibility
        top_k: Number of recommendations to generate
    
    Returns:
        Dictionary of experimental results
    """
    print(f"=== Starting ablation comparison on {dataset_name.upper()} dataset ===")
    print(f"Sample size: {num_users} users, Random seed: {random_state}")
    start_time = time.time()
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    if dataset_name.lower() == 'frappe':
        user_data, item_data, interactions, test_interactions = load_or_create_frappe_sample()
    elif dataset_name.lower() == 'musicincar':
        user_data, item_data, interactions, test_interactions = load_or_create_musicincar_sample()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 2. Initialize and train HallAgent4Rec (full model)
    print("\n2. Initializing and training HallAgent4Rec (full model)...")
    if dataset_name.lower() == 'frappe':
        num_clusters = 15
    else:  # MusicInCar
        num_clusters = 10
    
    hall_agent_full = HallAgent4Rec(
        num_clusters=num_clusters,
        latent_dim=20,
        lambda_u=0.1,
        lambda_v=0.1,
        lambda_h=1.0,
        learning_rate=0.01,
        decay_rate=0.0001,
        max_iterations=50,
        similarity_threshold=0.5,
        relevance_threshold=0.1
    )
    
    hall_agent_full.load_data(user_data, item_data, interactions)
    hall_agent_full.train()
    print("Full model training complete!")
    
    # 3. Initialize and train HallAgent4Rec without clustering
    print("\n3. Initializing and training HallAgent4Rec without clustering...")
    hall_agent_no_clustering = HallAgent4Rec(
        num_clusters=1,  # Set to 1 to effectively disable clustering
        latent_dim=20,
        lambda_u=0.1,
        lambda_v=0.1,
        lambda_h=1.0,
        learning_rate=0.01,
        decay_rate=0.0001,
        max_iterations=50,
        similarity_threshold=0.5,
        relevance_threshold=0.1
    )
    
    hall_agent_no_clustering.load_data(user_data, item_data, interactions)
    hall_agent_no_clustering.train()
    print("No clustering model training complete!")
    
    # 4. Initialize and train HallAgent4Rec without hallucination regularization
    print("\n4. Initializing and training HallAgent4Rec without hallucination regularization...")
    hall_agent_no_hallucination = HallAgent4Rec(
        num_clusters=num_clusters,
        latent_dim=20,
        lambda_u=0.1,
        lambda_v=0.1,
        lambda_h=0.0,  # Set to 0 to disable hallucination regularization
        learning_rate=0.01,
        decay_rate=0.0001,
        max_iterations=50,
        similarity_threshold=0.5,
        relevance_threshold=0.1
    )
    
    hall_agent_no_hallucination.load_data(user_data, item_data, interactions)
    hall_agent_no_hallucination.train()
    print("No hallucination regularization model training complete!")
    
    # 5. Initialize baseline models
    print("\n5. Initializing baseline models...")
    baseline = BaselineModels()
    
    # Convert data to RecBole format
    dataset_name_recbole = baseline.convert_data_to_recbole_format(
        user_data, item_data, interactions, test_interactions)
    
    # Load and prepare data
    baseline.load_and_prepare_data(dataset_name_recbole)
    
    # 6. Train LightGCN as a representative baseline
    print("\n6. Training LightGCN as baseline...")
    lightgcn_config = {
        'embedding_size': 20,
        'n_layers': 3
    }
    baseline.train_model('LightGCN', lightgcn_config)
    
    # 7. Perform stratified sampling
    print("\n7. Performing stratified sampling...")
    exp_framework = ExperimentalFramework(hall_agent_full)
    sampled_users = exp_framework.stratified_sampling(n_samples=num_users, random_state=random_state)
    
    # 8. Evaluate all models on sampled users
    print("\n8. Evaluating all models on sampled users...")
    
    # Prepare ground truth from test interactions
    ground_truth = {}
    for user_id in sampled_users:
        user_test_items = test_interactions[test_interactions['user_id'] == user_id]['item_id'].tolist()
        if user_test_items:
            ground_truth[user_id] = set(user_test_items)
    
    # List of models to evaluate
    models = [
        ('HallAgent4Rec (Full)', hall_agent_full),
        ('HallAgent4Rec (No Clustering)', hall_agent_no_clustering),
        ('HallAgent4Rec (No Hallucination Reg)', hall_agent_no_hallucination)
    ]
    
    # Evaluation results
    evaluation_results = []
    
    # Evaluate HallAgent4Rec variants
    for model_name, model in models:
        print(f"Evaluating {model_name}...")
        
        precision_values = []
        recall_values = []
        hallucination_count = 0
        total_recommendations = 0
        
        for user_id in tqdm(sampled_users, desc=f"Users for {model_name}"):
            # Get recommendations
            recommendations = model.generate_recommendations(user_id, num_recommendations=top_k)
            if not recommendations:
                continue
                
            # Extract recommended items
            rec_items = [item['item_id'] for item in recommendations]
            
            # Calculate precision and recall if ground truth exists
            if user_id in ground_truth:
                hits = len(set(rec_items) & ground_truth[user_id])
                precision = hits / len(rec_items) if rec_items else 0
                recall = hits / len(ground_truth[user_id]) if ground_truth[user_id] else 0
                
                precision_values.append(precision)
                recall_values.append(recall)
            
            # Detect hallucinations
            query, query_embedding = model.construct_rag_query(user_id)
            knowledge_base, _ = model.construct_knowledge_base(user_id)
            retrieved_items = model.retrieve_items(user_id, query_embedding, knowledge_base)
            
            retrieved_item_ids = set(item['item_id'] for item in retrieved_items)
            
            for item_id in rec_items:
                total_recommendations += 1
                if item_id not in retrieved_item_ids:
                    hallucination_count += 1
        
        # Calculate average metrics
        avg_precision = np.mean(precision_values) if precision_values else 0
        avg_recall = np.mean(recall_values) if recall_values else 0
        hallucination_rate = hallucination_count / total_recommendations if total_recommendations > 0 else 0
        
        # Add to results
        evaluation_results.append({
            'Model': model_name,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'Hallucination_Rate': hallucination_rate
        })
    
    # Evaluate LightGCN
    print("Evaluating LightGCN...")
    lightgcn_recommendations = baseline.get_recommendations('LightGCN', list(ground_truth.keys()), top_k)
    
    precision_values = []
    recall_values = []
    
    for user_id, items in lightgcn_recommendations.items():
        if user_id not in ground_truth:
            continue
            
        hits = len(set(items) & ground_truth[user_id])
        precision = hits / len(items) if items else 0
        recall = hits / len(ground_truth[user_id]) if ground_truth[user_id] else 0
        
        precision_values.append(precision)
        recall_values.append(recall)
    
    # Detect hallucinations
    hallucinations = baseline.detect_hallucination('LightGCN', lightgcn_recommendations, item_data)
    hallucination_count = sum(sum(h) for h in hallucinations.values())
    total_count = sum(len(h) for h in hallucinations.values())
    hallucination_rate = hallucination_count / total_count if total_count > 0 else 0
    
    # Add to results
    evaluation_results.append({
        'Model': 'LightGCN',
        'Precision': np.mean(precision_values) if precision_values else 0,
        'Recall': np.mean(recall_values) if recall_values else 0,
        'Hallucination_Rate': hallucination_rate
    })
    
    # 9. Create results DataFrame
    results_df = pd.DataFrame(evaluation_results)
    
    # 10. Plot comparison results
    plot_ablation_comparison(results_df, dataset_name)
    
    # 11. Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(results_df, f"{dataset_name}_ablation_comparison_{timestamp}")
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExperiments completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    return results_df

def plot_ablation_comparison(results_df, dataset_name):
    """
    Plot ablation comparison results.
    
    Args:
        results_df: DataFrame with ablation comparison results
        dataset_name: Name of the dataset used
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot precision
    sns.barplot(x='Model', y='Precision', data=results_df, ax=axes[0])
    axes[0].set_title(f'Precision Comparison ({dataset_name.upper()})')
    axes[0].set_ylim(0, max(results_df['Precision']) * 1.2)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot recall
    sns.barplot(x='Model', y='Recall', data=results_df, ax=axes[1])
    axes[1].set_title(f'Recall Comparison ({dataset_name.upper()})')
    axes[1].set_ylim(0, max(results_df['Recall']) * 1.2)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot hallucination rate
    sns.barplot(x='Model', y='Hallucination_Rate', data=results_df, ax=axes[2])
    axes[2].set_title(f'Hallucination Rate Comparison ({dataset_name.upper()})')
    axes[2].set_ylim(0, max(results_df['Hallucination_Rate']) * 1.2)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{dataset_name}_ablation_comparison.png', dpi=300)
    plt.show()

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
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Set PyTorch seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run experiments based on mode
    if args.mode == 'comparative':
        results = run_comparative_experiments(
            args.dataset, 
            num_users=args.users, 
            random_state=args.seed,
            top_k=args.topk
        )
    elif args.mode == 'ablation':
        results = run_ablation_comparison(
            args.dataset, 
            num_users=args.users, 
            random_state=args.seed,
            top_k=args.topk
        )
    
    print("\nComparative experiments completed successfully!")