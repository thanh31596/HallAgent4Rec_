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
import random

# Import the HallAgent4Rec extension for MovieLens
from movielens_adapter import MovieLensAgent4Rec
from experimental_framework import ExperimentalFramework
from baseline_integration import BaselineModels
from utilities import save_results, load_results

def run_movielens_experiments(num_users=50, random_state=42, top_k=10, 
                             use_selective_memory=True, memory_strategy='diverse'):
    """
    Run experiments on the MovieLens dataset with HallAgent4Rec.
    
    Args:
        num_users: Number of users to sample for evaluation
        random_state: Random seed for reproducibility
        top_k: Number of recommendations to generate
        use_selective_memory: Whether to use selective memory for agents
        memory_strategy: Strategy for memory selection ('recent', 'diverse', 'important')
        
    Returns:
        Dictionary of experimental results
    """
    print(f"=== Starting MovieLens experiments with HallAgent4Rec ===")
    print(f"Sample size: {num_users} users, Random seed: {random_state}")
    start_time = time.time()
    
    # 1. Create and train MovieLensAgent4Rec
    print("\n1. Initializing and training MovieLensAgent4Rec...")
    agent = MovieLensAgent4Rec(
        num_clusters=20,
        latent_dim=50,
        lambda_u=0.1,
        lambda_v=0.1,
        lambda_h=1.0,
        learning_rate=0.01,
        decay_rate=0.0001,
        max_iterations=50,  # Reduced for faster experiments
        similarity_threshold=0.5,
        relevance_threshold=0.1,
        use_selective_memory=use_selective_memory,
        max_memories_per_user=15,
        memory_strategy=memory_strategy
    )
    
    # Load MovieLens data
    agent.load_movielens_data()
    
    # Train the model
    print("Training HallAgent4Rec on MovieLens data...")
    agent.train()
    print("HallAgent4Rec training complete!")
    
    # 2. Create experimental framework
    print("\n2. Creating experimental framework...")
    exp_framework = ExperimentalFramework(agent)
    
    # 3. Perform stratified sampling
    print("\n3. Performing stratified sampling...")
    sampled_users = exp_framework.stratified_sampling(n_samples=num_users, random_state=random_state)
    print(f"Sampled {len(sampled_users)} users for evaluation")
    
    # 4. Initialize baseline models with RecBole
    print("\n4. Initializing baseline models...")
    baseline = BaselineModels()
    
    # Convert data to RecBole format
    dataset_name_recbole = baseline.convert_data_to_recbole_format(
        agent.user_data, agent.item_data, agent.interactions, agent.test_interactions)
    
    # Load and prepare data
    print("Loading and preparing data for RecBole...")
    baseline.load_and_prepare_data(dataset_name_recbole)
    
    # 5. Train baseline models
    print("\n5. Training baseline models...")
    baseline_models = []
    
    # Train PMF (Probabilistic Matrix Factorization)
    print("Training PMF model...")
    try:
        pmf_config = {
            'loss_type': 'square',
            'embedding_size': 50,
            'negative_sampling': None
        }
        baseline.train_model('BPR', pmf_config)
        baseline_models.append('BPR')
    except Exception as e:
        print(f"Error training PMF model: {e}")
    
    # Train NMF (Non-negative Matrix Factorization)
    print("Training NMF model...")
    try:
        nmf_config = {
            'embedding_size': 50,
            'dropout_prob': 0.0
        }
        baseline.train_model('NeuMF', nmf_config)
        baseline_models.append('NeuMF')
    except Exception as e:
        print(f"Error training NMF model: {e}")
    
    # Train LightGCN
    print("Training LightGCN model...")
    try:
        lightgcn_config = {
            'embedding_size': 50,
            'n_layers': 3
        }
        baseline.train_model('LightGCN', lightgcn_config)
        baseline_models.append('LightGCN')
    except Exception as e:
        print(f"Error training LightGCN model: {e}")
    
    # 6. Compare models
    print("\n6. Comparing HallAgent4Rec with baseline models...")
    comparison_df = baseline.compare_with_hallagent4rec(
        agent, 
        sampled_users, 
        baseline_models,
        top_k=top_k
    )
    
    # 7. Evaluate hallucination rates
    print("\n7. Evaluating hallucination rates...")
    
    # HallAgent4Rec evaluation
    hall_metrics = agent.evaluate_recommendations(num_users=len(sampled_users), num_recommendations=top_k)
    
    # Record detailed metrics
    hall_detailed_metrics = {
        'precision': hall_metrics['precision'],
        'recall': hall_metrics['recall'],
        'ndcg': hall_metrics['ndcg'],
        'hallucination_rate': hall_metrics['hallucination_rate']
    }
    
    # 8. Ablation studies
    print("\n8. Running ablation studies...")
    
    # Create variants of HallAgent4Rec for ablation studies
    variants = {}
    
    # Variant 1: No clustering (single cluster)
    print("Creating and training variant: No clustering")
    agent_no_clustering = MovieLensAgent4Rec(
        num_clusters=1,  # Only one cluster
        latent_dim=50,
        lambda_u=0.1,
        lambda_v=0.1,
        lambda_h=1.0,
        learning_rate=0.01,
        decay_rate=0.0001,
        max_iterations=50,
        similarity_threshold=0.5,
        relevance_threshold=0.1,
        use_selective_memory=use_selective_memory,
        max_memories_per_user=15,
        memory_strategy=memory_strategy
    )
    agent_no_clustering.load_movielens_data()
    agent_no_clustering.train()
    variants['No Clustering'] = agent_no_clustering
    
    # Variant 2: No hallucination regularization
    print("Creating and training variant: No hallucination regularization")
    agent_no_hallucination = MovieLensAgent4Rec(
        num_clusters=20,
        latent_dim=50,
        lambda_u=0.1,
        lambda_v=0.1,
        lambda_h=0.0,  # Disable hallucination regularization
        learning_rate=0.01,
        decay_rate=0.0001,
        max_iterations=50,
        similarity_threshold=0.5,
        relevance_threshold=0.1,
        use_selective_memory=use_selective_memory,
        max_memories_per_user=15,
        memory_strategy=memory_strategy
    )
    agent_no_hallucination.load_movielens_data()
    agent_no_hallucination.train()
    variants['No Hallucination Reg'] = agent_no_hallucination
    
    # Variant 3: No selective memory (if using selective memory)
    if use_selective_memory:
        print("Creating and training variant: No selective memory")
        agent_no_selective = MovieLensAgent4Rec(
            num_clusters=20,
            latent_dim=50,
            lambda_u=0.1,
            lambda_v=0.1,
            lambda_h=1.0,
            learning_rate=0.01,
            decay_rate=0.0001,
            max_iterations=50,
            similarity_threshold=0.5,
            relevance_threshold=0.1,
            use_selective_memory=False  # Use all memories
        )
        agent_no_selective.load_movielens_data()
        agent_no_selective.train()
        variants['No Selective Memory'] = agent_no_selective
    
    # Evaluate all variants
    ablation_results = []
    
    # Add full model results
    ablation_results.append({
        'Model': 'HallAgent4Rec (Full)',
        'Precision': hall_detailed_metrics['precision'],
        'Recall': hall_detailed_metrics['recall'],
        'NDCG': hall_detailed_metrics['ndcg'],
        'Hallucination_Rate': hall_detailed_metrics['hallucination_rate']
    })
    
    # Evaluate variants
    for variant_name, variant_model in variants.items():
        print(f"Evaluating variant: {variant_name}")
        variant_metrics = variant_model.evaluate_recommendations(
            num_users=min(20, len(sampled_users)),  # Use smaller subset for efficiency
            num_recommendations=top_k
        )
        
        ablation_results.append({
            'Model': f'HallAgent4Rec ({variant_name})',
            'Precision': variant_metrics['precision'],
            'Recall': variant_metrics['recall'],
            'NDCG': variant_metrics['ndcg'],
            'Hallucination_Rate': variant_metrics['hallucination_rate']
        })
    
    # Convert to DataFrame
    ablation_df = pd.DataFrame(ablation_results)
    
    # 9. Save all results
    results = {
        'comparison': comparison_df,
        'hall_detailed': hall_detailed_metrics,
        'ablation': ablation_df
    }
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(results, f"movielens_experiments_{timestamp}")
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExperiments completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # 10. Plot results
    plot_movielens_results(results)
    
    return results, agent

def run_progressive_scaling_analysis(max_users=50, step=10, random_state=42):
    """
    Run progressive scaling analysis to determine stabilization of metrics.
    
    Args:
        max_users: Maximum number of users to evaluate
        step: Step size for increasing the number of users
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of results by sample size
    """
    print(f"=== Running progressive scaling analysis on MovieLens dataset ===")
    start_time = time.time()
    
    # 1. Create MovieLensAgent4Rec
    agent = MovieLensAgent4Rec(
        num_clusters=20,
        latent_dim=50,
        use_selective_memory=True,
        max_memories_per_user=15
    )
    
    # 2. Load MovieLens data
    agent.load_movielens_data()
    
    # 3. Train the model
    agent.train()
    
    # 4. Create experimental framework
    exp_framework = ExperimentalFramework(agent)
    
    # 5. Generate sample sizes to evaluate
    sample_sizes = list(range(step, max_users + 1, step))
    if sample_sizes[-1] != max_users:
        sample_sizes.append(max_users)
    
    # 6. Get full set of sampled users
    all_sampled_users = exp_framework.stratified_sampling(n_samples=max_users, random_state=random_state)
    
    # 7. Evaluate each sample size
    results = {}
    for size in sample_sizes:
        print(f"\nEvaluating with {size} users...")
        
        # Use subset of users
        users_subset = all_sampled_users[:size]
        
        # Evaluate metrics
        metrics = agent.evaluate_recommendations(
            num_users=len(users_subset),
            num_recommendations=10
        )
        
        # Store results
        results[size] = {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'ndcg': metrics['ndcg'],
            'hallucination_rate': metrics['hallucination_rate']
        }
    
    # 8. Plot results
    plot_progressive_scaling(results)
    
    # 9. Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(results, f"movielens_progressive_scaling_{timestamp}")
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nProgressive scaling analysis completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    return results

def run_memory_strategy_comparison(num_users=30, random_state=42):
    """
    Compare different memory selection strategies for movie recommendations.
    
    Args:
        num_users: Number of users to sample for evaluation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of results by memory strategy
    """
    print(f"=== Comparing memory selection strategies on MovieLens dataset ===")
    start_time = time.time()
    
    # Strategies to compare
    strategies = {
        'all': False,  # No selective memory (use all)
        'random': True,  # Random selection
        'recent': True,  # Most recent interactions
        'diverse': True,  # Diverse across genres
        'important': True  # Most significant (high/low ratings)
    }
    
    # Results storage
    results = {}
    
    # Sample users (use same set for fair comparison)
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Create a base agent to get user data and sample
    base_agent = MovieLensAgent4Rec()
    base_agent.load_movielens_data()
    
    # Create experimental framework
    exp_framework = ExperimentalFramework(base_agent)
    
    # Perform stratified sampling
    sampled_users = exp_framework.stratified_sampling(n_samples=num_users, random_state=random_state)
    print(f"Sampled {len(sampled_users)} users for evaluation")
    
    # Evaluate each strategy
    for strategy_name, use_selective in strategies.items():
        print(f"\nEvaluating memory strategy: {strategy_name}")
        
        # Create and train agent with this strategy
        agent = MovieLensAgent4Rec(
            num_clusters=20,
            latent_dim=50,
            use_selective_memory=use_selective,
            max_memories_per_user=15,
            memory_strategy=strategy_name if use_selective else None
        )
        
        # Load data and train
        agent.load_movielens_data()
        agent.train()
        
        # Evaluate
        metrics = agent.evaluate_recommendations(
            num_users=len(sampled_users),
            num_recommendations=10
        )
        
        # Store results
        results[strategy_name] = {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'ndcg': metrics['ndcg'],
            'hallucination_rate': metrics['hallucination_rate']
        }
    
    # Plot results
    plot_memory_strategies(results)
    
    # Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(results, f"movielens_memory_strategies_{timestamp}")
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nMemory strategy comparison completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    return results

def plot_movielens_results(results):
    """
    Plot the results of MovieLens experiments.
    
    Args:
        results: Dictionary of experiment results
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Comparison with baseline models
    if 'comparison' in results:
        comparison_df = results['comparison']
        
        # Precision
        sns.barplot(x='Model', y='Precision', data=comparison_df, ax=axes[0, 0])
        axes[0, 0].set_title('Precision Comparison', fontsize=16)
        axes[0, 0].set_ylim(0, max(comparison_df['Precision']) * 1.2)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        sns.barplot(x='Model', y='Recall', data=comparison_df, ax=axes[0, 1])
        axes[0, 1].set_title('Recall Comparison', fontsize=16)
        axes[0, 1].set_ylim(0, max(comparison_df['Recall']) * 1.2)
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 2: Ablation study
    if 'ablation' in results:
        ablation_df = results['ablation']
        
        # Create a separate figure for ablation results
        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
        
        # Performance metrics
        performance_df = ablation_df[['Model', 'Precision', 'Recall', 'NDCG']].melt(
            id_vars=['Model'],
            var_name='Metric',
            value_name='Value'
        )
        
        sns.barplot(x='Model', y='Value', hue='Metric', data=performance_df, ax=axes2[0])
        axes2[0].set_title('Performance Metrics by Model Variant', fontsize=16)
        axes2[0].set_ylim(0, max(performance_df['Value']) * 1.2)
        axes2[0].tick_params(axis='x', rotation=45)
        
        # Hallucination rate
        sns.barplot(x='Model', y='Hallucination_Rate', data=ablation_df, ax=axes2[1])
        axes2[1].set_title('Hallucination Rate by Model Variant', fontsize=16)
        axes2[1].set_ylim(0, max(ablation_df['Hallucination_Rate']) * 1.2)
        axes2[1].tick_params(axis='x', rotation=45)
        
        # Set common title
        fig2.suptitle('Ablation Study Results', fontsize=20)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save ablation figure
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/movielens_ablation_results.png', dpi=300)
    
    # Plot 3 & 4: Use for other analysis if needed
    if 'hall_detailed' in results:
        # Extract metrics
        metrics = results['hall_detailed']
        
        # Create bar chart of all metrics
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        sns.barplot(x='Metric', y='Value', data=metrics_df, ax=axes[1, 0])
        axes[1, 0].set_title('HallAgent4Rec Detailed Metrics', fontsize=16)
        axes[1, 0].set_ylim(0, max(metrics_df['Value']) * 1.2)
        
        # Add text labels on top of bars
        for i, v in enumerate(metrics_df['Value']):
            axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    # Set common title
    fig.suptitle('MovieLens Experiment Results with HallAgent4Rec', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/movielens_experiment_results.png', dpi=300)

def plot_progressive_scaling(results):
    """
    Plot the results of progressive scaling analysis.
    
    Args:
        results: Dictionary mapping sample sizes to metrics
    """
    # Extract sample sizes and metrics
    sample_sizes = sorted(results.keys())
    metrics = ['precision', 'recall', 'ndcg', 'hallucination_rate']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = [results[size][metric] for size in sample_sizes]
        
        axes[i].plot(sample_sizes, values, 'o-', linewidth=2)
        axes[i].set_xlabel('Sample Size (Number of Users)', fontsize=12)
        axes[i].set_ylabel(metric.capitalize(), fontsize=12)
        axes[i].set_title(f'{metric.capitalize()} vs Sample Size', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        
        # Add text labels for each point
        for j, (x, y) in enumerate(zip(sample_sizes, values)):
            axes[i].annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                           xytext=(0, 10), ha='center')
    
    # Set common title
    fig.suptitle('Progressive Scaling Analysis on MovieLens Dataset', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/movielens_progressive_scaling.png', dpi=300)

def plot_memory_strategies(results):
    """
    Plot comparison of different memory selection strategies.
    
    Args:
        results: Dictionary mapping strategies to metrics
    """
    # Convert results to DataFrame for easier plotting
    data = []
    for strategy, metrics in results.items():
        for metric, value in metrics.items():
            data.append({
                'Strategy': strategy,
                'Metric': metric,
                'Value': value
            })
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot each metric
    metrics = ['precision', 'recall', 'ndcg', 'hallucination_rate']
    for i, metric in enumerate(metrics):
        metric_df = df[df['Metric'] == metric]
        
        sns.barplot(x='Strategy', y='Value', data=metric_df, ax=axes[i])
        axes[i].set_title(f'{metric.capitalize()} by Memory Strategy', fontsize=14)
        axes[i].set_ylim(0, max(metric_df['Value']) * 1.2)
        
        # Add text labels on top of bars
        for j, v in enumerate(metric_df['Value']):
            axes[i].text(j, v + 0.01, f"{v:.3f}", ha='center')
    
    # Set common title
    fig.suptitle('Comparison of Memory Selection Strategies', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/movielens_memory_strategies.png', dpi=300)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MovieLens experiments with HallAgent4Rec')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'scaling', 'memory'],
                       help='Experiment mode')
    parser.add_argument('--users', type=int, default=50, 
                       help='Number of users to sample')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--topk', type=int, default=10, 
                       help='Number of recommendations to generate')
    
    args = parser.parse_args()
    
    # Set PyTorch seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run experiments based on mode
    if args.mode == 'full':
        results, agent = run_movielens_experiments(
            num_users=args.users, 
            random_state=args.seed,
            top_k=args.topk
        )
    elif args.mode == 'scaling':
        results = run_progressive_scaling_analysis(
            max_users=args.users, 
            step=10,
            random_state=args.seed
        )
    elif args.mode == 'memory':
        results = run_memory_strategy_comparison(
            num_users=args.users, 
            random_state=args.seed
        )
    
    print("\nMovieLens experiments completed successfully!")