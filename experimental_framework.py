import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from scipy import stats
from typing import Dict, List, Set, Tuple, Optional
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from collections import defaultdict
from tqdm import tqdm


class ExperimentalFramework:
    """
    Implements the experimental framework for evaluating HallAgent4Rec
    based on the methodology described in the paper.
    """
    
    def __init__(self, hall_agent):
        """
        Initialize the experimental framework with a trained HallAgent4Rec instance.
        
        Args:
            hall_agent: A trained HallAgent4Rec instance
        """
        self.hall_agent = hall_agent
        self.sample_results = {}
        self.full_results = {}
        self.bootstrap_results = {}
        self.ablation_results = {}
        self.progressive_scaling_results = {}
        
    def compute_icrd_score(self, cluster_id: int, recommendations_by_user: Dict[int, List[Dict]]) -> float:
        """
        Compute the Intra-Cluster Recommendation Diversity (ICRD) score for a cluster
        as defined in equation 28 of the paper.
        
        Args:
            cluster_id: The cluster ID
            recommendations_by_user: Dictionary mapping user IDs to their recommendations
            
        Returns:
            ICRD score for the cluster
        """
        # Get users in this cluster
        users_in_cluster = [
            user_id for user_id, user_idx in self.hall_agent.user_id_map.items()
            if np.argmax(self.hall_agent.user_cluster_matrix[user_idx]) == cluster_id
        ]
        
        if len(users_in_cluster) <= 1:
            return 1.0  # Perfect diversity if only one user
        
        # Compute Jaccard similarity between recommendation sets
        jaccard_sum = 0
        comparison_count = 0
        
        for i, user_i in enumerate(users_in_cluster):
            if user_i not in recommendations_by_user:
                continue
                
            items_i = set(item['item_id'] for item in recommendations_by_user[user_i])
            
            for j, user_j in enumerate(users_in_cluster[i+1:], i+1):
                if user_j not in recommendations_by_user:
                    continue
                    
                items_j = set(item['item_id'] for item in recommendations_by_user[user_j])
                
                # Compute Jaccard similarity
                if items_i or items_j:  # Avoid division by zero
                    similarity = len(items_i & items_j) / len(items_i | items_j)
                else:
                    similarity = 0
                
                jaccard_sum += similarity
                comparison_count += 1
        
        # Compute ICRD score
        if comparison_count > 0:
            avg_similarity = jaccard_sum / comparison_count
            icrd = 1.0 - avg_similarity
        else:
            icrd = 1.0  # Perfect diversity if no comparisons possible
            
        return icrd
    
    def stratified_sampling(self, n_samples: int = 50, random_state: int = 42) -> List[int]:
        """
        Implement stratified sampling to select a representative subset of users
        based on their activity level and preference diversity.
        
        Args:
            n_samples: Number of users to sample
            random_state: Random seed for reproducibility
            
        Returns:
            List of sampled user IDs
        """
        np.random.seed(random_state)
        
        # Calculate user features for stratification
        user_features = {}
        for user_id, user_idx in self.hall_agent.user_id_map.items():
            # Activity level: number of interactions
            interactions = np.sum(self.hall_agent.interaction_matrix[user_idx])
            
            # Preference diversity: entropy of cluster preferences
            cluster_prefs = self.hall_agent.user_cluster_matrix[user_idx]
            cluster_prefs = cluster_prefs / (np.sum(cluster_prefs) + 1e-10)  # Normalize
            entropy = -np.sum(cluster_prefs * np.log(cluster_prefs + 1e-10))
            
            user_features[user_id] = {
                'activity': interactions,
                'diversity': entropy
            }
        
        # Define strata based on activity and diversity
        activity_thresholds = [10, 50]  # Low: <10, Medium: 10-50, High: >50
        diversity_thresholds = [0.4, 0.8]  # Low: <0.4, Medium: 0.4-0.8, High: >0.8
        
        strata = {}
        for user_id, features in user_features.items():
            # Determine activity level
            if features['activity'] < activity_thresholds[0]:
                activity_level = 'low'
            elif features['activity'] < activity_thresholds[1]:
                activity_level = 'medium'
            else:
                activity_level = 'high'
                
            # Determine diversity level
            if features['diversity'] < diversity_thresholds[0]:
                diversity_level = 'low'
            elif features['diversity'] < diversity_thresholds[1]:
                diversity_level = 'medium'
            else:
                diversity_level = 'high'
                
            stratum = f"{activity_level}_{diversity_level}"
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(user_id)
        
        # Calculate proportion of users in each stratum in the full dataset
        total_users = len(self.hall_agent.user_id_map)
        stratum_proportions = {
            stratum: len(users) / total_users
            for stratum, users in strata.items()
        }
        
        # Allocate sample sizes proportionally
        stratum_samples = {
            stratum: max(1, int(n_samples * proportion))
            for stratum, proportion in stratum_proportions.items()
        }
        
        # Adjust to ensure total is n_samples
        while sum(stratum_samples.values()) != n_samples:
            if sum(stratum_samples.values()) < n_samples:
                # Increase sample size for largest stratum
                largest_stratum = max(stratum_proportions.items(), key=lambda x: x[1])[0]
                stratum_samples[largest_stratum] += 1
            else:
                # Decrease sample size for smallest non-zero stratum
                smallest_stratum = min(
                    [(s, n) for s, n in stratum_samples.items() if n > 1],
                    key=lambda x: x[1]
                )[0]
                stratum_samples[smallest_stratum] -= 1
        
        # Sample users from each stratum
        sampled_users = []
        for stratum, sample_size in stratum_samples.items():
            if sample_size > 0:
                stratum_users = strata[stratum]
                if len(stratum_users) <= sample_size:
                    sampled_users.extend(stratum_users)
                else:
                    sampled_users.extend(np.random.choice(stratum_users, sample_size, replace=False))
        
        return sampled_users
    
    def compute_surrogate_metrics(self, user_id: int, recommendations: List[Dict]) -> Dict[str, float]:
        """
        Compute surrogate metrics for a user's recommendations as defined in Section 4.2.
        
        Args:
            user_id: User ID
            recommendations: List of recommended items
            
        Returns:
            Dictionary of surrogate metrics
        """
        # Get user index
        user_idx = self.hall_agent.user_id_map[user_id]
        
        # Get top cluster for the user
        top_cluster = np.argmax(self.hall_agent.user_cluster_matrix[user_idx])
        cluster_items = set(self.hall_agent.items_by_cluster[top_cluster])
        
        # Grounding Violation Rate (GVR) - Equation 29
        recommended_item_indices = []
        for item in recommendations:
            item_id = item['item_id']
            item_idx = self.hall_agent.item_id_map.get(item_id)
            if item_idx is not None:
                recommended_item_indices.append(item_idx)
        
        violations = sum(1 for idx in recommended_item_indices if idx not in cluster_items)
        gvr = violations / len(recommendations) if recommendations else 0
        
        # Retrieval-Prediction Alignment (RPA) - Equation 30
        # First get query embedding
        query, query_embedding = self.hall_agent.construct_rag_query(user_id)
        
        # Compute similarity between recommended items and query
        rpa_scores = []
        for item_idx in recommended_item_indices:
            item_id = self.hall_agent.idx_to_item_id[item_idx]
            item_row = self.hall_agent.item_data[self.hall_agent.item_data['item_id'] == item_id]
            if not item_row.empty:
                item_info = {}
                for col in item_row.columns:
                    item_info[col] = item_row.iloc[0][col]
                
                # Convert item to string and compute embedding
                item_str = ", ".join([f"{k}: {v}" for k, v in item_info.items() if k != 'item_id'])
                item_embedding = self.hall_agent.embeddings_model.embed_query(item_str)
                
                # Compute similarity
                similarity = np.dot(query_embedding, item_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
                )
                rpa_scores.append(similarity)
        
        rpa = np.mean(rpa_scores) if rpa_scores else 0
        
        # Cluster Consistency (CC) - Equation 31
        cluster_counts = defaultdict(int)
        for idx in recommended_item_indices:
            for cluster_id, items in self.hall_agent.items_by_cluster.items():
                if idx in items:
                    cluster_counts[cluster_id] += 1
                    break
        
        cc = max(cluster_counts.values()) / len(recommendations) if recommendations else 0
        
        return {
            'grounding_violation_rate': gvr,
            'retrieval_prediction_alignment': rpa,
            'cluster_consistency': cc
        }
    
    def progressive_scaling_analysis(self, sample_sizes: List[int], num_recommendations: int = 10, random_state: int = 42):
        """
        Perform progressive scaling analysis to establish statistical stability.
        
        Args:
            sample_sizes: List of sample sizes to evaluate
            num_recommendations: Number of recommendations to generate per user
            random_state: Random seed for reproducibility
        """
        print("Performing progressive scaling analysis...")
        
        # Full set of sampled users
        max_sample_size = max(sample_sizes)
        all_sampled_users = self.stratified_sampling(max_sample_size, random_state)
        
        results = {}
        for size in tqdm(sample_sizes, desc="Sample sizes"):
            # Take subset of users for this sample size
            users = all_sampled_users[:size]
            
            # Evaluate metrics for this sample
            precision_sum = 0
            recall_sum = 0
            hallucination_count = 0
            total_recommendations = 0
            surrogate_metrics = {
                'grounding_violation_rate': 0,
                'retrieval_prediction_alignment': 0,
                'cluster_consistency': 0
            }
            
            for user_id in tqdm(users, desc=f"Users (n={size})", leave=False):
                # Generate recommendations
                recommendations = self.hall_agent.generate_recommendations(user_id, num_recommendations)
                
                if not recommendations:
                    continue
                
                # Compute surrogate metrics
                user_surrogate = self.compute_surrogate_metrics(user_id, recommendations)
                for metric, value in user_surrogate.items():
                    surrogate_metrics[metric] += value
                
                # Evaluate hallucination
                query, query_embedding = self.hall_agent.construct_rag_query(user_id)
                knowledge_base, _ = self.hall_agent.construct_knowledge_base(user_id)
                retrieved_items = self.hall_agent.retrieve_items(user_id, query_embedding, knowledge_base)
                
                retrieved_item_names = [item['name'] for item in retrieved_items]
                for item in recommendations:
                    total_recommendations += 1
                    item_name = item['name'] if 'name' in item else str(item['item_id'])
                    
                    is_hallucination = True
                    for retrieved_name in retrieved_item_names:
                        if item_name == retrieved_name:
                            is_hallucination = False
                            break
                    
                    if is_hallucination:
                        hallucination_count += 1
            
            # Calculate average metrics
            hallucination_rate = hallucination_count / total_recommendations if total_recommendations > 0 else 0
            for metric in surrogate_metrics:
                surrogate_metrics[metric] /= len(users)
            
            # Store results for this sample size
            results[size] = {
                'hallucination_rate': hallucination_rate,
                **surrogate_metrics
            }
            
            print(f"Sample size {size}: Hallucination rate = {hallucination_rate:.4f}")
        
        self.progressive_scaling_results = results
        return results
    
    def compute_effect_size(self, metric: str, baseline: str, experiment: str, sample_size: int) -> float:
        """
        Compute Cohen's d effect size between baseline and experimental conditions.
        
        Args:
            metric: The metric to compare
            baseline: Name of baseline method
            experiment: Name of experimental method
            sample_size: Sample size to use
            
        Returns:
            Cohen's d effect size
        """
        if baseline not in self.sample_results or experiment not in self.sample_results:
            return 0
            
        baseline_values = self.sample_results[baseline].get(sample_size, {}).get(metric, [])
        experiment_values = self.sample_results[experiment].get(sample_size, {}).get(metric, [])
        
        if not baseline_values or not experiment_values:
            return 0
            
        # Compute Cohen's d
        mean_diff = np.mean(experiment_values) - np.mean(baseline_values)
        pooled_std = np.sqrt((np.var(baseline_values) + np.var(experiment_values)) / 2)
        
        if pooled_std == 0:
            return 0
            
        return mean_diff / pooled_std
    
    def statistical_power_analysis(self, sample_sizes: List[int], baseline: str, experiment: str, metric: str, alpha: float = 0.05):
        """
        Perform statistical power analysis to ensure adequate sample size.
        
        Args:
            sample_sizes: List of sample sizes to evaluate
            baseline: Name of baseline method
            experiment: Name of experimental method
            metric: The metric to analyze
            alpha: Significance level
            
        Returns:
            Dictionary of power values for each sample size
        """
        power_results = {}
        
        for size in sample_sizes:
            # Get effect size for this sample size
            effect_size = self.compute_effect_size(metric, baseline, experiment, size)
            
            # Compute statistical power
            power = stats.power.TTestIndPower().power(
                effect_size=effect_size,
                nobs=size,
                alpha=alpha
            )
            
            power_results[size] = power
            
        return power_results
    
    def correlation_analysis(self, sampled_users: List[int], full_users: List[int], metrics: List[str]):
        """
        Perform correlation analysis between sample and full dataset metrics.
        
        Args:
            sampled_users: List of sampled user IDs
            full_users: List of all user IDs
            metrics: List of metrics to analyze
            
        Returns:
            Dictionary of correlation coefficients for each metric
        """
        correlation_results = {}
        
        for metric in metrics:
            sample_values = []
            full_values = []
            
            # Compute surrogate metrics for sampled users
            for user_id in sampled_users:
                recommendations = self.hall_agent.generate_recommendations(user_id, num_recommendations=10)
                if recommendations:
                    surrogate = self.compute_surrogate_metrics(user_id, recommendations)
                    sample_values.append(surrogate[metric])
            
            # Compute surrogate metrics for a random subset of full users (for efficiency)
            subset_full_users = np.random.choice(full_users, min(100, len(full_users)), replace=False)
            for user_id in subset_full_users:
                recommendations = self.hall_agent.generate_recommendations(user_id, num_recommendations=10)
                if recommendations:
                    surrogate = self.compute_surrogate_metrics(user_id, recommendations)
                    full_values.append(surrogate[metric])
            
            # Compute correlations
            if sample_values and full_values:
                pearson_r, pearson_p = stats.pearsonr(sample_values, full_values)
                spearman_rho, spearman_p = stats.spearmanr(sample_values, full_values)
                
                correlation_results[metric] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_rho': spearman_rho,
                    'spearman_p': spearman_p
                }
            
        return correlation_results
    
    def bootstrap_validation(self, sampled_users: List[int], metric: str, n_iterations: int = 1000):
        """
        Perform bootstrap resampling to estimate confidence intervals for the full dataset.
        
        Args:
            sampled_users: List of sampled user IDs
            metric: The metric to analyze
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Mean, standard error, and 95% confidence interval for the metric
        """
        print(f"Performing bootstrap validation for {metric} with {n_iterations} iterations...")
        
        # Calculate metric for each user in the sample
        user_metrics = []
        for user_id in tqdm(sampled_users, desc="Computing user metrics"):
            recommendations = self.hall_agent.generate_recommendations(user_id, num_recommendations=10)
            if recommendations:
                if metric in ['hallucination_rate', 'precision', 'recall']:
                    # Compute hallucination rate
                    if metric == 'hallucination_rate':
                        query, query_embedding = self.hall_agent.construct_rag_query(user_id)
                        knowledge_base, _ = self.hall_agent.construct_knowledge_base(user_id)
                        retrieved_items = self.hall_agent.retrieve_items(user_id, query_embedding, knowledge_base)
                        
                        retrieved_item_names = [item['name'] for item in retrieved_items]
                        hallucinations = 0
                        for item in recommendations:
                            item_name = item['name'] if 'name' in item else str(item['item_id'])
                            is_hallucination = True
                            for retrieved_name in retrieved_item_names:
                                if item_name == retrieved_name:
                                    is_hallucination = False
                                    break
                            if is_hallucination:
                                hallucinations += 1
                        
                        user_metrics.append(hallucinations / len(recommendations))
                else:
                    # Compute surrogate metrics
                    surrogate = self.compute_surrogate_metrics(user_id, recommendations)
                    if metric in surrogate:
                        user_metrics.append(surrogate[metric])
        
        # Bootstrap resampling
        bootstrap_means = []
        
        for _ in tqdm(range(n_iterations), desc="Bootstrap iterations"):
            # Resample with replacement
            bootstrap_sample = np.random.choice(user_metrics, size=len(user_metrics), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate statistics
        mean_estimate = np.mean(bootstrap_means)
        std_error = np.std(bootstrap_means)
        conf_interval = (
            np.percentile(bootstrap_means, 2.5),
            np.percentile(bootstrap_means, 97.5)
        )
        
        result = {
            'mean': mean_estimate,
            'std_error': std_error,
            'conf_interval': conf_interval
        }
        
        print(f"Bootstrap results for {metric}:")
        print(f"  Mean: {mean_estimate:.4f}")
        print(f"  95% CI: ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
        
        return result
    
    def perform_ablation_studies(self, sampled_users: List[int], num_recommendations: int = 10):
        """
        Perform ablation studies by removing key components of HallAgent4Rec.
        
        Args:
            sampled_users: List of sampled user IDs
            num_recommendations: Number of recommendations to generate per user
            
        Returns:
            Dictionary of results for each variant
        """
        print("Performing ablation studies...")
        
        # Create variants of HallAgent4Rec with components removed
        variants = {
            'full': self.hall_agent,  # Original model
            'no_clustering': self._create_no_clustering_variant(),
            'no_rag': self._create_no_rag_variant(),
            'no_hallucination_reg': self._create_no_hallucination_reg_variant()
        }
        
        results = {}
        
        for variant_name, variant_model in variants.items():
            print(f"Evaluating {variant_name} variant...")
            
            # Evaluate the variant
            hallucination_count = 0
            total_recommendations = 0
            precision_sum = 0
            recall_sum = 0
            user_count = 0
            
            for user_id in tqdm(sampled_users, desc=f"Users for {variant_name}"):
                # Generate recommendations
                recommendations = variant_model.generate_recommendations(user_id, num_recommendations)
                
                if not recommendations:
                    continue
                
                # Evaluate hallucination
                query, query_embedding = variant_model.construct_rag_query(user_id)
                knowledge_base, _ = variant_model.construct_knowledge_base(user_id)
                retrieved_items = variant_model.retrieve_items(user_id, query_embedding, knowledge_base)
                
                retrieved_item_names = [item['name'] for item in retrieved_items]
                for item in recommendations:
                    total_recommendations += 1
                    item_name = item['name'] if 'name' in item else str(item['item_id'])
                    
                    is_hallucination = True
                    for retrieved_name in retrieved_item_names:
                        if item_name == retrieved_name:
                            is_hallucination = False
                            break
                    
                    if is_hallucination:
                        hallucination_count += 1
                
                # Evaluate precision and recall (with ground truth from original model)
                ground_truth = self.hall_agent.generate_recommendations(user_id, num_recommendations)
                if ground_truth:
                    ground_truth_ids = set(item['item_id'] for item in ground_truth)
                    recommended_ids = set(item['item_id'] for item in recommendations)
                    
                    hit_count = len(ground_truth_ids & recommended_ids)
                    precision = hit_count / len(recommended_ids) if recommended_ids else 0
                    recall = hit_count / len(ground_truth_ids) if ground_truth_ids else 0
                    
                    precision_sum += precision
                    recall_sum += recall
                    user_count += 1
            
            # Calculate average metrics
            hallucination_rate = hallucination_count / total_recommendations if total_recommendations > 0 else 0
            precision = precision_sum / user_count if user_count > 0 else 0
            recall = recall_sum / user_count if user_count > 0 else 0
            
            results[variant_name] = {
                'hallucination_rate': hallucination_rate,
                'precision': precision,
                'recall': recall
            }
            
            print(f"{variant_name} results:")
            print(f"  Hallucination Rate: {hallucination_rate:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
        
        self.ablation_results = results
        return results
    
    def _create_no_clustering_variant(self):
        """Create a variant of HallAgent4Rec without clustering."""
        variant = deepcopy(self.hall_agent)
        
        # Modify implementation to bypass clustering
        variant.num_clusters = 1
        
        # Create a single cluster containing all items
        variant.items_by_cluster = {0: list(range(len(variant.item_features)))}
        
        # Assign all items to cluster 0
        variant.cluster_assignments = np.zeros(len(variant.item_features), dtype=int)
        
        return variant
    
    def _create_no_rag_variant(self):
        """Create a variant of HallAgent4Rec without RAG (retrieval-augmented generation)."""
        variant = deepcopy(self.hall_agent)
        
        # Override retrieve_items method to return all items in the knowledge base
        original_retrieve_items = variant.retrieve_items
        
        def no_rag_retrieve_items(self, user_id, query_embedding, knowledge_base):
            # Return all items without filtering by similarity
            return knowledge_base
        
        variant.retrieve_items = types.MethodType(no_rag_retrieve_items, variant)
        
        return variant
    
    def _create_no_hallucination_reg_variant(self):
        """Create a variant of HallAgent4Rec without hallucination regularization."""
        variant = deepcopy(self.hall_agent)
        
        # Set hallucination penalty to zero
        variant.lambda_h = 0.0
        
        # Reset hallucination scores to zero
        variant.hallucination_scores = np.zeros_like(variant.hallucination_scores)
        
        return variant
    
    def run_full_evaluation(self, n_samples: int = 50, random_state: int = 42):
        """
        Run the complete experimental evaluation as described in the paper.
        
        Args:
            n_samples: Number of users to sample for evaluation
            random_state: Random seed for reproducibility
        """
        print(f"Starting full experimental evaluation with {n_samples} sampled users...")
        
        # 1. Perform stratified sampling
        sampled_users = self.stratified_sampling(n_samples, random_state)
        print(f"Sampled {len(sampled_users)} users for evaluation")
        
        # 2. Progressive scaling analysis
        sample_sizes = [10, 20, 30, 40, 50]
        progressive_results = self.progressive_scaling_analysis(sample_sizes)
        
        # 3. Correlation analysis between sample and full dataset
        all_users = list(self.hall_agent.user_id_map.keys())
        correlation_results = self.correlation_analysis(
            sampled_users, 
            [u for u in all_users if u not in sampled_users],
            ['grounding_violation_rate', 'retrieval_prediction_alignment', 'cluster_consistency']
        )
        print("Correlation analysis results:")
        for metric, corr in correlation_results.items():
            print(f"  {metric}: Pearson r = {corr['pearson_r']:.4f} (p = {corr['pearson_p']:.4f}), " 
                  f"Spearman ρ = {corr['spearman_rho']:.4f} (p = {corr['spearman_p']:.4f})")
        
        # 4. Bootstrap validation
        bootstrap_results = {}
        for metric in ['hallucination_rate', 'grounding_violation_rate', 'retrieval_prediction_alignment']:
            bootstrap_results[metric] = self.bootstrap_validation(sampled_users, metric)
        
        # 5. Ablation studies
        ablation_results = self.perform_ablation_studies(sampled_users)
        
        # 6. Store all results
        self.evaluation_results = {
            'sampled_users': sampled_users,
            'progressive_scaling': progressive_results,
            'correlation': correlation_results,
            'bootstrap': bootstrap_results,
            'ablation': ablation_results
        }
        
        print("Full experimental evaluation complete!")
        
        return self.evaluation_results
    
    def plot_progressive_scaling_results(self):
        """Plot the results of progressive scaling analysis."""
        if not self.progressive_scaling_results:
            print("No progressive scaling results to plot.")
            return
        
        metrics = ['hallucination_rate', 'grounding_violation_rate', 
                   'retrieval_prediction_alignment', 'cluster_consistency']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        sample_sizes = sorted(self.progressive_scaling_results.keys())
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [self.progressive_scaling_results[size][metric] for size in sample_sizes]
            
            ax.plot(sample_sizes, values, 'o-', linewidth=2)
            ax.set_xlabel('Sample Size')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Sample Size')
            ax.grid(True, alpha=0.3)
            
            # Add labels for each point
            for x, y in zip(sample_sizes, values):
                ax.text(x, y, f"{y:.3f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('progressive_scaling_results.png', dpi=300)
        plt.show()
    
    def plot_ablation_results(self):
        """Plot the results of ablation studies."""
        if not self.ablation_results:
            print("No ablation results to plot.")
            return
        
        metrics = ['hallucination_rate', 'precision', 'recall']
        variants = list(self.ablation_results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [self.ablation_results[variant][metric] for variant in variants]
            
            ax.bar(variants, values)
            ax.set_xlabel('Model Variant')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Model Variant')
            ax.set_ylim(0, max(values) * 1.2)
            
            # Add labels for each bar
            for x, y in zip(variants, values):
                ax.text(x, y, f"{y:.3f}", ha='center', va='bottom')
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('ablation_results.png', dpi=300)
        plt.show()
    
    def plot_bootstrap_results(self):
        """Plot the bootstrap validation results with confidence intervals."""
        if not hasattr(self, 'evaluation_results') or 'bootstrap' not in self.evaluation_results:
            print("No bootstrap results to plot.")
            return
        
        bootstrap_results = self.evaluation_results['bootstrap']
        metrics = list(bootstrap_results.keys())
        
        plt.figure(figsize=(10, 6))
        
        means = [bootstrap_results[metric]['mean'] for metric in metrics]
        errors = [bootstrap_results[metric]['std_error'] for metric in metrics]
        conf_intervals = [bootstrap_results[metric]['conf_interval'] for metric in metrics]
        
        # Plot means with error bars
        plt.errorbar(metrics, means, yerr=errors, fmt='o', capsize=5)
        
        # Plot confidence intervals
        for i, (metric, ci) in enumerate(zip(metrics, conf_intervals)):
            plt.plot([i, i], ci, 'r-')
            plt.plot([i-0.1, i+0.1], [ci[0], ci[0]], 'r-')
            plt.plot([i-0.1, i+0.1], [ci[1], ci[1]], 'r-')
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Bootstrap Estimates with 95% Confidence Intervals')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(range(len(metrics)), [m.replace('_', ' ').title() for m in metrics], rotation=45)
        
        plt.tight_layout()
        plt.savefig('bootstrap_results.png', dpi=300)
        plt.show()


# Add this code to the main HallAgent4Rec class to integrate the experimental framework

def add_to_hallagent4rec_class():
    """
    Additional methods to add to the HallAgent4Rec class for experimental evaluation.
    """
    
    def create_experimental_framework(self):
        """
        Create an experimental framework for evaluating the model.
        
        Returns:
            ExperimentalFramework instance
        """
        from experimental_framework import ExperimentalFramework
        return ExperimentalFramework(self)
    
    def evaluate_with_surrogate_metrics(self, user_ids, num_recommendations=10):
        """
        Evaluate the model using surrogate metrics.
        
        Args:
            user_ids: List of user IDs to evaluate
            num_recommendations: Number of recommendations to generate per user
            
        Returns:
            Dictionary of evaluation metrics
        """
        framework = self.create_experimental_framework()
        
        surrogate_metrics = {
            'grounding_violation_rate': 0,
            'retrieval_prediction_alignment': 0,
            'cluster_consistency': 0
        }
        
        user_count = 0
        for user_id in user_ids:
            # Generate recommendations
            recommendations = self.generate_recommendations(user_id, num_recommendations)
            
            if not recommendations:
                continue
                
            # Compute surrogate metrics
            user_surrogate = framework.compute_surrogate_metrics(user_id, recommendations)
            for metric, value in user_surrogate.items():
                surrogate_metrics[metric] += value
                
            user_count += 1
        
        # Calculate average metrics
        for metric in surrogate_metrics:
            surrogate_metrics[metric] /= user_count if user_count > 0 else 1
            
        return surrogate_metrics
    
    def run_statistical_validation(self, n_samples=50, random_state=42):
        """
        Run the complete statistical validation methodology described in the paper.
        
        Args:
            n_samples: Number of users to sample for validation
            random_state: Random seed for reproducibility
            
        Returns:
            ExperimentalFramework instance with evaluation results
        """
        framework = self.create_experimental_framework()
        results = framework.run_full_evaluation(n_samples, random_state)
        return framework


# Example of how to use the experimental framework
def example_experimental_usage():
    # Assuming the HallAgent4Rec instance is already trained
    hall_agent = HallAgent4Rec(num_clusters=3, latent_dim=10)
    
    # Load sample data and train (assuming example_usage() has been adapted)
    from utils import load_sample_data
    user_data, item_data, interactions, test_interactions = load_sample_data()
    hall_agent.load_data(user_data, item_data, interactions)
    hall_agent.train()
    
    # Create experimental framework
    exp_framework = ExperimentalFramework(hall_agent)
    
    # Run complete evaluation
    results = exp_framework.run_full_evaluation(n_samples=50)
    
    # Plot results
    exp_framework.plot_progressive_scaling_results()
    exp_framework.plot_ablation_results()
    exp_framework.plot_bootstrap_results()
    
    return exp_framework


# Helper function to load real datasets for experiments
def load_datasets():
    """
    Load the Frappe and MusicInCar datasets mentioned in the paper.
    
    Returns:
        Dictionary with dataset configurations
    """
    import pandas as pd
    import os
    
    datasets = {}
    
    # Try to load Frappe dataset
    try:
        frappe_dir = "data/frappe/"
        
        if os.path.exists(frappe_dir):
            frappe_users = pd.read_csv(f"{frappe_dir}/users.csv")
            frappe_items = pd.read_csv(f"{frappe_dir}/items.csv")
            frappe_interactions = pd.read_csv(f"{frappe_dir}/interactions.csv")
            
            datasets["frappe"] = {
                "user_data": frappe_users,
                "item_data": frappe_items,
                "interactions": frappe_interactions
            }
            
            print(f"Loaded Frappe dataset: {len(frappe_users)} users, {len(frappe_items)} items, {len(frappe_interactions)} interactions")
    except Exception as e:
        print(f"Could not load Frappe dataset: {e}")
    
    # Try to load MusicInCar dataset
    try:
        music_dir = "data/musicincar/"
        
        if os.path.exists(music_dir):
            music_users = pd.read_csv(f"{music_dir}/users.csv")
            music_items = pd.read_csv(f"{music_dir}/items.csv")
            music_interactions = pd.read_csv(f"{music_dir}/interactions.csv")
            
            datasets["musicincar"] = {
                "user_data": music_users,
                "item_data": music_items,
                "interactions": music_interactions
            }
            
            print(f"Loaded MusicInCar dataset: {len(music_users)} users, {len(music_items)} items, {len(music_interactions)} interactions")
    except Exception as e:
        print(f"Could not load MusicInCar dataset: {e}")
    
    return datasets


if __name__ == "__main__":
    # Example usage of the experimental framework
    example_experimental_usage()