import numpy as np
import pandas as pd
import faiss
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from pydantic import BaseModel, Field
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)






# =========== HallAgent4Rec Implementation Based on the Paper ===========

class HallAgent4Rec:
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
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_h = lambda_h
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold
        self.relevance_threshold = relevance_threshold
        
        # Initialize components
        self.agents = {}  # Dictionary to store generative agents
        self.clusters = None  # Will store the cluster model
        self.cluster_assignments = None  # Will store item cluster assignments
        self.user_embeddings = None  # Will store user embeddings from PMF
        self.item_embeddings = None  # Will store item embeddings from PMF
        self.user_cluster_matrix = None  # Will store user-cluster interaction matrix
        self.item_features = None  # Will store item features for clustering
        self.item_database = None  # Will store the item database
        self.hallucination_scores = None  # Will store hallucination likelihood scores
        self.items_by_cluster = {}  # Will store items grouped by cluster

    def load_data(self, user_data: pd.DataFrame, item_data: pd.DataFrame, interactions: pd.DataFrame):
        """
        Load user, item, and interaction data.
        
        Args:
            user_data: DataFrame with user information (id, features, etc.)
            item_data: DataFrame with item information (id, features, etc.)
            interactions: DataFrame with user-item interactions
        """
        self.user_data = user_data
        self.item_data = item_data
        self.interactions = interactions
        
        # Create a set of all users and items
        self.all_users = set(user_data['user_id'].unique())
        self.all_items = set(item_data['item_id'].unique())
        
        # Extract item features for clustering
        feature_columns = [col for col in item_data.columns if col != 'item_id']
        self.item_features = item_data[feature_columns].values
        self.item_id_map = {id: idx for idx, id in enumerate(item_data['item_id'])}
        self.idx_to_item_id = {idx: id for id, idx in self.item_id_map.items()}
        
        # Create user-item interaction matrix
        self.create_interaction_matrix()
        print(f"Loaded data: {len(self.all_users)} users, {len(self.all_items)} items, {len(interactions)} interactions")

    def create_interaction_matrix(self):
        """Create the user-item interaction matrix from interaction data."""
        # Create user and item ID mappings
        self.user_id_map = {id: idx for idx, id in enumerate(self.user_data['user_id'])}
        self.idx_to_user_id = {idx: id for id, idx in self.user_id_map.items()}
        
        # Initialize interaction matrix
        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)
        self.interaction_matrix = np.zeros((num_users, num_items))
        
        # Fill the interaction matrix
        for _, row in self.interactions.iterrows():
            user_idx = self.user_id_map.get(row['user_id'])
            item_idx = self.item_id_map.get(row['item_id'])
            if user_idx is not None and item_idx is not None:
                # For implicit feedback, use 1 to indicate interaction
                self.interaction_matrix[user_idx, item_idx] = 1
        
        print(f"Created interaction matrix of shape {self.interaction_matrix.shape}")

    def cluster_items(self):
        """Cluster items based on their features."""
        print(f"Clustering {len(self.item_features)} items into {self.num_clusters} clusters...")
        
        # Apply k-means clustering
        self.clusters = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.cluster_assignments = self.clusters.fit_predict(self.item_features)
        
        # Group items by cluster
        self.items_by_cluster = {}
        for idx, cluster_id in enumerate(self.cluster_assignments):
            if cluster_id not in self.items_by_cluster:
                self.items_by_cluster[cluster_id] = []
            self.items_by_cluster[cluster_id].append(idx)
        
        # Create user-cluster interaction matrix
        self.create_user_cluster_matrix()
        
        print(f"Item clustering complete. Cluster distribution:")
        for cluster_id, items in self.items_by_cluster.items():
            print(f"Cluster {cluster_id}: {len(items)} items")

    def create_user_cluster_matrix(self):
        """
        Create user-cluster interaction matrix by aggregating 
        user-item interactions within each cluster.
        """
        num_users = self.interaction_matrix.shape[0]
        self.user_cluster_matrix = np.zeros((num_users, self.num_clusters))
        
        # For each user-cluster pair, aggregate interactions
        for user_idx in range(num_users):
            for cluster_id in range(self.num_clusters):
                # Get items in this cluster
                cluster_items = self.items_by_cluster[cluster_id]
                # Sum interactions with items in this cluster
                cluster_interactions = sum(self.interaction_matrix[user_idx, item_idx] for item_idx in cluster_items)
                # Normalize by cluster size to get average interaction
                self.user_cluster_matrix[user_idx, cluster_id] = cluster_interactions / max(1, len(cluster_items))
        
        print(f"Created user-cluster matrix of shape {self.user_cluster_matrix.shape}")

    def matrix_factorization(self):
        """
        Perform hallucination-aware matrix factorization to learn
        user and item latent factors.
        """
        print("Starting hallucination-aware matrix factorization...")
        
        # Initialize user and item embeddings randomly
        num_users = self.interaction_matrix.shape[0]
        num_items = self.interaction_matrix.shape[1]
        
        # Initialize embeddings
        self.user_embeddings = np.random.normal(0, 0.1, (num_users, self.latent_dim))
        self.item_embeddings = np.random.normal(0, 0.1, (num_items, self.latent_dim))
        
        # Initialize hallucination likelihood scores
        # Initially, all items have equal hallucination likelihood
        self.hallucination_scores = np.ones((num_users, num_items)) * 0.5
        
        # Get user-item pairs with observed interactions
        user_indices, item_indices = np.where(self.interaction_matrix > 0)
        
        # Prepare for optimization
        learning_rate = self.learning_rate
        
        # Implement mini-batch SGD as described in Algorithm 1 of the paper
        for iteration in range(self.max_iterations):
            # Shuffle the observed interactions
            indices = np.arange(len(user_indices))
            np.random.shuffle(indices)
            
            # Mini-batch size
            batch_size = min(1024, len(indices))
            
            # Process mini-batches
            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]
                
                # Get user and item indices for this batch
                batch_user_indices = user_indices[batch_indices]
                batch_item_indices = item_indices[batch_indices]
                
                # Update user embeddings
                for i, user_idx in enumerate(batch_user_indices):
                    item_idx = batch_item_indices[i]
                    
                    # Get observed interaction
                    rating = self.interaction_matrix[user_idx, item_idx]
                    
                    # Compute prediction
                    prediction = np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx])
                    
                    # Compute error
                    error = rating - prediction
                    
                    # Hallucination penalty term (from equation 22 in the paper)
                    h_penalty = self.lambda_h * self.hallucination_scores[user_idx, item_idx] * prediction
                    
                    # Compute gradients
                    user_grad = -error * self.item_embeddings[item_idx] + self.lambda_u * self.user_embeddings[user_idx] + h_penalty * self.item_embeddings[item_idx]
                    item_grad = -error * self.user_embeddings[user_idx] + self.lambda_v * self.item_embeddings[item_idx] + h_penalty * self.user_embeddings[user_idx]
                    
                    # Update embeddings
                    self.user_embeddings[user_idx] -= learning_rate * user_grad
                    self.item_embeddings[item_idx] -= learning_rate * item_grad
            
            # Decay learning rate (equation 27 in the paper)
            learning_rate = self.learning_rate / (1 + self.decay_rate * iteration)
            
            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                # Compute training error
                error = self.compute_training_error()
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Error: {error:.4f}, Learning Rate: {learning_rate:.6f}")
        
        print("Matrix factorization complete")

    def compute_training_error(self):
        """Compute the training error of the current matrix factorization model."""
        # Compute predictions for all observed interactions
        user_indices, item_indices = np.where(self.interaction_matrix > 0)
        ratings = self.interaction_matrix[user_indices, item_indices]
        
        # Compute predictions
        predictions = np.sum(self.user_embeddings[user_indices] * self.item_embeddings[item_indices], axis=1)
        
        # Compute mean squared error
        mse = np.mean((ratings - predictions) ** 2)
        return mse

    def initialize_agents(self):
        """Initialize generative agents for each user."""
        print("Initializing generative agents...")
        
        for user_id, user_idx in self.user_id_map.items():
            # Get user data
            user_row = self.user_data[self.user_data['user_id'] == user_id].iloc[0]
            
            # Extract user traits from user data
            traits = {}
            for col in self.user_data.columns:
                if col != 'user_id':
                    traits[col] = user_row[col]
            
            # Format traits as a string
            traits_str = ", ".join([f"{k}: {v}" for k, v in traits.items()])
            
            # Create agent memory
            memory = GenerativeAgentMemory(
                llm=LLM,
                memory_retriever=create_new_memory_retriever(),
                verbose=False,
                reflection_threshold=30,
            )
            
            # Create generative agent
            agent = GenerativeAgent(
                name=f"User_{user_id}",
                age=traits.get('age', 30),  # Default to 30 if age not provided
                traits=traits_str,
                status="looking for recommendations",
                memory_retriever=create_new_memory_retriever(),
                llm=LLM,
                memory=memory,
            )
            
            # Store agent
            self.agents[user_id] = agent
            
            # Add user interactions as memories to the agent
            user_interactions = self.interactions[self.interactions['user_id'] == user_id]
            for _, interaction in user_interactions.iterrows():
                item_id = interaction['item_id']
                if 'timestamp' in interaction:
                    timestamp = interaction['timestamp']
                    created_at = datetime.fromtimestamp(timestamp) if isinstance(timestamp, (int, float)) else datetime.now()
                else:
                    created_at = datetime.now()
                
                # Get item details
                item_row = self.item_data[self.item_data['item_id'] == item_id]
                if not item_row.empty:
                    item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                    memory_content = f"I interacted with {item_name} (ID: {item_id})"
                    
                    # Add to agent memory
                    agent.memory.add_memory(memory_content)
        
        print(f"Initialized {len(self.agents)} generative agents")

    def construct_rag_query(self, user_id):
        """
        Construct a retrieval query integrating user traits and memory.
        This implements equation 12 from the paper.
        """
        # Get agent for the user
        agent = self.agents[user_id]
        
        # Get user traits
        user_traits = agent.traits
        
        # Get relevant memories
        relevant_memories = agent.memory.memory_retriever.get_relevant_documents("What do I like?")
        memory_contents = " ".join([mem.page_content for mem in relevant_memories])
        
        # Construct the query
        query = f"User traits: {user_traits}. User memories: {memory_contents}"
        
        # Encode the query
        query_embedding = embeddings_model.embed_query(query)
        
        return query, query_embedding

    def construct_knowledge_base(self, user_id):
        """
        Construct a knowledge base of relevant items for the user.
        This implements equation 14 from the paper.
        """
        # Get user index
        user_idx = self.user_id_map[user_id]
        
        # Predict user's cluster preferences
        cluster_scores = np.dot(self.user_embeddings[user_idx], self.item_embeddings.T)
        
        # Get the top cluster
        top_cluster = np.argmax(np.mean([cluster_scores[user_idx] for user_idx in self.items_by_cluster[cluster_id]]) 
                                for cluster_id in range(self.num_clusters))
        
        # Get items in the top cluster
        cluster_items = self.items_by_cluster[top_cluster]
        
        # Get item scores
        item_scores = np.dot(self.user_embeddings[user_idx], self.item_embeddings[cluster_items].T)
        
        # Filter items by relevance threshold
        relevant_items = [cluster_items[i] for i, score in enumerate(item_scores) if score >= self.relevance_threshold]
        
        # Create knowledge base with item details
        knowledge_base = []
        for item_idx in relevant_items:
            item_id = self.idx_to_item_id[item_idx]
            item_row = self.item_data[self.item_data['item_id'] == item_id]
            if not item_row.empty:
                item_info = {}
                for col in item_row.columns:
                    item_info[col] = item_row.iloc[0][col]
                knowledge_base.append(item_info)
        
        return knowledge_base, top_cluster

    def retrieve_items(self, user_id, query_embedding, knowledge_base):
        """
        Retrieve items from the knowledge base based on similarity to query.
        This implements equation 15 from the paper.
        """
        # Encode each item in the knowledge base
        item_embeddings = []
        for item in knowledge_base:
            # Convert item to string representation
            item_str = ", ".join([f"{k}: {v}" for k, v in item.items() if k != 'item_id'])
            item_embedding = embeddings_model.embed_query(item_str)
            item_embeddings.append((item, item_embedding))
        
        # Compute similarities
        similarities = []
        for item, item_embedding in item_embeddings:
            sim = cosine_similarity([query_embedding], [item_embedding])[0][0]
            similarities.append((item, sim))
        
        # Sort by similarity and filter by threshold
        retrieved_items = [item for item, sim in sorted(similarities, key=lambda x: x[1], reverse=True) 
                          if sim >= self.similarity_threshold]
        
        return retrieved_items

    def generate_recommendations(self, user_id, num_recommendations=5):
        """
        Generate recommendations for a user using the HallAgent4Rec methodology.
        This implements the full recommendation pipeline from the paper.
        """
        print(f"Generating recommendations for user {user_id}...")
        
        # Step 1: Construct RAG query (eq. 12-13)
        query, query_embedding = self.construct_rag_query(user_id)
        
        # Step 2: Construct knowledge base (eq. 14)
        knowledge_base, top_cluster = self.construct_knowledge_base(user_id)
        
        # Step 3: Retrieve relevant items (eq. 15)
        retrieved_items = self.retrieve_items(user_id, query_embedding, knowledge_base)
        
        # If no items retrieved, return empty list
        if not retrieved_items:
            print("No relevant items retrieved. Cannot generate recommendations.")
            return []
        
        # Step 4: Generate recommendations using LLM
        # Format retrieved items for prompt
        item_descriptions = "\n".join([f"- {item['name']}" for item in retrieved_items[:10]])
        
        # Get agent
        agent = self.agents[user_id]
        
        # Create prompt for LLM
        prompt = f"""
        You are a recommendation system for a user with the following traits:
        {agent.traits}
        
        Based on the user's profile and past behavior, you have retrieved the following relevant items:
        {item_descriptions}
        
        Please recommend {num_recommendations} items from the list above that would be most relevant for this user.
        For each recommendation, provide a brief explanation of why it matches the user's preferences.
        
        IMPORTANT: You must ONLY recommend items from the provided list. Do not suggest any items that are not in the list.
        
        Format your response as:
        1. [Item Name]: [Explanation]
        2. [Item Name]: [Explanation]
        ...
        """
        
        # Generate recommendations
        response = LLM.invoke(prompt)
        recommendations_text = response.content
        
        # Step 5: Detect hallucinations
        # Extract recommended items from response
        recommended_items = []
        lines = recommendations_text.strip().split('\n')
        for line in lines:
            if line.strip() and any(char.isdigit() for char in line[:5]):
                # Extract item name from line (format: "1. [Item Name]: [Explanation]")
                parts = line.split(':', 1)
                if len(parts) > 0:
                    item_name_part = parts[0].strip()
                    # Extract text inside brackets if present
                    item_name = item_name_part.split('.', 1)[1].strip() if '.' in item_name_part else item_name_part
                    recommended_items.append(item_name)
        
        # Check for hallucinations (items not in retrieved set)
        retrieved_item_names = [item['name'] for item in retrieved_items]
        hallucinations = []
        valid_recommendations = []
        
        for item_name in recommended_items:
            is_hallucination = True
            # Check if recommended item is in retrieved items (allowing for minor text differences)
            for retrieved_name in retrieved_item_names:
                print("item_name: ", item_name)
                print("retrieved_name: ", retrieved_name)
                if item_name == retrieved_name:
                    is_hallucination = False
                    # Find the actual item
                    for item in retrieved_items:
                        if item['name'] == item_name:
                            valid_recommendations.append(item)
                            break
                    break
            
            if is_hallucination:
                hallucinations.append(item_name)
        
        # Report hallucinations
        if hallucinations:
            print(f"Detected {len(hallucinations)} hallucinations: {hallucinations}")
            
            # Update hallucination scores
            user_idx = self.user_id_map[user_id]
            for hallucination in hallucinations:
                # For each hallucination, increase hallucination scores for similar items
                for item_idx in range(self.item_embeddings.shape[0]):
                    item_id = self.idx_to_item_id[item_idx]
                    item_row = self.item_data[self.item_data['item_id'] == item_id]
                    if not item_row.empty:
                        item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                        # If item name is similar to hallucination, increase score
                        if hallucination==item_name:
                            self.hallucination_scores[user_idx, item_idx] += 0.1
        
        # If not enough valid recommendations, fill with top predicted items
        if len(valid_recommendations) < num_recommendations:
            user_idx = self.user_id_map[user_id]
            cluster_items = self.items_by_cluster[top_cluster]
            
            # Get predicted scores for items in the cluster
            item_scores = [(idx, np.dot(self.user_embeddings[user_idx], self.item_embeddings[idx])) 
                           for idx in cluster_items]
            
            # Sort by score and remove items already recommended
            recommended_ids = [item['item_id'] for item in valid_recommendations]
            additional_items = []
            
            for item_idx, score in sorted(item_scores, key=lambda x: x[1], reverse=True):
                item_id = self.idx_to_item_id[item_idx]
                if item_id not in recommended_ids:
                    item_row = self.item_data[self.item_data['item_id'] == item_id]
                    if not item_row.empty:
                        item_info = {}
                        for col in item_row.columns:
                            item_info[col] = item_row.iloc[0][col]
                        additional_items.append(item_info)
                        if len(valid_recommendations) + len(additional_items) >= num_recommendations:
                            break
            
            valid_recommendations.extend(additional_items)
        
        # Limit to requested number
        valid_recommendations = valid_recommendations[:num_recommendations]
        
        print(f"Generated {len(valid_recommendations)} valid recommendations")
        return valid_recommendations

    def train(self):
        """Train the complete HallAgent4Rec model."""
        # Step 1: Cluster items
        self.cluster_items()
        
        # Step 2: Initialize matrix factorization
        self.matrix_factorization()
        
        # Step 3: Initialize generative agents
        self.initialize_agents()
        
        print("HallAgent4Rec training complete!")

    def evaluate(self, test_interactions, metrics=['precision', 'recall', 'hallucination_rate']):
        """
        Evaluate the recommendation system using test interactions.
        
        Args:
            test_interactions: DataFrame with test user-item interactions
            metrics: List of metrics to compute
        
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        if 'precision' in metrics or 'recall' in metrics:
            # Group test interactions by user
            user_test_items = {}
            for _, row in test_interactions.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                if user_id not in user_test_items:
                    user_test_items[user_id] = set()
                user_test_items[user_id].add(item_id)
            
            # Generate recommendations for each user
            precision_sum = 0
            recall_sum = 0
            user_count = 0
            
            for user_id, test_items in user_test_items.items():
                if user_id in self.user_id_map:
                    # Generate recommendations
                    recommendations = self.generate_recommendations(user_id, num_recommendations=10)
                    if recommendations:
                        recommended_items = [item['item_id'] for item in recommendations]
                        
                        # Compute precision and recall
                        hit_count = len(set(recommended_items) & test_items)
                        precision = hit_count / len(recommended_items) if recommended_items else 0
                        recall = hit_count / len(test_items) if test_items else 0
                        
                        precision_sum += precision
                        recall_sum += recall
                        user_count += 1
            
            # Compute average precision and recall
            if user_count > 0:
                results['precision'] = precision_sum / user_count
                results['recall'] = recall_sum / user_count
        
        if 'hallucination_rate' in metrics:
            # Evaluate hallucination rate on a sample of users
            hallucination_count = 0
            total_recommendations = 0
            
            sample_users = np.random.choice(list(self.user_id_map.keys()), 
                                           size=min(50, len(self.user_id_map)), 
                                           replace=False)
            
            for user_id in sample_users:
                # Construct query and knowledge base
                query, query_embedding = self.construct_rag_query(user_id)
                knowledge_base, _ = self.construct_knowledge_base(user_id)
                retrieved_items = self.retrieve_items(user_id, query_embedding, knowledge_base)
                
                # Format retrieved items for prompt
                item_descriptions = "\n".join([f"- {item['name']}" for item in retrieved_items[:10]])
                
                # Get agent
                agent = self.agents[user_id]
                
                # Create prompt for LLM
                prompt = f"""
                You are a recommendation system for a user with the following traits:
                {agent.traits}
                
                Based on the user's profile and past behavior, you have retrieved the following relevant items:
                {item_descriptions}
                
                Please recommend 5 items from the list above that would be most relevant for this user.
                Provide a brief explanation for each recommendation.
                
                IMPORTANT: You must ONLY recommend items from the provided list. Do not suggest any items that are not in the list.
                
                Format your response as:
                1. [Item Name]: [Explanation]
                2. [Item Name]: [Explanation]
                ...
                """
                
                # Generate recommendations
                response = LLM.invoke(prompt)
                recommendations_text = response.content
                
                # Extract recommended items from response
                recommended_items = []
                lines = recommendations_text.strip().split('\n')
                for line in lines:
                    if line.strip() and any(char.isdigit() for char in line[:5]):
                        parts = line.split(':', 1)
                        if len(parts) > 0:
                            item_name_part = parts[0].strip()
                            item_name = item_name_part.split('.', 1)[1].strip() if '.' in item_name_part else item_name_part
                            recommended_items.append(item_name)
                
                # Check for hallucinations
                retrieved_item_names = [item['name'] for item in retrieved_items]
                for item_name in recommended_items:
                    total_recommendations += 1
                    is_hallucination = True
                    for retrieved_name in retrieved_item_names:
                        if item_name == retrieved_name:
                            is_hallucination = False
                            break
                    
                    if is_hallucination:
                        hallucination_count += 1
            
            # Compute hallucination rate
            if total_recommendations > 0:
                results['hallucination_rate'] = hallucination_count / total_recommendations
        
        return results

class HybridHallAgent4Rec(HallAgent4Rec):
    def __init__(self, agent_usage_threshold=0.8, cold_start_threshold=5, **kwargs):
        """
        Initialize hybrid approach that combines matrix factorization with selective agent usage.
        
        Args:
            agent_usage_threshold: Score threshold above which to use matrix factorization directly
                                  (avoiding agent API calls)
            cold_start_threshold: Number of interactions below which a user is considered "cold start"
                                 and will use an agent
            **kwargs: Other parameters for HallAgent4Rec
        """
        super().__init__(**kwargs)
        self.agent_usage_threshold = agent_usage_threshold
        self.cold_start_threshold = cold_start_threshold
        self.prototype_agent = None  # Single agent used for cold start users
    
    def train(self):
        """Train without initializing agents for all users."""
        # Step 1: Cluster items
        self.cluster_items()
        
        # Step 2: Initialize matrix factorization
        self.matrix_factorization()
        
        # Step 3: Create one prototype agent for cold start users
        self._initialize_prototype_agent()
        
        print("HybridHallAgent4Rec training complete!")
    
    def _initialize_prototype_agent(self):
        """Initialize a single prototype agent for cold start cases."""
        print("Initializing prototype agent for cold start users...")
        
        # Create a generic profile based on average user
        traits = {}
        
        # Average numeric traits
        numeric_columns = [col for col in self.user_data.columns 
                          if col != 'user_id' and pd.api.types.is_numeric_dtype(self.user_data[col])]
        
        for col in numeric_columns:
            avg_value = self.user_data[col].mean()
            traits[col] = avg_value
        
        # Mode for categorical traits
        categorical_columns = [col for col in self.user_data.columns 
                             if col != 'user_id' and not pd.api.types.is_numeric_dtype(self.user_data[col])]
        
        for col in categorical_columns:
            if not self.user_data[col].empty:
                mode_value = self.user_data[col].mode().iloc[0]
                traits[col] = mode_value
        
        # Format traits as a string
        traits_str = ", ".join([f"{k}: {v}" for k, v in traits.items()])
        
        # Create agent memory
        memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=30,
        )
        
        # Create prototype agent
        self.prototype_agent = GenerativeAgent(
            name="Prototype_Agent",
            age=int(traits.get('age', 30)) if 'age' in traits else 30,
            traits=traits_str,
            status="helping users find recommendations",
            memory_retriever=create_new_memory_retriever(),
            llm=LLM,
            memory=memory,
        )
        
        # Add some generic memories about popular items
        # Find top 5 most popular items
        item_popularity = {}
        for _, row in self.interactions.iterrows():
            item_id = row['item_id']
            item_popularity[item_id] = item_popularity.get(item_id, 0) + 1
        
        popular_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for item_id, _ in popular_items:
            item_row = self.item_data[self.item_data['item_id'] == item_id]
            if not item_row.empty:
                item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                memory_content = f"Many users like {item_name} (ID: {item_id})"
                
                # Add to agent memory
                self.prototype_agent.memory.add_memory(memory_content)
        
        print("Prototype agent initialized")
    
    def _is_cold_start_user(self, user_id):
        """Determine if a user is a cold start case (few interactions)."""
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        return len(user_interactions) < self.cold_start_threshold
    
    def _create_user_specific_agent(self, user_id):
        """Create an agent specific to a user (for cold start cases)."""
        # Get user data
        user_row = self.user_data[self.user_data['user_id'] == user_id].iloc[0]
        
        # Extract user traits
        traits = {}
        for col in self.user_data.columns:
            if col != 'user_id':
                traits[col] = user_row[col]
        
        # Format traits as a string
        traits_str = ", ".join([f"{k}: {v}" for k, v in traits.items()])
        
        # Create agent memory (reusing prototype agent's memory retriever to save API calls)
        memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=self.prototype_agent.memory.memory_retriever,
            verbose=False,
            reflection_threshold=30,
        )
        
        # Create generative agent
        agent = GenerativeAgent(
            name=f"User_{user_id}",
            age=traits.get('age', 30),
            traits=traits_str,
            status="looking for recommendations",
            memory_retriever=self.prototype_agent.memory.memory_retriever,  # Reuse retriever
            llm=LLM,
            memory=memory,
        )
        
        # Add user interactions as memories to the agent
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        for _, interaction in user_interactions.iterrows():
            item_id = interaction['item_id']
            # Get item details
            item_row = self.item_data[self.item_data['item_id'] == item_id]
            if not item_row.empty:
                item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                memory_content = f"I interacted with {item_name} (ID: {item_id})"
                
                # Add to agent memory
                agent.memory.add_memory(memory_content)
        
        return agent
    
    def _get_mf_recommendations(self, user_id, num_recommendations=5):
        """Get recommendations using only matrix factorization."""
        print(f"Using matrix factorization for user {user_id} (skipping agent)...")
        
        # Get user index
        user_idx = self.user_id_map[user_id]
        
        # Get prediction scores for all items
        predicted_scores = np.dot(self.user_embeddings[user_idx], self.item_embeddings.T)
        
        # Apply hallucination penalty from the hallucination scores
        if self.hallucination_scores is not None:
            hallucination_penalties = self.hallucination_scores[user_idx]
            predicted_scores -= self.lambda_h * hallucination_penalties * predicted_scores
        
        # Get top items
        top_item_indices = np.argsort(predicted_scores)[::-1][:num_recommendations]
        
        # Create recommendation list
        recommendations = []
        for item_idx in top_item_indices:
            item_id = self.idx_to_item_id[item_idx]
            item_row = self.item_data[self.item_data['item_id'] == item_id]
            if not item_row.empty:
                item_info = {}
                for col in item_row.columns:
                    item_info[col] = item_row.iloc[0][col]
                recommendations.append(item_info)
        
        return recommendations
    
    def generate_recommendations(self, user_id, num_recommendations=5):
        """Generate recommendations using either matrix factorization or agent-based approach."""
        # Check if user exists
        if user_id not in self.user_id_map:
            print(f"User {user_id} not found. Cannot generate recommendations.")
            return []
        
        # Check if user is a cold start case
        is_cold_start = self._is_cold_start_user(user_id)
        
        if is_cold_start:
            print(f"User {user_id} is a cold start case. Using agent-based approach.")
            # Create a user-specific agent for this cold start user
            agent = self._create_user_specific_agent(user_id)
            
            # Use full agent-based recommendation approach
            # This follows same pattern as original implementation but with the new agent
            
            # Construct query using agent
            user_traits = agent.traits
            relevant_memories = agent.memory.memory_retriever.get_relevant_documents("What do I like?")
            memory_contents = " ".join([mem.page_content for mem in relevant_memories])
            query = f"User traits: {user_traits}. User memories: {memory_contents}"
            query_embedding = embeddings_model.embed_query(query)
            
            knowledge_base, top_cluster = self.construct_knowledge_base(user_id)
            retrieved_items = self.retrieve_items(user_id, query_embedding, knowledge_base)
            
            # Process agent recommendations
            # ...
            
            # Format retrieved items for prompt
            item_descriptions = "\n".join([f"- {item['name']}" for item in retrieved_items[:10]])
            
            # Create prompt for LLM
            prompt = f"""
            You are a recommendation system for a user with the following traits:
            {agent.traits}
            
            Based on the user's profile and past behavior, you have retrieved the following relevant items:
            {item_descriptions}
            
            Please recommend {num_recommendations} items from the list above that would be most relevant for this user.
            For each recommendation, provide a brief explanation of why it matches the user's preferences.
            
            IMPORTANT: You must ONLY recommend items from the provided list. Do not suggest any items that are not in the list.
            
            Format your response as:
            1. [Item Name]: [Explanation]
            2. [Item Name]: [Explanation]
            ...
            """
            
            # Generate recommendations
            response = LLM.invoke(prompt)
            recommendations_text = response.content
            
            # Process recommendations and check for hallucinations
            # This follows the same pattern as the original implementation
            # ...
            
            # Extract recommended items and check for hallucinations
            recommended_items = []
            lines = recommendations_text.strip().split('\n')
            for line in lines:
                if line.strip() and any(char.isdigit() for char in line[:5]):
                    parts = line.split(':', 1)
                    if len(parts) > 0:
                        item_name_part = parts[0].strip()
                        item_name = item_name_part.split('.', 1)[1].strip() if '.' in item_name_part else item_name_part
                        recommended_items.append(item_name)
            
            # Check for hallucinations (items not in retrieved set)
            retrieved_item_names = [item['name'] for item in retrieved_items]
            hallucinations = []
            valid_recommendations = []
            
            for item_name in recommended_items:
                is_hallucination = True
                for retrieved_name in retrieved_item_names:
                    if item_name.lower() in retrieved_name.lower() or retrieved_name.lower() in item_name.lower():
                        is_hallucination = False
                        for item in retrieved_items:
                            if item['name'].lower() in item_name.lower() or item_name.lower() in item['name'].lower():
                                valid_recommendations.append(item)
                                break
                        break
                
                if is_hallucination:
                    hallucinations.append(item_name)
            
            # Fill with top predicted items if needed
            if len(valid_recommendations) < num_recommendations:
                # This follows the same pattern as the original implementation
                # ...
                user_idx = self.user_id_map[user_id]
                cluster_items = self.items_by_cluster[top_cluster]
                
                # Get predicted scores
                item_scores = [(idx, np.dot(self.user_embeddings[user_idx], self.item_embeddings[idx])) 
                            for idx in cluster_items]
                
                # Add additional items
                recommended_ids = [item['item_id'] for item in valid_recommendations]
                additional_items = []
                
                for item_idx, score in sorted(item_scores, key=lambda x: x[1], reverse=True):
                    item_id = self.idx_to_item_id[item_idx]
                    if item_id not in recommended_ids:
                        item_row = self.item_data[self.item_data['item_id'] == item_id]
                        if not item_row.empty:
                            item_info = {}
                            for col in item_row.columns:
                                item_info[col] = item_row.iloc[0][col]
                            additional_items.append(item_info)
                            if len(valid_recommendations) + len(additional_items) >= num_recommendations:
                                break
                
                valid_recommendations.extend(additional_items)
            
            # Limit to requested number
            recommendations = valid_recommendations[:num_recommendations]
            
        else:
            # For non-cold-start users, use matrix factorization directly
            recommendations = self._get_mf_recommendations(user_id, num_recommendations)
        
        print(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    

class ClusteredHallAgent4Rec(HallAgent4Rec):
    def __init__(self, num_user_clusters=10, **kwargs):
        """
        Initialize with user clustering to reduce the number of agents needed.
        
        Args:
            num_user_clusters: Number of user clusters to create (instead of one agent per user)
            **kwargs: Other parameters for HallAgent4Rec
        """
        super().__init__(**kwargs)
        self.num_user_clusters = num_user_clusters
        self.user_cluster_model = None
        self.user_cluster_assignments = None
        self.cluster_agents = {}  # Maps cluster_id -> agent
        self.user_to_cluster = {}  # Maps user_id -> cluster_id
    
    def cluster_users(self):
        """Cluster users based on their interaction patterns and traits."""
        print(f"Clustering {len(self.user_data)} users into {self.num_user_clusters} clusters...")
        
        # Create user feature matrix
        user_features = []
        
        # 1. Incorporate interaction patterns
        for user_id in self.user_data['user_id']:
            user_idx = self.user_id_map[user_id]
            # Get user's item interaction vector
            interaction_vector = self.interaction_matrix[user_idx]
            
            # 2. If user data contains demographic features, add them
            user_row = self.user_data[self.user_data['user_id'] == user_id].iloc[0]
            demographic_features = []
            
            for col in self.user_data.columns:
                if col != 'user_id':
                    # Convert categorical features to numeric if needed
                    val = user_row[col]
                    if isinstance(val, (str)):
                        continue  # Skip categorical for simplicity
                    if isinstance(val, (int, float)):
                        demographic_features.append(val)
            
            # Combine interaction and demographic features
            # You might want to normalize these features or apply weights
            combined_features = np.concatenate([interaction_vector, demographic_features])
            user_features.append(combined_features)
        
        # Convert to numpy array
        user_features = np.array(user_features)
        
        # Fill NaN values with 0
        user_features = np.nan_to_num(user_features)
        
        # Apply k-means clustering to users
        self.user_cluster_model = KMeans(n_clusters=self.num_user_clusters, random_state=42)
        self.user_cluster_assignments = self.user_cluster_model.fit_predict(user_features)
        
        # Map users to clusters
        for i, user_id in enumerate(self.user_data['user_id']):
            cluster_id = self.user_cluster_assignments[i]
            self.user_to_cluster[user_id] = cluster_id
        
        print(f"User clustering complete. Cluster distribution:")
        cluster_counts = {}
        for cluster_id in range(self.num_user_clusters):
            count = sum(1 for cid in self.user_cluster_assignments if cid == cluster_id)
            cluster_counts[cluster_id] = count
            print(f"Cluster {cluster_id}: {count} users")
    
    def initialize_representative_agents(self):
        """Initialize one agent per user cluster instead of per user."""
        print(f"Initializing {self.num_user_clusters} representative agents (instead of {len(self.user_data)} user agents)...")
        
        for cluster_id in range(self.num_user_clusters):
            # Find users in this cluster
            cluster_users = [user_id for user_id, cid in self.user_to_cluster.items() if cid == cluster_id]
            
            if not cluster_users:
                continue
                
            # Compute average traits and interactions for this cluster
            cluster_traits = {}
            
            # 1. Average numeric traits
            numeric_columns = [col for col in self.user_data.columns 
                              if col != 'user_id' and pd.api.types.is_numeric_dtype(self.user_data[col])]
            
            for col in numeric_columns:
                avg_value = self.user_data[self.user_data['user_id'].isin(cluster_users)][col].mean()
                cluster_traits[col] = avg_value
            
            # 2. Mode for categorical traits
            categorical_columns = [col for col in self.user_data.columns 
                                 if col != 'user_id' and not pd.api.types.is_numeric_dtype(self.user_data[col])]
            
            for col in categorical_columns:
                mode_value = self.user_data[self.user_data['user_id'].isin(cluster_users)][col].mode().iloc[0]
                cluster_traits[col] = mode_value
            
            # Format traits as a string
            traits_str = ", ".join([f"{k}: {v}" for k, v in cluster_traits.items()])
            
            # Create agent memory
            memory = GenerativeAgentMemory(
                llm=LLM,
                memory_retriever=create_new_memory_retriever(),
                verbose=False,
                reflection_threshold=30,
            )
            
            # Create representative generative agent for this cluster
            agent = GenerativeAgent(
                name=f"Cluster_{cluster_id}",
                age=int(cluster_traits.get('age', 30)) if 'age' in cluster_traits else 30,
                traits=traits_str,
                status="representing a group of similar users",
                memory_retriever=create_new_memory_retriever(),
                llm=LLM,
                memory=memory,
            )
            
            # Store the agent
            self.cluster_agents[cluster_id] = agent
            
            # Add representative memories from this cluster
            # Take a sample of interactions from users in this cluster
            cluster_interactions = self.interactions[self.interactions['user_id'].isin(cluster_users)]
            if len(cluster_interactions) > 10:
                # Sample up to 10 interactions per cluster to avoid excessive API calls
                sampled_interactions = cluster_interactions.sample(10, random_state=42)
            else:
                sampled_interactions = cluster_interactions
                
            for _, interaction in sampled_interactions.iterrows():
                item_id = interaction['item_id']
                if 'timestamp' in interaction:
                    timestamp = interaction['timestamp']
                    created_at = datetime.fromtimestamp(timestamp) if isinstance(timestamp, (int, float)) else datetime.now()
                else:
                    created_at = datetime.now()
                
                # Get item details
                item_row = self.item_data[self.item_data['item_id'] == item_id]
                if not item_row.empty:
                    item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                    memory_content = f"A user in my cluster interacted with {item_name} (ID: {item_id})"
                    
                    # Add to agent memory
                    agent.memory.add_memory(memory_content, created_at=created_at)
        
        print(f"Initialized {len(self.cluster_agents)} cluster representative agents")
    
    def train(self):
        """Train the modified HallAgent4Rec model with user clustering."""
        # Step 1: Cluster items (same as original)
        self.cluster_items()
        
        # Step 2: Initialize matrix factorization (same as original)
        self.matrix_factorization()
        
        # Step 3: Cluster users (new step)
        self.cluster_users()
        
        # Step 4: Initialize representative agents (instead of per-user agents)
        self.initialize_representative_agents()
        
        print("ClusteredHallAgent4Rec training complete!")
    
    def generate_recommendations(self, user_id, num_recommendations=5):
        """
        Generate recommendations for a user using the representative agent for their cluster.
        """
        print(f"Generating recommendations for user {user_id}...")
        
        # Get user's cluster
        cluster_id = self.user_to_cluster.get(user_id)
        if cluster_id is None:
            print(f"User {user_id} not found in clustering. Cannot generate recommendations.")
            return []
        
        # Get the representative agent for this cluster
        agent = self.cluster_agents.get(cluster_id)
        if agent is None:
            print(f"No agent for cluster {cluster_id}. Cannot generate recommendations.")
            return []
        
        # Continue with the recommendation process using the representative agent
        # Construct query using the cluster agent, but user-specific interaction data
        user_idx = self.user_id_map[user_id]
        
        # Step 1: Construct RAG query
        # Start with the cluster agent's traits
        query = f"User traits: {agent.traits}."
        
        # Add user-specific recent interactions if available
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        if not user_interactions.empty:
            recent_interactions = user_interactions.sort_values('timestamp', ascending=False).head(5)
            interaction_texts = []
            for _, interaction in recent_interactions.iterrows():
                item_id = interaction['item_id']
                item_row = self.item_data[self.item_data['item_id'] == item_id]
                if not item_row.empty:
                    item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                    interaction_texts.append(f"interacted with {item_name}")
            
            if interaction_texts:
                query += f" Recent activities: user {', '.join(interaction_texts)}."
        
        # Get query embedding
        query_embedding = embeddings_model.embed_query(query)
        
        # Steps 2-5: Continue with knowledge base construction, retrieval, and recommendation generation
        # These steps remain the same as in the original implementation
        knowledge_base, top_cluster = self.construct_knowledge_base(user_id)
        retrieved_items = self.retrieve_items(user_id, query_embedding, knowledge_base)
        
        # If no items retrieved, return empty list
        if not retrieved_items:
            print("No relevant items retrieved. Cannot generate recommendations.")
            return []
        
        # Format retrieved items for prompt
        item_descriptions = "\n".join([f"- {item['name']}" for item in retrieved_items[:10]])
        
        # Create prompt for LLM
        prompt = f"""
        You are a recommendation system for a user with the following traits:
        {agent.traits}
        
        Based on the user's profile and past behavior, you have retrieved the following relevant items:
        {item_descriptions}
        
        Please recommend {num_recommendations} items from the list above that would be most relevant for this user.
        For each recommendation, provide a brief explanation of why it matches the user's preferences.
        
        IMPORTANT: You must ONLY recommend items from the provided list. Do not suggest any items that are not in the list.
        
        Format your response as:
        1. [Item Name]: [Explanation]
        2. [Item Name]: [Explanation]
        ...
        """
        
        # Generate recommendations
        response = LLM.invoke(prompt)
        recommendations_text = response.content
        
        # Process the response to extract recommendations and check for hallucinations
        # This part remains the same as in the original implementation
        # ...
        
        # Extract recommended items and check for hallucinations
        recommended_items = []
        lines = recommendations_text.strip().split('\n')
        for line in lines:
            if line.strip() and any(char.isdigit() for char in line[:5]):
                parts = line.split(':', 1)
                if len(parts) > 0:
                    item_name_part = parts[0].strip()
                    item_name = item_name_part.split('.', 1)[1].strip() if '.' in item_name_part else item_name_part
                    recommended_items.append(item_name)
        
        # Check for hallucinations (items not in retrieved set)
        retrieved_item_names = [item['name'] for item in retrieved_items]
        hallucinations = []
        valid_recommendations = []
        
        for item_name in recommended_items:
            is_hallucination = True
            for retrieved_name in retrieved_item_names:
                if item_name.lower() in retrieved_name.lower() or retrieved_name.lower() in item_name.lower():
                    is_hallucination = False
                    for item in retrieved_items:
                        if item['name'].lower() in item_name.lower() or item_name.lower() in item['name'].lower():
                            valid_recommendations.append(item)
                            break
                    break
            
            if is_hallucination:
                hallucinations.append(item_name)
        
        # Fill with top predicted items if needed
        if len(valid_recommendations) < num_recommendations:
            # This remains the same as the original implementation
            # ...
            user_idx = self.user_id_map[user_id]
            cluster_items = self.items_by_cluster[top_cluster]
            
            # Get predicted scores
            item_scores = [(idx, np.dot(self.user_embeddings[user_idx], self.item_embeddings[idx])) 
                        for idx in cluster_items]
            
            # Add additional items
            recommended_ids = [item['item_id'] for item in valid_recommendations]
            additional_items = []
            
            for item_idx, score in sorted(item_scores, key=lambda x: x[1], reverse=True):
                item_id = self.idx_to_item_id[item_idx]
                if item_id not in recommended_ids:
                    item_row = self.item_data[self.item_data['item_id'] == item_id]
                    if not item_row.empty:
                        item_info = {}
                        for col in item_row.columns:
                            item_info[col] = item_row.iloc[0][col]
                        additional_items.append(item_info)
                        if len(valid_recommendations) + len(additional_items) >= num_recommendations:
                            break
            
            valid_recommendations.extend(additional_items)
        
        # Limit to requested number
        valid_recommendations = valid_recommendations[:num_recommendations]
        
        print(f"Generated {len(valid_recommendations)} valid recommendations")
        return valid_recommendations
    

class BatchHallAgent4Rec(HallAgent4Rec):
    def __init__(self, batch_size=5, **kwargs):
        """
        Initialize with batch processing capabilities to optimize API calls.
        
        Args:
            batch_size: Number of users to process in a single batch
            **kwargs: Other parameters for HallAgent4Rec
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.embedding_cache = {}  # Cache for embeddings to avoid recomputation
        self.recommendation_cache = {}  # Cache for recommendations
    
    def _batch_embed_queries(self, queries):
        """Batch process embeddings to reduce API calls."""
        # Check cache first
        results = []
        queries_to_embed = []
        cache_indices = []
        
        for i, query in enumerate(queries):
            if query in self.embedding_cache:
                results.append(self.embedding_cache[query])
            else:
                queries_to_embed.append(query)
                cache_indices.append(i)
        
        # If we have queries that need embedding
        if queries_to_embed:
            # This is where you would batch process embeddings if the API supported it
            # For Google Embeddings, we currently have to process one at a time
            embeddings = []
            for query in queries_to_embed:
                embedding = embeddings_model.embed_query(query)
                self.embedding_cache[query] = embedding
                embeddings.append(embedding)
            
            # Insert embeddings back into results at the correct positions
            for i, idx in enumerate(cache_indices):
                results.insert(idx, embeddings[i])
        
        return results
    
    def batch_generate_recommendations(self, user_ids, num_recommendations=5):
        """
        Generate recommendations for multiple users in a batch to optimize API calls.
        
        Args:
            user_ids: List of user IDs to generate recommendations for
            num_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_ids to their recommendations
        """
        print(f"Generating recommendations for {len(user_ids)} users in batches of {self.batch_size}...")
        
        all_recommendations = {}
        
        # Process users in batches
        for i in range(0, len(user_ids), self.batch_size):
            batch_user_ids = user_ids[i:i+self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}: users {batch_user_ids}")
            
            # 1. Batch prepare queries
            queries = []
            query_embeddings = []
            
            for user_id in batch_user_ids:
                # Check cache first
                if user_id in self.recommendation_cache:
                    all_recommendations[user_id] = self.recommendation_cache[user_id]
                    continue
                
                # Get user index
                if user_id not in self.user_id_map:
                    print(f"User {user_id} not found. Skipping.")
                    all_recommendations[user_id] = []
                    continue
                
                # Construct query for this user (simplified version without agent)
                user_idx = self.user_id_map[user_id]
                user_row = self.user_data[self.user_data['user_id'] == user_id].iloc[0]
                
                # Build query from user traits
                user_traits = ", ".join([f"{k}: {v}" for k, v in user_row.items() if k != 'user_id'])
                
                # Add recent interactions
                user_interactions = self.interactions[self.interactions['user_id'] == user_id]
                if not user_interactions.empty:
                    recent_interactions = user_interactions.sort_values('timestamp', ascending=False).head(3)
                    interaction_texts = []
                    for _, interaction in recent_interactions.iterrows():
                        item_id = interaction['item_id']
                        item_row = self.item_data[self.item_data['item_id'] == item_id]
                        if not item_row.empty:
                            item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                            interaction_texts.append(f"interacted with {item_name}")
                    
                    interaction_str = ", ".join(interaction_texts)
                    query = f"User traits: {user_traits}. Recent activities: {interaction_str}"
                else:
                    query = f"User traits: {user_traits}."
                
                queries.append((user_id, query))
            
            # 2. Batch embed queries
            if queries:
                query_texts = [q[1] for q in queries]
                batch_embeddings = self._batch_embed_queries(query_texts)
                
                # 3. For each user in the batch, process recommendations
                for idx, (user_id, query) in enumerate(queries):
                    query_embedding = batch_embeddings[idx]
                    
                    # Get knowledge base and retrieved items
                    knowledge_base, top_cluster = self.construct_knowledge_base(user_id)
                    retrieved_items = self.retrieve_items(user_id, query_embedding, knowledge_base)
                    
                    # If no items retrieved
                    if not retrieved_items:
                        all_recommendations[user_id] = []
                        continue
                    
                    # Format retrieved items for prompt
                    item_descriptions = "\n".join([f"- {item['name']}" for item in retrieved_items[:10]])
                    
                    # Create prompt for LLM
                    prompt = f"""
                    You are a recommendation system for a user with the following traits:
                    {query.split('User traits: ')[1]}
                    
                    Based on the user's profile and past behavior, you have retrieved the following relevant items:
                    {item_descriptions}
                    
                    Please recommend {num_recommendations} items from the list above that would be most relevant for this user.
                    For each recommendation, provide a brief explanation of why it matches the user's preferences.
                    
                    IMPORTANT: You must ONLY recommend items from the provided list. Do not suggest any items that are not in the list.
                    
                    Format your response as:
                    1. [Item Name]: [Explanation]
                    2. [Item Name]: [Explanation]
                    ...
                    """
                    
                    # Generate recommendations
                    # Here we can't easily batch LLM calls, so process one at a time
                    response = LLM.invoke(prompt)
                    recommendations_text = response.content
                    
                    # Process the response (same as original implementation)
                    recommended_items = []
                    lines = recommendations_text.strip().split('\n')
                    for line in lines:
                        if line.strip() and any(char.isdigit() for char in line[:5]):
                            parts = line.split(':', 1)
                            if len(parts) > 0:
                                item_name_part = parts[0].strip()
                                item_name = item_name_part.split('.', 1)[1].strip() if '.' in item_name_part else item_name_part
                                recommended_items.append(item_name)
                    
                    # Check for hallucinations (items not in retrieved set)
                    retrieved_item_names = [item['name'] for item in retrieved_items]
                    valid_recommendations = []
                    
                    for item_name in recommended_items:
                        for retrieved_name in retrieved_item_names:
                            if item_name.lower() in retrieved_name.lower() or retrieved_name.lower() in item_name.lower():
                                for item in retrieved_items:
                                    if item['name'].lower() in item_name.lower() or item_name.lower() in item['name'].lower():
                                        valid_recommendations.append(item)
                                        break
                                break
                    
                    # Fill with top predicted items if needed
                    if len(valid_recommendations) < num_recommendations:
                        user_idx = self.user_id_map[user_id]
                        cluster_items = self.items_by_cluster[top_cluster]
                        
                        # Get predicted scores
                        item_scores = [(idx, np.dot(self.user_embeddings[user_idx], self.item_embeddings[idx])) 
                                    for idx in cluster_items]
                        
                        # Add additional items
                        recommended_ids = [item['item_id'] for item in valid_recommendations]
                        additional_items = []
                        
                        for item_idx, score in sorted(item_scores, key=lambda x: x[1], reverse=True):
                            item_id = self.idx_to_item_id[item_idx]
                            if item_id not in recommended_ids:
                                item_row = self.item_data[self.item_data['item_id'] == item_id]
                                if not item_row.empty:
                                    item_info = {}
                                    for col in item_row.columns:
                                        item_info[col] = item_row.iloc[0][col]
                                    additional_items.append(item_info)
                                    if len(valid_recommendations) + len(additional_items) >= num_recommendations:
                                        break
                        
                        valid_recommendations.extend(additional_items)
                    
                    # Limit to requested number
                    recommendations = valid_recommendations[:num_recommendations]
                    
                    # Store in results and cache
                    all_recommendations[user_id] = recommendations
                    self.recommendation_cache[user_id] = recommendations
        
        return all_recommendations
    
    def clear_caches(self):
        """Clear embedding and recommendation caches."""
        self.embedding_cache = {}
        self.recommendation_cache = {}
        print("Caches cleared")


class EnhancedClusterHallAgent4Rec(HallAgent4Rec):
    def __init__(self, num_user_clusters=10, personalization_weight=0.7, **kwargs):
        """
        Initialize with enhanced user clustering that preserves personalization.
        
        Args:
            num_user_clusters: Number of user clusters to create
            personalization_weight: Weight given to user-specific preferences vs. cluster preferences
                                    (higher = more personalized, range 0-1)
            **kwargs: Other parameters for HallAgent4Rec
        """
        super().__init__(**kwargs)
        self.num_user_clusters = num_user_clusters
        self.user_cluster_model = None
        self.user_cluster_assignments = None
        self.cluster_agents = {}  # Maps cluster_id -> agent
        self.user_to_cluster = {}  # Maps user_id -> cluster_id
        self.personalization_weight = personalization_weight
        self.cluster_item_ratings = {}  # Average ratings per cluster for each item
    
    def cluster_users(self):
        """Cluster users based on their interaction patterns and traits."""
        print(f"Clustering {len(self.user_data)} users into {self.num_user_clusters} clusters...")
        
        # Create user feature matrix
        user_features = []
        
        # 1. Incorporate interaction patterns
        for user_id in self.user_data['user_id']:
            user_idx = self.user_id_map[user_id]
            # Get user's item interaction vector
            interaction_vector = self.interaction_matrix[user_idx]
            
            # 2. If user data contains demographic features, add them
            user_row = self.user_data[self.user_data['user_id'] == user_id].iloc[0]
            demographic_features = []
            
            for col in self.user_data.columns:
                if col != 'user_id':
                    # Convert categorical features to numeric if needed
                    val = user_row[col]
                    if isinstance(val, (str)):
                        continue  # Skip categorical for simplicity
                    if isinstance(val, (int, float)):
                        demographic_features.append(val)
            
            # Combine interaction and demographic features
            # You might want to normalize these features or apply weights
            combined_features = np.concatenate([interaction_vector, demographic_features])
            user_features.append(combined_features)
        
        # Convert to numpy array
        user_features = np.array(user_features)
        
        # Fill NaN values with 0
        user_features = np.nan_to_num(user_features)
        
        # Apply k-means clustering to users
        self.user_cluster_model = KMeans(n_clusters=self.num_user_clusters, random_state=42)
        self.user_cluster_assignments = self.user_cluster_model.fit_predict(user_features)
        
        # Map users to clusters
        for i, user_id in enumerate(self.user_data['user_id']):
            cluster_id = self.user_cluster_assignments[i]
            self.user_to_cluster[user_id] = cluster_id
        
        print(f"User clustering complete. Cluster distribution:")
        cluster_counts = {}
        for cluster_id in range(self.num_user_clusters):
            count = sum(1 for cid in self.user_cluster_assignments if cid == cluster_id)
            cluster_counts[cluster_id] = count
            print(f"Cluster {cluster_id}: {count} users")
            
        # Compute cluster-level item ratings
        self.compute_cluster_item_ratings()
    
    def compute_cluster_item_ratings(self):
        """
        Compute average item ratings for each cluster.
        These act as a starting point that will be personalized for each user.
        """
        print("Computing cluster-level item ratings...")
        
        for cluster_id in range(self.num_user_clusters):
            # Get users in this cluster
            cluster_users = [self.user_id_map[user_id] for user_id, cid in self.user_to_cluster.items() 
                            if cid == cluster_id and user_id in self.user_id_map]
            
            if not cluster_users:
                continue
            
            # Get average interaction matrix for this cluster
            cluster_interactions = np.zeros(self.interaction_matrix.shape[1])
            for user_idx in cluster_users:
                cluster_interactions += self.interaction_matrix[user_idx]
            
            # Normalize by number of users
            cluster_interactions /= len(cluster_users)
            
            # Store cluster-level ratings
            self.cluster_item_ratings[cluster_id] = cluster_interactions
    
    def initialize_representative_agents(self):
        """Initialize one agent per user cluster instead of per user."""
        print(f"Initializing {self.num_user_clusters} representative agents (instead of {len(self.user_data)} user agents)...")
        
        for cluster_id in range(self.num_user_clusters):
            # Find users in this cluster
            cluster_users = [user_id for user_id, cid in self.user_to_cluster.items() if cid == cluster_id]
            
            if not cluster_users:
                continue
                
            # Compute average traits and interactions for this cluster
            cluster_traits = {}
            
            # 1. Average numeric traits
            numeric_columns = [col for col in self.user_data.columns 
                              if col != 'user_id' and pd.api.types.is_numeric_dtype(self.user_data[col])]
            
            for col in numeric_columns:
                avg_value = self.user_data[self.user_data['user_id'].isin(cluster_users)][col].mean()
                cluster_traits[col] = avg_value
            
            # 2. Mode for categorical traits
            categorical_columns = [col for col in self.user_data.columns 
                                 if col != 'user_id' and not pd.api.types.is_numeric_dtype(self.user_data[col])]
            
            for col in categorical_columns:
                mode_value = self.user_data[self.user_data['user_id'].isin(cluster_users)][col].mode().iloc[0]
                cluster_traits[col] = mode_value
            
            # Format traits as a string
            traits_str = ", ".join([f"{k}: {v}" for k, v in cluster_traits.items()])
            
            # Create agent memory
            memory = GenerativeAgentMemory(
                llm=LLM,
                memory_retriever=create_new_memory_retriever(),
                verbose=False,
                reflection_threshold=30,
            )
            
            # Create representative generative agent for this cluster
            agent = GenerativeAgent(
                name=f"Cluster_{cluster_id}",
                age=int(cluster_traits.get('age', 30)) if 'age' in cluster_traits else 30,
                traits=traits_str,
                status="representing a group of similar users",
                memory_retriever=create_new_memory_retriever(),
                llm=LLM,
                memory=memory,
            )
            
            # Store the agent
            self.cluster_agents[cluster_id] = agent
            
            # Add representative memories from this cluster
            # Take a sample of interactions from users in this cluster
            cluster_interactions = self.interactions[self.interactions['user_id'].isin(cluster_users)]
            if len(cluster_interactions) > 10:
                # Sample up to 10 interactions per cluster to avoid excessive API calls
                sampled_interactions = cluster_interactions.sample(10, random_state=42)
            else:
                sampled_interactions = cluster_interactions
                
            for _, interaction in sampled_interactions.iterrows():
                item_id = interaction['item_id']
                if 'timestamp' in interaction:
                    timestamp = interaction['timestamp']
                    created_at = datetime.fromtimestamp(timestamp) if isinstance(timestamp, (int, float)) else datetime.now()
                else:
                    created_at = datetime.now()
                
                # Get item details
                item_row = self.item_data[self.item_data['item_id'] == item_id]
                if not item_row.empty:
                    item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                    memory_content = f"A user in my cluster interacted with {item_name} (ID: {item_id})"
                    
                    # Add to agent memory
                    agent.memory.add_memory(memory_content, created_at=created_at)
        
        print(f"Initialized {len(self.cluster_agents)} cluster representative agents")
    
    def train(self):
        """Train the modified HallAgent4Rec model with user clustering."""
        # Step 1: Cluster items (same as original)
        self.cluster_items()
        
        # Step 2: Initialize matrix factorization (same as original)
        self.matrix_factorization()
        
        # Step 3: Cluster users (new step)
        self.cluster_users()
        
        # Step 4: Initialize representative agents (instead of per-user agents)
        self.initialize_representative_agents()
        
        print("EnhancedClusterHallAgent4Rec training complete!")
    
    def get_personalized_item_scores(self, user_id):
        """
        Calculate personalized item scores by blending:
        1. User-specific preferences from matrix factorization
        2. Cluster-level preferences
        
        This demonstrates how personalization is maintained even with shared agents.
        """
        # Get user and cluster info
        user_idx = self.user_id_map[user_id]
        cluster_id = self.user_to_cluster[user_id]
        
        # 1. Get user-specific scores from matrix factorization
        user_specific_scores = np.dot(self.user_embeddings[user_idx], self.item_embeddings.T)
        
        # 2. Get cluster-level scores
        cluster_scores = self.cluster_item_ratings.get(cluster_id, np.zeros_like(user_specific_scores))
        
        # 3. Normalize both score arrays to 0-1 range
        user_specific_scores = (user_specific_scores - user_specific_scores.min()) / (user_specific_scores.max() - user_specific_scores.min() + 1e-10)
        if cluster_scores.max() > cluster_scores.min():
            cluster_scores = (cluster_scores - cluster_scores.min()) / (cluster_scores.max() - cluster_scores.min())
        
        # 4. Blend user-specific and cluster scores using personalization weight
        # Higher weight = more emphasis on individual user preferences
        blended_scores = (self.personalization_weight * user_specific_scores + 
                           (1 - self.personalization_weight) * cluster_scores)
        
        return blended_scores
    
    def generate_recommendations(self, user_id, num_recommendations=5):
        """
        Generate recommendations for a user using the representative agent for their cluster,
        while maintaining personalization through individual user embeddings.
        """
        print(f"Generating recommendations for user {user_id}...")
        
        # Get user's cluster
        cluster_id = self.user_to_cluster.get(user_id)
        if cluster_id is None:
            print(f"User {user_id} not found in clustering. Cannot generate recommendations.")
            return []
        
        # Get the representative agent for this cluster
        agent = self.cluster_agents.get(cluster_id)
        if agent is None:
            print(f"No agent for cluster {cluster_id}. Cannot generate recommendations.")
            return []
        
        # PERSONALIZATION PART 1: Get user-specific item scores
        # This ensures recommendations are personalized even with shared agents
        personalized_scores = self.get_personalized_item_scores(user_id)
        
        # Get user index
        user_idx = self.user_id_map[user_id]
        
        # Step 1: Construct RAG query
        # Start with the cluster agent's traits
        query = f"User traits: {agent.traits}."
        
        # PERSONALIZATION PART 2: Add user-specific recent interactions
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        if not user_interactions.empty:
            recent_interactions = user_interactions.sort_values('timestamp', ascending=False).head(5)
            interaction_texts = []
            for _, interaction in recent_interactions.iterrows():
                item_id = interaction['item_id']
                item_row = self.item_data[self.item_data['item_id'] == item_id]
                if not item_row.empty:
                    item_name = item_row.iloc[0].get('name', f"Item_{item_id}")
                    interaction_texts.append(f"interacted with {item_name}")
            
            if interaction_texts:
                query += f" Recent activities: user {', '.join(interaction_texts)}."
        
        # Get query embedding
        query_embedding = embeddings_model.embed_query(query)
        
        # PERSONALIZATION PART 3: User-specific knowledge base
        # Instead of using the generic construct_knowledge_base method, we'll create a personalized version
        # Get the top cluster based on personalized scores
        item_indices = np.argsort(personalized_scores)[::-1][:100]  # Get top 100 items
        
        # Group these items by their clusters
        cluster_item_counts = {}
        for item_idx in item_indices:
            item_cluster = self.cluster_assignments[item_idx] if item_idx < len(self.cluster_assignments) else 0
            cluster_item_counts[item_cluster] = cluster_item_counts.get(item_cluster, 0) + 1
        
        # Get the top cluster with the most high-scoring items
        top_cluster = max(cluster_item_counts.items(), key=lambda x: x[1])[0]
        
        # Get items in the top cluster
        cluster_items = self.items_by_cluster[top_cluster]
        
        # PERSONALIZATION PART 4: Filter by personalized relevance
        # Filter items by personalized relevance threshold
        relevant_items = [item_idx for item_idx in cluster_items 
                         if personalized_scores[item_idx] >= self.relevance_threshold]
        
        # Create knowledge base with item details
        knowledge_base = []
        for item_idx in relevant_items:
            item_id = self.idx_to_item_id[item_idx]
            item_row = self.item_data[self.item_data['item_id'] == item_id]
            if not item_row.empty:
                item_info = {}
                for col in item_row.columns:
                    item_info[col] = item_row.iloc[0][col]
                knowledge_base.append(item_info)
        
        # Perform retrieval on the personalized knowledge base
        retrieved_items = self.retrieve_items(user_id, query_embedding, knowledge_base)
        
        # If no items retrieved, return empty list
        if not retrieved_items:
            print("No relevant items retrieved. Cannot generate recommendations.")
            return []
        
        # Format retrieved items for prompt
        item_descriptions = "\n".join([f"- {item['name']}" for item in retrieved_items[:10]])
        
        # PERSONALIZATION PART 5: Create a prompt that includes user-specific information
        # Get user-specific traits to blend with cluster traits
        user_row = self.user_data[self.user_data['user_id'] == user_id].iloc[0]
        user_specific_traits = ", ".join([f"{k}: {v}" for k, v in user_row.items() if k != 'user_id'])
        
        # Create prompt for LLM that includes both cluster and user-specific information
        prompt = f"""
        You are a recommendation system for a user with the following cluster traits:
        {agent.traits}
        
        This specific user has these characteristics:
        {user_specific_traits}
        
        Based on this user's specific profile and past behavior, you have retrieved the following relevant items:
        {item_descriptions}
        
        Please recommend {num_recommendations} items from the list above that would be most relevant for this specific user.
        For each recommendation, provide a brief explanation of why it matches the user's preferences.
        
        IMPORTANT: You must ONLY recommend items from the provided list. Do not suggest any items that are not in the list.
        
        Format your response as:
        1. [Item Name]: [Explanation]
        2. [Item Name]: [Explanation]
        ...
        """
        
        # Generate recommendations
        response = LLM.invoke(prompt)
        recommendations_text = response.content
        
        # Extract recommended items and check for hallucinations
        recommended_items = []
        lines = recommendations_text.strip().split('\n')
        for line in lines:
            if line.strip() and any(char.isdigit() for char in line[:5]):
                parts = line.split(':', 1)
                if len(parts) > 0:
                    item_name_part = parts[0].strip()
                    item_name = item_name_part.split('.', 1)[1].strip() if '.' in item_name_part else item_name_part
                    recommended_items.append(item_name)
        
        # Check for hallucinations (items not in retrieved set)
        retrieved_item_names = [item['name'] for item in retrieved_items]
        hallucinations = []
        valid_recommendations = []
        
        for item_name in recommended_items:
            is_hallucination = True
            for retrieved_name in retrieved_item_names:
                if item_name.lower() in retrieved_name.lower() or retrieved_name.lower() in item_name.lower():
                    is_hallucination = False
                    for item in retrieved_items:
                        if item['name'].lower() in item_name.lower() or item_name.lower() in item['name'].lower():
                            valid_recommendations.append(item)
                            break
                    break
            
            if is_hallucination:
                hallucinations.append(item_name)
        
        # PERSONALIZATION PART 6: Fill with top personalized items if needed
        if len(valid_recommendations) < num_recommendations:
            # Get top items based on personalized scores
            top_item_indices = np.argsort(personalized_scores)[::-1]
            
            # Add additional items
            recommended_ids = [item['item_id'] for item in valid_recommendations]
            additional_items = []
            
            for item_idx in top_item_indices:
                if item_idx >= len(self.idx_to_item_id):
                    continue
                    
                item_id = self.idx_to_item_id[item_idx]
                if item_id not in recommended_ids:
                    item_row = self.item_data[self.item_data['item_id'] == item_id]
                    if not item_row.empty:
                        item_info = {}
                        for col in item_row.columns:
                            item_info[col] = item_row.iloc[0][col]
                        additional_items.append(item_info)
                        if len(valid_recommendations) + len(additional_items) >= num_recommendations:
                            break
            
            valid_recommendations.extend(additional_items)
        
        # Limit to requested number
        valid_recommendations = valid_recommendations[:num_recommendations]
        
        print(f"Generated {len(valid_recommendations)} personalized recommendations for user {user_id}")
        return valid_recommendations
    
    def visualize_personalization(self, user_id1, user_id2):
        """
        Visualize how two users in the same cluster still get different recommendations.
        This helps explain the personalization aspect of the system.
        """
        # Check if both users are in the same cluster
        cluster_id1 = self.user_to_cluster.get(user_id1)
        cluster_id2 = self.user_to_cluster.get(user_id2)
        
        if cluster_id1 is None or cluster_id2 is None:
            print("One or both users not found.")
            return
        
        if cluster_id1 != cluster_id2:
            print(f"Users {user_id1} and {user_id2} are in different clusters ({cluster_id1} and {cluster_id2}).")
            print("Please select two users from the same cluster to visualize personalization.")
            return
        
        print(f"Users {user_id1} and {user_id2} are both in cluster {cluster_id1}.")
        print("Generating recommendations for both users to demonstrate personalization...")
        
        # Generate recommendations for both users
        recs1 = self.generate_recommendations(user_id1, num_recommendations=5)
        recs2 = self.generate_recommendations(user_id2, num_recommendations=5)
        
        # Compare recommendations
        rec_ids1 = [item['item_id'] for item in recs1]
        rec_ids2 = [item['item_id'] for item in recs2]
        
        common_items = set(rec_ids1).intersection(set(rec_ids2))
        
        print("\nDemonstrating Personalization:")
        print(f"Recommendations for User {user_id1}:")
        for i, item in enumerate(recs1):
            print(f"  {i+1}. {item['name']} (ID: {item['item_id']})")
        
        print(f"\nRecommendations for User {user_id2}:")
        for i, item in enumerate(recs2):
            print(f"  {i+1}. {item['name']} (ID: {item['item_id']})")
        
        print(f"\nCommon recommendations: {len(common_items)} items")
        print(f"Different recommendations: {5 - len(common_items)} items")
        print(f"Personalization rate: {((5 - len(common_items)) / 5) * 100:.1f}%")
        
        # Show personalized scores for a few items
        print("\nPersonalized scores for sample items:")
        sample_items = list(set(rec_ids1 + rec_ids2))[:5]
        
        scores1 = self.get_personalized_item_scores(user_id1)
        scores2 = self.get_personalized_item_scores(user_id2)
        
        print(f"{'Item ID':<10} | {'User '+str(user_id1):<15} | {'User '+str(user_id2):<15} | Difference")
        print("-" * 55)
        
        for item_id in sample_items:
            item_idx = list(self.idx_to_item_id.keys())[list(self.idx_to_item_id.values()).index(item_id)]
            score1 = scores1[item_idx]
            score2 = scores2[item_idx]
            diff = abs(score1 - score2)
            
            print(f"{item_id:<10} | {score1:<15.4f} | {score2:<15.4f} | {diff:.4f}")