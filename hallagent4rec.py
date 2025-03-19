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
from langchain.callbacks.base import Callbacks 

# Base memory structure (as provided in the notebook)
class MemoryItem(BaseModel):
    content: str
    created_at: datetime
    importance: Optional[float] = 0.0


class BaseCache(BaseModel):
    memories: Dict[str, MemoryItem] = Field(default_factory=dict)
# Initialize the LLM with Google Generative AI
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Initialize the vectorstore as empty
    embedding_size = 768  # fixed to match with GoogleEmbedding
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

# Helper functions for agent creation (from the notebook)
def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # Convert euclidean norm of normalized embeddings to similarity function
    return 1.0 - score / math.sqrt(2)



def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help interact with the agent."""
    new_message = f"{agent.name} says {message}"
    return agent.generate_dialogue_response(new_message)[1]
GenerativeAgentMemory.model_rebuild()








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


# Example usage
def example_usage():
    # Create sample data
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    # User data
    user_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'age': [25, 30, 35, 40, 45],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'occupation': ['student', 'engineer', 'doctor', 'artist', 'teacher']
    })
    
    # Item data
    item_data = pd.DataFrame({
        'item_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'name': ['Item A', 'Item B', 'Item C', 'Item D', 'Item E', 
                 'Item F', 'Item G', 'Item H', 'Item I', 'Item J'],
        'category': ['books', 'electronics', 'books', 'clothing', 'electronics',
                    'books', 'clothing', 'electronics', 'books', 'clothing'],
        'price': [10, 50, 15, 30, 100, 20, 25, 80, 12, 35],
        'popularity': [0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.7, 0.6, 0.3, 0.8]
    })
    label_encoders = {}
    for col in ['gender', 'occupation']:
        le = LabelEncoder()
        user_data[col] = le.fit_transform(user_data[col])
        label_encoders[col] = le

    for col in ['category','name']:
        le = LabelEncoder()
        item_data[col] = le.fit_transform(item_data[col])
        label_encoders[col] = le
    print(item_data.head(2))
    # Interaction data
    interactions = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
        'item_id': [101, 103, 105, 102, 104, 103, 106, 109, 105, 107, 101, 108, 110],
        'timestamp': [1615000000, 1615100000, 1615200000, 1615300000, 1615400000,
                     1615500000, 1615600000, 1615700000, 1615800000, 1615900000,
                     1616000000, 1616100000, 1616200000]
    })
    
    # Test interactions
    test_interactions = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'item_id': [106, 101, 102, 108, 103]
    })
    
    # Initialize HallAgent4Rec
    hall_agent = HallAgent4Rec(num_clusters=3, latent_dim=10)
    
    # Load data
    hall_agent.load_data(user_data, item_data, interactions)
    
    # Train the system
    hall_agent.train()
    
    # Generate recommendations for a user
    recommendations = hall_agent.generate_recommendations(user_id=1, num_recommendations=3)
    print("Recommendations for User 1:")
    for i, item in enumerate(recommendations):
        print(f"{i+1}. {item['name']} - {item['category']} (${item['price']})")
    
    # Evaluate the system
    eval_results = hall_agent.evaluate(test_interactions)
    print("\nEvaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    example_usage()