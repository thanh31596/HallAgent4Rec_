import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import os
import time
from datetime import datetime

from movielens_loader import (
    prepare_movielens_for_hallagent,
    create_movie_prompts,
    create_user_profile_for_llm,
    get_user_genre_preferences
)

from hallagent4rec import HallAgent4Rec


class MovieLensAgent4Rec(HallAgent4Rec):
    """
    Extension of HallAgent4Rec specifically optimized for MovieLens dataset.
    """
    
    def __init__(
        self,
        num_clusters: int = 20,  # More clusters for better movie genre separation
        latent_dim: int = 50,    # Higher dimension for movie embeddings
        lambda_u: float = 0.1,
        lambda_v: float = 0.1,
        lambda_h: float = 1.0,
        learning_rate: float = 0.01,
        decay_rate: float = 0.0001,
        max_iterations: int = 100,
        similarity_threshold: float = 0.5,
        relevance_threshold: float = 0.1,
        use_selective_memory: bool = True,
        max_memories_per_user: int = 10,
        memory_strategy: str = 'diverse'  # 'recent', 'diverse', or 'important'
    ):
        """
        Initialize the MovieLensAgent4Rec system.
        
        Args:
            num_clusters: Number of clusters for movie grouping (typically by genre/style)
            latent_dim: Dimensionality of user and movie embeddings
            lambda_u: User regularization coefficient
            lambda_v: Item regularization coefficient
            lambda_h: Hallucination penalty coefficient
            learning_rate: Initial learning rate for optimization
            decay_rate: Learning rate decay parameter
            max_iterations: Maximum number of iterations for matrix factorization
            similarity_threshold: Threshold for item similarity in retrieval
            relevance_threshold: Threshold for item relevance in knowledge base
            use_selective_memory: Whether to use selective memory for agents
            max_memories_per_user: Maximum number of memories per user if using selective memory
            memory_strategy: Strategy for selecting memories ('recent', 'diverse', 'important')
        """
        super().__init__(
            num_clusters=num_clusters,
            latent_dim=latent_dim,
            lambda_u=lambda_u,
            lambda_v=lambda_v,
            lambda_h=lambda_h,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            max_iterations=max_iterations,
            similarity_threshold=similarity_threshold,
            relevance_threshold=relevance_threshold
        )
        
        # MovieLens specific attributes
        self.movie_prompts = {}
        self.use_selective_memory = use_selective_memory
        self.max_memories_per_user = max_memories_per_user
        self.memory_strategy = memory_strategy
        self.user_genre_preferences = {}
    
    def load_movielens_data(self, data_path: str = "./data", test_ratio: float = 0.2, random_state: int = 42):
        """
        Load MovieLens dataset and prepare it for HallAgent4Rec.
        
        Args:
            data_path: Path to the dataset
            test_ratio: Ratio of interactions to use for testing
            random_state: Random seed for reproducibility
        """
        print("Loading MovieLens dataset...")
        user_data, item_data, train_interactions, test_interactions = prepare_movielens_for_hallagent(
            data_path, test_ratio, random_state
        )
        
        # Create movie prompts for better LLM understanding
        self.movie_prompts = create_movie_prompts(item_data)
        
        # Load data into HallAgent4Rec
        super().load_data(user_data, item_data, train_interactions)
        
        # Store test interactions for evaluation
        self.test_interactions = test_interactions
        
        # Precompute user genre preferences for more efficient agent memory creation
        self._precompute_user_genre_preferences()
        
        print(f"MovieLens data loaded: {len(user_data)} users, {len(item_data)} movies, " 
              f"{len(train_interactions)} training interactions, {len(test_interactions)} test interactions")
    
    def _precompute_user_genre_preferences(self):
        """Precompute genre preferences for each user to use in agent memories."""
        print("Precomputing user genre preferences...")
        for user_id in tqdm(self.user_id_map.keys(), desc="Computing genre preferences"):
            self.user_genre_preferences[user_id] = get_user_genre_preferences(
                user_id, self.interactions, self.item_data
            )
    
    def add_selective_memories(self, agent, user_id, max_memories=None, strategy=None):
        """
        Add selected memories to the agent based on the specified strategy.
        
        Args:
            agent: The agent to add memories to
            user_id: User ID to get interactions for
            max_memories: Maximum number of memories to add (overrides instance setting if provided)
            strategy: Selection strategy ('recent', 'diverse', 'important')
        """
        if max_memories is None:
            max_memories = self.max_memories_per_user
            
        if strategy is None:
            strategy = self.memory_strategy
            
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        
        # If fewer interactions than max memories, use all of them
        if len(user_interactions) <= max_memories:
            selected_interactions = user_interactions
        else:
            if strategy == 'recent':
                # Sort by timestamp and take the most recent
                if 'timestamp' in user_interactions.columns:
                    selected_interactions = user_interactions.sort_values(
                        'timestamp', ascending=False
                    ).head(max_memories)
                else:
                    selected_interactions = user_interactions.iloc[-max_memories:]
                    
            elif strategy == 'diverse':
                # Select movies from different genres
                movie_ids = user_interactions['item_id'].values
                genres = []
                
                for movie_id in movie_ids:
                    movie_row = self.item_data[self.item_data['item_id'] == movie_id]
                    if not movie_row.empty and 'primary_genre' in movie_row.columns:
                        genre = movie_row['primary_genre'].iloc[0]
                        genres.append(genre)
                    else:
                        genres.append('unknown')
                
                user_interactions['genre'] = genres
                
                # Sample from each genre proportionally
                genre_counts = user_interactions['genre'].value_counts()
                selected_indices = []
                
                for genre, count in genre_counts.items():
                    n_samples = max(1, int(count / len(user_interactions) * max_memories))
                    n_samples = min(n_samples, count)  # Don't sample more than available
                    
                    genre_interactions = user_interactions[user_interactions['genre'] == genre]
                    
                    # For each genre, prioritize higher ratings
                    if 'rating' in genre_interactions.columns:
                        sampled = genre_interactions.sort_values('rating', ascending=False).head(n_samples)
                    else:
                        sampled = genre_interactions.sample(n_samples)
                        
                    selected_indices.extend(sampled.index)
                    
                    # Stop if we've reached max_memories
                    if len(selected_indices) >= max_memories:
                        break
                
                # If we need more, add highest rated remaining movies
                if len(selected_indices) < max_memories and 'rating' in user_interactions.columns:
                    remaining = max_memories - len(selected_indices)
                    remaining_interactions = user_interactions[~user_interactions.index.isin(selected_indices)]
                    
                    if len(remaining_interactions) > 0:
                        additional = remaining_interactions.sort_values('rating', ascending=False).head(remaining)
                        selected_indices.extend(additional.index)
                
                selected_interactions = user_interactions.loc[selected_indices[:max_memories]]
                
            elif strategy == 'important':
                # Prioritize movies with extreme ratings (very high or very low)
                if 'rating' in user_interactions.columns:
                    avg_rating = user_interactions['rating'].mean()
                    user_interactions['rating_distance'] = abs(user_interactions['rating'] - avg_rating)
                    selected_interactions = user_interactions.sort_values(
                        'rating_distance', ascending=False
                    ).head(max_memories)
                else:
                    # Fall back to random if no ratings
                    selected_interactions = user_interactions.sample(max_memories)
            else:
                # Default to random sampling
                selected_interactions = user_interactions.sample(max_memories)
        
        # Add selected memories to the agent
        for _, interaction in selected_interactions.iterrows():
            movie_id = interaction['item_id']
            rating = interaction.get('rating', None)
            
            # Get movie details
            movie_row = self.item_data[self.item_data['item_id'] == movie_id]
            if not movie_row.empty:
                movie_title = movie_row['title'].iloc[0]
                
                # Create memory content
                if rating is not None:
                    sentiment = "really liked" if rating >= 4 else (
                        "liked" if rating >= 3 else (
                            "disliked" if rating >= 2 else "really disliked"
                        )
                    )
                    memory_content = f"I watched the movie '{movie_title}' and {sentiment} it (rated {rating}/5)."
                else:
                    memory_content = f"I watched the movie '{movie_title}'."
                
                # Add to agent memory
                agent.memory.add_memory(memory_content)
    
    def initialize_agents(self):
        """Initialize generative agents for each user with MovieLens-specific customization."""
        print("Initializing generative agents for MovieLens users...")
        
        for user_id, user_idx in tqdm(self.user_id_map.items(), desc="Initializing agents"):
            # Get user data
            user_row = self.user_data[self.user_data['user_id'] == user_id].iloc[0]
            
            # Extract user traits from user data
            traits = {}
            for col in self.user_data.columns:
                if col != 'user_id':
                    traits[col] = user_row[col]
            
            # For MovieLens, add genre preferences to traits
            if user_id in self.user_genre_preferences:
                genre_prefs = self.user_genre_preferences[user_id]
                # Add top 3 liked genres to traits
                liked_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
                for i, (genre, score) in enumerate(liked_genres):
                    if score > 0:
                        traits[f'liked_genre_{i+1}'] = genre
            
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
                name=f"MovieLover_{user_id}",
                age=traits.get('age', 30),
                traits=traits_str,
                status="looking for movie recommendations",
                memory_retriever=create_new_memory_retriever(),
                llm=LLM,
                memory=memory,
            )
            
            # Store agent
            self.agents[user_id] = agent
            
            # Add memories - either selective or all
            if self.use_selective_memory:
                self.add_selective_memories(
                    agent=agent,
                    user_id=user_id,
                    max_memories=self.max_memories_per_user,
                    strategy=self.memory_strategy
                )
            else:
                # Add all user interactions as memories (original approach)
                user_interactions = self.interactions[self.interactions['user_id'] == user_id]
                for _, interaction in user_interactions.iterrows():
                    movie_id = interaction['item_id']
                    rating = interaction.get('rating', None)
                    
                    # Get movie details
                    movie_row = self.item_data[self.item_data['item_id'] == movie_id]
                    if not movie_row.empty:
                        movie_title = movie_row['title'].iloc[0]
                        
                        # Create memory content
                        if rating is not None:
                            memory_content = f"I watched '{movie_title}' and rated it {rating}/5"
                        else:
                            memory_content = f"I watched '{movie_title}'"
                        
                        # Add to agent memory
                        agent.memory.add_memory(memory_content)
        
        print(f"Initialized {len(self.agents)} generative agents for MovieLens users")
    
    def construct_rag_query(self, user_id):
        """
        Construct a movie-specific retrieval query integrating user traits and memory.
        
        Args:
            user_id: User ID to construct query for
            
        Returns:
            Tuple of (query, query_embedding)
        """
        # Get agent for the user
        agent = self.agents[user_id]
        
        # Get user traits
        user_traits = agent.traits
        
        # Get relevant memories about movie preferences
        memory_query = "What movies have I watched? What kinds of movies do I like and dislike?"
        relevant_memories = agent.memory.memory_retriever.get_relevant_documents(memory_query)
        memory_contents = " ".join([mem.page_content for mem in relevant_memories])
        
        # Create a more movie-specific query
        query = f"""
        User Profile:
        {user_traits}
        
        Movie Watching History:
        {memory_contents}
        
        Based on this user's profile and movie watching history, what types of movies would they enjoy?
        """
        
        # Encode the query
        query_embedding = self.embeddings_model.embed_query(query)
        
        return query, query_embedding
    
    def generate_movie_recommendations(self, user_id, num_recommendations=5):
        """
        Generate movie recommendations for a user using the HallAgent4Rec methodology.
        
        Args:
            user_id: User ID to generate recommendations for
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommended movies with explanations
        """
        print(f"Generating movie recommendations for user {user_id}...")
        
        # Step 1: Construct RAG query
        query, query_embedding = self.construct_rag_query(user_id)
        
        # Step 2: Construct knowledge base
        knowledge_base, top_cluster = self.construct_knowledge_base(user_id)
        
        # Step 3: Retrieve relevant items
        retrieved_movies = self.retrieve_items(user_id, query_embedding, knowledge_base)
        
        # If no movies retrieved, return empty list
        if not retrieved_movies:
            print("No relevant movies retrieved. Cannot generate recommendations.")
            return []
        
        # Step 4: Generate recommendations using LLM
        # Format retrieved movies for prompt
        movie_descriptions = "\n".join([
            f"- {movie['title']} ({movie.get('year', 'Unknown')}) - {movie.get('primary_genre', 'Unknown genre')}"
            for movie in retrieved_movies[:15]  # Use up to 15 movies for better diversity
        ])
        
        # Get agent
        agent = self.agents[user_id]
        
        # Create a movie-specific prompt for LLM
        prompt = f"""
        You are a personalized movie recommendation system for a user with the following profile:
        {agent.traits}
        
        Based on this user's profile and past behavior, you have retrieved the following relevant movies:
        {movie_descriptions}
        
        Please recommend {num_recommendations} movies from the list above that would be most relevant for this user.
        For each recommendation, provide a brief explanation of why you think the user would enjoy it based on their 
        preferences, watching history, and demographic information.
        
        IMPORTANT: You must ONLY recommend movies from the provided list. Do not suggest any movies that are not in the list.
        
        Format your response as:
        1. [Movie Title] ([Year]): [Explanation]
        2. [Movie Title] ([Year]): [Explanation]
        ...
        """
        
        # Generate recommendations
        response = self.LLM.invoke(prompt)
        recommendations_text = response.content
        
        # Step 5: Extract recommended movies and detect hallucinations
        recommended_movies = []
        lines = recommendations_text.strip().split('\n')
        
        for line in lines:
            if line.strip() and any(char.isdigit() for char in line[:5]):
                # Extract movie title and explanation from line
                parts = line.split(':', 1)
                if len(parts) > 1:
                    title_part = parts[0].strip()
                    explanation = parts[1].strip()
                    
                    # Extract movie title (remove numbering)
                    if '.' in title_part:
                        title_part = title_part.split('.', 1)[1].strip()
                    
                    # Look for the movie in retrieved items
                    movie_info = None
                    for movie in retrieved_movies:
                        if movie['title'] in title_part:
                            movie_info = movie.copy()
                            movie_info['explanation'] = explanation
                            recommended_movies.append(movie_info)
                            break
                    
                    # If movie not found (hallucination), log it
                    if movie_info is None:
                        print(f"Hallucination detected: {title_part}")
        
        # If not enough valid recommendations, fill with top predicted movies
        if len(recommended_movies) < num_recommendations:
            user_idx = self.user_id_map[user_id]
            cluster_items = self.items_by_cluster[top_cluster]
            
            # Get predicted scores for movies in the cluster
            item_scores = [(idx, np.dot(self.user_embeddings[user_idx], self.item_embeddings[idx])) 
                           for idx in cluster_items]
            
            # Sort by score and remove movies already recommended
            recommended_ids = [movie['item_id'] for movie in recommended_movies]
            additional_movies = []
            
            for item_idx, score in sorted(item_scores, key=lambda x: x[1], reverse=True):
                item_id = self.idx_to_item_id[item_idx]
                if item_id not in recommended_ids:
                    item_row = self.item_data[self.item_data['item_id'] == item_id]
                    if not item_row.empty:
                        movie_info = {}
                        for col in item_row.columns:
                            movie_info[col] = item_row.iloc[0][col]
                        movie_info['explanation'] = f"This movie matches your preferences based on your viewing history."
                        additional_movies.append(movie_info)
                        if len(recommended_movies) + len(additional_movies) >= num_recommendations:
                            break
            
            recommended_movies.extend(additional_movies)
        
        # Limit to requested number
        recommended_movies = recommended_movies[:num_recommendations]
        
        print(f"Generated {len(recommended_movies)} movie recommendations")
        return recommended_movies
    
    def evaluate_recommendations(self, num_users=None, num_recommendations=10):
        """
        Evaluate movie recommendations against test set.
        
        Args:
            num_users: Number of users to evaluate (None for all)
            num_recommendations: Number of recommendations to generate per user
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating movie recommendations...")
        
        # Select users to evaluate
        if num_users is None:
            eval_users = list(self.user_id_map.keys())
        else:
            eval_users = np.random.choice(list(self.user_id_map.keys()), 
                                         size=min(num_users, len(self.user_id_map)), 
                                         replace=False)
        
        # Prepare metrics
        precision_sum = 0
        recall_sum = 0
        ndcg_sum = 0
        hallucination_count = 0
        total_recommendations = 0
        user_count = 0
        
        # Create ground truth from test interactions
        ground_truth = {}
        for user_id in eval_users:
            user_test = self.test_interactions[self.test_interactions['user_id'] == user_id]
            if not user_test.empty:
                # Consider items with high ratings (>= 4) as relevant
                relevant_items = user_test[user_test['rating'] >= 4]['item_id'].tolist()
                if relevant_items:
                    ground_truth[user_id] = set(relevant_items)
        
        # Evaluate each user
        for user_id in tqdm(eval_users, desc="Evaluating users"):
            if user_id not in ground_truth:
                continue
                
            # Generate recommendations
            recommendations = self.generate_movie_recommendations(user_id, num_recommendations)
            if not recommendations:
                continue
                
            # Extract recommended movie IDs
            rec_items = [movie['item_id'] for movie in recommendations]
            
            # Calculate precision and recall
            hits = len(set(rec_items) & ground_truth[user_id])
            precision = hits / len(rec_items)
            recall = hits / len(ground_truth[user_id])
            
            # Calculate NDCG
            # Create relevance array (1 if item is relevant, 0 otherwise)
            relevance = [1 if item in ground_truth[user_id] else 0 for item in rec_items]
            
            # Calculate DCG
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
            
            # Calculate ideal DCG (all relevant items at the top)
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # Detect hallucinations
            query, query_embedding = self.construct_rag_query(user_id)
            knowledge_base, _ = self.construct_knowledge_base(user_id)
            retrieved_items = self.retrieve_items(user_id, query_embedding, knowledge_base)
            
            retrieved_item_ids = set(item['item_id'] for item in retrieved_items)
            
            for item_id in rec_items:
                total_recommendations += 1
                if item_id not in retrieved_item_ids:
                    hallucination_count += 1
            
            # Update sums
            precision_sum += precision
            recall_sum += recall
            ndcg_sum += ndcg
            user_count += 1
        
        # Calculate average metrics
        avg_precision = precision_sum / user_count if user_count > 0 else 0
        avg_recall = recall_sum / user_count if user_count > 0 else 0
        avg_ndcg = ndcg_sum / user_count if user_count > 0 else 0
        hallucination_rate = hallucination_count / total_recommendations if total_recommendations > 0 else 0
        
        # Return metrics
        metrics = {
            'precision': avg_precision,
            'recall': avg_recall,
            'ndcg': avg_ndcg,
            'hallucination_rate': hallucination_rate,
            'users_evaluated': user_count
        }
        
        print(f"Evaluation results:")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  NDCG: {avg_ndcg:.4f}")
        print(f"  Hallucination Rate: {hallucination_rate:.4f}")
        print(f"  Users Evaluated: {user_count}")
        
        return metrics
    
    def generate_movie_explanations(self, user_id, movie_ids, personalized=True):
        """
        Generate personalized explanations for why a user might like certain movies.
        
        Args:
            user_id: User ID to generate explanations for
            movie_ids: List of movie IDs to explain
            personalized: Whether to personalize explanations to the user
            
        Returns:
            Dictionary mapping movie IDs to explanations
        """
        # Get agent
        agent = self.agents.get(user_id)
        if agent is None:
            print(f"Agent not found for user {user_id}")
            return {}
        
        # Get movie information
        movies_info = []
        for movie_id in movie_ids:
            movie_row = self.item_data[self.item_data['item_id'] == movie_id]
            if not movie_row.empty:
                movie_info = {
                    'item_id': movie_id,
                    'title': movie_row['title'].iloc[0]
                }
                
                # Add other movie details if available
                for col in ['year', 'primary_genre']:
                    if col in movie_row.columns:
                        movie_info[col] = movie_row[col].iloc[0]
                        
                movies_info.append(movie_info)
        
        if not movies_info:
            print(f"No movie information found for the provided IDs")
            return {}
            
        # Format movie information for the prompt
        movie_list = "\n".join([
            f"- {movie['title']} ({movie.get('year', 'Unknown')}) - {movie.get('primary_genre', 'Unknown genre')}"
            for movie in movies_info
        ])
        
        # Create prompt
        if personalized:
            # Get user profile for more personalized explanations
            user_traits = agent.traits
            
            # Get relevant memories
            memory_query = "What movies have I watched? What kinds of movies do I like and dislike?"
            relevant_memories = agent.memory.memory_retriever.get_relevant_documents(memory_query)
            memory_contents = " ".join([mem.page_content for mem in relevant_memories])
            
            prompt = f"""
            You are a personalized movie recommendation system for a user with the following profile:
            {user_traits}
            
            The user's movie viewing history:
            {memory_contents}
            
            For each of the following movies, provide a personalized explanation of why this specific user might enjoy it,
            based on their preferences, history, and demographic information:
            
            {movie_list}
            
            Format your response as:
            Movie: [Movie Title]
            Explanation: [Your detailed explanation why this user specifically would enjoy this movie]
            
            Make each explanation unique, personalized, and insightful.
            """
        else:
            # Generic explanations not personalized to the user
            prompt = f"""
            You are a movie recommendation system. For each of the following movies, provide an explanation of what 
            type of viewer might enjoy it and what makes this movie appealing:
            
            {movie_list}
            
            Format your response as:
            Movie: [Movie Title]
            Explanation: [Your detailed explanation of the movie's appeal]
            
            Make each explanation unique and insightful.
            """
        
        # Generate explanations
        response = self.LLM.invoke(prompt)
        explanations_text = response.content
        
        # Parse explanations
        explanations = {}
        current_movie_id = None
        current_explanation = []
        
        for line in explanations_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Movie:'):
                # Save previous movie and explanation
                if current_movie_id is not None and current_explanation:
                    explanations[current_movie_id] = ' '.join(current_explanation)
                
                # Find the movie ID for this title
                movie_title = line[6:].strip()
                current_movie_id = None
                for movie in movies_info:
                    if movie['title'] in movie_title:
                        current_movie_id = movie['item_id']
                        break
                
                current_explanation = []
            elif line.startswith('Explanation:'):
                if current_movie_id is not None:
                    current_explanation.append(line[12:].strip())
            elif current_movie_id is not None:
                current_explanation.append(line)
        
        # Add the last movie
        if current_movie_id is not None and current_explanation:
            explanations[current_movie_id] = ' '.join(current_explanation)
        
        return explanations


# Example usage
def example_usage():
    # Create and train MovieLensAgent4Rec
    agent = MovieLensAgent4Rec(
        num_clusters=20,
        latent_dim=50,
        use_selective_memory=True,
        max_memories_per_user=15,
        memory_strategy='diverse'
    )
    
    # Load MovieLens data
    agent.load_movielens_data()
    
    # Train the model
    agent.train()
    
    # Generate recommendations for a user
    user_id = 1
    recommendations = agent.generate_movie_recommendations(user_id, num_recommendations=5)
    
    # Print recommendations
    print(f"\nRecommendations for User {user_id}:")
    for i, movie in enumerate(recommendations):
        print(f"{i+1}. {movie['title']} ({movie.get('year', 'Unknown')}) - {movie.get('primary_genre', 'Unknown')}")
        print(f"   Explanation: {movie.get('explanation', 'No explanation available')}")
    
    # Evaluate on a sample of users
    metrics = agent.evaluate_recommendations(num_users=20, num_recommendations=10)
    
    return agent, recommendations, metrics


if __name__ == "__main__":
    agent, recommendations, metrics = example_usage()