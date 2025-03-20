import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import zipfile
import requests
from typing import Dict, List, Tuple, Optional, Union

def download_movielens_100k(data_path: str = "./data") -> str:
    """
    Download MovieLens 100K dataset if not already available.
    
    Args:
        data_path: Directory to save the dataset
        
    Returns:
        Path to the dataset
    """
    os.makedirs(data_path, exist_ok=True)
    ml_100k_path = os.path.join(data_path, "ml-100k")
    zip_path = os.path.join(data_path, "ml-100k.zip")
    
    # Check if the dataset already exists
    if os.path.exists(ml_100k_path) and os.path.isdir(ml_100k_path):
        print(f"MovieLens 100K dataset already exists at {ml_100k_path}")
        return ml_100k_path
    
    # Download the dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    print(f"Downloading MovieLens 100K dataset from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save the zip file
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)
    
    # Remove the zip file
    os.remove(zip_path)
    
    print(f"MovieLens 100K dataset downloaded and extracted to {ml_100k_path}")
    return ml_100k_path

def load_movielens_100k(data_path: str = "./data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens 100K dataset and format it for HallAgent4Rec.
    
    Args:
        data_path: Path to the dataset
        
    Returns:
        Tuple of (user_data, item_data, interactions)
    """
    # Download dataset if needed
    ml_100k_path = download_movielens_100k(data_path)
    
    # Load user data
    user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users_path = os.path.join(ml_100k_path, "u.user")
    user_data = pd.read_csv(users_path, sep='|', names=user_cols, encoding='latin-1')
    
    # Load movie data
    movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 
                 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation', 
                 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                 'Thriller', 'War', 'Western']
    movies_path = os.path.join(ml_100k_path, "u.item")
    item_data = pd.read_csv(movies_path, sep='|', names=movie_cols, encoding='latin-1')
    
    # Process movie data to create a cleaner format
    # Extract year from title and create primary genre
    item_data['year'] = item_data['title'].str.extract(r'\((\d{4})\)$')
    
    # Create a primary genre column
    genre_columns = movie_cols[5:]  # All genre columns
    
    def get_primary_genre(row):
        for genre in genre_columns:
            if row[genre] == 1:
                return genre
        return "unknown"
    
    item_data['primary_genre'] = item_data.apply(get_primary_genre, axis=1)
    
    # Simplify the item data
    item_data_simplified = item_data[['item_id', 'title', 'year', 'primary_genre']].copy()
    
    # Convert year to numeric
    item_data_simplified['year'] = pd.to_numeric(item_data_simplified['year'], errors='coerce')
    
    # Load ratings data
    rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_path = os.path.join(ml_100k_path, "u.data")
    interactions = pd.read_csv(ratings_path, sep='\t', names=rating_cols)
    
    # Encode categorical variables for model compatibility
    label_encoders = {}
    for col in ['gender', 'occupation']:
        le = LabelEncoder()
        user_data[col] = le.fit_transform(user_data[col])
        label_encoders[col] = le
    
    for col in ['primary_genre']:
        le = LabelEncoder()
        item_data_simplified[col] = le.fit_transform(item_data_simplified[col])
        label_encoders[col] = le
    
    print(f"Loaded MovieLens 100K dataset:")
    print(f" - Users: {len(user_data)}")
    print(f" - Movies: {len(item_data_simplified)}")
    print(f" - Ratings: {len(interactions)}")
    
    return user_data, item_data_simplified, interactions

def split_train_test(interactions: pd.DataFrame, test_ratio: float = 0.2, 
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into training and testing sets.
    
    Args:
        interactions: DataFrame of user-item interactions
        test_ratio: Ratio of interactions to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_interactions, test_interactions)
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Group by user to ensure each user has both training and testing data
    train_interactions = []
    test_interactions = []
    
    for user_id, user_ratings in interactions.groupby('user_id'):
        # Shuffle user ratings
        user_ratings = user_ratings.sample(frac=1, random_state=random_state)
        
        # Calculate split point
        split_idx = int(len(user_ratings) * (1 - test_ratio))
        
        # Split into train and test
        train_interactions.append(user_ratings.iloc[:split_idx])
        test_interactions.append(user_ratings.iloc[split_idx:])
    
    # Combine all users' train and test interactions
    train_df = pd.concat(train_interactions)
    test_df = pd.concat(test_interactions)
    
    print(f"Split into {len(train_df)} training and {len(test_df)} testing interactions")
    
    return train_df, test_df

def prepare_movielens_for_hallagent(data_path: str = "./data", test_ratio: float = 0.2, 
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                  pd.DataFrame, pd.DataFrame]:
    """
    Prepare MovieLens dataset for HallAgent4Rec.
    
    Args:
        data_path: Path to store the dataset
        test_ratio: Ratio of interactions to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (user_data, item_data, train_interactions, test_interactions)
    """
    # Load the dataset
    user_data, item_data, interactions = load_movielens_100k(data_path)
    
    # Split into training and testing sets
    train_interactions, test_interactions = split_train_test(interactions, test_ratio, random_state)
    
    return user_data, item_data, train_interactions, test_interactions

def create_movie_prompts(movie_data: pd.DataFrame) -> Dict[int, str]:
    """
    Create detailed prompts for each movie for better LLM understanding.
    
    Args:
        movie_data: DataFrame with movie information
        
    Returns:
        Dictionary mapping movie IDs to detailed descriptions
    """
    movie_prompts = {}
    
    # Get original movie data with all genre columns
    for _, movie in movie_data.iterrows():
        movie_id = movie['item_id']
        title = movie['title']
        year = movie.get('year', 'Unknown')
        
        # Create a more detailed prompt for each movie
        prompt = f"Movie: {title} ({year}). "
        
        # Add genre information if available
        if 'primary_genre' in movie:
            genre = movie['primary_genre']
            prompt += f"Genre: {genre}. "
        
        movie_prompts[movie_id] = prompt
    
    return movie_prompts

def get_user_movie_preferences(user_id: int, interactions: pd.DataFrame, 
                               item_data: pd.DataFrame, min_rating: int = 4) -> Dict[str, List[str]]:
    """
    Get a user's movie preferences based on ratings.
    
    Args:
        user_id: User ID
        interactions: DataFrame with user-item interactions
        item_data: DataFrame with movie information
        min_rating: Minimum rating to consider as positive preference
        
    Returns:
        Dictionary with liked and disliked movies
    """
    user_ratings = interactions[interactions['user_id'] == user_id]
    
    liked_movies = []
    disliked_movies = []
    
    for _, rating in user_ratings.iterrows():
        movie_id = rating['item_id']
        rating_value = rating['rating']
        
        movie_row = item_data[item_data['item_id'] == movie_id]
        if not movie_row.empty:
            movie_title = movie_row['title'].iloc[0]
            
            if rating_value >= min_rating:
                liked_movies.append(movie_title)
            else:
                disliked_movies.append(movie_title)
    
    return {
        'liked': liked_movies,
        'disliked': disliked_movies
    }

def get_user_genre_preferences(user_id: int, interactions: pd.DataFrame, 
                              item_data: pd.DataFrame, min_rating: int = 4) -> Dict[str, float]:
    """
    Calculate a user's genre preferences based on ratings.
    
    Args:
        user_id: User ID
        interactions: DataFrame with user-item interactions
        item_data: DataFrame with movie information
        min_rating: Minimum rating to consider as positive preference
        
    Returns:
        Dictionary mapping genres to preference scores
    """
    user_ratings = interactions[interactions['user_id'] == user_id]
    
    # Initialize genre counts
    genre_ratings = {}
    genre_counts = {}
    
    for _, rating in user_ratings.iterrows():
        movie_id = rating['item_id']
        rating_value = rating['rating']
        
        movie_row = item_data[item_data['item_id'] == movie_id]
        if not movie_row.empty and 'primary_genre' in movie_row.columns:
            genre = movie_row['primary_genre'].iloc[0]
            
            if genre not in genre_ratings:
                genre_ratings[genre] = 0
                genre_counts[genre] = 0
            
            genre_ratings[genre] += rating_value
            genre_counts[genre] += 1
    
    # Calculate average rating per genre
    genre_preferences = {}
    for genre, total_rating in genre_ratings.items():
        count = genre_counts[genre]
        if count > 0:
            avg_rating = total_rating / count
            # Normalize to a -1 to 1 scale
            preference = (avg_rating - 3) / 2  # 1 to 5 -> -1 to 1
            genre_preferences[genre] = round(preference, 2)
    
    return genre_preferences

def create_user_profile_for_llm(user_id: int, user_data: pd.DataFrame, 
                               interactions: pd.DataFrame, item_data: pd.DataFrame) -> str:
    """
    Create a comprehensive user profile for LLM interactions.
    
    Args:
        user_id: User ID
        user_data: DataFrame with user information
        interactions: DataFrame with user-item interactions
        item_data: DataFrame with movie information
        
    Returns:
        String representation of user profile for LLM
    """
    # Get user demographic information
    user_row = user_data[user_data['user_id'] == user_id]
    if user_row.empty:
        return f"User {user_id}: No demographic information available."
    
    user_info = user_row.iloc[0]
    age = user_info.get('age', 'Unknown')
    gender = user_info.get('gender', 'Unknown')
    occupation = user_info.get('occupation', 'Unknown')
    
    # Decode encoded categorical variables if label encoders are available
    # This is just a placeholder - in practice you'd need to pass the label encoders
    if isinstance(gender, (int, np.integer)):
        gender_map = {0: 'Female', 1: 'Male'}
        gender = gender_map.get(gender, gender)
    
    if isinstance(occupation, (int, np.integer)):
        occupation_list = ['administrator', 'artist', 'doctor', 'educator', 'engineer', 
                          'entertainment', 'executive', 'healthcare', 'homemaker', 
                          'lawyer', 'librarian', 'marketing', 'none', 'other', 
                          'programmer', 'retired', 'salesman', 'scientist', 
                          'student', 'technician', 'writer']
        if 0 <= occupation < len(occupation_list):
            occupation = occupation_list[occupation]
    
    # Get user movie preferences
    preferences = get_user_movie_preferences(user_id, interactions, item_data)
    genre_prefs = get_user_genre_preferences(user_id, interactions, item_data)
    
    # Create user profile text
    profile = f"User Profile - ID: {user_id}\n"
    profile += f"Demographics: {age} year old {gender}, occupation: {occupation}\n\n"
    
    # Add genre preferences
    profile += "Genre Preferences:\n"
    sorted_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)
    for genre, score in sorted_genres:
        sentiment = "likes" if score > 0 else "dislikes"
        strength = abs(score)
        if strength > 0.7:
            intensity = "strongly"
        elif strength > 0.3:
            intensity = "moderately"
        else:
            intensity = "slightly"
        
        profile += f"- {intensity} {sentiment} {genre}\n"
    
    # Add some example liked and disliked movies
    profile += "\nMovie Preferences:\n"
    
    if preferences['liked']:
        profile += "Liked movies: " + ", ".join(preferences['liked'][:5])
        if len(preferences['liked']) > 5:
            profile += f" and {len(preferences['liked']) - 5} more"
        profile += "\n"
    
    if preferences['disliked']:
        profile += "Disliked movies: " + ", ".join(preferences['disliked'][:3])
        if len(preferences['disliked']) > 3:
            profile += f" and {len(preferences['disliked']) - 3} more"
        profile += "\n"
    
    return profile

if __name__ == "__main__":
    # Example usage
    user_data, item_data, interactions = load_movielens_100k()
    
    # Example: Create user profile
    user_id = 1
    user_profile = create_user_profile_for_llm(user_id, user_data, interactions, item_data)
    print(user_profile)
    
    # Example: Split into train and test
    train_interactions, test_interactions = split_train_test(interactions)
    print(f"Training set size: {len(train_interactions)}")
    print(f"Testing set size: {len(test_interactions)}")