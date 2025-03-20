import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import pickle
from typing import Dict, Tuple, List
import datetime
import json

def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create sample data for testing the experimental framework when real datasets aren't available.
    
    Returns:
        user_data: DataFrame with user information
        item_data: DataFrame with item information
        interactions: DataFrame with user-item interactions
        test_interactions: DataFrame with test user-item interactions
    """
    # User data
    user_data = pd.DataFrame({
        'user_id': list(range(1, 101)),  # 100 users
        'age': np.random.randint(18, 70, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'occupation': np.random.choice(['student', 'engineer', 'doctor', 'artist', 'teacher'], 100)
    })
    
    # Item data
    categories = ['books', 'electronics', 'clothing', 'food', 'sports']
    item_data = pd.DataFrame({
        'item_id': list(range(101, 501)),  # 400 items
        'name': [f'Item_{i}' for i in range(101, 501)],
        'category': np.random.choice(categories, 400),
        'price': np.random.uniform(5, 200, 400).round(2),
        'popularity': np.random.uniform(0, 1, 400).round(2)
    })
    
    # Encode categorical features
    label_encoders = {}
    for col in ['gender', 'occupation']:
        le = LabelEncoder()
        user_data[col] = le.fit_transform(user_data[col])
        label_encoders[col] = le

    for col in ['category', 'name']:
        le = LabelEncoder()
        item_data[col] = le.fit_transform(item_data[col])
        label_encoders[col] = le
    
    # Generate interactions (each user interacts with ~10% of items)
    interactions_list = []
    test_interactions_list = []
    
    for user_id in user_data['user_id']:
        # Sample items for this user (training)
        num_items = np.random.randint(20, 50)  
        train_items = np.random.choice(item_data['item_id'], size=num_items, replace=False)
        
        # Create timestamps spread over a month
        current_time = datetime.datetime.now()
        timestamps = np.random.randint(
            int((current_time - datetime.timedelta(days=30)).timestamp()),
            int(current_time.timestamp()),
            num_items
        )
        
        # Add training interactions
        for item_id, timestamp in zip(train_items, timestamps):
            interactions_list.append({
                'user_id': user_id,
                'item_id': item_id,
                'timestamp': timestamp
            })
        
        # Sample items for testing (non-overlapping with training)
        remaining_items = list(set(item_data['item_id']) - set(train_items))
        test_items = np.random.choice(remaining_items, size=min(5, len(remaining_items)), replace=False)
        
        # Add test interactions
        for item_id in test_items:
            test_interactions_list.append({
                'user_id': user_id,
                'item_id': item_id
            })
    
    interactions = pd.DataFrame(interactions_list)
    test_interactions = pd.DataFrame(test_interactions_list)
    
    return user_data, item_data, interactions, test_interactions

def load_or_create_frappe_sample() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a synthetic sample dataset that mimics the structure of the Frappe dataset.
    
    Returns:
        user_data: DataFrame with user information
        item_data: DataFrame with item information
        interactions: DataFrame with user-item interactions
        test_interactions: DataFrame with test user-item interactions
    """
    # User data with Frappe-like features
    num_users = 100
    user_data = pd.DataFrame({
        'user_id': list(range(1, num_users + 1)),
        'age': np.random.randint(18, 65, num_users),
        'gender': np.random.choice(['M', 'F'], num_users),
        'country': np.random.choice(['US', 'UK', 'JP', 'CN', 'IN', 'DE', 'FR'], num_users)
    })
    
    # App data (items in Frappe are mobile apps)
    num_apps = 200
    app_categories = ['Games', 'Social', 'Productivity', 'Entertainment', 'Tools', 'News', 'Education']
    item_data = pd.DataFrame({
        'item_id': list(range(101, 101 + num_apps)),
        'name': [f'App_{i}' for i in range(101, 101 + num_apps)],
        'category': np.random.choice(app_categories, num_apps),
        'rating': np.random.uniform(1, 5, num_apps).round(1),
        'downloads': np.random.choice([1000, 5000, 10000, 50000, 100000, 500000, 1000000], num_apps),
        'free': np.random.choice([True, False], num_apps, p=[0.7, 0.3])
    })
    
    # Encode categorical features
    label_encoders = {}
    for col in ['gender', 'country']:
        le = LabelEncoder()
        user_data[col] = le.fit_transform(user_data[col])
        label_encoders[col] = le

    for col in ['category', 'name']:
        le = LabelEncoder()
        item_data[col] = le.fit_transform(item_data[col])
        label_encoders[col] = le
    
    item_data['free'] = item_data['free'].astype(int)
    
    # Generate app usage interactions with context (Frappe specific)
    interactions_list = []
    test_interactions_list = []
    
    contexts = ['home', 'work', 'commuting', 'travelling', 'morning', 'afternoon', 'evening']
    
    for user_id in user_data['user_id']:
        # User's app preferences - some categories they like more
        preferred_categories = np.random.choice(app_categories, size=np.random.randint(1, 4), replace=False)
        preferred_apps = item_data[item_data['category'].isin([app_cat for app_cat in preferred_categories 
                                                           if app_cat in item_data['category'].values])]['item_id'].values
        
        # Sample apps for this user (training), with higher probability for preferred categories
        all_apps = item_data['item_id'].values
        probs = np.ones(len(all_apps))
        
        for i, app_id in enumerate(all_apps):
            if app_id in preferred_apps:
                probs[i] = 3  # Higher probability for preferred categories
        
        probs = probs / probs.sum()
        
        num_interactions = np.random.randint(10, 30)
        user_apps = np.random.choice(all_apps, size=num_interactions, replace=True, p=probs)
        
        # Create timestamps spread over a month
        current_time = datetime.datetime.now()
        timestamps = np.random.randint(
            int((current_time - datetime.timedelta(days=30)).timestamp()),
            int(current_time.timestamp()),
            num_interactions
        )
        
        # Add training interactions with context
        for app_id, timestamp in zip(user_apps, timestamps):
            context = np.random.choice(contexts)
            interactions_list.append({
                'user_id': user_id,
                'item_id': app_id,
                'timestamp': timestamp,
                'context': context
            })
        
        # Sample apps for testing (could be overlapping with training, as in real-world scenarios)
        test_apps = np.random.choice(all_apps, size=5, replace=False, p=probs)
        
        # Add test interactions (without context)
        for app_id in test_apps:
            test_interactions_list.append({
                'user_id': user_id,
                'item_id': app_id
            })
    
    interactions = pd.DataFrame(interactions_list)
    test_interactions = pd.DataFrame(test_interactions_list)
    
    return user_data, item_data, interactions, test_interactions

def load_or_create_musicincar_sample() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a synthetic sample dataset that mimics the structure of the MusicInCar dataset.
    
    Returns:
        user_data: DataFrame with user information
        item_data: DataFrame with item information
        interactions: DataFrame with user-item interactions
        test_interactions: DataFrame with test user-item interactions
    """
    # User data with MusicInCar-like features
    num_users = 80
    user_data = pd.DataFrame({
        'user_id': list(range(1, num_users + 1)),
        'age': np.random.randint(18, 65, num_users),
        'gender': np.random.choice(['M', 'F'], num_users),
        'driving_style': np.random.choice(['calm', 'moderate', 'aggressive'], num_users),
        'car_type': np.random.choice(['sedan', 'SUV', 'compact', 'sports'], num_users)
    })
    
    # Music data (items in MusicInCar are songs)
    num_songs = 150
    music_genres = ['Rock', 'Pop', 'Classical', 'Jazz', 'Electronic', 'Hip-hop', 'Country', 'R&B']
    item_data = pd.DataFrame({
        'item_id': list(range(101, 101 + num_songs)),
        'name': [f'Song_{i}' for i in range(101, 101 + num_songs)],
        'artist': [f'Artist_{np.random.randint(1, 50)}' for _ in range(num_songs)],
        'genre': np.random.choice(music_genres, num_songs),
        'tempo': np.random.randint(60, 180, num_songs),  # BPM
        'release_year': np.random.randint(1970, 2023, num_songs),
        'duration': np.random.randint(120, 420, num_songs)  # seconds
    })
    
    # Encode categorical features
    label_encoders = {}
    for col in ['gender', 'driving_style', 'car_type']:
        le = LabelEncoder()
        user_data[col] = le.fit_transform(user_data[col])
        label_encoders[col] = le

    for col in ['genre', 'artist', 'name']:
        le = LabelEncoder()
        item_data[col] = le.fit_transform(item_data[col])
        label_encoders[col] = le
    
    # Generate music listening interactions with driving context
    interactions_list = []
    test_interactions_list = []
    
    driving_conditions = ['city', 'highway', 'countryside', 'traffic']
    weather_conditions = ['sunny', 'rainy', 'cloudy', 'night']
    
    for user_id in user_data['user_id']:
        # User's music preferences based on driving style
        user_row = user_data[user_data['user_id'] == user_id].iloc[0]
        driving_style = user_row['driving_style']
        
        # Define genre preferences based on driving style
        if driving_style == 0:  # calm
            preferred_genres = ['Classical', 'Jazz', 'Country']
        elif driving_style == 1:  # moderate
            preferred_genres = ['Pop', 'R&B', 'Rock']
        else:  # aggressive
            preferred_genres = ['Rock', 'Electronic', 'Hip-hop']
            
        preferred_songs = item_data[item_data['genre'].isin([genre for genre in preferred_genres 
                                                         if genre in item_data['genre'].values])]['item_id'].values
        
        # Sample songs for this user (training), with higher probability for preferred genres
        all_songs = item_data['item_id'].values
        probs = np.ones(len(all_songs))
        
        for i, song_id in enumerate(all_songs):
            if song_id in preferred_songs:
                probs[i] = 3  # Higher probability for preferred genres
        
        probs = probs / probs.sum()
        
        num_interactions = np.random.randint(15, 40)
        user_songs = np.random.choice(all_songs, size=num_interactions, replace=True, p=probs)
        
        # Create timestamps spread over a month
        current_time = datetime.datetime.now()
        timestamps = np.random.randint(
            int((current_time - datetime.timedelta(days=30)).timestamp()),
            int(current_time.timestamp()),
            num_interactions
        )
        
        # Add training interactions with context
        for song_id, timestamp in zip(user_songs, timestamps):
            driving_condition = np.random.choice(driving_conditions)
            weather_condition = np.random.choice(weather_conditions)
            interactions_list.append({
                'user_id': user_id,
                'item_id': song_id,
                'timestamp': timestamp,
                'driving_condition': driving_condition,
                'weather': weather_condition
            })
        
        # Sample songs for testing
        test_songs = np.random.choice(all_songs, size=5, replace=False, p=probs)
        
        # Add test interactions (without context)
        for song_id in test_songs:
            test_interactions_list.append({
                'user_id': user_id,
                'item_id': song_id
            })
    
    interactions = pd.DataFrame(interactions_list)
    test_interactions = pd.DataFrame(test_interactions_list)
    
    return user_data, item_data, interactions, test_interactions

def save_results(results, filename):
    """
    Save experimental results to disk.
    
    Args:
        results: Dictionary of results to save
        filename: Name of the file to save results to
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save results as pickle file
    with open(f'results/{filename}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Also save as JSON if possible (for human readability)
    try:
        # Convert numpy arrays and other non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            else:
                return str(obj)
        
        # Convert results for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: convert_for_json(v) for k, v in value.items()}
            else:
                json_results[key] = convert_for_json(value)
        
        with open(f'results/{filename}.json', 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        print(f"Could not save results as JSON: {e}")

def load_results(filename):
    """
    Load experimental results from disk.
    
    Args:
        filename: Name of the file to load results from
        
    Returns:
        Dictionary of results
    """
    with open(f'results/{filename}.pkl', 'rb') as f:
        return pickle.load(f)

def generate_synthetic_dataset(dataset_name, num_users, num_items, num_interactions):
    """
    Generate a synthetic dataset that mimics the structure of a real dataset.
    
    Args:
        dataset_name: Name of the dataset to mimic ('frappe' or 'musicincar')
        num_users: Number of users to generate
        num_items: Number of items to generate
        num_interactions: Number of interactions to generate
        
    Returns:
        Tuple of DataFrames (user_data, item_data, interactions, test_interactions)
    """
    if dataset_name.lower() == 'frappe':
        return generate_synthetic_frappe(num_users, num_items, num_interactions)
    elif dataset_name.lower() == 'musicincar':
        return generate_synthetic_musicincar(num_users, num_items, num_interactions)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def generate_synthetic_frappe(num_users, num_items, num_interactions):
    """Generate a synthetic Frappe-like dataset with specified sizes."""
    # Scale the existing function
    user_data, item_data, interactions, test_interactions = load_or_create_frappe_sample()
    
    # Expand user data
    user_data = pd.concat([user_data] * (num_users // len(user_data) + 1), ignore_index=True)
    user_data = user_data.iloc[:num_users]
    user_data['user_id'] = range(1, num_users + 1)
    
    # Expand item data
    item_data = pd.concat([item_data] * (num_items // len(item_data) + 1), ignore_index=True)
    item_data = item_data.iloc[:num_items]
    item_data['item_id'] = range(101, 101 + num_items)
    
    # Generate new interactions
    new_interactions = []
    for _ in range(num_interactions):
        user_id = np.random.choice(user_data['user_id'])
        item_id = np.random.choice(item_data['item_id'])
        context = np.random.choice(['home', 'work', 'commuting', 'travelling', 'morning', 'afternoon', 'evening'])
        timestamp = int(datetime.datetime.now().timestamp() - np.random.randint(0, 30*24*3600))
        
        new_interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'timestamp': timestamp,
            'context': context
        })
    
    interactions = pd.DataFrame(new_interactions)
    
    # Generate test interactions (10% of total)
    test_size = num_interactions // 10
    test_interactions = []
    for _ in range(test_size):
        user_id = np.random.choice(user_data['user_id'])
        item_id = np.random.choice(item_data['item_id'])
        
        test_interactions.append({
            'user_id': user_id,
            'item_id': item_id
        })
    
    test_interactions = pd.DataFrame(test_interactions)
    
    return user_data, item_data, interactions, test_interactions

def generate_synthetic_musicincar(num_users, num_items, num_interactions):
    """Generate a synthetic MusicInCar-like dataset with specified sizes."""
    # Scale the existing function
    user_data, item_data, interactions, test_interactions = load_or_create_musicincar_sample()
    
    # Expand user data
    user_data = pd.concat([user_data] * (num_users // len(user_data) + 1), ignore_index=True)
    user_data = user_data.iloc[:num_users]
    user_data['user_id'] = range(1, num_users + 1)
    
    # Expand item data
    item_data = pd.concat([item_data] * (num_items // len(item_data) + 1), ignore_index=True)
    item_data = item_data.iloc[:num_items]
    item_data['item_id'] = range(101, 101 + num_items)
    
    # Generate new interactions
    new_interactions = []
    for _ in range(num_interactions):
        user_id = np.random.choice(user_data['user_id'])
        item_id = np.random.choice(item_data['item_id'])
        driving_condition = np.random.choice(['city', 'highway', 'countryside', 'traffic'])
        weather_condition = np.random.choice(['sunny', 'rainy', 'cloudy', 'night'])
        timestamp = int(datetime.datetime.now().timestamp() - np.random.randint(0, 30*24*3600))
        
        new_interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'timestamp': timestamp,
            'driving_condition': driving_condition,
            'weather': weather_condition
        })
    
    interactions = pd.DataFrame(new_interactions)
    
    # Generate test interactions (10% of total)
    test_size = num_interactions // 10
    test_interactions = []
    for _ in range(test_size):
        user_id = np.random.choice(user_data['user_id'])
        item_id = np.random.choice(item_data['item_id'])
        
        test_interactions.append({
            'user_id': user_id,
            'item_id': item_id
        })
    
    test_interactions = pd.DataFrame(test_interactions)
    
    return user_data, item_data, interactions, test_interactions

def plot_convergence_analysis(metrics_by_sample_size, title="Convergence of Metrics with Sample Size"):
    """
    Plot the convergence of metrics as sample size increases.
    
    Args:
        metrics_by_sample_size: Dictionary mapping sample sizes to metric values
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    sample_sizes = sorted(metrics_by_sample_size.keys())
    metrics = list(metrics_by_sample_size[sample_sizes[0]].keys())
    
    for metric in metrics:
        values = [metrics_by_sample_size[size][metric] for size in sample_sizes]
        plt.plot(sample_sizes, values, 'o-', label=metric)
    
    plt.xlabel('Sample Size')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('results/convergence_analysis.png', dpi=300)
    plt.show()

def analyze_user_distributions(user_data, item_data, interactions):
    """
    Analyze the distributions of user characteristics in the dataset.
    
    Args:
        user_data: DataFrame with user information
        item_data: DataFrame with item information
        interactions: DataFrame with user-item interactions
        
    Returns:
        Dictionary of user distribution statistics
    """
    # Count interactions per user
    interactions_per_user = interactions.groupby('user_id').size()
    
    # User activity levels
    low_activity = sum(interactions_per_user < 10)
    medium_activity = sum((interactions_per_user >= 10) & (interactions_per_user <= 50))
    high_activity = sum(interactions_per_user > 50)
    
    # Calculate preference diversity (entropy of item categories)
    diversity_scores = []
    
    for user_id in user_data['user_id']:
        user_items = interactions[interactions['user_id'] == user_id]['item_id']
        item_categories = item_data[item_data['item_id'].isin(user_items)]['category']
        
        if len(item_categories) > 0:
            # Count occurrences of each category
            category_counts = item_categories.value_counts()
            category_probs = category_counts / category_counts.sum()
            
            # Calculate entropy
            entropy = -np.sum(category_probs * np.log(category_probs))
            diversity_scores.append(entropy)
        else:
            diversity_scores.append(0)
    
    # User diversity levels
    diversity_array = np.array(diversity_scores)
    low_diversity = sum(diversity_array < 0.4)
    medium_diversity = sum((diversity_array >= 0.4) & (diversity_array <= 0.8))
    high_diversity = sum(diversity_array > 0.8)
    
    # Item popularity distribution
    item_popularity = interactions.groupby('item_id').size()
    item_popularity.sort_values(ascending=False, inplace=True)
    
    # Calculate Gini coefficient for popularity
    item_popularity_array = np.array(item_popularity)
    gini = 1 - 2 * np.sum((np.arange(1, len(item_popularity_array) + 1) / len(item_popularity_array)) * 
                         (item_popularity_array / item_popularity_array.sum())) + 1 / len(item_popularity_array)
    
    return {
        'user_activity': {
            'low': low_activity,
            'medium': medium_activity,
            'high': high_activity
        },
        'user_diversity': {
            'low': low_diversity,
            'medium': medium_diversity,
            'high': high_diversity
        },
        'item_popularity': {
            'gini': gini,
            'top10_share': item_popularity.iloc[:10].sum() / item_popularity.sum()
        }
    }

def plot_user_distributions(distribution_stats, title="User Characteristic Distributions"):
    """
    Plot the distributions of user characteristics.
    
    Args:
        distribution_stats: Dictionary of distribution statistics from analyze_user_distributions
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Activity distribution
    plt.subplot(2, 2, 1)
    activity = distribution_stats['user_activity']
    plt.bar(['Low', 'Medium', 'High'], [activity['low'], activity['medium'], activity['high']])
    plt.title('User Activity Distribution')
    plt.ylabel('Number of Users')
    
    # Diversity distribution
    plt.subplot(2, 2, 2)
    diversity = distribution_stats['user_diversity']
    plt.bar(['Low', 'Medium', 'High'], [diversity['low'], diversity['medium'], diversity['high']])
    plt.title('User Diversity Distribution')
    plt.ylabel('Number of Users')
    
    # Item popularity metrics
    plt.subplot(2, 2, 3)
    plt.bar(['Gini Coefficient', 'Top 10% Share'], 
           [distribution_stats['item_popularity']['gini'], distribution_stats['item_popularity']['top10_share']])
    plt.title('Item Popularity Metrics')
    plt.ylabel('Value')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/user_distributions.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Test the utility functions
    user_data, item_data, interactions, test_interactions = create_sample_data()
    print(f"Created sample data: {len(user_data)} users, {len(item_data)} items, {len(interactions)} interactions")
    
    # Test Frappe sample
    frappe_user, frappe_item, frappe_inter, frappe_test = load_or_create_frappe_sample()
    print(f"Created Frappe sample: {len(frappe_user)} users, {len(frappe_item)} items, {len(frappe_inter)} interactions")
    
    # Test MusicInCar sample
    music_user, music_item, music_inter, music_test = load_or_create_musicincar_sample()
    print(f"Created MusicInCar sample: {len(music_user)} users, {len(music_item)} items, {len(music_inter)} interactions")
    
    # Test distribution analysis
    dist_stats = analyze_user_distributions(user_data, item_data, interactions)
    print("Distribution statistics:", dist_stats)
    
    # Test plotting
    plot_user_distributions(dist_stats)