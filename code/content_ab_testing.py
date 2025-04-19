# simple_ab_test.py - revised version
import pandas as pd
import os
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from datetime import datetime

class SimpleContentTester:
    def __init__(self, api_key):
        """Initialize with YouTube API key"""
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    def get_video_stats(self, video_ids):
        """Get statistics for a list of videos"""
        results = []
        
        # Process videos in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            
            # Get video details
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(batch)
            ).execute()
            
            # Process each video
            for item in response.get('items', []):
                video_data = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'publish_date': item['snippet']['publishedAt'],
                    'channel_id': item['snippet']['channelId'],
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'like_count': int(item['statistics'].get('likeCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0)),
                    'has_question': '?' in item['snippet']['title'],
                    'has_number': any(c.isdigit() for c in item['snippet']['title']),
                    'title_length': len(item['snippet']['title']),
                    'is_uppercase': sum(1 for c in item['snippet']['title'] if c.isupper()) > len(item['snippet']['title'])/3,
                    'has_emoji': any(ord(c) > 9000 for c in item['snippet']['title']),
                    'category_id': item['snippet'].get('categoryId', '')
                }
                
                # Calculate engagement metrics
                if video_data['view_count'] > 0:
                    video_data['likes_per_view'] = video_data['like_count'] / video_data['view_count']
                    video_data['comments_per_view'] = video_data['comment_count'] / video_data['view_count']
                else:
                    video_data['likes_per_view'] = 0
                    video_data['comments_per_view'] = 0
                
                results.append(video_data)
        
        return pd.DataFrame(results)
    
    def compare_groups(self, group_a_df, group_b_df, experiment_name="comparison"):
        """Compare metrics between two groups of videos"""
        metrics = ['view_count', 'like_count', 'comment_count', 
                  'likes_per_view', 'comments_per_view']
        
        results = {}
        
        # Calculate stats for each metric
        for metric in metrics:
            results[metric] = {
                'group_a_mean': group_a_df[metric].mean(),
                'group_b_mean': group_b_df[metric].mean(),
                'difference': group_b_df[metric].mean() - group_a_df[metric].mean(),
                'percent_change': ((group_b_df[metric].mean() - group_a_df[metric].mean()) 
                                  / group_a_df[metric].mean() * 100) if group_a_df[metric].mean() != 0 else 0
            }
        
        # Create output directory
        os.makedirs('ab_test_results', exist_ok=True)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ab_test_results/{experiment_name}_{timestamp}.csv"
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'metric': list(results.keys()),
            'group_a_mean': [results[m]['group_a_mean'] for m in results],
            'group_b_mean': [results[m]['group_b_mean'] for m in results],
            'difference': [results[m]['difference'] for m in results],
            'percent_change': [results[m]['percent_change'] for m in results]
        })
        
        # Save to CSV
        results_df.to_csv(filename, index=False)
        
        # Create visualization
        self.plot_results(results, experiment_name, timestamp)
        
        return results_df
    
    def plot_results(self, results, experiment_name, timestamp):
        """Create visualization of A/B test results"""
        metrics = list(results.keys())
        percent_changes = [results[m]['percent_change'] for m in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, percent_changes)
        
        # Color bars based on positive/negative change
        for i, bar in enumerate(bars):
            bar.set_color('green' if percent_changes[i] > 0 else 'red')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'A/B Test Results: {experiment_name}')
        plt.ylabel('Percent Change (Group B vs Group A)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"ab_test_results/{experiment_name}_{timestamp}.png")
        plt.close()
    
    def classify_by_title_length(self, videos_df):
        """Classify videos based on title length (simpler classification)"""
        # Calculate median title length
        median_length = videos_df['title_length'].median()
        
        # Group A: Shorter titles
        group_a = videos_df[videos_df['title_length'] <= median_length]
        
        # Group B: Longer titles
        group_b = videos_df[videos_df['title_length'] > median_length]
        
        return group_a, group_b
    
    def classify_by_any_feature(self, videos_df, feature, label_a, label_b):
        """Generic function to classify videos by any boolean feature"""
        # Group A: Videos without the feature
        group_a = videos_df[~videos_df[feature]]
        
        # Group B: Videos with the feature
        group_b = videos_df[videos_df[feature]]
        
        return group_a, group_b, label_a, label_b

# Example usage
def run_simple_ab_test(api_key, channel_id=None, custom_video_ids=None):
    """Run a simple A/B test on YouTube content"""
    tester = SimpleContentTester(api_key)
    
    # Get videos - either from a channel or from custom list
    if channel_id:
        # Get videos from channel
        response = tester.youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            type="video",
            order="date"
        ).execute()
        
        video_ids = [item['id']['videoId'] for item in response.get('items', [])]
    elif custom_video_ids:
        video_ids = custom_video_ids
    else:
        raise ValueError("Either channel_id or custom_video_ids must be provided")
    
    print(f"Analyzing {len(video_ids)} videos...")
    
    # Get video stats
    videos_df = tester.get_video_stats(video_ids)
    
    # Print all videos and their features for debugging
    print("\nDetected Video Features:")
    for _, row in videos_df.iterrows():
        print(f"Video: {row['title']}")
        print(f"  - Length: {row['title_length']} chars")
        print(f"  - Has numbers: {row['has_number']}")
        print(f"  - Has question: {row['has_question']}")
        print(f"  - Uppercase: {row['is_uppercase']}")
        print(f"  - Views: {row['view_count']}")
        print(f"  - Likes: {row['like_count']}")
    
    # Try different classification methods until we find one that works
    
    # 1. Try classifying by title length (always works)
    group_a, group_b = tester.classify_by_title_length(videos_df)
    
    print(f"\nClassification by title length:")
    print(f"Group A (Shorter titles): {len(group_a)} videos")
    print(f"Group B (Longer titles): {len(group_b)} videos")
    
    # Compare groups
    if len(group_a) > 0 and len(group_b) > 0:
        print("\nRunning A/B test on title length...")
        results = tester.compare_groups(group_a, group_b, "title_length_test")
        
        print("\nA/B Test Results:")
        print(results)
        print("\nRecommendations:")
        
        # Generate simple recommendations
        for metric, values in results.iterrows():
            if abs(values['percent_change']) > 10:
                direction = "higher" if values['percent_change'] > 0 else "lower"
                print(f"- {metric}: Longer titles perform {abs(values['percent_change']):.1f}% {direction}")
    else:
        print("Not enough videos for comparison")
    
    # 2. Try other features if we have enough videos
    if len(videos_df) >= 4:
        features_to_try = [
            ('has_number', 'Without numbers', 'With numbers'),
            ('is_uppercase', 'Normal case', 'UPPERCASE heavy'),
            ('has_question', 'Without questions', 'With questions')
        ]
        
        for feature, label_a, label_b in features_to_try:
            group_a, group_b, label_a, label_b = tester.classify_by_any_feature(videos_df, feature, label_a, label_b)
            
            if len(group_a) > 0 and len(group_b) > 0:
                print(f"\nClassification by {feature}:")
                print(f"Group A ({label_a}): {len(group_a)} videos")
                print(f"Group B ({label_b}): {len(group_b)} videos")
                
                # Run comparison
                print(f"\nRunning A/B test on {feature}...")
                results = tester.compare_groups(group_a, group_b, f"{feature}_test")
                
                print("\nA/B Test Results:")
                print(results)
    
    return videos_df

# If running directly
if __name__ == "__main__":
    API_KEY = 'AIzaSyCPGnczZG-sn_f6i6iNh5qvEcmSsXN5LOA'
    # Option 1: Run test on a specific channel
    # run_simple_ab_test(API_KEY, channel_id="UCxxx")  # Replace with actual channel ID
    
    # Option 2: Run test on custom list of videos
    custom_videos = [
        # Add your video IDs here - these are just examples
        "dQw4w9WgXcQ",  
        "xvFZjo5PgG0"   
    ]
    run_simple_ab_test(API_KEY, custom_video_ids=custom_videos)