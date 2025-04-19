# YouTube Trending Videos Analysis

## Project Overview

This data science project analyzes global YouTube trending videos to uncover patterns and insights about what makes videos trend on the platform. The analysis explores various aspects including video categories, channel characteristics, publishing patterns, engagement metrics, and regional trends. Additionally, the project implements machine learning models to predict video view counts and categories based on various features, providing content creators with valuable insights for optimizing their videos. The project culminates in a real-time monitoring system and A/B testing framework that can predict trending potential for new videos and optimize posting strategies.

## Dataset

The project uses the `youtube_trending_videos_global.csv` dataset, which contains information about trending videos across multiple countries. The dataset includes:

- Video metadata (ID, title, description, tags, duration, etc.)
- Channel information (title, subscriber count, country, age, etc.)
- Engagement metrics (views, likes, comments)
- Trending information (trending date, trending country)

**Dataset Size:** 2,058,048 rows × 26 columns (dataset size is more than 3.5GB hence download it from [Kaggle]([https://website-name.com](https://www.kaggle.com/datasets/canerkonuk/youtube-trending-videos-global)))

## Project Architecture

The project consists of several interconnected components:

### 1. Data Collection & Processing
- YouTube trending videos dataset ingestion
- Data cleaning and preprocessing pipeline
- Feature engineering for both numerical and text data

### 2. Exploratory Analysis
- Statistical analysis of video metrics and trends
- Visualization of key patterns and relationships
- Geographic and temporal trend analysis

### 3. Predictive Modeling
- View count prediction with Random Forest and Gradient Boosting
- Category classification using ensemble methods and text features
- Deep learning title analysis with LSTM networks

### 4. Real-time Monitoring System
- YouTube Data API integration
- Scheduled fetching of popular videos across countries and categories
- Feature extraction and trending potential prediction
- Ranked recommendations of videos likely to trend

### 5. A/B Testing Framework
- Posting time analysis and optimization
- Channel-specific performance testing
- Multi-metric evaluation of content strategies

## Features

The analysis includes both original dataset features and engineered features:

### Original Features
- `video_id`, `video_title`, `video_description`, etc.
- `channel_id`, `channel_title`, `channel_subscriber_count`, etc.
- `video_view_count`, `video_like_count`, `video_comment_count`
- `video_trending_date`, `video_trending_country`

### Engineered Features
- `video_duration_seconds`: Duration in seconds extracted from ISO 8601 format
- `days_to_trending`: Days between publishing and trending
- `channel_age_days`: Age of channel at time of video publishing
- `publish_day_of_week`: Day of week when video was published
- `publish_hour`: Hour of day when video was published
- `tag_count`: Number of tags used in the video
- `likes_per_view`: Ratio of likes to views
- `comments_per_view`: Ratio of comments to views
- `channel_size_category`: Categorization of channels by subscriber count

## Analysis Components

### 1. Data Cleaning & Preprocessing
- Handling missing values in video metadata and engagement metrics
- Converting datetime columns to appropriate format
- Removing irrelevant or redundant columns

### 2. Exploratory Data Analysis (EDA)
- Basic statistics of trending videos
- Distribution of trending videos across countries
- Video category analysis
- Top channels with most trending videos
- Publishing patterns (day of week and hour of day)
- Video duration distribution
- Engagement metrics analysis (views, likes, comments)
- Time-to-trending analysis

### 3. Correlation Analysis
- Correlation between engagement metrics
- Relationship between channel metrics and video performance
- Impact of content characteristics on engagement

### 4. Time Series Analysis
- Category popularity trends over time
- Engagement patterns over time

### 5. Predictive Modeling
- View count prediction using Random Forest Regressor
- Improved model with log transformation and additional features
- Advanced model using Gradient Boosting with hyperparameter tuning
- Video category classification using Random Forest Classifier with feature selection
- Enhanced category prediction with text feature integration
- Deep learning approach for title-based classification
- Learning curves and confusion matrix analysis
- Feature importance analysis for both prediction models
- Model performance comparison

## Key Findings

### Engagement Patterns
- Strong correlation (0.89) exists between video views and likes
- Channel subscriber count moderately correlates with both video views (0.48) and likes (0.60)
- Comment count has a weaker correlation with views (0.26) than likes (0.36), suggesting high viewership doesn't necessarily translate to high comment engagement
- The likes_per_view and comments_per_view metrics show moderate positive correlation (0.29)

### Content Characteristics
- Shorter videos tend to perform slightly better in terms of engagement (negative correlations of -0.14 with views, -0.13 with likes)
- Tag count and description length have a moderate positive correlation (0.31), indicating creators who use more tags also tend to write longer descriptions

### Publishing Patterns
- Friday is the most popular day for publishing trending videos
- Peak publishing hours are between 15:00-17:00 (3-5 PM)
- Videos typically trend within 5-7 days after publishing

### Channel Analysis
- Channel view count and subscriber count have a strong correlation (0.78)
- Larger channels (Large and Mega categories) tend to achieve higher likes for the same view count compared to smaller ones
- Channel age has weak correlations with most metrics, suggesting older channels don't necessarily perform better just because of their age

### Category Insights
- Entertainment is the dominant category among trending videos
- People & Blogs, Music, Sports, and Gaming round out the top 5 categories
- Category popularity fluctuates over time, with some seasonal patterns observed

### Predictive Modeling Insights
- Comments per view emerged as the most important feature for predicting view counts in both models
- Channel subscriber count is the second strongest predictor of view counts
- Video duration is another significant feature in determining view counts
- Channel video count has moderate importance
- Content-related features (tag_count, title_length) have relatively lower importance for view count prediction
- For category prediction, channel video count and subscriber count are the most important features
- Description length and video duration have significant importance for category prediction
- Adding text features to category prediction dramatically improved accuracy from 50% to 92%
- Text processing using TF-IDF vectorization of titles and descriptions provides essential context for categorization
- Ensemble approach combining text and numerical features outperforms single-model approaches
- Deep learning with LSTM networks achieves the highest accuracy (94%) for category prediction, using only video titles
- GPU acceleration with Metal framework enables efficient training on large datasets
- Model simplification and proper regularization (increased dropout) helped prevent overfitting in deep learning model
- Log transformation of view counts significantly improves model performance
- Random Forest model dramatically outperforms Gradient Boosting for view count prediction, achieving 99.49% R² score compared to 78.44% for Gradient Boosting
- The performance difference is even more pronounced in original scale metrics, with Random Forest RMSE being 33x better than Gradient Boosting
- Mean Absolute Percentage Error of 4.34% for Random Forest indicates high prediction accuracy

### A/B Testing Results
- **View Count**: Morning posts (8 AM - 12 PM) receive 35% more views than evening posts
- **Like Count**: Morning posts receive 17.8% more likes than evening posts
- **Comment Count**: Evening posts (5 PM - 9 PM) receive 82.9% more comments than morning posts
- **Engagement Ratios**: Evening posts show higher engagement ratios (likes per view: +2.6%, comments per view: +20.8%)
- **Posting Distribution**: Most creators post around 12-1 PM, but highest views occur at 6-7 PM (potentially indicating an opportunity)

## Machine Learning Models

### 1. View Count Prediction

#### Model Comparison

##### Random Forest (Best Performer)
- **Log-scale RMSE:** 0.1340 (6.5x better than Gradient Boosting)
- **Log-scale R²:** 0.9949 (27% higher than Gradient Boosting)
- **Original-scale RMSE:** 508,072.78 (35x better than Gradient Boosting)
- **Original-scale R²:** 0.9997 (50% higher than Gradient Boosting)
- **Mean Absolute Percentage Error:** 4.34%
- **Strengths:** Excellent performance across all metrics, better handles the complex relationships between features and view counts, resistant to overfitting despite the large feature set

##### Gradient Boosting
- **Log-scale RMSE:** 0.8707
- **Log-scale R²:** 0.7844
- **Original-scale RMSE:** 17,657,302.94
- **Original-scale R²:** 0.6672
- **Strengths:** Typically performs well on structured data, but in this case was outperformed by Random Forest

Despite expecting Gradient Boosting to perform better due to its typical advantages with structured data and hyperparameter tuning, Random Forest demonstrated substantially superior performance on this dataset. This suggests that the ensemble approach of Random Forest, which builds multiple independent decision trees and averages their predictions, is better suited to handle the complex, non-linear relationships in YouTube trending video data.

### 2. Video Category Classification

#### Base Model
- **Features:** video_duration_seconds, video_view_count, video_like_count, video_comment_count, title_length, description_length, tag_count, publish_hour, channel_subscriber_count
- **Target:** video_category_id (15 categories with at least 1000 samples each)
- **Model:** RandomForestClassifier with feature selection, class balancing, and parameter tuning
- **Performance:** 
  - Cross-validation Accuracy: 0.4979 ± 0.0049
  - Test Accuracy: 0.5014
  - Macro Average F1-score: 0.44
  - Weighted Average F1-score: 0.52

#### Improved Model with Text Features
- **Text Features:** TF-IDF vectors from cleaned video titles and descriptions
- **Numerical Features:** 17 features including original metrics and additional engineered features:
  - Basic engagement metrics (views, likes, comments)
  - Channel metrics (subscriber count, video count)
  - Content characteristics (title length, description length, etc.)
  - Advanced features (engagement ratio, title word count, has question/exclamation)
  - Video duration features (duration in minutes, is_short_video, is_long_video)
- **Modeling Approach:** Ensemble of three models
  - SGDClassifier trained on title text features
  - SGDClassifier trained on description text features
  - RandomForestClassifier trained on numerical features
  - Final prediction: Weighted average (50% numerical, 30% title, 20% description)
- **Performance:**
  - Test Accuracy: 0.9207 (92.07%)
  - Macro Average F1-score: 0.92
  - Weighted Average F1-score: 0.92

#### Category Classification Insights:
- Adding text features dramatically improved classification accuracy from 50% to 92%
- Pets & Animals (1.00 precision) and Howto & Style (0.99 precision) categories have the highest prediction accuracy
- Entertainment (0.76 recall) category remains the most difficult to classify correctly
- Channel video count and channel subscriber count are the most important numerical features
- Video duration and description length significantly contribute to category classification
- Title characteristics (word count, presence of question/exclamation marks) provide valuable signals

### 3. Deep Learning for Title-Based Classification

#### Initial LSTM Model
- **Architecture:** Complex model with dual Bidirectional LSTM layers
- **Features:** Video titles converted to token sequences
- **Target:** Top 5 video categories (Entertainment, People & Blogs, Music, Sports, Gaming)
- **Performance:**
  - Initial accuracy: ~89%
  - Exhibited catastrophic forgetting after epoch 2
  - Training accuracy declined from ~90% to ~66%
  - Unstable validation pattern

#### Metal GPU-Accelerated Model (Improved)
- **Architecture:** Simplified with single Bidirectional LSTM layer and increased dropout (0.7)
- **Hardware Acceleration:** Leveraged Metal GPU capabilities for faster training
- **Training Approach:** Implemented early stopping and learning rate reduction
- **Performance:**
  - Test Accuracy: 93.78% across 5 categories
  - Training time: 894.27 seconds (~15 minutes) for 1.65 million videos
  - Excellent stability with proper early stopping after epoch 4
  - Category F1-Scores: 0.92-0.96

#### Sample Title Predictions:
| Sample Title | Predicted Category | Confidence |
|-------------|-------------------|------------|
| "How to Make the Perfect Chocolate Cake" | People & Blogs | 99.99% |
| "EPIC Gaming Moments That Will Blow Your Mind" | Entertainment | 100.00% |
| "Latest News: Breaking World Events Today" | People & Blogs | 97.24% |
| "Music Video - Summer Hit 2024" | Music | 99.73% |

## Feature Importance Findings

### For View Count Prediction
- **Comments per view:** Most important predictor in both Random Forest and Gradient Boosting models
- **Channel subscriber count:** Second most important feature, suggesting that creator popularity is a key factor
- **Video duration:** Significant importance in predicting views, with shorter videos generally performing better
- **Channel video count:** Moderate importance, indicating that channel activity level impacts viewership
- **Likes per view:** Shows moderate importance in the model
- **Content features:** Title length, description length, and tag count have lesser importance for view prediction

### For Category Classification
#### Numerical Features Only Model:
- **Channel subscriber count:** Top feature for distinguishing between content categories
- **Video duration:** Strong indicator of content type, as different categories tend to have characteristic lengths
- **Description length:** Highly important for category prediction, unlike in view count prediction
- **Title length:** Moderate importance, helps distinguish between types of content

#### Improved Model (Text + Numerical Features):
- **Channel video count:** Most important numerical feature in the enhanced model
- **Channel subscriber count:** Second most important feature, consistent with previous findings
- **Video duration metrics:** Both seconds and minutes versions show high importance
- **Description length and word count:** Strong indicators of content type
- **Engagement metrics:** Views, likes, and comments all contribute significantly
- **Content characteristics:** Title metrics, tag count, and presence of question/exclamation marks provide additional signal
- **Text features:** TF-IDF vectors from titles and descriptions (not shown in feature importance chart) provide essential context that dramatically improves classification accuracy

## Real-time Monitoring System

The project includes a production-ready system for continuous monitoring of YouTube trends, designed to leverage the insights gained from the analysis and predictive models.

### Key Features
- **Continuous Monitoring**: Automatically fetches popular videos at configurable intervals (default: 6 hours)
- **Geographic Coverage**: Tracks videos across multiple countries (US, UK, India, Canada, Australia)
- **Category Targeting**: Focuses on specific content categories of interest
- **Feature Extraction**: Processes video and channel metadata to extract predictive features
- **ML-Powered Prediction**: Uses trained models to calculate trending probability scores
- **Ranked Recommendations**: Identifies and ranks videos by their likelihood of trending
- **Reporting**: Generates timestamped CSV reports and console output with actionable insights

### Implementation Details
- Uses YouTube Data API for real-time data collection
- Implements efficient API quota management
- Extracts the same feature set used in the predictive models
- Provides direct links to high-potential videos
- Supports automated scheduling for continuous operation

### Use Cases
- Content strategy planning for creators and marketers
- Competitive analysis and trend identification
- Early discovery of potential viral content
- Performance tracking across regions and categories

This system transforms the project from a retrospective analysis to an actionable tool that can identify trending video candidates before they achieve widespread popularity.

## A/B Testing Framework

The project includes a robust A/B testing framework for evaluating optimal video posting strategies:

### Key Features
- **Posting Time Analysis**: Compares morning (8 AM - 12 PM) vs evening (5 PM - 9 PM) posting performance
- **Channel-Specific Testing**: Analyze posting patterns from specific channels (e.g., MrBeast, PewDiePie)
- **Trending Video Testing**: Apply tests to videos identified by the trending prediction system
- **Multi-Metric Evaluation**: Assesses impact on views, likes, comments, and engagement ratios
- **Statistical Visualization**: Automatically generates bar charts showing percent changes between test groups
- **Result Documentation**: Saves test results as timestamped CSV files for historical comparison

### Implementation Details
- Uses YouTube Data API to fetch video metadata and statistics
- Converts posting timestamps to local time for accurate time-based analysis
- Calculates key engagement metrics including likes per view and comments per view
- Generates distribution analysis of posting hours across channels
- Creates hour-by-hour performance charts showing optimal posting times

### Use Cases
- Data-driven posting schedule optimization
- Channel-specific strategy development
- Target audience timing analysis
- Engagement type prioritization (views vs. comments)
- A/B testing additional content variables

## Technological Stack

- **Programming Language:** Python 3.x
- **Data Manipulation:** pandas, NumPy
- **Data Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn (RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier)
- **Natural Language Processing:** NLTK, TF-IDF Vectorization
- **Deep Learning:** TensorFlow, Keras (Bidirectional LSTM)
- **GPU Acceleration:** Metal GPU support for Apple Silicon
- **Text Processing:** Regular expressions, tokenization, sequence padding
- **Evaluation Metrics:** RMSE, MAE, R², accuracy, F1-score, confusion matrices
- **API Integration:** Google API Client Library, requests

## Model Evolution

### View Count Prediction
1. **Base Model:** Simple Random Forest with raw features
2. **Improved Model:** Random Forest with log-transformed target and feature engineering
3. **Advanced Model:** Gradient Boosting with hyperparameter tuning (underperformed compared to Random Forest)

### Category Classification
1. **Base Model:** Random Forest with numerical features only (50% accuracy)
2. **Enhanced Model:** Ensemble of models with text features integration (92% accuracy)
3. **Deep Learning Model:** Bidirectional LSTM for title-based classification (94% accuracy)
   - **Initial Version:** Complex dual LSTM architecture (unstable training)
   - **Improved Version:** Simplified architecture with Metal GPU acceleration (stable and efficient)

## Installation & Usage

### Dependencies
```
pandas
numpy
matplotlib
seaborn
datetime
re
nltk
wordcloud
scikit-learn (RandomForestRegressor, GradientBoostingRegressor, GridSearchCV, etc.)
plotly
tensorflow
google-api-python-client
requests
```

### Running the Analysis
1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place the dataset file in the appropriate directory
4. Run the main analysis script:
   ```
   python youtube_trending_analysis.py
   ```

### Running the Monitoring System
1. Set up a YouTube Data API key
2. Configure the monitoring parameters in the script:
   ```python
   API_KEY = 'your_api_key'
   CATEGORIES_TO_MONITOR = [1, 10, 20, 23, 24]  # Adjust as needed
   COUNTRIES_TO_MONITOR = ['US', 'GB', 'IN', 'CA', 'AU']  # Adjust as needed
   MONITORING_INTERVAL_HOURS = 6  # Adjust as needed
   ```
3. Run the monitoring script:
   ```
   python youtube_monitoring.py
   ```

### Running the A/B Testing Framework
1. Configure API settings in the script:
   ```python
   API_KEY = 'your_api_key'
   ```
2. Choose one of the testing methods:
   - Test with specific channels:
     ```python
     channels_to_analyze = [
         "UCX6OQ3DkcsbYNE6H8uQQuVA",  # MrBeast
         "UC-lHJZR3Gqxm24_Vd_AJ5Yw"    # PewDiePie
     ]
     videos_df, results = run_posting_time_test(API_KEY, channel_ids=channels_to_analyze)
     ```
   - Test with trending prediction results:
     ```python
     trending_df = pd.read_csv('trending_predictions_timestamp.csv')
     video_ids = trending_df['video_id'].tolist()
     videos_df, results = run_posting_time_test(API_KEY, video_ids=video_ids)
     ```
3. Run the script to generate visualizations and CSV reports:
   ```
   python youtube_ab_testing.py
   ```

## File Structure
```
youtube_trending_analysis/
│
├── data/
│   └── youtube_trending_videos_global.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_correlation_analysis.ipynb
│   ├── 04_time_series_analysis.ipynb
│   ├── 05_view_count_prediction.ipynb
│   ├── 06_category_classification.ipynb
│   └── 07_deep_learning_models.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   ├── ml_models.py
│   └── deep_learning.py
│
├── models/
│   ├── trending_prediction_model.pkl
│   ├── category_classifier.pkl
│   └── title_lstm_model.h5
│
├── output/
│   ├── figures/
│   ├── predictions/
│   └── ab_test_results/
│
├── youtube_monitoring.py
├── youtube_ab_testing.py
├── requirements.txt
└── README.md
```

## 🚀 Potential Use Cases

### 📈 **For Content Creators & Influencers**
- **Optimize Posting Time**: Use A/B testing results to schedule posts for maximum engagement (e.g., post in the morning for higher views and likes).
- **Trend Forecasting**: Leverage real-time monitoring to identify which videos are likely to trend next.
- **Title Optimization**: Use the deep learning model to suggest trending-worthy video titles.
- **Category Targeting**: Identify high-performing categories and adjust content themes accordingly.

---

### 💼 **For Marketing & Brand Teams**
- **Competitor Benchmarking**: Track performance of competitor channels and replicate successful content strategies.
- **Ad Strategy Alignment**: Use category popularity trends and engagement metrics to guide ad placement or influencer partnerships.
- **Audience Segmentation**: Tailor content by analyzing regional preferences and engagement ratios.


## Author
- Aaryan Akshay Shah
- Maehika Hegde

