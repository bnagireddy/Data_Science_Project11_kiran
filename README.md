# Data Science Project: YouTube Data Analysis

Welcome to the **Data Science Project: YouTube Data Analysis**! This repository demonstrates a comprehensive data analysis pipeline, including cleaning, exploratory data analysis (EDA), visualization, machine learning, and natural language processing (NLP) on YouTube data.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Files](#project-files)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Results](#results)
6. [License](#license)

---

## Overview

This project analyzes YouTube data to uncover insights into video popularity, trends, and user engagement. It includes:

- **Data Cleaning:** Removing duplicates, combining fields, and handling missing values.
- **Exploratory Data Analysis (EDA):** Visualizing distributions, correlations, and trends in the data.
- **Machine Learning:** Building a regression model to predict video views based on attributes like likes, dislikes, and comment counts.
- **NLP Analysis:** Performing sentiment analysis and generating word clouds to understand the textual content of video titles and tags.

---

## Project Files

### 1. `cleaned_youtube.csv`
The cleaned dataset generated after data preprocessing in `data_cleaning.py`.

### 2. `data_cleaning.py`
- **Purpose:** Cleans the raw YouTube dataset.
- **Key Steps:**
  - Removes duplicates.
  - Combines `publish_date` and `time_frame` into a single datetime field.
  - Extracts new features like `publish_hour`, `publish_day`, and `publish_month`.
  - Cleans string fields and fills missing numerical values.
  - Saves the cleaned dataset as `cleaned_youtube.csv`.

### 3. `eda_visualization.py`
- **Purpose:** Conducts exploratory data analysis and generates insightful visualizations.
- **Visualizations:**
  - Distribution of views.
  - Count of videos by category.
  - Correlation heatmap.
  - Scatter plot for views vs likes.

### 4. `model_training.py`
- **Purpose:** Trains a linear regression model to predict video views.
- **Key Steps:**
  - Splits the dataset into training and testing sets.
  - Trains a linear regression model using `likes`, `dislikes`, and `comment_count` as features.
  - Evaluates the model using Mean Squared Error (MSE) and R-squared (R2) metrics.
  - Saves the trained model as `linear_regression_model.pkl`.

### 5. `nlp_text_analysis.py`
- **Purpose:** Performs Natural Language Processing (NLP) on video titles and tags.
- **Key Steps:**
  - Combines text fields for analysis.
  - Tokenizes and cleans text to remove stopwords and non-alphanumeric tokens.
  - Generates word frequencies and a word cloud.
  - Performs sentiment analysis on titles using the VADER sentiment analysis tool.
  - Saves the analyzed data as `nlp_analysis_results.csv`.

---

## Getting Started

### Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/bnagireddy/Data_Science_Project11_kiran.git
   cd Data_Science_Project11_kiran
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**:
   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:
   Install all required Python libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Resources**:
   Ensure the required NLTK resources are downloaded:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

---

## Usage

### Run Data Cleaning
```bash
python data_cleaning.py
```

### Run Exploratory Data Analysis and Visualization
```bash
python eda_visualization.py
```

### Train and Evaluate the Machine Learning Model
```bash
python model_training.py
```

### Perform NLP Analysis
```bash
python nlp_text_analysis.py
```

---

## Results

### Insights and Findings
- **Data Cleaning:** The dataset was processed to remove duplicates and handle missing values.
- **EDA:** Visualizations revealed patterns in video views, likes, and categories.
- **Machine Learning:** A regression model was trained to predict video views with an R2 score of ~0.8.
- **NLP Analysis:** Sentiment analysis and word clouds provided insights into the textual content of video titles and tags.

---

## License
