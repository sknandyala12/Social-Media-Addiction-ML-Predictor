# Social-Media-Addiction-ML-Predictor
ML models predicting social media addiction and productivity loss using behavioral data and demographic features, with a focus on fairness and interpretability.

Social Media Addiction & Productivity Loss Analysis

This project analyzes a Kaggle dataset titled "Time Wasters on Social Media" 
and an NYC.gov dataset exploring behavioral indicators and addiction levels among users. 
The analysis includes descriptive statistics, rule-based predictions, and visualizations to understand 
the impact of social media usage on productivity and self-control.

Project Goals
- Load and explore the dataset for patterns and correlations.
- Implement rule-based heuristics for predicting Addiction Level.
- Visualize distribution patterns for key behavioral metrics.
- Lay the groundwork for future machine learning model experimentation.

Dataset:
The dataset used is publicly available on Kaggle:
https://www.kaggle.com/datasets/muhammadroshaanriaz/time-wasters-on-social-media

You must have a Kaggle API token to download it programmatically.

Requirements:
Ensure you have Python 3 and the following libraries installed:
- pandas
- numpy
- matplotlib
- scikit-learn
- kaggle

Also, ensure you have your Kaggle API key set up:
- mkdir ~/.kaggle
- cp kaggle.json ~/.kaggle/
- chmod 600 ~/.kaggle/kaggle.json

How to Run:
- Clone this repository and navigate to the folder.
- Run the main analysis script:
- python3 analysis.py > output.txt

This will:
- Download the dataset from Kaggle.
- Unzip the dataset.
- Load and process the data.
- Print various descriptive statistics.
- Apply simple rule-based prediction heuristics.
- Generate and show visualizations for exploratory analysis.
- The standard output is redirected to output.txt for easy inspection.

Outputs:
- Descriptive statistics for all numerical and categorical features
- Value counts for categorical fields (Platform, Profession, etc.)
- Histograms and bar plots (e.g., Time Spent, Platform usage)
- Predictions using two rule-based baseline models:
- heuristic_baseline()
- rules_based_addiction()

File structure:
├── analysis.py          # Main script with analysis logic
├── output.txt           # Captures script output
├── README.md            # General overview
├── requirements.txt     # Python dependencies
├── time-wasters-on-social-media.zip  # Downloaded dataset
└── Time-Wasters on Social Media.csv  # Unzipped CSV

Future Work:
- Replace rule-based predictors with machine learning models (e.g., RandomForest, Logistic Regression).
- Add productivity loss prediction and correlation matrix analysis.
- Develop a front-end interface or API.

Contact:
For questions or suggestions, feel free to reach out to our team @:
suhithn@umich.edu
sforet@umich.edu
jjusko@umich.edu
thentes@umich.edu
navpreet@umich.edu
