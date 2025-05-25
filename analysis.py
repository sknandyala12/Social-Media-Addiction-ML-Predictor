import os
import zipfile
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc


pd.set_option('display.max_rows', None)     
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 1000)        
pd.set_option('display.max_colwidth', None) 

def download_dataset():
    
    dataset_url = "muhammadroshaanriaz/time-wasters-on-social-media"
    
    print("Downloading dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_url])

def unzip_dataset(zip_file="time-wasters-on-social-media.zip"):
    if not os.path.exists(zip_file):
        print(f"'{zip_file}' not found. Make sure the dataset is downloaded.")
        return

    print(f"Unzipping '{zip_file}'...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(".")  
    print("Dataset unzipped.")

def load_data(csv_file="Time Wasters Social Media.csv"):
    
    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' not found!")
        return None

    print(f"Loading data from '{csv_file}'...")
    df = pd.read_csv(csv_file)
    print("Data loaded successfully!")
    return df

def heuristic_baseline(df, threshold=5):
    """
    A simple rule:
      - If 'Total Time Spent' > threshold, predict classes >= 4 (high addiction).
      - Otherwise predict classes 0-3 (low addiction).

    For a multi-class scenario (0..7):
    Let's say we split it:
      - "low" = [0,1,2,3]
      - "high" = [4,5,6,7]
    We'll arbitrarily pick class 0 for 'low' and class 5 for 'high' 
    (or any single 'high' class).
    """
    # This is just one possible naive approach; adapt to your data.
    predictions = []
    for _, row in df.iterrows():
        if row["Total Time Spent"] > threshold:
            # arbitrarily picking class 5 for "high" addiction
            predictions.append(6)
        elif row["Total Time Spent"] > 2: 
            predictions.append(3)
        else: 
            predictions.append(0)
            # arbitrarily picking class 0 for "low" addiction
    return predictions

def rules_based_addiction(df):
    """
    Rule-based prediction for Addiction Level.
    Adjust rules based on exploratory analysis.
    """
    predictions = []
    for _, row in df.iterrows():
        time_spent = row.get("Total Time Spent", 0)
        sessions = row.get("Number of Sessions", 0)
        self_control = row.get("Self Control", 5)  # assume mid if missing

        # Example logic:
        if time_spent > 6 and self_control < 3:
            predictions.append(7)
        elif time_spent > 4 and self_control < 4:
            predictions.append(6)
        elif time_spent > 3 and sessions > 5:
            predictions.append(5)
        elif time_spent > 2:
            predictions.append(3)
        else:
            predictions.append(1)
    return predictions

def main():
    download_dataset()
    unzip_dataset("time-wasters-on-social-media.zip")  # Adjust if the ZIP has a different name
    df = load_data("Time-Wasters on Social Media.csv")  

    if df is not None:
        print(df.head())
        print("=== DataFrame Shape ===")
        print(df.shape)
        print("\n=== DataFrame Columns ===")
        print(df.columns)
        print("\n=== DataFrame Info ===")
        print(df.info())
        print("\n=== Statistical Summary (Numeric Only) ===")
        print(df.describe)

        #printing out stats for each individual numerical category
        if 'Age' in df.columns:
            print("\n=== Age Distribution Stats ===")
            print(df['Age'].describe())
        
        if 'Total Time Spent' in df.columns:
            print("\n=== Total Time Spent Distribution Stats ===")
            print(df['Total Time Spent'].describe())

        if 'Number of Sessions' in df.columns:
            print("\n=== Total Number of Sessions Distribution Stats ===")
            print(df['Number of Sessions'].describe())

        if 'Total Time Spent' in df.columns:
            print("\n=== Total Time Spent Distribution Stats ===")
            print(df['Total Time Spent'].describe())

        if 'Scroll Rate' in df.columns:
            print("\n=== Scroll Rate Distribution Stats ===")
            print(df['Scroll Rate'].describe())

        if 'Video Length' in df.columns:
            print("\n=== Video Length Distribution Stats ===")
            print(df['Video Length'].describe())

        if 'Frequency' in df.columns:
            print("\n=== Frequency Distribution Stats ===")
            print(df['Frequency'].describe())
        
        if 'Engagement' in df.columns:
            print("\n=== Engagement Distribution Stats ===")
            print(df['Engagement'].describe())

        if 'Productivity Loss' in df.columns:
            print("\n=== Productivity Loss Distribution Stats ===")
            print(df['Productivity Loss'].describe())

        if 'Watch Time' in df.columns:
            print("\n=== Watch Time Distribution Stats ===")
            print(df['Watch Time'].describe())

        if 'Self Control' in df.columns:
            print("\n=== Self Control Distribution Stats ===")
            print(df['Self Control'].describe())

        if 'Addiction Level' in df.columns:
            print("\n=== Addiction Level Distribution Stats ===")
            print(df['Addiction Level'].describe())

        #categorical data stats
        for col in ['Gender', 'Location', 'Platform', 'DeviceType', 'Video Category', 'Demographics', 'Profession', 'Watch Reason']:
            if col in df.columns:
                print(f"\n=== Value Counts: {col} ===")
                print(df[col].value_counts())

        #visualizations
        if 'Total Time Spent' in df.columns:
            plt.figure()
            df['Total Time Spent'].hist()
            plt.title("Distribution of Total Time Spent")
            plt.xlabel("Units")
            plt.ylabel("Frequency")
            plt.show()

        if 'Platform' in df.columns:
            plt.figure()
            df['Platform'].value_counts().plot(kind='bar')
            plt.title("Number of Users by Platform")
            plt.xlabel("Platform")
            plt.ylabel("Count")
            plt.show()

        numeric_cols = df.select_dtypes(include=['int', 'float']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            print("\n=== Correlation Matrix (Numeric Columns) ===")
            print(corr_matrix)
        
        
            plt.figure()
            cax = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
            plt.title("Correlation Heatmap")
            plt.colorbar(cax)
            
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            plt.show()
    
   
        if all(x in df.columns for x in ['Gender', 'Total Time Spent']):
            group_stats = df.groupby('Gender')['Total Time Spent'].mean()
            print("\n=== Average Total Time Spent by Gender ===")
            print(group_stats)

    

        #correlation with addiction level
        columns_of_interest = [
            'Addiction Level',
            'Total Time Spent',
            'Number of Sessions',
            'ProductivityLoss',
            'Self Control'
        ]
    
        
        df_filtered = df[columns_of_interest].dropna()

        # Calculate correlation matrix
        corr_matrix = df_filtered.corr()
    
        print("=== Correlation Matrix for Selected Features ===")
        print(corr_matrix)
    
        print("\n=== Correlation with Addiction Level ===")
        print(corr_matrix['Addiction Level'])

        plt.figure()
        plt.scatter(df_filtered['Total Time Spent'], df_filtered['Addiction Level'])
        plt.xlabel("Total Time Spent")
        plt.ylabel("Addiction Level")
        plt.title("Scatter Plot: Time Spent vs. Addiction Level")
        plt.show()



        #prediction trial? 
        features = [
            "Total Time Spent", 
            "Number of Sessions", 
            "Self Control",
            "Scroll Rate",
            "Number of Videos Watched",
            "Time Spent On Video"
        ]

        target = "Addiction Level" 

        X = df[features]
        y = df[target]

       
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(random_state=42)
    
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, None]
        }
        grid_search = GridSearchCV(
            model, 
            param_grid=param_grid, 
            cv=5, 
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
    
        best_model = grid_search.best_estimator_
        print("Best Params:", grid_search.best_params_)

        y_heuristic_pred = heuristic_baseline(X_test, threshold=5)

        heuristic_acc = accuracy_score(y_test, y_heuristic_pred)
        print(f"Heuristic Baseline Accuracy: {heuristic_acc}")
        print("Heuristic Classification Report:")
        print(classification_report(y_test, y_heuristic_pred, zero_division=0))

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Random Forest")
        plt.show()
        print("Accuracy:", acc)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # === Rules-Based Predictions ===
        print("\n=== Rules-Based Heuristics ===")
        rules_based_preds = rules_based_addiction(X_test.copy())  # Or df.loc[X_test.index] if it needs more columns
        rules_acc = accuracy_score(y_test, rules_based_preds)
        print(f"Rules-Based Accuracy: {rules_acc}")
        print("Rules-Based Classification Report:")
        print(classification_report(y_test, rules_based_preds, zero_division=0))

        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = np.array(features)

        plt.figure()
        plt.title("Feature Importances - Random Forest")
        plt.bar(range(len(importances)), importances[indices], color="skyblue", align="center")
        plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


        # Binarize the output
        y_bin = label_binarize(y, classes=sorted(y.unique()))
        n_classes = y_bin.shape[1]

        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)

        classifier = OneVsRestClassifier(RandomForestClassifier(**grid_search.best_params_, random_state=42))
        y_score = classifier.fit(X_train_bin, y_train_bin).predict_proba(X_test_bin)

        # Plot ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure()
        colors = plt.cm.get_cmap('tab10', n_classes)
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=colors(i))

        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.title('Multi-Class ROC Curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
