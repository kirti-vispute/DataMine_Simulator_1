#!/usr/bin/env python3
"""
Data Mining Demo Script
Demonstrates the capabilities of the data mining applications without GUI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns

def load_sample_data(dataset_name):
    """Load sample datasets"""
    if dataset_name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['species'] = [data.target_names[i] for i in data.target]
        return df
    elif dataset_name == "wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['wine_type'] = [data.target_names[i] for i in data.target]
        return df
    elif dataset_name == "diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df

def demo_kmeans(df):
    """Demonstrate K-Means clustering"""
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING DEMO")
    print("="*50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        print("❌ Need at least 2 numeric columns for clustering")
        return
        
    X = df[numeric_cols[:2]]  # Use first 2 numeric columns
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, clusters)
    inertia = kmeans.inertia_
    
    print(f"✓ Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Features used: {numeric_cols[0]}, {numeric_cols[1]}")
    print(f"✓ Number of clusters: 3")
    print(f"✓ Silhouette Score: {silhouette:.3f}")
    print(f"✓ Inertia: {inertia:.2f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title('K-Means Clustering Results')
    plt.colorbar(label='Cluster')
    
    plt.subplot(1, 2, 2)
    plt.hist(clusters, bins=3, alpha=0.7, edgecolor='black')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Cluster Distribution')
    
    plt.tight_layout()
    plt.savefig('kmeans_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization saved as 'kmeans_demo.png'")

def demo_classification(df):
    """Demonstrate classification"""
    print("\n" + "="*50)
    print("CLASSIFICATION DEMO")
    print("="*50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) == 0 or len(numeric_cols) == 0:
        print("❌ Need both numeric and categorical columns for classification")
        return
        
    X = df[numeric_cols]
    y = df[categorical_cols[0]]
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train multiple models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"✓ {name} Accuracy: {accuracy:.3f}")
    
    # Feature importance for Random Forest
    rf = models['Random Forest']
    importance = rf.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Feature Importance (Random Forest):")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    accuracies = list(results.values())
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    plt.barh(feature_importance['feature'][:5], feature_importance['importance'][:5])
    plt.xlabel('Importance')
    plt.title('Top 5 Feature Importances')
    
    plt.tight_layout()
    plt.savefig('classification_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization saved as 'classification_demo.png'")

def demo_regression(df):
    """Demonstrate regression"""
    print("\n" + "="*50)
    print("REGRESSION DEMO")
    print("="*50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        print("❌ Need at least 2 numeric columns for regression")
        return
        
    # Use first column as target, rest as features
    target_col = numeric_cols[-1]
    feature_cols = numeric_cols[:-1]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    
    print(f"✓ Target variable: {target_col}")
    print(f"✓ Features: {', '.join(feature_cols)}")
    print(f"✓ R² Score: {r2:.3f}")
    print(f"✓ Intercept: {lr.intercept_:.3f}")
    
    # Show coefficients
    print("\n📊 Regression Coefficients:")
    for feature, coef in zip(feature_cols, lr.coef_):
        print(f"  {feature}: {coef:.3f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.title('Linear Regression Results')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(lr.coef_)), lr.coef_)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Regression Coefficients')
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=45)
    
    plt.tight_layout()
    plt.savefig('regression_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization saved as 'regression_demo.png'")

def demo_pca(df):
    """Demonstrate PCA"""
    print("\n" + "="*50)
    print("PCA DEMO")
    print("="*50)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        print("❌ Need at least 2 numeric columns for PCA")
        return
        
    X = df[numeric_cols]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"✓ Original dimensions: {X.shape[1]}")
    print(f"✓ Reduced dimensions: {X_pca.shape[1]}")
    print(f"✓ Explained variance ratio: {explained_variance}")
    print(f"✓ Cumulative variance: {cumulative_variance}")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
    plt.title('PCA Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(explained_variance)), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization saved as 'pca_demo.png'")

def main():
    """Main demo function"""
    print("🚀 Data Mining Algorithm Demo")
    print("=" * 60)
    
    # Load sample datasets
    datasets = {
        'iris': load_sample_data('iris'),
        'wine': load_sample_data('wine'),
        'diabetes': load_sample_data('diabetes')
    }
    
    # Run demos
    for name, df in datasets.items():
        print(f"\n📊 Dataset: {name.upper()}")
        print(f"Shape: {df.shape}")
        
        if name == 'iris':
            demo_kmeans(df)
            demo_classification(df)
            demo_pca(df)
        elif name == 'wine':
            demo_classification(df)
            demo_pca(df)
        elif name == 'diabetes':
            demo_regression(df)
            demo_pca(df)
    
    print("\n" + "="*60)
    print("✅ Demo completed! Check the generated PNG files for visualizations.")
    print("💡 To run the interactive applications, use:")
    print("   python launcher.py")
    print("   python data_mining_app.py")
    print("   python advanced_data_mining_app.py")

if __name__ == "__main__":
    main()
