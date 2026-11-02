#!/usr/bin/env python3
"""
Data Mining & Machine Learning Algorithm Demonstrator
A user-friendly Python application for exploring various algorithms with interactive graphs.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataMiningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Mining Algorithm Explorer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.current_algorithm = None
        
        # Results storage
        self.results = {}
        
        self.setup_ui()
        self.load_sample_data()
        
    def setup_ui(self):
        """Create the main user interface"""
        # Create main frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for plots
        self.plot_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_data_controls()
        self.setup_algorithm_controls()
        self.setup_plot_area()
        
    def setup_data_controls(self):
        """Setup data loading and preview controls"""
        # Data loading section
        data_section = ttk.LabelFrame(self.control_frame, text="Data Management", padding=5)
        data_section.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(data_section, text="Load CSV File", command=self.load_csv).pack(fill=tk.X, pady=2)
        ttk.Button(data_section, text="Load Sample Dataset", command=self.load_sample_data).pack(fill=tk.X, pady=2)
        
        # Sample data selection
        self.sample_var = tk.StringVar(value="iris")
        sample_combo = ttk.Combobox(data_section, textvariable=self.sample_var, 
                                  values=["iris", "wine", "diabetes", "sales", "custom"])
        sample_combo.pack(fill=tk.X, pady=2)
        
        # Data info
        self.data_info = ttk.Label(data_section, text="No data loaded", wraplength=200)
        self.data_info.pack(pady=5)
        
    def setup_algorithm_controls(self):
        """Setup algorithm selection and parameters"""
        algo_section = ttk.LabelFrame(self.control_frame, text="Algorithm Selection", padding=5)
        algo_section.pack(fill=tk.X, pady=(0, 10))
        
        # Algorithm selection
        self.algo_var = tk.StringVar(value="kmeans")
        algo_combo = ttk.Combobox(algo_section, textvariable=self.algo_var, 
                                values=["kmeans", "agglomerative", "decision_tree", 
                                       "random_forest", "linear_regression", "data_exploration"])
        algo_combo.pack(fill=tk.X, pady=2)
        algo_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)
        
        # Parameter controls frame
        self.params_frame = ttk.Frame(algo_section)
        self.params_frame.pack(fill=tk.X, pady=5)
        
        # Run button
        ttk.Button(algo_section, text="Run Algorithm", command=self.run_algorithm).pack(fill=tk.X, pady=2)
        
        # Results display
        self.results_text = tk.Text(algo_section, height=8, width=30)
        self.results_text.pack(fill=tk.X, pady=5)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(algo_section, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.setup_algorithm_parameters()
        
    def setup_algorithm_parameters(self):
        """Setup parameter controls for different algorithms"""
        # Clear existing parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        algorithm = self.algo_var.get()
        
        if algorithm == "kmeans":
            ttk.Label(self.params_frame, text="Number of clusters:").pack(anchor=tk.W)
            self.k_clusters = tk.IntVar(value=3)
            ttk.Spinbox(self.params_frame, from_=2, to=10, textvariable=self.k_clusters, width=10).pack(fill=tk.X)
            
        elif algorithm == "agglomerative":
            ttk.Label(self.params_frame, text="Number of clusters:").pack(anchor=tk.W)
            self.n_clusters = tk.IntVar(value=3)
            ttk.Spinbox(self.params_frame, from_=2, to=10, textvariable=self.n_clusters, width=10).pack(fill=tk.X)
            
            ttk.Label(self.params_frame, text="Linkage:").pack(anchor=tk.W)
            self.linkage = tk.StringVar(value="ward")
            ttk.Combobox(self.params_frame, textvariable=self.linkage, 
                        values=["ward", "complete", "average", "single"]).pack(fill=tk.X)
                        
        elif algorithm == "decision_tree":
            ttk.Label(self.params_frame, text="Max depth:").pack(anchor=tk.W)
            self.max_depth = tk.IntVar(value=3)
            ttk.Spinbox(self.params_frame, from_=1, to=10, textvariable=self.max_depth, width=10).pack(fill=tk.X)
            
        elif algorithm == "random_forest":
            ttk.Label(self.params_frame, text="Number of trees:").pack(anchor=tk.W)
            self.n_estimators = tk.IntVar(value=100)
            ttk.Spinbox(self.params_frame, from_=10, to=500, textvariable=self.n_estimators, width=10).pack(fill=tk.X)
            
        elif algorithm == "linear_regression":
            ttk.Label(self.params_frame, text="Features (comma-separated):").pack(anchor=tk.W)
            self.features = tk.StringVar(value="")
            ttk.Entry(self.params_frame, textvariable=self.features).pack(fill=tk.X)
            
    def setup_plot_area(self):
        """Setup the plotting area"""
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self.show_welcome_plot()
        
    def show_welcome_plot(self):
        """Show welcome message and instructions"""
        self.ax.clear()
        self.ax.text(0.5, 0.7, "Welcome to Data Mining Explorer!", 
                    ha='center', va='center', fontsize=20, fontweight='bold')
        self.ax.text(0.5, 0.5, "1. Load a dataset (CSV file or sample data)\n"
                               "2. Select an algorithm from the dropdown\n"
                               "3. Adjust parameters as needed\n"
                               "4. Click 'Run Algorithm' to see results", 
                    ha='center', va='center', fontsize=12)
        self.ax.text(0.5, 0.2, "Available algorithms:\n"
                               "• K-Means Clustering\n"
                               "• Agglomerative Clustering\n"
                               "• Decision Tree Classification\n"
                               "• Random Forest\n"
                               "• Linear Regression\n"
                               "• Data Exploration", 
                    ha='center', va='center', fontsize=10)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()
        
    def load_csv(self):
        """Load CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.update_data_info()
                self.log_message(f"Loaded CSV: {file_path}")
                self.log_message(f"Shape: {self.df.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
                
    def load_sample_data(self):
        """Load sample dataset"""
        sample_type = self.sample_var.get()
        
        try:
            if sample_type == "iris":
                from sklearn.datasets import load_iris
                data = load_iris()
                self.df = pd.DataFrame(data.data, columns=data.feature_names)
                self.df['species'] = data.target
                
            elif sample_type == "wine":
                from sklearn.datasets import load_wine
                data = load_wine()
                self.df = pd.DataFrame(data.data, columns=data.feature_names)
                self.df['wine_type'] = data.target
                
            elif sample_type == "diabetes":
                from sklearn.datasets import load_diabetes
                data = load_diabetes()
                self.df = pd.DataFrame(data.data, columns=data.feature_names)
                self.df['target'] = data.target
                
            elif sample_type == "sales":
                # Generate synthetic sales data
                np.random.seed(42)
                n_samples = 200
                self.df = pd.DataFrame({
                    'advertising_spend': np.random.exponential(1000, n_samples),
                    'social_media_followers': np.random.poisson(1000, n_samples),
                    'price': np.random.normal(50, 15, n_samples),
                    'sales': np.random.normal(1000, 300, n_samples)
                })
                
            elif sample_type == "custom":
                # Generate custom clustering data
                np.random.seed(42)
                n_samples = 300
                # Create 3 clusters
                cluster1 = np.random.normal([2, 2], 0.5, (n_samples//3, 2))
                cluster2 = np.random.normal([-2, -2], 0.5, (n_samples//3, 2))
                cluster3 = np.random.normal([2, -2], 0.5, (n_samples//3, 2))
                
                X = np.vstack([cluster1, cluster2, cluster3])
                self.df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
                self.df['cluster'] = [0] * (n_samples//3) + [1] * (n_samples//3) + [2] * (n_samples//3)
                
            self.update_data_info()
            self.log_message(f"Loaded sample dataset: {sample_type}")
            self.log_message(f"Shape: {self.df.shape}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample data: {str(e)}")
            
    def update_data_info(self):
        """Update data information display"""
        if self.df is not None:
            info_text = f"Rows: {self.df.shape[0]}\nColumns: {self.df.shape[1]}\n"
            info_text += f"Numeric: {len(self.df.select_dtypes(include=[np.number]).columns)}\n"
            info_text += f"Categorical: {len(self.df.select_dtypes(include=['object']).columns)}"
            self.data_info.config(text=info_text)
            
            # Update column lists
            self.numeric_columns = list(self.df.select_dtypes(include=[np.number]).columns)
            self.categorical_columns = list(self.df.select_dtypes(include=['object']).columns)
            
    def on_algorithm_change(self, event=None):
        """Handle algorithm selection change"""
        self.setup_algorithm_parameters()
        
    def log_message(self, message):
        """Add message to results log"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        
    def run_algorithm(self):
        """Run the selected algorithm"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
            
        algorithm = self.algo_var.get()
        self.log_message(f"\n=== Running {algorithm.upper()} ===")
        
        try:
            if algorithm == "kmeans":
                self.run_kmeans()
            elif algorithm == "agglomerative":
                self.run_agglomerative()
            elif algorithm == "decision_tree":
                self.run_decision_tree()
            elif algorithm == "random_forest":
                self.run_random_forest()
            elif algorithm == "linear_regression":
                self.run_linear_regression()
            elif algorithm == "data_exploration":
                self.run_data_exploration()
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Algorithm failed: {str(e)}")
            
    def run_kmeans(self):
        """Run K-Means clustering"""
        if len(self.numeric_columns) < 2:
            raise ValueError("Need at least 2 numeric columns for clustering")
            
        k = self.k_clusters.get()
        X = self.df[self.numeric_columns].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, clusters)
        inertia = kmeans.inertia_
        
        self.log_message(f"Number of clusters: {k}")
        self.log_message(f"Silhouette Score: {silhouette:.3f}")
        self.log_message(f"Inertia: {inertia:.2f}")
        
        # Plot results
        self.plot_clustering_results(X, clusters, kmeans.cluster_centers_, 
                                   "K-Means Clustering", scaler)
        
    def run_agglomerative(self):
        """Run Agglomerative clustering"""
        if len(self.numeric_columns) < 2:
            raise ValueError("Need at least 2 numeric columns for clustering")
            
        n_clusters = self.n_clusters.get()
        linkage = self.linkage.get()
        X = self.df[self.numeric_columns].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run Agglomerative clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        clusters = clustering.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, clusters)
        
        self.log_message(f"Number of clusters: {n_clusters}")
        self.log_message(f"Linkage: {linkage}")
        self.log_message(f"Silhouette Score: {silhouette:.3f}")
        
        # Plot results
        self.plot_clustering_results(X, clusters, None, 
                                   f"Agglomerative Clustering ({linkage})", scaler)
        
    def run_decision_tree(self):
        """Run Decision Tree classification"""
        if len(self.categorical_columns) == 0:
            raise ValueError("Need at least one categorical column for classification")
            
        max_depth = self.max_depth.get()
        X = self.df[self.numeric_columns].dropna()
        y_col = self.categorical_columns[0]
        y = self.df[y_col].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                          test_size=0.2, random_state=42)
        
        # Train model
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(X_train, y_train)
        
        # Predictions
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.log_message(f"Target variable: {y_col}")
        self.log_message(f"Max depth: {max_depth}")
        self.log_message(f"Test accuracy: {accuracy:.3f}")
        
        # Plot decision tree
        self.plot_decision_tree(dt, X.columns, le.classes_)
        
    def run_random_forest(self):
        """Run Random Forest classification"""
        if len(self.categorical_columns) == 0:
            raise ValueError("Need at least one categorical column for classification")
            
        n_estimators = self.n_estimators.get()
        X = self.df[self.numeric_columns].dropna()
        y_col = self.categorical_columns[0]
        y = self.df[y_col].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                          test_size=0.2, random_state=42)
        
        # Train model
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.log_message(f"Target variable: {y_col}")
        self.log_message(f"Number of trees: {n_estimators}")
        self.log_message(f"Test accuracy: {accuracy:.3f}")
        
        # Plot feature importance
        self.plot_feature_importance(rf, X.columns)
        
    def run_linear_regression(self):
        """Run Linear Regression"""
        if len(self.numeric_columns) < 2:
            raise ValueError("Need at least 2 numeric columns for regression")
            
        # Get feature columns
        features_text = self.features.get().strip()
        if features_text:
            feature_cols = [col.strip() for col in features_text.split(',')]
            feature_cols = [col for col in feature_cols if col in self.numeric_columns]
        else:
            feature_cols = self.numeric_columns[:-1]  # All except last
            
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns specified")
            
        # Target is the last numeric column
        target_col = self.numeric_columns[-1]
        
        X = self.df[feature_cols].dropna()
        y = self.df[target_col].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                          test_size=0.2, random_state=42)
        
        # Train model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Predictions
        y_pred = lr.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        self.log_message(f"Target variable: {target_col}")
        self.log_message(f"Features: {', '.join(feature_cols)}")
        self.log_message(f"R² Score: {r2:.3f}")
        self.log_message(f"Intercept: {lr.intercept_:.3f}")
        
        # Plot regression results
        self.plot_regression_results(X_test, y_test, y_pred, feature_cols[0], target_col)
        
    def run_data_exploration(self):
        """Run data exploration and visualization"""
        self.log_message("Performing data exploration...")
        
        # Create subplots
        self.fig.clear()
        
        n_cols = min(3, len(self.numeric_columns))
        n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
        
        for i, col in enumerate(self.numeric_columns[:6]):  # Limit to 6 plots
            ax = self.fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Histogram
            ax.hist(self.df[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Log statistics
        self.log_message("Dataset Statistics:")
        for col in self.numeric_columns:
            stats = self.df[col].describe()
            self.log_message(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
            
    def plot_clustering_results(self, X, clusters, centroids, title, scaler):
        """Plot clustering results"""
        self.ax.clear()
        
        # Use first two features for 2D visualization
        if len(X.columns) >= 2:
            x_col, y_col = X.columns[0], X.columns[1]
            x_data, y_data = X[x_col], X[y_col]
            
            # Plot clusters
            scatter = self.ax.scatter(x_data, y_data, c=clusters, cmap='viridis', alpha=0.7)
            
            # Plot centroids if available
            if centroids is not None:
                centroids_original = scaler.inverse_transform(centroids)
                self.ax.scatter(centroids_original[:, 0], centroids_original[:, 1], 
                              c='red', marker='x', s=200, linewidths=3, label='Centroids')
                
            self.ax.set_xlabel(x_col)
            self.ax.set_ylabel(y_col)
            self.ax.set_title(title)
            self.ax.legend()
            
            # Add colorbar
            plt.colorbar(scatter, ax=self.ax, label='Cluster')
            
        else:
            # 1D visualization
            self.ax.hist(X.iloc[:, 0], bins=20, alpha=0.7)
            self.ax.set_xlabel(X.columns[0])
            self.ax.set_ylabel('Frequency')
            self.ax.set_title(title)
            
        self.canvas.draw()
        
    def plot_decision_tree(self, dt, feature_names, class_names):
        """Plot decision tree"""
        self.fig.clear()
        
        # Create a new figure for the tree
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(dt, feature_names=feature_names, class_names=class_names, 
                 filled=True, rounded=True, ax=ax)
        ax.set_title("Decision Tree Visualization")
        
        # Replace the main figure
        self.fig.clear()
        self.fig.add_subplot(111)
        
        # Copy the tree plot to main figure
        plot_tree(dt, feature_names=feature_names, class_names=class_names, 
                 filled=True, rounded=True, ax=self.fig.gca())
        self.fig.gca().set_title("Decision Tree Visualization")
        
        self.canvas.draw()
        
    def plot_feature_importance(self, rf, feature_names):
        """Plot feature importance"""
        self.ax.clear()
        
        importance = rf.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Plot top 10 features
        top_n = min(10, len(feature_names))
        self.ax.barh(range(top_n), importance[indices][:top_n])
        self.ax.set_yticks(range(top_n))
        self.ax.set_yticklabels([feature_names[i] for i in indices[:top_n]])
        self.ax.set_xlabel('Feature Importance')
        self.ax.set_title('Random Forest Feature Importance')
        
        self.canvas.draw()
        
    def plot_regression_results(self, X_test, y_test, y_pred, feature_col, target_col):
        """Plot regression results"""
        self.ax.clear()
        
        # Scatter plot of actual vs predicted
        self.ax.scatter(y_test, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        self.ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        self.ax.set_xlabel(f'Actual {target_col}')
        self.ax.set_ylabel(f'Predicted {target_col}')
        self.ax.set_title('Linear Regression Results')
        self.ax.legend()
        
        self.canvas.draw()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = DataMiningApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
