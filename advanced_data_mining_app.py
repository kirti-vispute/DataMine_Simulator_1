#!/usr/bin/env python3
"""
Advanced Data Mining & Machine Learning Algorithm Demonstrator
Enhanced version with more algorithms, better visualizations, and interactive features.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

class AdvancedDataMiningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Data Mining Algorithm Explorer")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.current_algorithm = None
        
        # Results storage
        self.results = {}
        self.history = []
        
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
        self.setup_status_bar()
        
    def setup_data_controls(self):
        """Setup data loading and preview controls"""
        # Data loading section
        data_section = ttk.LabelFrame(self.control_frame, text="Data Management", padding=5)
        data_section.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        file_frame = ttk.Frame(data_section)
        file_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(file_frame, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(file_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Sample data selection
        sample_frame = ttk.Frame(data_section)
        sample_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(sample_frame, text="Sample Data:").pack(anchor=tk.W)
        self.sample_var = tk.StringVar(value="iris")
        sample_combo = ttk.Combobox(sample_frame, textvariable=self.sample_var, 
                                  values=["iris", "wine", "diabetes", "breast_cancer", 
                                         "sales", "custom_clusters", "synthetic_regression"])
        sample_combo.pack(fill=tk.X, pady=2)
        
        ttk.Button(sample_frame, text="Load Sample", command=self.load_sample_data).pack(fill=tk.X, pady=2)
        
        # Data info
        self.data_info = ttk.Label(data_section, text="No data loaded", wraplength=200)
        self.data_info.pack(pady=5)
        
        # Quick stats
        self.stats_frame = ttk.Frame(data_section)
        self.stats_frame.pack(fill=tk.X, pady=5)
        
    def setup_algorithm_controls(self):
        """Setup algorithm selection and parameters"""
        algo_section = ttk.LabelFrame(self.control_frame, text="Algorithm Selection", padding=5)
        algo_section.pack(fill=tk.X, pady=(0, 10))
        
        # Algorithm selection
        self.algo_var = tk.StringVar(value="kmeans")
        algo_combo = ttk.Combobox(algo_section, textvariable=self.algo_var, 
                                values=["kmeans", "agglomerative", "dbscan", "decision_tree", 
                                       "random_forest", "gradient_boosting", "svm", "knn",
                                       "linear_regression", "ridge_regression", "lasso_regression",
                                       "pca", "data_exploration", "correlation_analysis"])
        algo_combo.pack(fill=tk.X, pady=2)
        algo_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)
        
        # Parameter controls frame
        self.params_frame = ttk.Frame(algo_section)
        self.params_frame.pack(fill=tk.X, pady=5)
        
        # Advanced options
        self.advanced_frame = ttk.LabelFrame(algo_section, text="Advanced Options", padding=5)
        self.advanced_frame.pack(fill=tk.X, pady=5)
        
        # Cross-validation
        self.cv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.advanced_frame, text="Cross-validation", variable=self.cv_var).pack(anchor=tk.W)
        
        # Random state
        ttk.Label(self.advanced_frame, text="Random State:").pack(anchor=tk.W)
        self.random_state = tk.IntVar(value=42)
        ttk.Spinbox(self.advanced_frame, from_=0, to=1000, textvariable=self.random_state, width=10).pack(fill=tk.X)
        
        # Run button
        run_frame = ttk.Frame(algo_section)
        run_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(run_frame, text="Run Algorithm", command=self.run_algorithm).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(run_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Results display
        results_frame = ttk.LabelFrame(self.control_frame, text="Results & Log", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.results_text = tk.Text(results_frame, height=12, width=35, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.setup_algorithm_parameters()
        
    def setup_algorithm_parameters(self):
        """Setup parameter controls for different algorithms"""
        # Clear existing parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        algorithm = self.algo_var.get()
        
        if algorithm in ["kmeans", "agglomerative"]:
            ttk.Label(self.params_frame, text="Number of clusters:").pack(anchor=tk.W)
            self.k_clusters = tk.IntVar(value=3)
            ttk.Spinbox(self.params_frame, from_=2, to=20, textvariable=self.k_clusters, width=10).pack(fill=tk.X)
            
            if algorithm == "agglomerative":
                ttk.Label(self.params_frame, text="Linkage:").pack(anchor=tk.W)
                self.linkage = tk.StringVar(value="ward")
                ttk.Combobox(self.params_frame, textvariable=self.linkage, 
                            values=["ward", "complete", "average", "single"]).pack(fill=tk.X)
                            
        elif algorithm == "dbscan":
            ttk.Label(self.params_frame, text="Epsilon (eps):").pack(anchor=tk.W)
            self.eps = tk.DoubleVar(value=0.5)
            ttk.Spinbox(self.params_frame, from_=0.1, to=5.0, increment=0.1, 
                       textvariable=self.eps, width=10).pack(fill=tk.X)
            
            ttk.Label(self.params_frame, text="Min samples:").pack(anchor=tk.W)
            self.min_samples = tk.IntVar(value=5)
            ttk.Spinbox(self.params_frame, from_=2, to=50, textvariable=self.min_samples, width=10).pack(fill=tk.X)
            
        elif algorithm in ["decision_tree", "random_forest", "gradient_boosting"]:
            ttk.Label(self.params_frame, text="Max depth:").pack(anchor=tk.W)
            self.max_depth = tk.IntVar(value=3)
            ttk.Spinbox(self.params_frame, from_=1, to=20, textvariable=self.max_depth, width=10).pack(fill=tk.X)
            
            if algorithm in ["random_forest", "gradient_boosting"]:
                ttk.Label(self.params_frame, text="Number of estimators:").pack(anchor=tk.W)
                self.n_estimators = tk.IntVar(value=100)
                ttk.Spinbox(self.params_frame, from_=10, to=1000, textvariable=self.n_estimators, width=10).pack(fill=tk.X)
                
        elif algorithm == "svm":
            ttk.Label(self.params_frame, text="C (regularization):").pack(anchor=tk.W)
            self.C = tk.DoubleVar(value=1.0)
            ttk.Spinbox(self.params_frame, from_=0.1, to=100.0, increment=0.1, 
                       textvariable=self.C, width=10).pack(fill=tk.X)
            
            ttk.Label(self.params_frame, text="Kernel:").pack(anchor=tk.W)
            self.kernel = tk.StringVar(value="rbf")
            ttk.Combobox(self.params_frame, textvariable=self.kernel, 
                        values=["linear", "poly", "rbf", "sigmoid"]).pack(fill=tk.X)
                        
        elif algorithm == "knn":
            ttk.Label(self.params_frame, text="Number of neighbors:").pack(anchor=tk.W)
            self.n_neighbors = tk.IntVar(value=5)
            ttk.Spinbox(self.params_frame, from_=1, to=50, textvariable=self.n_neighbors, width=10).pack(fill=tk.X)
            
        elif algorithm in ["linear_regression", "ridge_regression", "lasso_regression"]:
            ttk.Label(self.params_frame, text="Features (comma-separated):").pack(anchor=tk.W)
            self.features = tk.StringVar(value="")
            ttk.Entry(self.params_frame, textvariable=self.features).pack(fill=tk.X)
            
            if algorithm in ["ridge_regression", "lasso_regression"]:
                ttk.Label(self.params_frame, text="Alpha:").pack(anchor=tk.W)
                self.alpha = tk.DoubleVar(value=1.0)
                ttk.Spinbox(self.params_frame, from_=0.01, to=10.0, increment=0.1, 
                           textvariable=self.alpha, width=10).pack(fill=tk.X)
                           
        elif algorithm == "pca":
            ttk.Label(self.params_frame, text="Number of components:").pack(anchor=tk.W)
            self.n_components = tk.IntVar(value=2)
            ttk.Spinbox(self.params_frame, from_=2, to=10, textvariable=self.n_components, width=10).pack(fill=tk.X)
            
    def setup_plot_area(self):
        """Setup the plotting area"""
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Initial plot
        self.show_welcome_plot()
        
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def show_welcome_plot(self):
        """Show welcome message and instructions"""
        self.ax.clear()
        self.ax.text(0.5, 0.7, "Advanced Data Mining Explorer", 
                    ha='center', va='center', fontsize=24, fontweight='bold')
        self.ax.text(0.5, 0.55, "Choose from 14+ algorithms:\n"
                               "• Clustering: K-Means, Agglomerative, DBSCAN\n"
                               "• Classification: Decision Tree, Random Forest, SVM, KNN\n"
                               "• Regression: Linear, Ridge, Lasso\n"
                               "• Dimensionality: PCA\n"
                               "• Analysis: Data Exploration, Correlation", 
                    ha='center', va='center', fontsize=12)
        self.ax.text(0.5, 0.25, "Instructions:\n"
                               "1. Load a dataset (CSV or sample)\n"
                               "2. Select an algorithm\n"
                               "3. Adjust parameters\n"
                               "4. Run and visualize results", 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
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
                self.status_var.set("Loading CSV file...")
                self.df = pd.read_csv(file_path)
                self.update_data_info()
                self.log_message(f"✓ Loaded CSV: {file_path}")
                self.log_message(f"✓ Shape: {self.df.shape}")
                self.status_var.set("CSV file loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
                self.status_var.set("Error loading CSV file")
                
    def load_sample_data(self):
        """Load sample dataset"""
        sample_type = self.sample_var.get()
        
        try:
            self.status_var.set(f"Loading {sample_type} dataset...")
            
            if sample_type == "iris":
                from sklearn.datasets import load_iris
                data = load_iris()
                self.df = pd.DataFrame(data.data, columns=data.feature_names)
                self.df['species'] = [data.target_names[i] for i in data.target]
                
            elif sample_type == "wine":
                from sklearn.datasets import load_wine
                data = load_wine()
                self.df = pd.DataFrame(data.data, columns=data.feature_names)
                self.df['wine_type'] = [data.target_names[i] for i in data.target]
                
            elif sample_type == "diabetes":
                from sklearn.datasets import load_diabetes
                data = load_diabetes()
                self.df = pd.DataFrame(data.data, columns=data.feature_names)
                self.df['target'] = data.target
                
            elif sample_type == "breast_cancer":
                from sklearn.datasets import load_breast_cancer
                data = load_breast_cancer()
                self.df = pd.DataFrame(data.data, columns=data.feature_names)
                self.df['diagnosis'] = [data.target_names[i] for i in data.target]
                
            elif sample_type == "sales":
                # Generate synthetic sales data
                np.random.seed(42)
                n_samples = 500
                self.df = pd.DataFrame({
                    'advertising_spend': np.random.exponential(1000, n_samples),
                    'social_media_followers': np.random.poisson(1000, n_samples),
                    'price': np.random.normal(50, 15, n_samples),
                    'seasonality': np.random.normal(0, 1, n_samples),
                    'sales': np.random.normal(1000, 300, n_samples)
                })
                
            elif sample_type == "custom_clusters":
                # Generate custom clustering data with noise
                np.random.seed(42)
                n_samples = 400
                # Create 4 clusters with different shapes
                cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0.3], [0.3, 0.5]], n_samples//4)
                cluster2 = np.random.multivariate_normal([-2, -2], [[0.8, -0.2], [-0.2, 0.8]], n_samples//4)
                cluster3 = np.random.multivariate_normal([2, -2], [[0.3, 0], [0, 0.8]], n_samples//4)
                cluster4 = np.random.multivariate_normal([-2, 2], [[0.6, 0.4], [0.4, 0.6]], n_samples//4)
                
                X = np.vstack([cluster1, cluster2, cluster3, cluster4])
                self.df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
                self.df['cluster'] = [0] * (n_samples//4) + [1] * (n_samples//4) + [2] * (n_samples//4) + [3] * (n_samples//4)
                
            elif sample_type == "synthetic_regression":
                # Generate synthetic regression data
                np.random.seed(42)
                n_samples = 300
                self.df = pd.DataFrame({
                    'feature_1': np.random.normal(0, 1, n_samples),
                    'feature_2': np.random.normal(0, 1, n_samples),
                    'feature_3': np.random.normal(0, 1, n_samples),
                    'target': 0
                })
                # Create non-linear relationship
                self.df['target'] = (2 * self.df['feature_1'] + 
                                   3 * self.df['feature_2'] + 
                                   0.5 * self.df['feature_3']**2 + 
                                   np.random.normal(0, 0.5, n_samples))
                
            self.update_data_info()
            self.log_message(f"✓ Loaded sample dataset: {sample_type}")
            self.log_message(f"✓ Shape: {self.df.shape}")
            self.status_var.set(f"Loaded {sample_type} dataset successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample data: {str(e)}")
            self.status_var.set("Error loading sample data")
            
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
            
            # Update quick stats
            self.update_quick_stats()
            
    def update_quick_stats(self):
        """Update quick statistics display"""
        # Clear existing stats
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
            
        if self.df is not None and len(self.numeric_columns) > 0:
            ttk.Label(self.stats_frame, text="Quick Stats:", font=('TkDefaultFont', 8, 'bold')).pack(anchor=tk.W)
            
            for col in self.numeric_columns[:3]:  # Show first 3 columns
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                ttk.Label(self.stats_frame, 
                         text=f"{col}: μ={mean_val:.2f}, σ={std_val:.2f}", 
                         font=('TkDefaultFont', 7)).pack(anchor=tk.W)
        
    def on_algorithm_change(self, event=None):
        """Handle algorithm selection change"""
        self.setup_algorithm_parameters()
        
    def log_message(self, message):
        """Add message to results log"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        
    def clear_results(self):
        """Clear results and log"""
        self.results_text.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()
        self.status_var.set("Results cleared")
        
    def export_results(self):
        """Export results to file"""
        if not self.results_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No results to export!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
                
    def run_algorithm(self):
        """Run the selected algorithm"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
            
        algorithm = self.algo_var.get()
        self.log_message(f"\n{'='*50}")
        self.log_message(f"RUNNING {algorithm.upper().replace('_', ' ')}")
        self.log_message(f"{'='*50}")
        
        try:
            self.status_var.set(f"Running {algorithm}...")
            
            if algorithm in ["kmeans", "agglomerative", "dbscan"]:
                self.run_clustering(algorithm)
            elif algorithm in ["decision_tree", "random_forest", "gradient_boosting", "svm", "knn"]:
                self.run_classification(algorithm)
            elif algorithm in ["linear_regression", "ridge_regression", "lasso_regression"]:
                self.run_regression(algorithm)
            elif algorithm == "pca":
                self.run_pca()
            elif algorithm in ["data_exploration", "correlation_analysis"]:
                self.run_analysis(algorithm)
                
            self.status_var.set(f"{algorithm} completed successfully")
            
        except Exception as e:
            self.log_message(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Algorithm failed: {str(e)}")
            self.status_var.set(f"Error running {algorithm}")
            
    def run_clustering(self, algorithm):
        """Run clustering algorithms"""
        if len(self.numeric_columns) < 2:
            raise ValueError("Need at least 2 numeric columns for clustering")
            
        X = self.df[self.numeric_columns].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if algorithm == "kmeans":
            k = self.k_clusters.get()
            clustering = KMeans(n_clusters=k, random_state=self.random_state.get(), n_init=10)
            clusters = clustering.fit_predict(X_scaled)
            centroids = clustering.cluster_centers_
            
            # Calculate metrics
            silhouette = silhouette_score(X_scaled, clusters)
            inertia = clustering.inertia_
            
            self.log_message(f"✓ Number of clusters: {k}")
            self.log_message(f"✓ Silhouette Score: {silhouette:.3f}")
            self.log_message(f"✓ Inertia: {inertia:.2f}")
            
            self.plot_clustering_results(X, clusters, centroids, "K-Means Clustering", scaler)
            
        elif algorithm == "agglomerative":
            n_clusters = self.k_clusters.get()
            linkage = self.linkage.get()
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            clusters = clustering.fit_predict(X_scaled)
            
            silhouette = silhouette_score(X_scaled, clusters)
            
            self.log_message(f"✓ Number of clusters: {n_clusters}")
            self.log_message(f"✓ Linkage: {linkage}")
            self.log_message(f"✓ Silhouette Score: {silhouette:.3f}")
            
            self.plot_clustering_results(X, clusters, None, 
                                       f"Agglomerative Clustering ({linkage})", scaler)
            
        elif algorithm == "dbscan":
            eps = self.eps.get()
            min_samples = self.min_samples.get()
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = clustering.fit_predict(X_scaled)
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            self.log_message(f"✓ Epsilon: {eps}")
            self.log_message(f"✓ Min samples: {min_samples}")
            self.log_message(f"✓ Number of clusters: {n_clusters}")
            self.log_message(f"✓ Number of noise points: {n_noise}")
            
            if n_clusters > 1:
                silhouette = silhouette_score(X_scaled, clusters)
                self.log_message(f"✓ Silhouette Score: {silhouette:.3f}")
            
            self.plot_clustering_results(X, clusters, None, "DBSCAN Clustering", scaler)
            
    def run_classification(self, algorithm):
        """Run classification algorithms"""
        if len(self.categorical_columns) == 0:
            raise ValueError("Need at least one categorical column for classification")
            
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
                                                          test_size=0.2, random_state=self.random_state.get())
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if algorithm == "decision_tree":
            max_depth = self.max_depth.get()
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_state.get())
            model.fit(X_train, y_train)
            
        elif algorithm == "random_forest":
            max_depth = self.max_depth.get()
            n_estimators = self.n_estimators.get()
            model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, 
                                         random_state=self.random_state.get())
            model.fit(X_train, y_train)
            
        elif algorithm == "gradient_boosting":
            max_depth = self.max_depth.get()
            n_estimators = self.n_estimators.get()
            model = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, 
                                             random_state=self.random_state.get())
            model.fit(X_train, y_train)
            
        elif algorithm == "svm":
            C = self.C.get()
            kernel = self.kernel.get()
            model = SVC(C=C, kernel=kernel, random_state=self.random_state.get())
            model.fit(X_train_scaled, y_train)
            
        elif algorithm == "knn":
            n_neighbors = self.n_neighbors.get()
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(X_train_scaled, y_train)
            
        # Predictions
        if algorithm in ["svm", "knn"]:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
            
        accuracy = accuracy_score(y_test, y_pred)
        
        self.log_message(f"✓ Target variable: {y_col}")
        self.log_message(f"✓ Test accuracy: {accuracy:.3f}")
        
        # Cross-validation if enabled
        if self.cv_var.get():
            if algorithm in ["svm", "knn"]:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            self.log_message(f"✓ Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
        
        # Plot results
        if algorithm == "decision_tree":
            self.plot_decision_tree(model, X.columns, le.classes_)
        elif algorithm in ["random_forest", "gradient_boosting"]:
            self.plot_feature_importance(model, X.columns)
        else:
            self.plot_classification_results(y_test, y_pred, le.classes_, algorithm)
            
    def run_regression(self, algorithm):
        """Run regression algorithms"""
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
                                                          test_size=0.2, random_state=self.random_state.get())
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if algorithm == "linear_regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            
        elif algorithm == "ridge_regression":
            alpha = self.alpha.get()
            model = Ridge(alpha=alpha, random_state=self.random_state.get())
            model.fit(X_train_scaled, y_train)
            
        elif algorithm == "lasso_regression":
            alpha = self.alpha.get()
            model = Lasso(alpha=alpha, random_state=self.random_state.get())
            model.fit(X_train_scaled, y_train)
            
        # Predictions
        if algorithm == "linear_regression":
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)
            
        r2 = r2_score(y_test, y_pred)
        
        self.log_message(f"✓ Target variable: {target_col}")
        self.log_message(f"✓ Features: {', '.join(feature_cols)}")
        self.log_message(f"✓ R² Score: {r2:.3f}")
        
        if hasattr(model, 'intercept_'):
            self.log_message(f"✓ Intercept: {model.intercept_:.3f}")
            
        # Cross-validation if enabled
        if self.cv_var.get():
            if algorithm == "linear_regression":
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            self.log_message(f"✓ Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
        
        # Plot regression results
        self.plot_regression_results(X_test, y_test, y_pred, feature_cols[0], target_col, algorithm)
        
    def run_pca(self):
        """Run Principal Component Analysis"""
        if len(self.numeric_columns) < 2:
            raise ValueError("Need at least 2 numeric columns for PCA")
            
        n_components = self.n_components.get()
        X = self.df[self.numeric_columns].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        self.log_message(f"✓ Number of components: {n_components}")
        self.log_message(f"✓ Explained variance ratio: {explained_variance}")
        self.log_message(f"✓ Cumulative variance: {cumulative_variance}")
        
        # Plot PCA results
        self.plot_pca_results(X_pca, explained_variance, X.columns)
        
    def run_analysis(self, analysis_type):
        """Run data analysis"""
        if analysis_type == "data_exploration":
            self.run_data_exploration()
        elif analysis_type == "correlation_analysis":
            self.run_correlation_analysis()
            
    def run_data_exploration(self):
        """Run comprehensive data exploration"""
        self.log_message("Performing comprehensive data exploration...")
        
        # Create subplots
        self.fig.clear()
        
        n_numeric = len(self.numeric_columns)
        if n_numeric == 0:
            self.log_message("❌ No numeric columns found for exploration")
            return
            
        # Create a grid of subplots
        n_cols = min(3, n_numeric)
        n_rows = (n_numeric + n_cols - 1) // n_cols
        
        for i, col in enumerate(self.numeric_columns[:6]):  # Limit to 6 plots
            ax = self.fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Histogram with KDE
            data = self.df[col].dropna()
            ax.hist(data, bins=20, alpha=0.7, density=True, edgecolor='black')
            
            # Add KDE curve
            try:
                from scipy import stats
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                ax.legend()
            except ImportError:
                pass
                
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Log comprehensive statistics
        self.log_message("\n📊 Dataset Statistics:")
        self.log_message(f"✓ Total rows: {len(self.df)}")
        self.log_message(f"✓ Total columns: {len(self.df.columns)}")
        self.log_message(f"✓ Numeric columns: {len(self.numeric_columns)}")
        self.log_message(f"✓ Categorical columns: {len(self.categorical_columns)}")
        
        self.log_message("\n📈 Numeric Column Statistics:")
        for col in self.numeric_columns:
            stats = self.df[col].describe()
            self.log_message(f"  {col}:")
            self.log_message(f"    Mean: {stats['mean']:.2f}")
            self.log_message(f"    Std: {stats['std']:.2f}")
            self.log_message(f"    Min: {stats['min']:.2f}")
            self.log_message(f"    Max: {stats['max']:.2f}")
            
    def run_correlation_analysis(self):
        """Run correlation analysis"""
        if len(self.numeric_columns) < 2:
            self.log_message("❌ Need at least 2 numeric columns for correlation analysis")
            return
            
        self.log_message("Performing correlation analysis...")
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_columns].corr()
        
        # Plot correlation heatmap
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        self.fig.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        # Set labels
        ax.set_xticks(range(len(self.numeric_columns)))
        ax.set_yticks(range(len(self.numeric_columns)))
        ax.set_xticklabels(self.numeric_columns, rotation=45, ha='right')
        ax.set_yticklabels(self.numeric_columns)
        
        # Add correlation values as text
        for i in range(len(self.numeric_columns)):
            for j in range(len(self.numeric_columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Correlation Matrix Heatmap')
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Log correlation insights
        self.log_message("\n🔗 Correlation Insights:")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                corr_pairs.append((abs(corr_val), corr_val, 
                                 corr_matrix.columns[i], corr_matrix.columns[j]))
        
        corr_pairs.sort(reverse=True)
        
        self.log_message("Strongest correlations:")
        for abs_corr, corr, col1, col2 in corr_pairs[:5]:
            strength = "Strong" if abs_corr > 0.7 else "Moderate" if abs_corr > 0.3 else "Weak"
            direction = "positive" if corr > 0 else "negative"
            self.log_message(f"  {col1} ↔ {col2}: {corr:.3f} ({strength} {direction})")
            
    # Plotting methods (similar to basic version but enhanced)
    def plot_clustering_results(self, X, clusters, centroids, title, scaler):
        """Plot clustering results with enhanced visualization"""
        self.fig.clear()
        
        # Use first two features for 2D visualization
        if len(X.columns) >= 2:
            x_col, y_col = X.columns[0], X.columns[1]
            x_data, y_data = X[x_col], X[y_col]
            
            ax = self.fig.add_subplot(111)
            
            # Plot clusters with different colors
            unique_clusters = np.unique(clusters)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster in enumerate(unique_clusters):
                if cluster == -1:  # Noise points for DBSCAN
                    mask = clusters == cluster
                    ax.scatter(x_data[mask], y_data[mask], c='black', marker='x', 
                             s=50, alpha=0.6, label='Noise')
                else:
                    mask = clusters == cluster
                    ax.scatter(x_data[mask], y_data[mask], c=[colors[i]], 
                             s=60, alpha=0.7, label=f'Cluster {cluster}')
            
            # Plot centroids if available
            if centroids is not None:
                centroids_original = scaler.inverse_transform(centroids)
                ax.scatter(centroids_original[:, 0], centroids_original[:, 1], 
                          c='red', marker='X', s=200, linewidths=3, 
                          label='Centroids', edgecolors='black')
                
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # 1D visualization
            ax = self.fig.add_subplot(111)
            ax.hist(X.iloc[:, 0], bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(X.columns[0], fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_decision_tree(self, dt, feature_names, class_names):
        """Plot decision tree with enhanced styling"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        plot_tree(dt, feature_names=feature_names, class_names=class_names, 
                 filled=True, rounded=True, ax=ax, fontsize=8)
        ax.set_title("Decision Tree Visualization", fontsize=14, fontweight='bold')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance with enhanced styling"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Plot top 10 features
        top_n = min(10, len(feature_names))
        y_pos = np.arange(top_n)
        
        bars = ax.barh(y_pos, importance[indices][:top_n], 
                      color=plt.cm.viridis(np.linspace(0, 1, top_n)))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices[:top_n]], fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, importance[indices][:top_n])):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_regression_results(self, X_test, y_test, y_pred, feature_col, target_col, algorithm):
        """Plot regression results with enhanced visualization"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Scatter plot of actual vs predicted
        ax.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {target_col}', fontsize=12)
        ax.set_ylabel(f'Predicted {target_col}', fontsize=12)
        ax.set_title(f'{algorithm.replace("_", " ").title()} Results', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_classification_results(self, y_test, y_pred, class_names, algorithm):
        """Plot classification results"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Confusion matrix-like visualization
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        class_counts = {cls: (y_test == cls).sum() for cls in unique_classes}
        
        bars = ax.bar(range(len(unique_classes)), [class_counts[cls] for cls in unique_classes],
                     color=plt.cm.Set3(np.linspace(0, 1, len(unique_classes))))
        
        ax.set_xticks(range(len(unique_classes)))
        ax.set_xticklabels([class_names[cls] for cls in unique_classes], rotation=45)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{algorithm.replace("_", " ").title()} - Class Distribution', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_pca_results(self, X_pca, explained_variance, feature_names):
        """Plot PCA results"""
        self.fig.clear()
        
        # Create subplots
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        # Plot PCA scatter
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)', fontsize=10)
        ax1.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)', fontsize=10)
        ax1.set_title('PCA Scatter Plot', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot explained variance
        ax2.bar(range(len(explained_variance)), explained_variance, 
               color=plt.cm.viridis(np.linspace(0, 1, len(explained_variance))))
        ax2.set_xlabel('Principal Component', fontsize=10)
        ax2.set_ylabel('Explained Variance Ratio', fontsize=10)
        ax2.set_title('Explained Variance by Component', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, val in enumerate(explained_variance):
            ax2.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = AdvancedDataMiningApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
