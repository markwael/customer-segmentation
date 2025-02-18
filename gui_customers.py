import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import os

# Main Application Class
class CustomerSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Segmentation Tool")

        # Dataset Placeholder
        self.data = None

        # GUI Layout
        self.create_widgets()

    def create_widgets(self):
        # File Upload Button
        self.upload_btn = tk.Button(self.root, text="Upload Dataset", command=self.upload_dataset)
        self.upload_btn.pack(pady=10)

        # Data Description Button
        self.describe_btn = tk.Button(self.root, text="Show Dataset Description", command=self.show_description, state=tk.DISABLED)
        self.describe_btn.pack(pady=10)

        # Heatmap Button
        self.heatmap_btn = tk.Button(self.root, text="Generate Heatmap", command=self.generate_heatmap, state=tk.DISABLED)
        self.heatmap_btn.pack(pady=10)

        # Feature Evaluation Button
        self.feature_eval_btn = tk.Button(self.root, text="Evaluate Features", command=self.evaluate_features, state=tk.DISABLED)
        self.feature_eval_btn.pack(pady=10)

        # PCA Button
        self.pca_btn = tk.Button(self.root, text="Perform PCA", command=self.perform_pca, state=tk.DISABLED)
        self.pca_btn.pack(pady=10)

        # View Dataset Button
        self.view_data_btn = tk.Button(self.root, text="View Dataset", command=self.view_dataset, state=tk.DISABLED)
        self.view_data_btn.pack(pady=10)

    def upload_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.data = pd.read_csv(file_path)
            self.data.drop(["Region", "Channel"], axis=1, inplace=True, errors='ignore')
            messagebox.showinfo("Success", f"Dataset loaded with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
            
            # Enable buttons
            self.describe_btn.config(state=tk.NORMAL)
            self.heatmap_btn.config(state=tk.NORMAL)
            self.feature_eval_btn.config(state=tk.NORMAL)
            self.pca_btn.config(state=tk.NORMAL)
            self.view_data_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def show_description(self):
        if self.data is not None:
            description = self.data.describe().to_string()
            self.show_output_window("Dataset Description", description)

    def generate_heatmap(self):
        if self.data is not None:
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                plt.title("Correlation Heatmap")
                plt.show()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate heatmap: {e}")

    def evaluate_features(self):
        if self.data is not None:
            output = []
            features = self.data.columns

            try:
                for feature in features:
                    new_data = self.data.drop(columns=[feature])
                    X_train, X_test, y_train, y_test = train_test_split(new_data, self.data[feature], test_size=0.25, random_state=42)
                    regressor = DecisionTreeRegressor(random_state=42)
                    regressor.fit(X_train, y_train)
                    score = r2_score(y_test, regressor.predict(X_test))
                    output.append(f"R^2 score for {feature}: {score:.2f}")

                self.show_output_window("Feature Evaluation", "\n".join(output))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to evaluate features: {e}")

    def perform_pca(self):
        if self.data is not None:
            try:
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(self.data)
                explained_variance = pca.explained_variance_ratio_

                # Plot PCA Results
                plt.figure(figsize=(8, 6))
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.5)
                plt.title("PCA Result: First 2 Principal Components")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.show()

                variance_info = f"Explained Variance Ratios: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}"
                self.show_output_window("PCA Results", variance_info)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to perform PCA: {e}")

    def view_dataset(self):
        if self.data is not None:
            try:
                data_preview = self.data.head(10).to_string()
                self.show_output_window("Dataset Preview (First 10 Rows)", data_preview)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to display dataset: {e}")

    def show_output_window(self, title, content):
        output_window = tk.Toplevel(self.root)
        output_window.title(title)
        text_widget = tk.Text(output_window, wrap=tk.WORD, width=80, height=20)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(padx=10, pady=10)

# Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = CustomerSegmentationApp(root)
    root.mainloop()
