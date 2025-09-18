# 👩‍🔬 Breast Cancer Diagnosis with Logistic Regression

This project applies **Logistic Regression** to classify breast tumors as **malignant (cancerous)** or **benign (non-cancerous)** using the **Breast Cancer Wisconsin Diagnosis** dataset. This is a compact, custom-built machine learning model designed for a binary classification task.

---

### 🚀 Key Features

- **Data Preprocessing**: Handles data cleaning, including feature normalization and converting categorical labels into a binary format.
- **Custom Model Implementation**: A logistic regression model built from scratch using NumPy, demonstrating the core principles of the algorithm.
- **Forward & Backward Propagation**: The model includes functions for calculating predictions (forward propagation) and updating weights/bias based on the cost gradient (backward propagation).
- **High Accuracy**: The model achieves high accuracy, proving the effectiveness of logistic regression for this medical prediction task.

---

### 🛠️ How It Works

1.  **Load & Prepare Data**: The dataset is loaded, cleaned (irrelevant columns are dropped), and the 'diagnosis' column is mapped to `0` for benign and `1` for malignant.
2.  **Normalize Features**: Features are scaled to a range between 0 and 1 to ensure a consistent contribution to the model's training.
3.  **Split Data**: The dataset is divided into training (85%) and testing (15%) sets to evaluate the model's performance on unseen data.
4.  **Train Model**: The custom logistic regression model is trained using **Gradient Descent** to minimize the prediction error. The sigmoid function is at the core of this process, turning the model's output into a probability.
5.  **Evaluate**: The model's accuracy is calculated on both the training and test sets. Similar accuracy scores indicate that the model generalizes well and is not overfitting.

---

### 📂 Repository Contents

- `data.csv`: The Breast Cancer Wisconsin Diagnosis dataset.
- `main.py` or `notebook.ipynb`: The Python code implementing the logistic regression model.
- `README.md`: This file, providing an overview of the project.

---

### ⚙️ Dependencies

To run this project, ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
