# CSCI316: Real Estate Valuation using Ensemble Learning

This project was developed for the course *CSCI316 - Big Data Mining Techniques and Implementation* at NYU. We built a predictive analytics pipeline using PySpark and implemented two ensemble learning models—Bagging and Boosting—from scratch to predict real estate property values.

## 🔍 Problem Statement
Accurately predicting property prices is critical for real estate developers, investors, and urban planners. Traditional models fall short in capturing complex nonlinear market dynamics.

## 💡 Solution
We applied large-scale data preprocessing using PySpark and implemented custom ensemble models (Bagging and XGBoost-like Boosting). The models were evaluated using RMSE, R², and MAE with 10-fold cross-validation.

## 📁 Project Structure
- `notebooks/`: Jupyter notebook implementing both ensemble methods from scratch.
- `data/`: Cleaned dataset used for model training and evaluation.
- `report/`: Final project report and slide presentation explaining the entire workflow.

## 🛠️ Technologies
- PySpark
- Pandas, NumPy
- Scikit-learn (for decision trees)
- Jupyter Notebook

## 🧠 Key Learnings
- Built XGBoost from scratch using gradient calculations and tree-based residual training.
- Applied the IQR method for outlier detection.
- Learned trade-offs between Bagging (stability) and Boosting (accuracy).
- Applied 10-fold cross-validation to ensure robust model performance.

## 📈 Results
- Boosting achieved higher predictive accuracy (R² ≈ 1.00, RMSE ≈ 0.05).
- Bagging demonstrated better generalization and variance reduction.

## 📄 Report & Slides
See the full [Final Report](./report/CSCI316%20Final%20Report.pdf) and [Presentation](./report/CSCI316%20Project%201%20Presentation.pdf) for more details.

## 👥 Team Members
See the report for a full list of contributors.
