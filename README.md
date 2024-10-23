
# Quality Prediction in Mining Process

#### Overview:
This project focuses on predicting impurity levels in the mining process, particularly targeting the percentage of silica in iron ore concentrate. Using machine learning (ML) techniques, the project aims to optimize the flotation process, ensuring higher quality and reducing environmental impacts. The dataset used is from a real-world mining plant, containing process variables and key quality measures.

#### Objectives:
- Predict impurity levels in ore concentrate.
- Enable proactive decision-making and process optimization.
- Contribute to the advancement of sustainability in mining by minimizing impurities.
  
#### Dataset:
- **Source:** Real-world flotation plant data (March 2017 - September 2017).
- **Size:** 24 columns and 736,282 rows.
- **Key Variables:** Percentage of silica concentrate, iron concentrate, airflows, and flotation column levels.

#### Dependencies:
- Python 3.x
- Jupyter Notebook or Google Colab
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`

#### Instructions to Run the Code:
1. **Setup Environment:**
   - Ensure Python is installed on your system (Windows/macOS/Linux).
   - Install and open Jupyter Notebook or Google Colab.
   
2. **Download the Dataset:**
   - The dataset is provided via a link in the project documentation.

3. **Clone or Download the Notebook:**
   - Download the Jupyter Notebook file or access the Colab notebook using the provided [Colab link](https://colab.research.google.com/drive/1btbcHc7bHtpc-iTGatQdl6Nqm9UCB1u5?usp=sharing).

4. **Install Required Libraries:**
   Run the following command to install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```

5. **Upload the Dataset:**
   - In Jupyter/Colab, navigate to the folder where the dataset is saved.
   - Load the dataset in the notebook for analysis.

6. **Run the Notebook:**
   - Execute the cells in the notebook to perform data preprocessing, feature selection, model training, and evaluation.

7. **Model Selection:**
   - Four models are used: Linear Regression, Lasso, Ridge, and XGBoost.
   - The best hyperparameters are selected via GridSearchCV.

#### Key Outputs:
- **Correlation Analysis:** Shows correlations between silica concentrate and other variables like iron concentrate and air flows.
- **Outliers Removal:** A threshold of 95% upper and 5% lower limits were set to remove outliers.
- **Data Scaling:** StandardScaler is applied to normalize the features.
- **Model Performance:** The performance is measured using MSE, MAE, R-squared, and RMSE.
- **Hyperparameter Tuning:** Best parameters are identified and applied to improve model accuracy.

#### Results:
- The XGBoost model yielded the best accuracy (69.23%).
- Data-driven models help in optimizing the mining process, improving product quality, and reducing environmental impact.

#### Conclusions:
- **Optimized Mining Process:** By implementing predictive models, mining plants can reduce impurities and enhance operational efficiency.
- **Environmental Impact:** Proactive impurity control contributes to more sustainable practices and reduces the mining industry's environmental footprint.
- **Sustainability:** Data-driven strategies allow for better resource management and align with long-term sustainability goals.

#### References:
- [1] Multi-Target Regression for Quality Prediction in a Mining Process.
- [2] Linear Regression - [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [3] Ridge Regression - [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [4] XGBoost - [XGBoost Docs](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)

