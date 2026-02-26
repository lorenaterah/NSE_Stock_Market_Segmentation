# Nairobi Securities Exchange (NSE) Market Intelligence System

## 🔗 Live Demo & Repository

- **GitHub Repository:** [![Python](https://img.shields.io/badge/Python-3.9-blue
)](https://github.com/lorenaterah/NSE_Stock_Market_Segmentation) 


- **Streamlit Dashboard:** [
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://nsestockmarketsegmentation.streamlit.app/)

## 📌 Project Overview
Despite the Nairobi Securities Exchange (NSE) seeing a significant surge in equity turnover, many retail investors in Kenya suffer from "herding behavior," relying on social intuition rather than technical data. This project bridges that information gap by converting raw historical stock price data into behavioral risk clusters using machine learning.

The system groups stocks into risk-based categories (e.g., Stable, Moderate, High-Volatility), allowing investors to move from intuition to evidence-based decision-making through an interactive intelligence dashboard.

## 🎯 Key Objectives
* **Feature Engineering:** Quantify stock behavior using metrics like Rolling Volatility, Daily Returns, and Maximum Drawdowns.
* **Behavioral Segmentation:** Apply unsupervised learning (K-Means, DBSCAN) to group 57+ major stocks into distinct risk profiles.
* **Sector Risk Analysis:** Identify systemic risks and stability patterns across different economic sectors.
* **Interactive Deployment:** Provide a Streamlit dashboard for users to explore stocks and assess risk profiles.

## 📂 Project Structure
The project is organized into sequential notebooks:

1.  **`01_Data_understanding.ipynb`** Initial exploration of the raw NSE dataset (2021-2024) and merging of multi-year records.
2.  **`02_Data_cleaning.ipynb`** Handling missing values and ensuring data consistency to produce the final `df_all` dataset.
3.  **`03_feature_engineering.ipynb`** Transforming raw prices into 26 distinct features, including RSI, Sharpe Ratios, and volatility metrics.
4.  **`04_clustering.ipynb`** Implementation of K-Means clustering to segment stocks into 4 behavioral risk labels.
5.  **`05_Conclusion and Recommendations.ipynb`** Final synthesis of insights and strategic advice for retail investors and stakeholders.

## 🛠️ Methodology & Features
### Data Features
The system utilizes 26 engineered features to profile stocks, including:
* **Volatility:** Mean Volatility, 5-day Volatility, and Downside Deviation.
* **Risk:** Maximum Drawdown, Value at Risk (VaR 95), and Sharpe Ratio.
* **Technical:** RSI (Relative Strength Index), MACD, and Bollinger Band Width.
* **Liquidity:** Amihud Illiquidity and Trading Frequency.


### Machine Learning
* **K-Means Clustering:** Used as the primary model to segment stocks into "Low Risk," "Moderate Risk," and "High Risk."
* **Dimensionality Reduction:** PCA (Principal Component Analysis) used for visualizing high-dimensional stock clusters.

## 🚀 How to Use
1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn streamlit
    ```
2.  **Run the Pipeline:** Follow the notebooks in numerical order (`01` through `04`) to process data and train the clustering model.
3.  **Launch Dashboard:**
    ```bash
    streamlit run src/app.py
    ```

## 📈 Recommendations for Investors
* **Diversification:** Use the risk clusters to ensure your portfolio isn't concentrated in a single "High-Risk" segment.
* **Risk Alignment:** Match your investment choices to your personal risk tolerance (e.g., conservative investors should stick to the "Stable/Low-Risk" cluster).
* **Sector Awareness:** Monitor sector-level clustering to identify systemic risks and defensive investment opportunities.

## 🔮 Future Enhancements

* Incorporate real-time NSE price feeds

* Add sentiment analysis from financial news and social media

* Extend clustering to include macroeconomic indicators

* Implement portfolio optimization based on risk clusters

---

##  🧰  Tech Stack
| Categories | Technologies |
|:--- |:--- |
| **Languages** | Python |
| **Data Processing** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn |
| **Dimensionality Reduction** | PCA |
| **Web App Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Development Environment** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

---
##  👥 Authors
| Name | GitHub Profile |
|------|----------------|
|**Stephen Muema**|[Steve](https://github.com/Kaks753)|
| **Sharon Kipruto** | [SharonK](https://github.com/sharonkipruto-code) |
| **Lorena Terah**   | [Lorena](https://github.com/lorenaterah) |
| **Edgar Owuor**    | [Edgar](https://github.com/edgarowuor-tech) |
|**Salma Mwende**|[Salma](https://github.com/salmamwende-code)|
| **Dawa Jarso**     | [Dawa](https://github.com/Dawa-Jarso)|

