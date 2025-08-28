# Zomato Restaurant Data Analysis 🍽️

Data analysis of Zomato restaurant data using Python

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-red.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

*A comprehensive data science project analyzing restaurant trends, digital transformation impact, and customer preferences using real-world Zomato data*

</div>

## 📊 Project Overview

This project analyzes 148 restaurants to uncover insights about:
- Restaurant type distribution and pricing strategies
- Impact of digital presence (online ordering) on ratings and costs
- Customer behavior patterns and rating correlations
- Best value restaurants based on price-quality ratio

## 🔍 Key Findings

- **Digital Advantage**: Restaurants with online ordering have 10.6% higher ratings and command 42.2% premium pricing
- **Market Distribution**: Dining establishments dominate (74.3%), followed by Cafes (15.5%)
- **Price-Quality Correlation**: Premium restaurants (>₹600) achieve higher average ratings (3.79) compared to budget options (3.53)
- **Service Premium**: Only 5.4% offer table booking, but they show significantly higher ratings (4.19 vs 3.60)

### 🏆 Performance Metrics
| Metric | Value | Insight |
|--------|-------|---------|
| Average Rating | 3.63/5.0 | Room for industry improvement |
| Average Cost | ₹418 | Mid-range market positioning |
| Online Adoption | 39.2% | Significant growth opportunity |
| High-Rated Restaurants | 23% | Quality differentiation exists |

---

## 🛠️ Technical Implementation

### **Core Technologies**
- **Python 3.8+** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations and statistical operations
- **Matplotlib & Seaborn** - Advanced data visualization
- **Jupyter Notebook** - Interactive development environment

### **Data Science Techniques Applied**
- ✅ **Data Cleaning**: Rating format standardization, missing value handling
- ✅ **Exploratory Data Analysis (EDA)**: Statistical summaries and distribution analysis
- ✅ **Correlation Analysis**: Identifying key performance relationships
- ✅ **Segmentation Analysis**: Restaurant categorization and comparison
- ✅ **Business Intelligence**: Actionable insight generation

---

## 📈 Visualizations

The project includes comprehensive visualizations:
- Restaurant type distribution analysis
- Rating vs cost correlation plots
- Online vs offline performance comparisons
- Price category breakdowns
- Top performer identification

## 🚀 Getting Started

### Installation
 
1. Clone the repository
```
git clone https://github.com/Jrsandy26/zomato-data-analysis.git
cd zomato-data-analysis
```
2.Ensure you have Python 3.8+ installed
```
python --version
```
3. Install required packages
```
pip install pandas numpy matplotlib seaborn jupyter
```
4.Launch Jupyter Notebook
```
jupyter notebook notebooks/zomato_analysis.ipynb
```
Load and preprocess data
df = pd.read_csv('data/Zomato-data.csv')

Analyze specific restaurant types
cafes_analysis = analyze_specific_type('Cafes')

Find best value restaurants
best_value = find_best_value_restaurants(max_cost=400, min_rating=3.5)


---

## 📊 Key Visualizations

### Restaurant Type Distribution
- **Dining**: 110 restaurants (74.3%)
- **Cafes**: 23 restaurants (15.5%)
- **Buffet**: 7 restaurants (4.7%)
- **Other**: 8 restaurants (5.4%)

### Digital vs Traditional Performance
| Metric | Online Restaurants | Offline Restaurants | Difference |
|--------|-------------------|-------------------|------------|
| Avg Rating | 3.86 | 3.49 | +10.6% ⬆️ |
| Avg Cost | ₹510 | ₹359 | +42.2% ⬆️ |
| Avg Votes | 559 | 75 | +645% ⬆️ |

---

## 💼 Business Applications

### **For Restaurant Owners**
- **Digital Strategy**: ROI analysis for online ordering implementation
- **Pricing Optimization**: Data-driven pricing strategies by restaurant type
- **Service Enhancement**: Table booking as differentiation opportunity

### **For Food Delivery Platforms**
- **Market Segmentation**: Targeted acquisition strategies
- **Partner Development**: Supporting offline restaurants' digital transition
- **Quality Metrics**: Rating improvement programs

### **For Investors & Analysts**
- **Market Trends**: Digital transformation impact quantification
- **Investment Decisions**: High-potential restaurant category identification
- **Risk Assessment**: Performance correlation analysis

---

## 🧠 Technical Skills Demonstrated

<table>
<tr>
<td valign="top" width="50%">

**Data Science Core**
- Data cleaning and preprocessing
- Statistical analysis and correlation
- Hypothesis testing and validation
- Business intelligence and insights

</td>
<td valign="top" width="50%">

**Programming & Tools**
- Python programming and libraries
- Jupyter Notebook development
- Git version control
- Documentation and presentation

</td>
</tr>
</table>

---

## 📈 Detailed Insights

### **Price Segmentation Analysis**
- **Budget (≤₹300)**: 67 restaurants, 3.53 avg rating
- **Mid-range (₹301-600)**: 54 restaurants, 3.69 avg rating  
- **Premium (>₹600)**: 27 restaurants, 3.79 avg rating

### **Correlation Findings**
- **Rating ↔ Votes**: 0.490 (Strong positive correlation)
- **Rating ↔ Cost**: 0.275 (Moderate positive correlation)
- **Digital Presence ↔ Performance**: Significant positive impact

### **Top Performers**
1. **Onesta** - 4.6/5.0 rating (2,556 votes)
2. **Empire Restaurant** - 4.4/5.0 rating (4,884 votes)
3. **Meghana Foods** - 4.4/5.0 rating (4,401 votes)

---

## 🔮 Future Enhancements

- [ ] **Predictive Modeling**: Rating prediction based on features
- [ ] **Sentiment Analysis**: Customer review text analysis
- [ ] **Location Analysis**: Geographic performance patterns
- [ ] **Temporal Trends**: Time-based performance evolution
- [ ] **Competitive Analysis**: Market positioning strategies

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 About the Author

**Sandeep Sai Kumar**

🎓 Aspiring Data Scientist  | 🐍 Python Developer

- 🌐 **Portfolio**: [sandeep26.vercel.app](https://sandeep26.vercel.app)
- 💼 **LinkedIn**: [linkedin.com/in/sandeepsai26](https://www.linkedin.com/in/sandeepsai26)
- 📧 **Email**: sandeepsai.work@gmail.com
- 🐱 **GitHub**: [@Jrsandy26](https://github.com/Jrsandy26)

---

## 🙏 Acknowledgments

- **Dataset Source**: Zomato restaurant data platform
- **Inspiration**: Restaurant industry digital transformation trends
- **Tools**: Python Data Science ecosystem and open-source community
- **Methodology**: Industry best practices for data analysis

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star! ⭐**

*Made with ❤️ and Python*

</div>

