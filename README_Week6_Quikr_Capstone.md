# YUVA INTERN – Week 6 Integrative Capstone
## Quikr Used-Car Price Prediction & Vehicle Segmentation

### Objective
Build a complete data-science pipeline covering acquisition, cleaning, EDA, feature engineering, supervised learning, evaluation and optional unsupervised learning.

### Dataset
File expected by the script:

`Quikr car price prediction.csv`

Week 1 documented:
- 892 original rows
- 6 columns
- 94 exact duplicates
- 725 final cleaned rows
- year, Price and kms_driven converted to numeric
- missing/invalid records handled explicitly
- IQR used for outlier investigation

### How to run

```bash
pip install pandas numpy matplotlib scikit-learn
python quikr_week6_capstone.py
```

Keep the CSV in the same folder as the Python script.

### Outputs
The script creates `week6_outputs/` containing:
- cleaned_data_week6.csv
- model_comparison.csv
- kmeans_clusters.csv
- 01_price_distribution.png
- 02_price_vs_kms.png
- 03_average_price_by_fuel.png
- 04_average_price_by_year.png
- 05_actual_vs_predicted.png
- 06_kmeans_clusters.png

### Important
The report deliberately does **not** invent model metrics. Run the script on the actual dataset and use the generated `model_comparison.csv` values in the final evaluation section.

### Suggested GitHub structure

```text
yuva-week6-quikr-capstone/
├── data/
│   └── Quikr car price prediction.csv
├── outputs/
│   ├── charts/
│   └── model_comparison.csv
├── quikr_week6_capstone.py
├── README.md
└── requirements.txt
```

Do not upload private/personal data. If the dataset is redistributed under restrictions, include only the source/reference and instructions for obtaining it.
