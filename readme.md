# 2024 US Presidential Election Forecast

A data-driven forecasting system that predicts daily outcomes for the 2024 US Presidential Election using advanced time series modeling and polling data analysis.

## Project Overview

This project implements a forecast system that predicts election outcomes for each day from October 23 through November 5, 2024. Using __Holt (double) exponential smoothing__ and rigorous cross-validation, the model processes FiveThirtyEight's polling data to generate both popular vote predictions and electoral college outcomes.

[__Project web page:__](https://wh33les.github.io/election-forecast/)

Final forecast within 2% of actual election results and correctly predicted winners for 47 out of 50 states.

## 🔬 Technical Methodology

### Time Series Modeling
- **Algorithm**: Holt (double) exponential smoothing with separate /(/alpha)/ and /(/beta/) parameters for each candidate
- **Hyperparameter tuning**: Grid search optimization using time series cross-validation
- **Baseline comparison**: Random walk with drift model for performance benchmarking
- **No data leakage**: Strict temporal validation ensuring no future information was used

### Data Processing Pipeline
- **Source**: FiveThirtyEigth polling data in `data/president_polls.csv` and swing state averages (no longer available)
- **Filtering**: Likely voters only, negative POLLSCORE ratings, post-Biden withdrawal polls, only swing states and national polls
- **Aggregation**: Daily polling averages with exponential smoothing
- **Validation**: 5-fold cross-validation with 7-day holdout periods

### Electoral College Calculation
- **Swing states**: Arizona, Georgia, Michigan, Nevada, North Carolina, Pennsylvania, Wisconsin
- **Allocation algorithm**: Proportional electoral vote distribution based on popular vote predictions
- **Safe state assumptions**: Used FiveThirtyEight's "safe" state classifications

## Project Structure

```
election-forecast/
│
├── src/
│   ├── config.py                 # Configuration management
│   ├── main.py                   # Main pipeline orchestration
│   │
│   ├── data/
│   │   ├── collectors.py         # Data collection and loading
│   │   └── processors.py         # Data cleaning and preprocessing
│   │
│   ├── models/
│   │   ├── holt_forecaster.py    # Holt exponential smoothing implementation
│   │   └── electoral_calculator.py # Electoral college calculations
│   │
│   └── visualization/
│       └── plotting.py           # Matplotlib plotting and visualization
│
├── data/
│   ├── raw/                      # Original FiveThirtyEight polling data
│   └── processed/                # Cleaned and processed datasets
│
├── forecast_images/              # Daily forecast visualizations
├── outputs/                      # Final analysis and historical plots
├── index.html                    # Interactive web presentation
└── README.md                     # This file
```

## Individual Components
```python
# Example: Generate single-day forecast
from src.models.holt_forecaster import HoltElectionForecaster
from src.config import ModelConfig

config = ModelConfig()
forecaster = HoltElectionForecaster(config)

# Train and forecast
best_params = forecaster.grid_search_hyperparameters(trump_data, harris_data, x_train)
fitted_models = forecaster.fit_final_models(trump_data, harris_data)
forecasts = forecaster.forecast(horizon=7)
```

## Visualizations

The system generates two types of visualizations:

- **Daily Forecast Plots**: Shows polling averages, model predictions, and baseline comparisons
- **Historical Performance**: Tracks how predictions evolved over the final two weeks
ofessional styling optimized for web presentation

## Applications & Impact

This project demonstrates:
- **Advanced time series modeling** for real-world prediction problems
- **Rigorous statistical methodology** with proper validation techniques
- **Production-ready code architecture** with modular, maintainable design
- **Data visualization skills** for communicating complex analytical results
- **End-to-end ML pipeline** from data collection to web deployment

## Future Enhancements

- **Real-time updates**: Automated daily data collection and forecast updates
- **Ensemble methods**: Combine multiple forecasting algorithms
- **Confidence intervals**: Add probabilistic uncertainty quantification
- **Interactive dashboard**: Enhanced web interface with dynamic visualizations
- **Feature engineering**: Incorporate economic indicators and social media sentiment

## Author

**Ashley K. W. Warren**  
*Mathematician → Machine Learning Engineer*

- 🔗 [LinkedIn](https://www.linkedin.com/in/ashleykwwarren)
- 🌐 [Portfolio Website](https://wh33les.github.io)
- 📧 [ashleykwwarren@gmail.com](mailto:ashleykwwarren@gmail.com)