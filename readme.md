# 2024 U.S. Presidential Election Forecast

A data-driven forecasting system that predicts daily outcomes for the 2024 U.S. presidential election using advanced time series modeling and polling data analysis. 

This project implements Holt (double) exponential smoothing with rigorous cross-validation to process FiveThirtyEight's polling data and generate both popular vote predictions and electoral college outcomes.

__Final Results:__ Achieved forecasts within 2.5% of actual election results and correctly predicted winners for 46 out of 50 states.

## Project Overview

Election outcome predictions for each day from Oct 23 through Nov 5, 2024 using polling data from FiveThirtyEight (`data/raw_polling_data.csv`). Uses Holt (double) exponential smoothing with a random walk with drift baseline and 5-fold cross-validation that optimizes MASE.

Polling data  is filtered by polling date (starts at Jul 21 when Biden dropped out), likely voters, swing state and national polls, and FiveThirtyEight's feature POLLSCORE (`preprocess_data.py`).  Training data is the average polling percentage for each day with these filters applied.  

__Project web page:__  [https://wh33les.github.io/us-election-forecast-2024/](https://wh33les.github.io/us-election-forecast-2024/)

## Technical Methodology

### Time Series Modeling
- Holt (double) exponential smoothing with separate hyperparameters for each candidate.
- Grid search optimization -- standard grid ($\alpha\in(0.05,0.5)$, $\beta\in(0.05,0.3)$, intervals of $0.05$), configurable in `src/config.py`.
- Random walk with drift model for performance benchmarking.
- Strict temporal validation ensuring no future information was used.

### Data Processing Pipeline
- **Source**: FiveThirtyEight polling data in `data/raw_polling_data.csv` and swing state averages (no longer available on FiveThirtyEight's site).
- **Filtering**: Likely voters only, negative POLLSCORE ratings, post-Biden withdrawal polls, only swing states and national polls.
- **Aggregation**: Daily polling averages with exponential smoothing.
- **Validation**: 5-fold cross-validation with 7-day holdout periods.
- **Quick updates**: Creates cache files so the raw data is not unnecessarily reprocessed for each prediction (`data/forecast_history.csv`, `data/polling_averages_cache.csv`).

### Electoral College Calculation
- **Swing states**: Arizona, Georgia, Michigan, Nevada, North Carolina, Pennsylvania, Wisconsin.
- **Allocation algorithm**: Proportional electoral vote distribution based on popular vote predictions.
- **Safe state assumptions**: Used FiveThirtyEight's "safe" state classifications.

## Usage
```python
python main.py
```

Optional attributes are `--date`, which gives the forecast for a single date, and `--start` and `--end` for a range of dates.  Date format is flexible (10-23, Oct 23, etc.).  The default is to forecast all dates from Oct 23 to Nov 5.

## Visualizations

The system generates two types of visualizations:

- **Daily forecast plots**: Shows polling averages, model predictions, and baseline comparisons (`outputs/forecast_images`).
- **Historical performance**: Tracks how predictions evolved over the final two weeks (`outputs/previous_forecasts`).

## Skills Demonstrated
This project showcases key machine learning engineering competencies:

- Advanced statistical modeling: Implementation of sophisticated time series forecasting techniques.
- Data pipeline engineering: End-to-end ML pipeline from data collection to web deployment.
- Model validation: Rigorous statistical methodology with proper cross-validation techniques.
- Production code architecture: Modular, maintainable design with proper configuration management (`src/`).
- Data visualization: Professional-quality plots optimized for stakeholder communication.
= Real-world application: Solving complex prediction problems with measurable business impact.

## Future Enhancements

- **Real-time updates**: Automated daily data collection and forecast updates.
- **Ensemble methods**: Combine multiple forecasting algorithms.
- **Confidence intervals**: Add probabilistic uncertainty quantification.
- **Interactive dashboard**: Enhanced web interface with dynamic visualizations.
- **Feature engineering**: Incorporate economic indicators and social media sentiment.

## Author

**Ashley K. W. Warren**  
*Mathematician → Machine Learning Engineer*

- [LinkedIn](https://www.linkedin.com/in/ashleykwwarren)
- [Website](https://wh33les.github.io)
- [ashleykwwarren@gmail.com](mailto:ashleykwwarren@gmail.com)