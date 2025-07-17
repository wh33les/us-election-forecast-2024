# 2024 U.S. Presidential Election Forecast

A data-driven forecasting system that predicts daily outcomes for the 2024 U.S. presidential election using advanced time series modeling and polling data analysis. 

This project implements Holt (double) exponential smoothing with rigorous cross-validation to process FiveThirtyEight's polling data and generate both popular vote predictions and electoral college outcomes.

__Final Results:__ Achieved forecasts within 2% of actual election results and correctly predicted winners for 47 out of 50 states.

## Project Overview

Election outcome predictions for each day from October 23 through November 5, 2024. Uses Holt (double) exponential smoothing with a random walk with drift baseline and rigorous cross-validation.

Polling data from FiveThirtyEight (`data/raw_polling_data.csv`) is filtered by polling date (starts at Jul 21 when Biden dropped out), likely voters, and swing state and national polls, and FiveThirtyEight's feature POLLSCORE.  Training data is the average polling percentage for each day.  

[__Project web page:__](https://wh33les.github.io/us-election-forecast-2024/)

## Technical Methodology

### Time Series Modeling
- Holt (double) exponential smoothing with separate /(/alpha)/ and /(/beta/) parameters for each candidate.
- Grid search optimization using time series cr
- Random walk with drift model for performance benchmarking.
- Strict temporal validation ensuring no future information was used.

### Data Processing Pipeline
- **Source**: FiveThirtyEigth polling data in `data/raw_polling_data.csv` and swing state averages (no longer available on FiveThirtyEight's site).
- **Filtering**: Likely voters only, negative POLLSCORE ratings, post-Biden withdrawal polls, only swing states and national polls.
- **Aggregation**: Daily polling averages with exponential smoothing.
- **Validation**: 5-fold cross-validation with 7-day holdout periods.

### Electoral College Calculation
- **Swing states**: Arizona, Georgia, Michigan, Nevada, North Carolina, Pennsylvania, Wisconsin.
- **Allocation algorithm**: Proportional electoral vote distribution based on popular vote predictions.
- **Safe state assumptions**: Used FiveThirtyEight's "safe" state classifications.

## Usage
```python
python main.py
```

Optional attributes are `--date`, which gives the forecast for a single date, and `--start` and `--end` for a range of dates.  Date format is flexible (e.g., 10-23, Oct 23, etc.). 

## Visualizations

The system generates two types of visualizations:

- **Daily Forecast Plots**: Shows polling averages, model predictions, and baseline comparisons.
- **Historical Performance**: Tracks how predictions evolved over the final two weeks.

## Future Enhancements

- **Real-time updates**: Automated daily data collection and forecast updates.
- **Ensemble methods**: Combine multiple forecasting algorithms.
- **Confidence intervals**: Add probabilistic uncertainty quantification.
- **Interactive dashboard**: Enhanced web interface with dynamic visualizations.
- **Feature engineering**: Incorporate economic indicators and social media sentiment.

## Author

**Ashley K. W. Warren**  
*Mathematician → Machine Learning Engineer*

- 🔗 [LinkedIn](https://www.linkedin.com/in/ashleykwwarren)
- 🌐 [Website](https://wh33les.github.io)
- 📧 [ashleykwwarren@gmail.com](mailto:ashleykwwarren@gmail.com)