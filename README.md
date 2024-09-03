# London Weather Prediction

This repository contains the complete workflow for predicting weather patterns in London using various machine learning models. The project demonstrates data preprocessing, exploratory data analysis (EDA), model development, and performance evaluation. It also incorporates Docker for containerization and MLFlow for tracking experiments.

## Project Overview

The objective of this project is to analyze historical weather data for London and develop predictive models to forecast future weather conditions, specifically the mean temperature. By leveraging historical data and applying various machine learning algorithms, we aim to provide insights that can assist in weather prediction and planning.

Key Aspects:
Data Preprocessing: Cleaning and transforming raw weather data for analysis.
Exploratory Data Analysis (EDA): Visualizing and understanding weather trends.
Model Development: Training machine learning models to predict mean temperature.
Model Evaluation: Assessing model performance using metrics like RMSE.
Containerization: Docker is used to encapsulate the project environment.
Experiment Tracking: MLFlow is integrated for tracking experiments and model versions.

Features:
Weather Analysis: Detailed analysis of weather features and trends.
Predictive Modeling: Implementation of regression models to predict future mean temperatures.
Visualization: Interactive and static visualizations to understand weather patterns.
Containerized Environment: Easily reproducible environment using Docker.
Experiment Tracking: MLFlow is used to track different model versions and experiments.

### Prerequisites

To run this project, you'll need to have the following software installed:

- Python 3.9 or higher
- pip (Python package installer)

You can install Python and pip by following the instructions [here](https://www.python.org/downloads/).

### Installing

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/phoenixak/london-weather-prediction.git
   cd london-weather-prediction

2. **Create a virtual environment:**

   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
Or 

1. **Build the Docker image:**

   ```bash
    docker build -t london_weather_prediction .
  
2. **Run the Docker container:**

   ```bash
   docker run -v "$(pwd)/dataset":/app/dataset -p 8888:8888 -p 5000:5000 -it london_weather_prediction

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any feature requests or bugs.




