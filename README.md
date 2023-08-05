# PracticalMachineLearningCourseProject
## Overview
This serves as the concluding report for the Practical Machine Learning course on Coursera, which is part of the Data Science Specialization track provided by John Hopkins University.

The objective of this project involves utilizing data collected from accelerometers placed on various body parts (belt, forearm, arm, and dumbell) of six participants. The goal is to predict the manner in which these individuals performed their exercises, which is indicated by the "classe" variable within the training dataset.

To achieve this, we employ four distinct models: Decision Tree, Random Forest, and Support Vector Machine. These models are trained using Weka_control as cross-validation on the training dataset. Subsequently, we apply these trained models to a validation dataset, which is randomly extracted from the training CSV data. This step allows us to calculate accuracy. By analyzing these metrics, we determine the optimal model. Once identified, we employ this model to predict outcomes for 20 cases using the test CSV dataset.
