# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

This dataset contains data about direct marketing campaigns conducted by a banking institution. We seek to predict whether a client will subscribe to a term deposit based on the provided data.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The best performing AutoML model was a `VotingEnsemble`, which is an ensemble of different ML algorithms. It achieved an accuracy of 92.03% when predicting the target variable.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The pipeline architecture involves creating, training, and optimizing a ML model to solve a classification problem. The pipeline architecture consists of the following components:

1. **Data Handling**: The project starts with the creation of a `TabularDatasetFactory` from an online CSV file, providing access to the dataset.

2. **Data Preprocessing**: The dataset undergoes data preprocessing, including cleaning and encoding categorical data into numeric format. Techniques such as one-hot encoding and ordinal encoding are applied as needed.

3. **Data Splitting**: The preprocessed data is split into two subsets: a training set and a test set, ensuring that the model is evaluated on a different subset than the one used for training.

4. **Classification Algorithm**: The chosen classification algorithm for this task is Logistic Regression, implemented using the `LogisticRegressionClassifier` from scikit-learn. Logistic Regression aims to discover the relationship between the target variable `'y'` and the dataset features.

5. **Model Saving**: After training, the model is saved as a pickle file, and the hyperparameters used during training are recorded for reproducibility.

6. **Hyperparameter Tuning**: Azure's *HyperDrive* service is used for hyperparameter tuning. The process is implemented in a Jupyter notebook named `'udacity-project.ipynb'`. It repeatedly executes the `'train.py'` script with different hyperparameter configurations sampled from specified ranges.

7. **Early Stopping Policy**: An early stopping policy, known as `BanditPolicy`, is defined to optimize resource usage by stopping experiments that do not show promise, based on a defined slack factor and evaluation interval.

In summary, the pipeline takes raw data, preprocesses it, trains a Logistic Regression model with various hyperparameter combinations, and selects the best-performing model using *HyperDrive* and the defined early stopping policy. This entire process aims to create a model that accurately predicts whether a client will subscribe to a term deposit based on the marketing campaign data.

**What are the benefits of the parameter sampler you chose?**

The parameter sampler chosen for *HyperDrive*, which includes a combination of uniform and choice distributions for the hyperparameters `C` and `max_iter`, offers several benefits. 

1. **Exploration of Hyperparameter Space**: The uniform distribution for `C` allows for a wide exploration of regularization strengths (in this case ranging from 0.1 to 10). This exploration helps in finding a balance between model complexity and overfitting, improving the chances of discovering the optimal value for this hyperparameter.

2. **Diverse Iteration Limits**: By using a choice distribution for `max_iter` (in this case with options such as 15, 25, 50, 75, and 100), the parameter sampler explores different iteration limits. This diversity allows the optimization process to adapt to different training convergence requirements, ensuring the best possible hyperparameter configuration for the Logistic Regression model.

3. **Enhanced Exploration**: The combination of these two distributions provides a balanced approach to explore the hyperparameter space. The sampler searches a broad range of `C` values and considers various `max_iter` options, which can lead to the discovery of an effective combination of hyperparameters for the given problem.

4. **Resource Efficiency**: `RandomParameterSampling` is relatively resource-efficient because it randomly samples hyperparameters, which can result in faster convergence to the optimal configuration while utilizing fewer computational resources compared to grid search.

**What are the benefits of the early stopping policy you chose?**

The early stopping policy chosen for *HyperDrive*, which is the `BanditPolicy`, offers several benefits:

1. **Resource Optimization**: `BanditPolicy` is effective in optimizing the allocation of computational resources. It allows for early termination of poorly performing runs, which conserves compute resources by avoiding the completion of experiments that are unlikely to yield significant improvements.

2. **Efficient Exploration**: Its `slack_factor` parameter (set to 0.1 in this case) determines the allowed slack in performance compared to the best performing run. When a run's performance falls below this threshold, it is terminated. This encourages the exploration of more promising hyperparameter combinations and speeds up the search for optimal configurations.

3. **Dynamic Adaptation**: The `BanditPolicy` is dynamically adaptive, which means it continuously evaluates the ongoing runs and terminates underperforming ones. This dynamic nature allows it to adjust to different phases of the optimization process, adapting to the quality of runs as they progress.

4. **Faster Convergence**: By eliminating runs that do not show promise early in the experimentation process, the BanditPolicy promotes faster convergence towards the optimal hyperparameter configuration. It minimizes the time spent on unproductive experiments, which is particularly valuable in scenarios with limited computational resources.

5. **Reduced Experimentation Costs**: Early stopping policies like `BanditPolicy` can significantly reduce experimentation costs by avoiding unnecessary computations. This is advantageous when working with cloud-based or expensive computing resources.


## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML generated a set of ML models, including 5 `XGBoostClassifier`s, 2 `LightGBMClassifier`s, and 1 `RandomForestClassifier`, with various hyperparameters. The primary metric for model evaluation is accuracy, and the models were trained and validated on the specified dataset with a timeout of 30 minutes.

## Pipeline comparison

**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The *HyperDrive* model used Logistic Regression with hyperparameter tuning and achieved an accuracy of approximately 0.915. In contrast, *AutoML* generated a set of ensemble models, including XGBoost, LightGBM, and RandomForest, with the best model achieving an accuracy of approximately 0.920. The difference in accuracy is relatively small, but *AutoML*'s ensemble approach likely benefited from a broader exploration of models and hyperparameters, resulting in slightly improved performance. *AutoML*'s architecture also automates the model selection process, making it more suitable for non-experts or scenarios where time is limited.

## Future work

**What are some areas of improvement for future experiments? Why might these improvements help the model?**
- To enhance the performance of the Logistic Regression model, it's advisable to consider applying **feature scaling, even though this type of model typically doesn't require it. Scaling may facilitate model convergence and optimization, leading to improved accuracy.

- The dataset exhibits a significant class imbalance, with only 11.2% of positive samples out of the total set. To better evaluate model performance, alternative performance metrics such as F1-score or AUC should be used, as they account for the performance of the model in each class separately. Techniques like resampling (e.g., downsampling the majority class or oversampling the minority class) should be explored to address the class imbalance.

- When selecting a model for production purposes, it's vital to consider factors beyond accuracy. The VotingEnsemble model, while slightly more accurate, may not be the ideal choice due to trade-offs in inference speed and model explainability. Models like XGBoost and LightGBM, which offer a balance between accuracy and interpretability, may be better suited for deployment.