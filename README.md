# Hyperparameter-Optimizer-SMBO-bayesian-Optimizer

## Implementation Steps : 

1. Define the optimization problem
The first step is to define the optimization problem, which means stating the objective function to be optimized and the hyperparameters to be fine-tuned. In this step, the objective function is usually a performance evaluation metric such as accuracy for classification problems or mean absolute error for regression problems. The aim of the objective function is either to maximize or minimize the final score. The hyperparameters are the variables that can be changed to optimize this evaluation metric.

2. Define the hyperparameter search space
The search space refers to the possible range of values that each defined hyperparameter can take at any given time. It is essential to select a search space that is broad enough to incorporate potentially good combinations that can produce the best performance but not so broad that the search becomes inefficient.

Examples of hyperparameters from the random decision tree algorithm available on the scikit-learn library (DecisionTreeClassifier()) are as follows:

splitter: This method is used to choose the split at each node.

criterion: This function measures the quality of a split.

max_depth: This is the maximum depth of the tree.

min_sample_split: This is the minimum number of samples required to split an internal node.

max_features: This is the number of features to consider when searching for the best split.

3. Choose an acquisition function
An acquisition function, also known as a selection function, is used to select the next combination of hyperparameters to evaluate based on the predictions of the probabilistic model. The most commonly used acquisition function is expected improvement (EI), which balances exploration and exploitation in the search space for optimal hyperparameters.

The expected improvement (EI) for a given combination of hyperparameters is calculated as follows:

It first calculates the best-observed performance so far, which is the maximum objective function value among all evaluated points.

Then, it calculates the probability that a new evaluation at the chosen hyperparameters will result in a better performance than the best-observed performance. This is done by considering the Gaussian Process model’s mean and variance predictions at that point.

The expected improvement is a weighted combination of the probability of improvement and the magnitude of the expected improvement. It seeks to find the trade-off between exploring unexplored regions (points with high uncertainty) and exploiting regions that are likely to yield better performance.

The higher the EI, the more promising the hyperparameters are, and the more likely they are to be chosen for evaluation.

4. Train the probabilistic model
The probabilistic model is trained on the results of the previous evaluations, which include the input hyperparameters and the corresponding output values of the objective function. Gaussian Process (GP) is the most common probabilistic or surrogate model. It is an effective tool for modeling complex, nonlinear relationships.

GP is a statistical model that is used to represent a probability distribution over functions. In Bayesian optimization, the GP is used to model the objective function, which is the performance of the ML model with respect to a given set of hyperparameters.

The GP model allows us to make predictions about the performance of the model for new hyperparameter settings that have not yet been evaluated.

5. Select the next combination to evaluate
The acquisition function defined in Step 3 is used to select the next combination of hyperparameters for evaluation based on the prediction of the probabilistic model, also known as a surrogate model. This combination is typically chosen to maximize the EI over the current best combination of hyperparameters.

6. Evaluate the objective function
The selected combination of hyperparameters is evaluated using the defined objective function to obtain a performance evaluation metric. The purpose of the objective function is to either maximize or minimize the final output score.

7. Update the probabilistic model
The results of the evaluation are used to update the probabilistic model, which incorporates the new information and refines the prediction of the probabilistic model. The process is then repeated from Step 5 to Step 7, with each new combination of hyperparameters selected based on the updated predictions from the probabilistic model.

8. Terminate the process
Finally, the process is terminated when a stopping criterion is met, such as when a maximum number of evaluations has been reached or when the improvement in the objective function is below a certain threshold.

Overall, SMBO is a powerful and flexible method for optimizing complex and expensive functions. It has been successfully applied to a wide range of different ML problems to produce ML models with the best performance.

## Implementation Example : A binary classification problem statement is optimized for Histogram Based Gradient Boosting algorithm and k-nearest algorithm

## Workflow of the Bayesian Optimization Algorithm : 

![image](https://github.com/user-attachments/assets/ac0519af-6b29-45f9-acd1-e3dcc038af24)

