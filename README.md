# grad-school-project

## Part 1: Data Exploration: Graduate School Admissions

The data we will be using is admission data on Grad school acceptances.

* `admit`: whether or not the applicant was admitted to graduate school
* `gpa`: undergraduate GPA
* `gre`: score of GRE test
* `rank`: prestige of undergraduate school (1 is highest prestige, ala Harvard)

We will use the GPA, GRE, and rank of the applicants to try to predict whether or not they will be accepted into graduate school.

Before we get to predictions, we should do some data exploration.

1. Load in the dataset into pandas: `data/grad.csv`.

2. Use the pandas `describe` method to get some preliminary summary statistics on the data. In particular look at the mean values of the features.

3. Use the pandas `crosstab` method to see how many applicants from each rank of school were accepted. You should get a dataframe that looks like this:

    ```
    rank    1   2   3   4
    admit
    0      28  ..  ..  ..
    1      33  ..  ..  ..
    ```

    Make a bar plot of the percent of applicants from each rank who were accepted. You can do `.plot(kind="bar")` on a pandas dataframe.

4. What does the distribution of the GPA and GRE scores look like? Do the distributions differ much?

    Hint: Use the pandas `hist` method.

5. Make a scatterplot of each predictor variable vs. the target (`admitted`).  This can be tricky, because the target in a classification model can only take on two possible values. You may wish to add some random noise to the y-coordinates (also called "jitter") to make the data density easier to see.

## Part 2: Predicting Graduate School Admissions

Now we're ready to try to fit our data with Logistic Regression.

1. Use sklearn to fit a [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to the raw data.  Your target variable should be admittance to graduate school.

2. Once we feel comfortable with our model, we can move on to cross validation.  Use sklearn's [KFold cross validation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) and `LogisticRegression` to calculate the average log-loss across a 10 fold cross validation. Try to gridsearch across various hyperparameters to increase the predictive power of your model.

3. Take some time to try to improve the models predictive power.  Make sure you use cross validation to evaluate whether adding in any predictors improves the model, as you cannot trust the training log-loss!  Here are some ideas:
  - Do some creative feature engineering.  Do you think any comparisons between the predictors could be predictive (i.e. a student with a good gpa but a poor gpa score, or the reverse)?
  - Add some quadratic terms or other feature transformations / interactions.

4. The `rank` column is ordinal where we assume an equal change between ranking levels, but we could also consider it to be more generally categorical. Use panda's [get_dummies](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.reshape.get_dummies.html) to binarize the column.

5. Compute the log-loss after categorizing the `rank` column. Does it do better or worse with the rank column binarized?

    From now on, use the version of the feature matrix that performed the best.

6. Say we are using this as a first step in the application process. We want to weed out clearly unqualified candidates, but not reject too many candidates. How may we use the predicted probabilities from our model to accomplish this?

7. Plot your final ROC curve, what should the cutoff be? What is the AUC score?

