# sklearn-regression

Regression models for Saltcorn based on scikit-learn.

## Installation

This requires Python 3.x to be available as `python3` in your environment. In
addition, install the following modules with `pip`:

```
pip install pandas scikit-learn nbconvert jupyter matplotlib
```

You may also need Pandoc installed, on Debian/Ubuntu run:

```
sudo apt install pandoc
```

Finally, install this module (sklearn-regression) in your Saltcorn
instance from the module store.

## Creating a model

After you have installed at least one module (for instance, this module)
that provides model patterns, you can create models based on tables. Scroll
all the way down to the bottom of the table settings page and there will be a
section for models, where are you can add a new model. To configure and run
a new predictive model you need to create two types of entities:

- You need to first create a _model_, which specifies the predictors and
  the outcome variables, and also the class of regression model (Linear,
  Lasso etc) you would like to use.
- Then for each model you need to define at least one _model instance_. The
  model instance will set particular values for the hyperparameters, if any.

When you create a model instance it will be trained on all the data in the table.
It will give values for an accuracy score which can be compared across different
model instances and different models, in this case R^2 which can be used to
compare all regression models.

Running a model ionstance also produces a report which is generated from Jupyter
notebook. This gives some more details and diagnostic plots to help you assess the model fit quality.

## Making predictions

Predictions can be made either by using functions, or by adding a calculated
field. The calculated field is the easiest as if it does not involve entering
a formula.

Create a new stored calculated expression of a Float type on the table on which
you have run a model instance. On the first settings page for the new field,
there is a drop-down box at the top where you can choose the formula of the
calculated field value. The default value is "JavaScript expression", change this
to "Model prediction". Then you will be able to choose a model, model instance
and the prediction required.
