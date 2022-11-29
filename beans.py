'''
Methods associated with in class exercises ex6
'''
import graphing
import numpy
import pandas
import numpy as np
import statsmodels.formula.api as smf

dataset = pandas.read_csv("data/beans_harvest.csv", sep="\t")

def print_data():
    print(dataset)

def show_graph(model=None):
    graphing.scatter_2D(dataset[["number_of_workers", "tons_of_beans_harvested"]], title="Number of Workers vs Tons of Beans Harvested", show=True, trendline=model)

def fit_model(data, formula):
    print("creating model object...")
    model = smf.ols(data=data, formula=formula)

    print("done!")

    print("Training model to find the best model parameters...")
    fit_result = model.fit()
    print("Done!\n")

    print("The best model parameters are:")
    print("Offset:")
    print(fit_result.params[0])

    print("Slope:")
    print(fit_result.params[1])

    # Graph
    show_graph(lambda x: x * fit_result.params[1] + fit_result.params[0])

