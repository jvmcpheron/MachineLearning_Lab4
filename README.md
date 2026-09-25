# Machine Learning Lab 4
## Comparing Perceptron and Multilayered Perceptron Models

I used k-fold validation with ten folds, and compared the average accuracies of
each model.

## Assignment
Resources: Perceptron and MLPClassifier.
For two of your data sets: Baseball and Sick
Pick at least three values for three of the parameters shown above.
hidden_layer_sizes
solver
max_iter
Perform experiments with each combination of these features to determine which combination is most accurate (remember, you want to do a fair comparison).  Also include the results from a Perceptron model.

## Notes
These are the parameters I ended up choosing:
hiddenLayerSizes = [2, 4, 6]
solvers = ['lbfgs', 'sgd', 'adam']
max_iterations = [500, 1000, 2000]

I chose to have my code do 10 folds and calculate the average accuracy of each model and then display those accuracies on a bar graph for comparison for my experiment. 

## Conclusion
For the baseball data, I was very surprised with how close all of the models were to each other. The bar graph representing accuracy suggested this flat line of 90% that the models can’t seem to pass. I did notice a few dips, and, checking my code, it would appear those dips happen for the adam solver, which suggests that the adam solver was a weak option for the baseball data. The adam solver with 500 max iterations and 2 hidden layer size parameters was the least accurate model. Interestingly, limiting max iterations and hidden layer sizes did not seem to affect the other solvers.

The sick data had even less variation! Multi Layered Perceptrons seemed to better fit the sick data compared to the baseball data, with accuracies ranging from .93 to .96, which is pretty good. The regular perceptron model did the worst here, with an average accuracy of .87. That being said, adjusting parameters yielded little change in resulting accuracy with the MLPs.



