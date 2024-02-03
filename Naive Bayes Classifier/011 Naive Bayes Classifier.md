
It is based on [[Bayes Theorem]].

Naive Bayes Classifiers makes the following assumptions:
- the features are independent of each other.
- each feature gives the same weight to the output.



![[lecture16.pdf]]

problem from https://www.geeksforgeeks.org/naive-bayes-classifiers/

**Naive Bayes Algorithm**
- we are given labeled data with features($X_1...X_n$) and labels($Y_1...Y_n$)
- calculate the prior probability of each class ($Y=y_i$)
- calculate the feature likelihoods $$P(X|Y)=\frac{C(X\cap Y)}{C(Y)}$$ for each feature $Y=y_i$ and feature $X=x_i$
- for new instance X, calculate $$P(Y|X) \propto P(X).P(Y)$$
- whichever has highest $P(Y|X)$ assign that label to the new instance $X$


**Advantage**
Training is fast compared to Perceptron, Logistic regression etc as it has a closed form solution.

Disadvantage
- real life features are not independent.
- sometimes $P(X_i|Y)=0$


### Generative Classifier
- example : Naive Bayes

![[Screenshot 2023-11-28 at 5.20.10 PM.png]]



### Discriminative Classifier

- Example : Logistic Regression

![[Screenshot 2023-11-28 at 5.21.09 PM.png]]










- https://www.youtube.com/watch?v=O2L2Uv9pdDA