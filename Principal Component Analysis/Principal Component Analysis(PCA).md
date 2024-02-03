
PCA is a technique which seeks a direction(unit vector $u$) which best captures the variance in the original data without much loss of information

Direction with largest projected variance is called First principal component.

The direction that maximises the variance is also the one which minimises the mean squared error.

**original data D**

- Covariance matrix of D is $\sum$

- Variance of the data matrix D
$$var(D) = tr(\sum)$$
- Projection of vector $x_i$ on u is $$x_i^` = u^Tx_i$$
**Projected data A**
- variance of the projected data is $$var(A) = \sigma^2 = u^T \sum u = \lambda$$
	$\sigma^2$ is nothing but the eigenvalue of $\sum$
	$\sum$ is the covariance matrix of D
	we want to find u which maximises $\sigma^2$

- Mean square error along u 
$$MSE(u) = var(D) - var(A)$$
$$\implies MSE(u) = var(D) - \lambda_1 - \lambda_2-...$$



- Covariance matrix of the transformed data
	
 ![[Screenshot 2023-10-24 at 2.21.32 PM.png|300]]

	The decomposition of covariance matrix of A yields:
	$$\sum = U\lambda U$$
	$U$ -> eigenvectors(they are orthogonal)
	$\lambda$ -> eigenvalues(sorted in decreasing order)



- choosing dimension 'r'
$$f(r) = \frac{\lambda_1+\lambda_2+...+\lambda_r}{\lambda_1+\lambda_2+...+\lambda_d}$$



**PCA pseudocode**

![[Screenshot 2023-10-24 at 2.22.32 PM.png]]


PCA is the special case of [[Singular Value Decomposition(SVD)]].


**notes**
![[PCA 1.pdf]]

