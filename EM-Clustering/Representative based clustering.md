Given dataset D and no of clusters 'k'
	- divide the dataset into 'k' clusters: $C = \{C_1, C_2...C_k\}$
	- for each cluster $C_i$ find representative point $\mu_i$
	- we try to minimise the objective function
		$$\sum_{i=1}^{k}\sum_{x_j \epsilon C_i}||(x_j-\mu_i)||^2 $$
	
		$$\mu_i=\frac{1}{|C_i|}\sum_{x_j\epsilon C_i}x_j$$

This is combinatorial optimisation problem difficult to solve. We solve it using greedy iterative approach called [[k-means]].


### K-means
- hard assignment where each point can belong to only 1 cluster
- K-means is greedy: Doesn't obtain global optimal solution but obtains a local optimal solution

- Algorithm:
	- randomly initialise k points in the dataset
	- assign each datapoint to the closest seed. Points assigned to the same seed form a cluster
	- recompute mean for each cluster
	- repeat 2 and 3 until convergence


![[Screenshot 2023-10-25 at 12.12.19 AM.png]]

complexity = $O(ndkt)$

- Limitations
	- the final clusters depend on the initial seeds
	- Assumes that the clusters are spherical
	- does not guarantee global optimal solution


To address the problem of producing only spherical shaped, [[EM-Clustering]] is used.

### EM-Clustering

- Expectation-Maximisation clustering is soft clustering (a point can belong to more than 1 cluster).
- It assumes that each cluster has a Gaussian or a multinomial distribution with parameters $\mu_i , \sum_i$
- It also assumes that each point is generated not by single cluster but a mixture of clusters, denoted by $P(C_i)$ 
	$\sum_{i=1}^{k}P(C_i) = 1$

- The aim of EM-clustering is to find the parameters of these clusters $\mu_i, \sum_i, P(C_i)$
- usually produces elliptical shaped clusters


EM in 1 dimension
- For each cluster initialise mean = random, variance = 1\ and $P(C_i) = \frac{1}{k}$
- Expectation step: Compute $w_{ij}$
	$$w_{ij} = P(C_i/x_j)=\frac{P(x_j/C_i).P(C_i)}{\sum_{a=1}^{k}P(x_j/C_a).P(C_a)}$$

- maximisation step : recompute the new means and new variance and cluster probabilities
	$$\mu_i = \frac{\sum_{j=1}^{n}w_{ij}.x_j}{\sum_{j=1}^{n}w_{ij}}$$
	$$\sigma_i^2 = \frac{\sum_{j=1}^{n}w_{ij}.(x_j-\mu_i)^2}{\sum_{j=1}^{n}w_{ij}}$$

	$$P(C_i) = \frac{\sum_{j=1}^{n}w_{ij}}{n}$$
	


EM Algorithm
- start with 'k' randomly placed Gaussians : $(\mu_1,\sigma_{1}^2), (\mu_2,\sigma_{2}^2)...(\mu_k,\sigma_{k}^2)$
- for each point calculate the probability that it came from cluster $C_i$ and assign the points to the cluster with which it has the max probability.
- adjust $(\mu_1,\sigma_{1}^2), (\mu_2,\sigma_{2}^2)...(\mu_k,\sigma_{k}^2)$ to fit the points assigned to them
- iterate until convergence


![[Clustering.pdf]]


for more than 1-D
$$P(C_i|x_j) = \frac{P(x_j|C_i).P(C_i)}{\sum_{a=1}^{k}P(x_j|C_a).P(C_a)} = w_{ij}
$$

where $$P(x_j|C_i)= \frac{1}{\sqrt{2\pi \sigma^2}}e^\frac{-(x_i-\mu_b)^2}{2\sigma_b^2}$$
finally $$P(C_i) = \frac{\sum_{j=1}^{n}w_{ij}}{n}$$

EM pseudocode

![[Screenshot 2023-12-07 at 8.30.16 AM.png]]





k means is a special case of EM algorithm:



Limitations
- has a hard time finding true clusters for non-convex situation or non- elliptical shaped clusters


To overcome this limitation we use [[Density based clustering - DBSCAN]].


Reference:
https://www.youtube.com/watch?v=REypj2sy_5U






