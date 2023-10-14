import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import sys

output_file = "assign2-iusaakell.txt"

with open(output_file, "w") as f:
    sys.stdout = f

    if len(sys.argv) < 3:
        print(f"Please provide a file path as a command-line argument.")
    else:
        # Get the file path from the command-line argument (sys.argv[1])
        file_path = sys.argv[1]
        

        try:
            # Read the data
            data = pd.read_csv(file_path)

            # Read 'k' value
            k= int(sys.argv[2])
            print(f"Data successfully loaded from: {file_path}")
            print('\n')
        except FileNotFoundError:
            print(f"File not found: {file_path}")

    #rename columns
    data.columns= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']

    #Extract features and labels
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    Y= data['label']

    n,d= X.shape
    subset_size = n // k

    # initialising means and covariances for each cluster
    initial_means = []
    initial_covariances = []

    for i in range(k):
        start = i * subset_size
        end = (i + 1) * subset_size
        subset = X[start:end]
    
        initial_means.append(np.mean(subset,axis=0))
        initial_covariances.append(np.cov(subset, rowvar=False))

    # EM Algorithm
    gmm = GaussianMixture(n_components=k, means_init=initial_means, precisions_init=initial_covariances, random_state=0,tol=1e-6)
    gmm.fit(X)
    cluster_assignments = gmm.predict(X)

    # final mean
    final_means = gmm.means_
    sorted_means = final_means[np.argsort(np.linalg.norm(final_means, axis=1))]

    print('a. Mean')
    for mean in sorted_means:
        print(str(mean))

    # final covariance
    final_covariances = gmm.covariances_
    sorted_covariances = final_covariances[np.argsort(np.linalg.norm(final_means, axis=1))]

    print("\nb. Covariance Matrices:")
    for covariance in sorted_covariances:
        print(str(covariance))

    # no of iterations to converge
    # n_iter = 0
    # while not gmm.converged_:
    #     gmm.step()
    #     n_iter += 1

    iteration_count = gmm.n_iter_
    print("\nc. Iteration count =", iteration_count)  

    # cluster probabilities for each datapoint
    cluster_probabilities = gmm.predict_proba(X)

    # Cluster membership
    print("\nd. Cluster Membership:")
    cluster_membership = [[] for _ in range(k)]

    for i, prob in enumerate(cluster_probabilities):
        max_cluster = np.argmax(prob)
        cluster_membership[max_cluster].append(i)

    for i, members in enumerate(cluster_membership):
        members_sorted = sorted(members)
        members_string = ', '.join(map(str, members_sorted))
        print(f"Cluster {i + 1}: {members_string}")  

    # Finding size of clusters
    cluster_sizes = [np.sum(cluster_assignments == i) for i in range(k)]
    size= ' '.join(map(str, cluster_sizes))
    print("\ne. Size:")
    print(size)

    # string to categorical variable
    label_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
    }
    numerical_labels = np.array([label_mapping[label] for label in Y])

    # purity
    conf_matrix = confusion_matrix(numerical_labels, cluster_assignments)
    purity = np.sum(np.max(conf_matrix, axis=0)) / len(data)

    print("\nPurity Score:")
    print(purity)
