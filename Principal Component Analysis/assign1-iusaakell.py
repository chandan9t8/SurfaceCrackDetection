import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

output_file = "assign1-iusaakell.txt"

with open(output_file, "w") as f:
    sys.stdout = f

    # Check if file path is provided in the command-line argument
    if len(sys.argv) < 2:
        f.write(f"Please provide a file path as a command-line argument.")
    else:
        # Get the file path from the command-line argument (sys.argv[1])
        file_path = sys.argv[1]

        try:
            original_data = pd.read_csv(file_path)
            f.write(f"Data successfully loaded from: {file_path}")
            f.write('\n')
        except FileNotFoundError:
            f.write(f"File not found: {file_path}")

    # f.write(original_data.head(5))
    # f.write("Dimensions of the data: ",original_data.shape)
    # f.write("Columns in the data",original_data.columns)

    #drop categorical column
    data=original_data.drop('g', axis=1)
    #f.write(data.columns)

    #(a)-applying z-normalization
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis =0 )
    data_normalized = (data-mean)/std
    f.write('\n')
    f.write("(a)------- Z-normalised data")
    f.write(str(data_normalized))
    f.write('\n')

    #converting pandas dataframe to numpy array
    numpy_array = data_normalized.values

    #generating covariance matrix
    covariance_matrix1 = (numpy_array.T @ data_normalized)/(len(data_normalized)-1)
    covariance_matrix1.columns = [''] * len(covariance_matrix1.columns)

    f.write('\n')
    f.write("(b)------- Sample covariance matrix without using np.cov()")
    f.write(str(covariance_matrix1))
    f.write('\n')

    covariance_matrix2 = np.cov(data_normalized, rowvar=False, bias=False)
    covariance_matrix2 = pd.DataFrame(covariance_matrix2)
    f.write('Covariance matrix using np.cov()')

    covariance_matrix2.columns = [''] * len(covariance_matrix2.columns)
    covariance_matrix2 = pd.DataFrame(covariance_matrix2)

    # covariance_matrix1.index.equals(covariance_matrix2.index)
    # covariance_matrix1.columns.equals(covariance_matrix2.columns)
    # covariance_matrix1.dtypes.equals(covariance_matrix2.dtypes)
    # np.array_equal(covariance_matrix1.values, covariance_matrix2.values)

    comparison_df = pd.concat([covariance_matrix1, covariance_matrix2], axis=1, keys=['df', 'other_df'])
    f.write(str(comparison_df))

    are_equal = covariance_matrix1.equals(pd.DataFrame(covariance_matrix2))

    # if are_equal:
    #     f.write('The two DataFrames are equal.')
    # else:
    #     f.write('The two DataFrames are not equal.')


    #(c)- power-iteration method

    num_iterations = 1000
    convergence_threshold = 0.000001

    eigenvector = np.random.rand(covariance_matrix1.shape[0])

    for i in range(num_iterations):

        prev_eigenvector = eigenvector
        product = np.dot(covariance_matrix1, eigenvector)
        eigenvector = product / np.linalg.norm(product)
        norm_diff = np.linalg.norm(eigenvector - prev_eigenvector)
        
        if norm_diff < convergence_threshold:
            break

    dominant_eigenvalue = np.dot(eigenvector, np.dot(covariance_matrix1, eigenvector)) / np.dot(prev_eigenvector, eigenvector)

    f.write('\n')
    f.write("(c)------- Power iteration")
    f.write('\n')
    f.write(f"Largest Eigenvalue: {dominant_eigenvalue}")
    f.write('\n')
    f.write(f"Corresponding Eigenvector: {eigenvector}")


    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix1)
    dominant_eigenvalue_linalg = max(eigenvalues)
    f.write('\n')
    f.write('np.linalg.eig()')
    f.write('\n')
    f.write(f"Dominant Eigenvalue: {dominant_eigenvalue_linalg}")
    

    #checking if results are similar
    if np.isclose(dominant_eigenvalue, dominant_eigenvalue_linalg, atol=1e-6):
        f.write('\n')
        f.write("Eigen values are the same")
    else:
        f.write('\n')
        f.write("Eigen values are different")

    #(d)-Variance of the datapoints in projected subspace

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix1)

    # Sorting eigenvalues and eigenvectors in descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Selecting the first two dominant eigenvectors
    first_eigenvector = eigenvectors[:, 0]
    second_eigenvector = eigenvectors[:, 1]

    # projecting the data points on the subspace spanned by the first two eigenvectors
    projected_data = np.dot(numpy_array, np.column_stack((first_eigenvector, second_eigenvector)))

    # Compute the variance of the data points in the projected subspace
    variance = np.var(projected_data, axis=0)

    # f.write the variance
    f.write('\n')
    f.write('\n')
    f.write(f"(d)---- Variance in the projected subspace: {variance}")


    #(e) - eigen decomposition form

    reconstructed_Sigma = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), eigenvectors.T)
    f.write('\n')
    f.write("\n(e)------- Reconstructed Covariance Matrix (UΛUT):")
    f.write('\n')
    f.write(str(reconstructed_Sigma))

    #(f) - mean square error
    def calculate_mse(original_data, projected_data):
        reconstructed_data = projected_data @ eigenvectors[:, :2].T
        squared_diff = (original_data - reconstructed_data)**2
        mse = np.mean(squared_diff)
        
        return mse


    n_components = 2
    mse = calculate_mse(numpy_array, projected_data)
    f.write('\n')
    f.write('\n')
    f.write(f'(f)---- The mean square error is: {mse}')


    selected_eigenvalues = eigenvalues[2:]  
    sum_of_selected_eigenvalues = np.sum(selected_eigenvalues)/np.sum(eigenvalues)
    f.write('\n')
    f.write(f'The sum of the Eigen values except the 1st two is: {sum_of_selected_eigenvalues}')


    #checking if results are similar
    if np.isclose(mse, sum_of_selected_eigenvalues, atol=1e-6):
        f.write('\n')
        f.write("MSE is the same")
    else:
        f.write('\n')
        f.write("MSE is different")

    projected_data = pd.DataFrame(projected_data)
    new_data = pd.concat([projected_data, original_data['g']], axis=1)

        #(h) -- Principal component analysis
    def pca(standardized_data, var_threshold):
        
        cov_matrix = np.cov(standardized_data, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)

        n_components = np.argmax(explained_variance_ratio >= var_threshold) + 1

        top_eigenvectors = eigenvectors[:, :n_components]

        # Project the data onto the new basis
        projected_data = np.dot(standardized_data, top_eigenvectors)

        return top_eigenvectors, projected_data


    # Perform PCA with a 95% variance threshold
    principle_vectors, new_basis_data = pca(numpy_array, var_threshold=0.95)

    # f.write the first 10 data points in the new basis
    f.write('\n')
    f.write('\n')
    f.write("(h)------- Coordinates of the first 10 data points in the new basis vectors:")
    f.write('\n')
    f.write(str(new_basis_data[:10]))

x_column_index = 0 
y_column_index = 1 
class_label = 2

# Extract the numerical columns and class labels
numerical_columns = new_data.iloc[:, :2]
class_labels = new_data['g']

# Create a scatter plot
plt.figure(figsize=(8, 6))  # Set the figure size

# Scatter plot for class 'g'
plt.scatter(
    numerical_columns[class_labels == 'g'][0],  # X-axis values for class 'g'
    numerical_columns[class_labels == 'g'][1],  # Y-axis values for class 'g'
    label='Class g',  # Label for class 'g'
    c='blue',  # Color for class 'g'
    marker='o'  # Marker style
)

# Scatter plot for class 'h'
plt.scatter(
    numerical_columns[class_labels == 'h'][0],  # X-axis values for class 'h'
    numerical_columns[class_labels == 'h'][1],  # Y-axis values for class 'h'
    label='Class h',  # Label for class 'h'
    c='red',  # Color for class 'h'
    marker='x'  # Marker style
)

# Set labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Features by Class')

# Show legend
plt.legend()

# Show the plot
plt.show()

