import numpy as np

def downsample(data: np.array, factor: float) -> np.array:
    # Downsample with uniform chance of emission
    
    # Calculate the number of samples to keep
    num_samples = int(len(data) * factor)
    # Generate random indices to select samples
    indices = np.random.choice(len(data), num_samples, replace=False)
    # Select the samples using the random indices
    downsampled_data = data[indices]
    # Sort the indices to maintain the original order
    downsampled_data.sort()
    return downsampled_data


if __name__ == "__main__":
    # Example usage
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    factor = .5
    downsampled_data = downsample(data, factor)
    print(f"Original data ({data.shape}):", data)
    print(f"Downsampled data ({downsampled_data.shape}):", downsampled_data)