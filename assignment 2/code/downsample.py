import numpy as np

# Downsample with uniform chance of emission
def downsample(data: np.array, factor: float) -> np.array:

    assert data.ndim == 2, "Data must be a 2D array (channels, time)"
    
    # Calculate the number of samples to keep
    time_len = data.shape[1]
    num_samples = int(time_len * factor)

    # Ensure the samples are not zero to prevent later errors
    if(num_samples <= 0):
        raise ValueError("Downsampling factor too small, results in zero samples.")
    
    indices = np.sort(np.random.choice(time_len, num_samples, replace = False))
    return data[:, indices]

# Below probably doesn't work anymore because the function 'downsample' treated the data as a 1D
# array, which is not the case with the MEG data we got.
# if __name__ == "__main__":
#     # Example usage
#     data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
#     factor = .5
#     downsampled_data = downsample(data, factor)
#     print(f"Original data ({data.shape}):", data)
#     print(f"Downsampled data ({downsampled_data.shape}):", downsampled_data)