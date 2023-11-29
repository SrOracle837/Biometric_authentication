import cv2
import numpy as np
import pickle

def gabor_filter(image, wavelength, orientation, kernel_size):
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), wavelength, orientation, kernel_size/3, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(image, cv2.CV_8UC1, kernel)
    return filtered


# Load the fingerprint image
image = cv2.imread('fingerprint.tif', 0)

# Apply Gabor filter to the image
wavelengths = [10, 20, 30]
orientations = [0, 45, 90, 135]
kernel_size = 9
features = []
final_values = []
for wavelength in wavelengths:
    for orientation in orientations:
        filtered = gabor_filter(image, wavelength, orientation, kernel_size)
        features.append(filtered)
        final_values.append((wavelength,orientation))

features = np.array(features)
# cv2.imshow('features', features[-4])
# cv2.waitKey(0)
cv2.imwrite('user_gabor_image.jpg', features[-4])
print('gabor features', len(features[-4]))
print(final_values[-4])

degree=10
# Flatten the features and perform Lagrange interpolation to generate the polynomial coefficients
feature_data = np.array([f.flatten() for f in features])
coefficients = np.zeros(degree + 1)
for i in range(len(feature_data)):
    term = feature_data[i]
    for j in range(len(feature_data)):
        if j != i:
            term = term*((i - j) / (i - j))
    term = np.array(term).reshape((11, 13192))
    term_sum = np.sum(term, axis=1)
    coefficients += term_sum

    # coefficients += term
coefficients /= np.max(coefficients)
print('coeff : ', coefficients)

def store(vault, filename):
    """
    Saves the fuzzy vault to a file.

    Parameters:
    vault (list): The fuzzy vault to be saved.
    filename (str): The name of the file to save the fuzzy vault to.

    Returns:
    None
    """

    with open(filename, 'wb') as f:
        pickle.dump(vault, f)


# Add noise to the polynomial coefficients
noise_mean = 0
noise_stddev = 0.05
noise = np.random.normal(noise_mean, noise_stddev, degree + 1)
noisy_coefficients = coefficients + noise

# Store the noisy coefficients in a secure vault
store(noisy_coefficients, 'user_vault.pickle')
