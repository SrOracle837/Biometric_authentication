import cv2
import numpy as np
import pickle
import hashlib

def gabor_filter(image, wavelength, orientation, kernel_size):
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), wavelength, orientation, kernel_size/3, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(image, cv2.CV_8UC1, kernel)
    return filtered

image = cv2.imread('wrong_print1.tif', 0)

# Apply Gabor filter 
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
cv2.imwrite('test_gabor_image.jpg', features[-4])
print('gabor features',len(features[-4]))
print(final_values[-4])

degree = 10
# Flatten the features and perform Lagrange interpolation to generate the polynomial coefficients
feature_data = np.array([f.flatten() for f in features])
coefficients = np.zeros(degree + 1)
for i in range(len(feature_data)):
    term = feature_data[i]
    for j in range(len(feature_data)):
        if j != i:
            term = term*((i - j) / (i - j))

    m = 11
    x = int(np.ceil(term.size / m))
    diff = (x*11)-len(term)
    term = np.append(term,[0 for i in range(diff)])
    term = np.array(term).reshape((11, int(len(term)/11)))
    
    term_sum = np.sum(term, axis=1)
    coefficients += term_sum
coefficients /= np.max(coefficients)
print('coeff : ', coefficients)

def store(vault, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vault, f)


def custom_prf(key, data, output_len):
    concatenated = key + data

    output = b''
    while len(output) < output_len:
        hash_value = hashlib.sha256(concatenated).digest()
        output += hash_value
        concatenated = hash_value

    output = output[:output_len]

    return output


def add_noise(features, key, noise_sd=0.1):
    output_len = 256
    noise = np.frombuffer(custom_prf(key, features, output_len), dtype=np.uint8)
    noise = noise[:features.size]
    noise = noise.reshape(features.shape) / 255.0 * noise_sd
    return features + noise


# Add PRF output to the polynomial coefficients
key = [100, 200, 300, 400, 1, 2, 3, 4, 5, 6, 7]
noisy_coefficients = add_noise(coefficients, key)

# Store the noisy coefficients in a secure vault
store(noisy_coefficients, 'test_vault.pickle')
