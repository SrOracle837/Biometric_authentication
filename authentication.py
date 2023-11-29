import pickle
import numpy as np

def retrieve(filename):
    with open(filename, 'rb') as f:
        vault = pickle.load(f)

    return vault

# # Retrieve from the vault
# vault = retrieve('og_vault.pickle')
# print('OG vault : ', vault)

# user_vault = retrieve('test_vault.pickle')
# print('Test_vault : ', user_vault)

# # Compare the stored polynomial coefficients with the user's coefficients
# threshold = 0.1
# difference = np.abs(vault - user_vault)
# if np.max(difference) < threshold:
#     print('Authentication successful')
# else:
#     print('Authentication failed')


def auth(og_vault_path, test_vault_path):
    # Retrieve from the vault
    og_vault = retrieve(og_vault_path)
    print('OG vault : ', og_vault)

    test_vault = retrieve(test_vault_path)
    print('Test vault : ', test_vault)
    # Compare the stored polynomial coefficients with the user's coefficients
    threshold = 0.1
    difference = np.abs(og_vault - test_vault)
    auth_result = 10
    if np.max(difference) < threshold:
        # print('Authentication successful')
        auth_result=1
    else:
        # print('Authentication failed')
        auth_result=0

    return auth_result