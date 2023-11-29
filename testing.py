from prf import *
from authentication import *
import os

filepath = 'CrossMatch_Sample_DB'
files = [file for file in os.listdir(filepath)]
accuracy_dict = {}
for i in range(51):
    set8 = files[:8]
    print(len(set8))
    files = files[8:]
    set8_acc = []
    for file in set8:
        og_vault_path = vault_gen(f'CrossMatch_Sample_DB/{file}', 0)      #creating vault
        test_vault_path = vault_gen(f'CrossMatch_Sample_DB/{file}', 1)      #testing vault
        result = auth(og_vault_path, test_vault_path)
        set8_acc.append(result)
    accuracy_dict[f'Set_{i}'] = sum(set8_acc)/8 

print(accuracy_dict)
