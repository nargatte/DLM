from get_dataset import * 
from collections import Counter
ds_train = get_dataset("train")
l = 0
c = Counter([y.numpy() for x, y in ds_train])
for x, y in ds_train:
    c[y.numpy()] += 1
    l = l + 1
    if l % 1000 == 0:
        print(f"### {l}")
print(f"### Found {l} records in train")
print(c)
ds_test = get_dataset("test")
l = 0
c = Counter([y.numpy() for x, y in ds_train])
for x, y in ds_test:
    c[y.numpy()] += 1
    l = l + 1
print(f"### Found {l} records in test")
print(c)
print(next(iter(ds_test.element_spec)))