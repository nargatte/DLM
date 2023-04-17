from get_dataset import * 
ds_train = get_dataset("train")
l = 0
for x in ds_train:
    l = l + 1
    if l % 1000 == 0:
        print(f"### {l}")
print(f"### Found {l} records in train")
ds_test = get_dataset("test")
l = 0
for x in ds_test:
    l = l + 1
print(f"### Found {l} records in test")
print(next(iter(ds_test.element_spec)))