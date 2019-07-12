from librosaprocess import process
from multifileprocess import remove_nums, multiprocess, data_label_write, data_label_load

print(remove_nums("hey/123aa1239.mp3"))

assert remove_nums("hey/123aa1239.mp3") == "aa"

files = []
for i in range(1,11):
    files.append(f"../../testaudio/ru{i}.mp3")

all_data, labels = multiprocess(files)
data_label_write(all_data,labels,"./saveddata/data")
d2,l2 = data_label_load("./saveddata/data")
assert labels == l2
