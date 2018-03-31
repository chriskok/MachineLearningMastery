# Load CSV using Pandas from URL
import pandas
url = "https://goo.gl/vhm1eU"
names = ['preg','plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
#print(data)
print(data.shape)
print("size: %d" % data.size)

# for x in range(0,3):
#     print (data.ix[x])
