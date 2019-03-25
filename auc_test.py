import pandas as pd

sub = pd.read_csv("C:\\Users\\18140\Desktop\\af2019-sr-devset-20190312\\submit.csv")
ano = pd.read_csv("C:\\Users\\18140\Desktop\\af2019-sr-devset-20190312\\annotation.csv")

data = pd.merge(sub , ano , on="FileID")

Y = data[((data["IsMember_x"] == 'Y') & (data["IsMember_y"] == 'Y'))].count()
N = data[((data["IsMember_x"] == 'N') & (data["IsMember_y"] == 'N'))].count()

print((Y+N)/len(data))