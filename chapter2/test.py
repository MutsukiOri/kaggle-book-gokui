# from sklearn.datasets import load_digits

# digits = load_digits()
# X = digits.data
# y = digits.target
# print(X.shape, y.shape)

# import torch

# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.int64)

# from torch import nn, optim

# model = nn.Sequential(
#   nn.Linear(64, 32),
#   nn.ReLU(),
#   nn.Linear(32, 16),
#   nn.ReLU(),
#   nn.Linear(16, 10),
# )
# model.train()
# lossfun = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# losses = []

# for eq in range(100):
#   optimizer.zero_grad()
#   # yの予測値を算出
#   out = model(X)

#   # 損失を計算
#   loss = lossfun(out, y)
#   loss.backward()

#   # 勾配を更新
#   optimizer.step()

#   losses.append(loss.item())

# _, pred = torch.max(out, 1)
# print((pred == y).sum().item() / len(y))

# import matplotlib.pyplot as plt

# plt.plot(losses)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('test.png')


# import torch
# from sklearn.datasets import load_digits
# from torch.utils.data import TensorDataset, DataLoader

# digits = load_digits()

# X = digits.data
# y = digits.target

# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.int64)
# dataset = TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
# print(torch.cuda.is_available())


import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

digits = load_digits()
X = digits.data
y = digits.target

# train　と　test　を分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# train から valid を切り出す
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_dataset = TensorDataset(X_valid, y_valid)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


