import torch
from torch.nn import Linear
import torch.nn.functional as F

# MLP网络定义
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        # torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)
    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

# MLP网络训练函数
def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x)   #输入初始节点表征：torch.Size([2708, 1433])
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
    
      return loss
    
# MLP网络测试函数
def test():
      model.eval()
      out = model(data.x)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        
      return test_acc, out


model = MLP(hidden_channels=16)
#MLP(
#  (lin1): Linear(in_features=1433, out_features=16, bias=True)
#  (lin2): Linear(in_features=16, out_features=7, bias=True)
#)

# 训练
criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss结合有LogSoftmax和NLLLoss两个类
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# 测试与可视化
test_acc, out = test()
print(f'Test Accuracy: {test_acc:.4f}')


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
%matplotlib inline

def visualize_two(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    
fig, ax1 = plt.subplots(figsize=(20, 10))
plt.subplot(1, 2, 1)
visualize_two(data.x, color=data.y) # MLP分类结果可视化
plt.title("Origin")
plt.subplot(1, 2, 2)
visualize_two(out, color=data.y) # 原始数据的可视化
plt.title("MLP")
plt.show()