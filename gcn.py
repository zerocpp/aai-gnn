# GCN
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# GCN网络定义
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# GCN网络训练函数   
def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

# GCN网络测试函数   
def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc, out
    


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_two(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")


model = GCN(hidden_channels=16)
#GCN(
#  (conv1): GCNConv(1433, 16)
#  (conv2): GCNConv(16, 7)
#)

# GCN未经训练时的输出——节点表征，及可视化
model.eval()
out = model(data.x, data.edge_index)

plt.figure(1)
fig, ax1 = plt.subplots(figsize=(20, 10))
plt.subplot(1, 2, 1)
visualize_two(data.x, color=data.y) # MLP分类结果可视化
plt.title("Origin")
plt.subplot(1, 2, 2)
visualize_two(out, color=data.y) # 原始数据的可视化
plt.title("GCN Before Train")
plt.show()


# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# 测试与可视化
test_acc, out = test()
print(f'Test Accuracy: {test_acc:.4f}')

plt.figure(2)
fig, ax1 = plt.subplots(figsize=(20, 10))
plt.subplot(1, 2, 1)
visualize_two(data.x, color=data.y) # MLP分类结果可视化
plt.title("Origin")
plt.subplot(1, 2, 2)
visualize_two(out, color=data.y) # 原始数据的可视化
plt.title("GCN After Train")
plt.show()