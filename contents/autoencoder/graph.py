import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GraphAutoencoder, self).__init__()

        # Graph Encoder layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

        # Graph Decoder layers
        self.conv3 = GCNConv(embedding_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, input_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def decode(self, x, edge_index):
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

    def forward(self, x, edge_index):
        # Encode
        encoding = self.encode(x, edge_index)
        # Decode
        reconstructed = self.decode(encoding, edge_index)

        return reconstructed


class GNNAutoencoder(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNAutoencoder, self).__init__(aggr='mean')
        self.conv1 = GNNLayer(in_channels, out_channels)
        self.conv2 = GNNLayer(out_channels, in_channels)

    def forward(self, x, edge_index):
        # 그래프 자동인코더의 순전파 함수
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='mean')
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 그래프 레이어의 순전파 함수
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x)
        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_j, edge_index, size):
        # 메시지 함수: 이웃 노드의 특성을 집계
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return x_j * norm.view(-1, 1)


class GATAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATAutoencoder, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads)
        self.conv2 = GATConv(out_channels, in_channels, heads=heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return


class GraphSAGEAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEAutoencoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x