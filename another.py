import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor
import torch.utils.data as Data
import math
from matplotlib import pyplot
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import math
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负

# 设置随机参数：保证实验结果可以重复
SEED=5000
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) # 适用于显卡训练
torch.cuda.manual_seed_all(SEED) # 适用于多显卡训练
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# 用30天的数据(包括这30天所有的因子和log_ret)预测下一天的log_ret
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv("data\ETTh1.csv")  # 1 3 7 是 预测列
print(data.columns)
['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
data.dropna(axis=0, how='any')
data_x = data[
    ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values
# ax=plt.subplots(figsize=(50,50))
# ax=sns.heatmap(data.corr(),vmax=.1,square=True,annot=True,annot_kws={"fontsize":8})
# plt.show()
# print(len(data_y))
# 31个数据划分为一组 用前30个预测后一个

data_31_x = []
data_31_y = []
window_size = 96  # 定义窗口大小

# for i in range(len(data_x) - window_size):
#     data_31_x.append(data_x[i:i + window_size])  # 包含95天的数据
#     data_31_y.append(data_x[i + window_size][-1])  # 第96天的OT值
# for i in range(len(data_x) - window_size - 239):  # 减去额外的239天
#     data_31_x.append(data_x[i:i + window_size])  # 包含96天的数据
#     data_31_y.append(data_x[i + window_size + 239][-1])  # 第336天的OT值
for i in range(len(data_x) - 432):
    data_31_x.append(data_x[i:i + window_size])  # 输入：96小时
    data_31_y.append(data_x[i + window_size:i + window_size +336])  # 输出：336小时

print(len(data_31_y), len(data_31_y[0]))
x_train, x_temp_test, y_train, y_temp_test = train_test_split(np.array(data_31_x), np.array(data_31_y), test_size=0.4, shuffle=False,random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_temp_test, y_temp_test, test_size=0.5,shuffle=False, random_state=1)
# print(len(x_train),x_train)
# print(len(x_test),x_test)
# print(len(x_val),x_val)
class DataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        #data_inputs=data_inputs.astype(float)
        self.inputs = torch.FloatTensor(data_inputs)
        self.label = torch.FloatTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)
Batch_Size = 8  #
#DataSet = DataSet(np.array(x_train), list(y_train))
#train_size = int(len(x_train) * 0.7)
#test_size = len(y_train) - train_size
#train_dataset, test_dataset = torch.utils.data.random_split(DataSet, [train_size, test_size])
train_dataset = DataSet(x_train, y_train)
val_dataset = DataSet(x_val, y_val)
test_dataset = DataSet(x_test, y_test)
TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)
ValDataLoader = Data.DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        chunk = x.chunk(x.size(-1), dim=2)
        out = torch.Tensor([]).to(x.device)
        for i in range(len(chunk)):
            out = torch.cat((out, chunk[i] + self.pe[:chunk[i].size(0), ...]), dim=2)
        return out


def transformer_generate_tgt_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class Transformer(nn.Module):
    """标准的Transformer编码器-解码器结构"""

    def __init__(self, n_encoder_inputs, n_decoder_inputs, Sequence_length, d_model=512, dropout=0.1, num_layer=8):
        """
        初始化
        :param n_encoder_inputs:    输入数据的特征维度
        :param n_decoder_inputs:    编码器输入的特征维度，其实等于编码器输出的特征维度
        :param d_model:             词嵌入特征维度
        :param dropout:             dropout
        :param num_layer:           Transformer块的个数
         Sequence_length:           transformer 输入数据 序列的长度
        """
        super(Transformer, self).__init__()

        self.input_pos_embedding = torch.nn.Embedding(500, embedding_dim=d_model)
        self.target_pos_embedding = torch.nn.Embedding(500, embedding_dim=d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_layer, dropout=dropout,dim_feedforward=4 * d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=num_layer, dropout=dropout,dim_feedforward=4 * d_model)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.input_projection = torch.nn.Linear(n_encoder_inputs, d_model)
        self.output_projection = torch.nn.Linear(n_decoder_inputs, d_model)

        self.linear = torch.nn.Linear(d_model, 1)
        self.ziji_add_linear = torch.nn.Linear(Sequence_length,1)

    def encode_in(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (torch.arange(0, in_sequence_len, device=src.device).unsqueeze(0).repeat(batch_size, 1))
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def decode_out(self, tgt, memory):
        tgt_start = self.output_projection(tgt).permute(1, 0, 2)
        out_sequence_len, batch_size = tgt_start.size(0), tgt_start.size(1)
        pos_decoder = (torch.arange(0, out_sequence_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1))
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        tgt = tgt_start + pos_decoder
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        out = self.linear(out)
        return out

    def forward(self, src, target_in):
        # print("src.shape", src.shape)
        src = self.encode_in(src)
        # print("src.shape",src.shape)#src.shape torch.Size([9, 8, 512])
        out = self.decode_out(tgt=target_in, memory=src)
        # print("out.shape",out.shape)
        # print("out.shape:",out.shape)# torch.Size([batch, 3, 1])

        # 下面是第一种方案
        # 使用全连接变成 [batch,1] 构成了基于transformer的回归单值预测
        # out = out.squeeze(2)
        # out = self.ziji_add_linear(out)
        # final_output = out[:, -1]
        return out

model = Transformer(n_encoder_inputs=7, n_decoder_inputs=7, Sequence_length=96).to(device)  # 3 表示Sequence_length  transformer 输入数据 序列的长度

def test_main(model,DataLoader):
    val_epoch_loss = []
    with torch.no_grad():
        for inputs, targets in DataLoader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.float()

            tgt_in = torch.rand((Batch_Size, 336, 7)).to(device)  # 确保tgt_in也在正确的设备上

            outputs = model(inputs, tgt_in)
            loss = criterion(outputs.float(), targets.float())
            val_epoch_loss.append(loss.item())
    return np.mean(val_epoch_loss)


epochs = 2 # 100 200
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss().to(device)

val_loss = []
train_loss = []
best_test_loss = 10000000
for epoch in tqdm(range(epochs)):
    model.train()  # 设置模型为训练模式
    train_epoch_loss = []

    for inputs, targets in TrainDataLoader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # 清除之前的梯度
        # print(targets.shape)
        # 前向传播
        tgt_in = torch.rand((Batch_Size, 336, 7)).to(device)
        # print(tgt_in.shape)
        outputs = model(inputs, tgt_in)
        # print(outputs.shape)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
    train_loss.append(np.mean(train_epoch_loss))

    # 在验证集上评估模型
    model.eval()  # 设置模型为评估模式
    val_loss_epoch = test_main(model, ValDataLoader)  # 评估模型在验证集上的表现
    val_loss.append(val_loss_epoch)


    # 检查是否是最好的模型
    if val_loss_epoch < best_test_loss:
        best_test_loss = val_loss_epoch
        best_model = model
        torch.save(best_model.state_dict(), 'best_Transformer_trainModel.pth')
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss[-1]}, Val Loss = {val_loss[-1]}")



# 画一下loss图
fig = plt.figure(facecolor='white', figsize=(10, 7))
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=len(val_loss), xmin=0)
plt.ylim(ymax=max(max(train_loss), max(val_loss)), ymin=0)
# 画两条（0-9）的坐标轴并设置轴标签x，y
x1 = [i for i in range(0, len(train_loss), 1)]  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
y1 = val_loss  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
x2 = [i for i in range(0, len(train_loss), 1)]
y2 = train_loss
colors1 = '#00CED4'  # 点的颜色
colors2 = '#DC143C'
area = np.pi * 4 ** 1  # 点面积
# 画散点图
plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='val_loss')
plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='train_loss')
plt.legend()
plt.show()

# 加载模型预测------
model = Transformer(n_encoder_inputs=7, n_decoder_inputs=7, Sequence_length=96).to(device)
model.load_state_dict(torch.load('best_Transformer_trainModel.pth'))
model.to(device)
model.eval()
# 在对模型进行评估时，应该配合使用with torch.no_grad() 与 model.eval()：
y_pred = []
y_true = []

test_epoch_loss = []
with torch.no_grad():
    for inputs, targets in TestDataLoader:
        inputs, targets = inputs.to(device), targets.to(device)
        tgt_in = torch.rand((inputs.size(0), 336, 7)).to(device)  # 确保输入的批次大小匹配
        outputs = model(inputs, tgt_in)
        loss = criterion(outputs, targets)  # 直接使用 PyTorch 张量
        test_epoch_loss.append(loss.item())
        y_pred.extend(outputs.detach().cpu().numpy().flatten())  # 使用 detach() 来避免梯度跟踪
        y_true.extend(targets.cpu().numpy().flatten())
# 训练完成后，评估模型在测试集上的表现
test_loss = np.mean(test_epoch_loss)
print(f"Test Loss: {test_loss}")


y_true = np.array(y_true)
y_pred = np.array(y_pred)
print(y_true.shape)
print(y_pred.shape)
# 画折线图显示----
# num_points = 300
# real_days = range(1, num_points + 1)  # 真实值的横坐标：从1开始
# pred_days = range(96, 96 + num_points)  # 预测值的横坐标：从31开始
#
# # 使用切片操作限制数据点数量
# y_true_limited = y_true[:num_points]
# y_pred_limited = y_pred[:num_points]
#
# plt.figure(figsize=(12, 6))
# plt.plot(real_days, y_true_limited, label='True Values', color='black')  # 真实值
# plt.plot(pred_days, y_pred_limited, label='Predictions', color='red')  # 预测值
#
# plt.xlabel('Day')
# plt.ylabel('OT Value')
# plt.title('True Values and Predictions')
# plt.legend()
# plt.show()

# 确保 y_pred 和 y_true 已准备好
num_points =300 # 设置要显示的数据点数量
days = range(1, num_points + 1)  # 两条曲线共用的横坐标：第1天到第300天

# 使用切片操作限制数据点数量
y_true_limited = y_true[:num_points]  # 真实值的前300天
y_pred_limited = y_pred[96:num_points]

plt.figure(figsize=(12, 6))
plt.plot(days, y_true_limited, label='True Values', color='orange')  # 真实值的前300天
plt.plot(days[96:], y_pred_limited, label='Predictions', color='blue')

plt.xlabel('Day')
plt.ylabel('OT Value')
plt.title('True Values and Predictions for the First 300 Days')
plt.legend()
plt.show()