#### https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582
import torch
import torch.nn as nn
import math

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size

        self.conv = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=4 * out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(num_features=4 * out_channels)
        self.w_ci = nn.Parameter(torch.Tensor(out_channels, *input_size))
        self.w_cf = nn.Parameter(torch.Tensor(out_channels, *input_size))
        self.w_co = nn.Parameter(torch.Tensor(out_channels, *input_size))

        self.initialize()

    def forward(self, x, hidden_prev, cell_prev):
        conv_out = self.conv(torch.cat([x, hidden_prev], dim=1))
        conv_out = self.bn(conv_out)
        input_conv, forget_conv, cell_conv, output_conv = torch.chunk(conv_out, chunks=4, dim=1)

        input_gate = torch.sigmoid(input_conv + self.w_ci * cell_prev)
        forget_gate = torch.sigmoid(forget_conv + self.w_cf * cell_prev)
        cell_current = forget_gate * cell_prev + input_gate * torch.tanh(cell_conv)
        output_gate = torch.sigmoid(output_conv + self.w_co * cell_current)
        hidden_current = output_gate * torch.tanh(cell_current)

        return hidden_current, cell_current

    def initialize(self):
        n = self.kernel_size * self.kernel_size * self.out_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / n))
        self.w_ci.data.normal_(0, math.sqrt(2. / n))
        self.w_cf.data.normal_(0, math.sqrt(2. / n))
        self.w_co.data.normal_(0, math.sqrt(2. / n))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

class ConvLSTMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size):
        super(ConvLSTMLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size

        # Will unroll this over time steps
        self.convlstmcell = ConvLSTMCell(in_channels, out_channels, kernel_size, input_size)

    def forward(self, x):

        # Get the dimensions
        batch_size, num_timesteps, num_channels, height, width = x.size()

        # Initialize output, hidden, cell
        output = torch.zeros(batch_size, num_timesteps, self.out_channels, height, width, device=x.device)
        hidden = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
        cell = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)

        # Unroll over time steps
        for time_step in range(num_timesteps):
            hidden, cell = self.convlstmcell(x[:, time_step, :, :, :], hidden, cell)
            output[:, time_step, :, :, :] = hidden
        return output

class ConvLSTM3(nn.Module):
    def __init__(self, num_classes=1000, num_timesteps=1):
        super(ConvLSTM3, self).__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        self.convlstm1 = ConvLSTMLayer(in_channels=3, out_channels=64, kernel_size=3, input_size=(100, 100))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.convlstm2 = ConvLSTMLayer(in_channels=64, out_channels=64, kernel_size=3, input_size=(50, 50))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.convlstm3 = ConvLSTMLayer(in_channels=64, out_channels=64, kernel_size=3, input_size=(25, 25))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.linear1 = nn.Linear(64*12*12, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1).repeat(1, self.num_timesteps, 1, 1, 1)

        x = self.convlstm1(x)
        x = self.maxpool1(x)
        x = self.convlstm2(x)
        x = self.maxpool2(x)
        x = self.convlstm3(x)
        x = self.maxpool3(x)

        outputs = []
        for t in range(self.num_timesteps):
            out = x[:,t,:,:,:]
            out = torch.flatten(out, 1)
            out = self.linear1(out)
            out = self.relu1(out)
            out = self.dropout(out)
            out = self.linear2(out)
            outputs.append(out)

        return outputs