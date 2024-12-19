import torch
import numpy as np
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        #x = self.lin(x)

        return x


class EEGNetLSTM_v1(nn.Module):
    def __init__(self, eegnet_params, lstm_hidden_dim, num_classes=1):
        super(EEGNetLSTM_v1, self).__init__()
        self.eegnet = EEGNet(**eegnet_params)
        self.eeg_feature_dim = self.eegnet.feature_dim()
        print('Seqs_bins: Features dim: ', self.eeg_feature_dim)
        self.lstm = nn.LSTM(input_size=self.eeg_feature_dim+1, hidden_size=lstm_hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x, prev_label):
        batch_size, num_seqs, channels, time_steps = x.shape
        # Redimensionar x para dividirlo en num_sequences
        x = x.view(batch_size * num_seqs, channels, time_steps)
        # (batch_size * num_sequences, 1, channels, sequence_length)
        x = x.unsqueeze(1)
        # (batch_size * num_sequences, eegnet_output_dim)
        eegnet_features = self.eegnet(x)  # (batch_size * sequence_size, eegnet_output_dim)
        # (batch_size, num_sequences, eegnet_output_dim)
        eegnet_features = eegnet_features.view(batch_size, num_seqs, -1)
        # prev_label: (batch_size, num_sequences, 1)
        #prev_label = prev_label.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_sequences, 1)
        concatenated = torch.cat((eegnet_features, prev_label), dim=2)
        lstm_output, _ = self.lstm(concatenated)
        final_output = lstm_output[:, -1, :]
        return self.fc(final_output)


class EEGNetLSTMwinLabels(nn.Module):
    def __init__(self, eegnet_params, lstm_hidden_dim, num_classes=1):
        super(EEGNetLSTMwinLabels, self).__init__()
        self.eegnet = EEGNet(**eegnet_params)
        self.eeg_feature_dim = self.eegnet.feature_dim()
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x, aGGObserved):
        x = x.unsqueeze(1)
        eegnet_features = self.eegnet(x)
        aGGObserved = aGGObserved.unsqueeze(1)
        concatenated = torch.cat((eegnet_features, aGGObserved), dim=1)
        concatenated = concatenated.unsqueeze(1)
        lstm_input = concatenated.permute(0, 2, 1)
        lstm_output, _ = self.lstm(lstm_input)
        final_output = lstm_output[:, -1, :]
        return self.fc(final_output)