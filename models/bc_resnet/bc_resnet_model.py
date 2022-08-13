import torch
from torch import nn
import torch.nn.functional as F
import constants


class BcResNetModel(nn.Module):
    def __init__(
        self,
        n_class: int = len(constants.LABELS),
        *,
        scale: int = 1,
        dropout: float = constants.DROPOUT,
        use_subspectral: bool = True
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(
            1, int(16 * scale), kernel_size=(5, 5), stride=(2, 1), padding=2
        )

        self.t1 = TransitionBlock(
            int(16 * scale), int(8 * scale), dropout=dropout, use_subspectral=use_subspectral
        )
        self.n11 = NormalBlock(
            int(8 * scale), dropout=dropout, use_subspectral=use_subspectral
        )

        self.t2 = TransitionBlock(
            int(8 * scale),
            int(12 * scale),
            dilation=2,
            stride=2,
            dropout=dropout,
            use_subspectral=use_subspectral,
        )
        self.n21 = NormalBlock(
            int(12 * scale), dilation=2, dropout=dropout, use_subspectral=use_subspectral
        )

        self.t3 = TransitionBlock(
            int(12 * scale),
            int(16 * scale),
            dilation=4,
            stride=2,
            dropout=dropout,
            use_subspectral=use_subspectral,
        )
        self.n31 = NormalBlock(
            int(16 * scale), dilation=4, dropout=dropout, use_subspectral=use_subspectral
        )
        self.n32 = NormalBlock(
            int(16 * scale), dilation=4, dropout=dropout, use_subspectral=use_subspectral
        )
        self.n33 = NormalBlock(
            int(16 * scale), dilation=4, dropout=dropout, use_subspectral=use_subspectral
        )

        self.t4 = TransitionBlock(
            int(16 * scale),
            int(20 * scale),
            dilation=8,
            dropout=dropout,
            use_subspectral=use_subspectral,
        )
        self.n41 = NormalBlock(
            int(20 * scale), dilation=8, dropout=dropout, use_subspectral=use_subspectral
        )
        self.n42 = NormalBlock(
            int(20 * scale), dilation=8, dropout=dropout, use_subspectral=use_subspectral
        )
        self.n43 = NormalBlock(
            int(20 * scale), dilation=8, dropout=dropout, use_subspectral=use_subspectral
        )

        self.dw_conv = nn.Conv2d(int(20 * scale), int(20 * scale), kernel_size=(5, 5), groups=int(20 * scale))
        self.onexone_conv = nn.Conv2d(int(20 * scale), int(32 * scale), kernel_size=1)

        self.head_conv = nn.Conv2d(int(32 * scale), n_class, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=3, keepdim=True)
        x = self.head_conv(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1).squeeze()


class BcResNetModelNoSoftMax(BcResNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=3, keepdim=True)
        x = self.head_conv(x)

        return x.squeeze()


class SubSpectralNorm(nn.Module):
    def __init__(self, channels, sub_bands, eps=1e-5):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands, eps=eps)

    def forward(self, x):
        N, C, F, T = x.size()
        x = x.view(N, C * self.sub_bands, F // self.sub_bands, T)
        x = self.bn(x)
        return x.view(N, C, F, T)


class NormalBlock(nn.Module):
    def __init__(
        self,
        n_chan: int,
        *,
        dilation: int = 1,
        dropout: float = constants.DROPOUT,
        use_subspectral: bool = True
    ):
        super().__init__()
        norm_layer = (
            SubSpectralNorm(n_chan, 5) if use_subspectral else nn.BatchNorm2d(n_chan)
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(
                n_chan, n_chan, kernel_size=(3, 1), padding="same", groups=n_chan
            ),
            norm_layer,
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(
                n_chan,
                n_chan,
                kernel_size=(1, 3),
                padding="same",
                groups=n_chan,
                dilation=(1, dilation),
            ),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            nn.Conv2d(n_chan, n_chan, kernel_size=1),
            nn.Dropout2d(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        x1 = self.f2(x)

        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1 + x2)


class TransitionBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        *,
        dilation: int = 1,
        stride: int = 1,
        dropout: float = constants.DROPOUT,
        use_subspectral: bool = True
    ):
        super().__init__()

        if stride == 1:
            conv = nn.Conv2d(
                out_chan, out_chan, kernel_size=(3, 1), groups=out_chan, padding="same"
            )
        else:
            conv = nn.Conv2d(
                out_chan,
                out_chan,
                kernel_size=(3, 1),
                stride=(stride, 1),
                groups=out_chan,
                padding=(1, 0),
            )

        norm_layer = (
            SubSpectralNorm(out_chan, 5)
            if use_subspectral
            else nn.BatchNorm2d(out_chan)
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            norm_layer,
        )

        self.f1 = nn.Sequential(
            nn.Conv2d(
                out_chan,
                out_chan,
                kernel_size=(1, 3),
                padding="same",
                groups=out_chan,
                dilation=(1, dilation),
            ),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(dropout),
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1)
