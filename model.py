import torch
import torch.nn as nn
import torch.nn.functional as F


SAMPLE_RATE = 32000          # даунсэмпл 44.1 -> 32 кГц: выше 16 кГц
                             # в MUSDB18 только шум AAC-кодека, анти-алиасинг
                             # ffmpeg отрезает его при декодировании

N_SOURCES = 4
SOURCE_NAMES = ["drums", "bass", "other", "vocals"]

WAVE_LEVELS = 5
WAVE_CHANNELS = 64
SPEC_NFFT = 4096
SPEC_HOP = 1024

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

_WINDOWS = {}


def get_window(n_fft, device):
    key = (n_fft, str(device))
    if key not in _WINDOWS:
        _WINDOWS[key] = torch.sqrt(
            torch.hann_window(n_fft, periodic=True, device=device)
        )
    return _WINDOWS[key]


def stft(x, fl, fs):
    """Комплексный STFT: (..., N) -> (..., F, T). Нужен только для loss."""
    shape = x.shape[:-1]
    s = torch.stft(
        x.reshape(-1, x.shape[-1]),
        n_fft=fl, hop_length=fs, win_length=fl,
        window=get_window(fl, x.device), return_complex=True,
    )
    return s.reshape(shape + s.shape[-2:])


def log_magnitude(s):
    return torch.log1p(s.abs())


class EncBlock(nn.Module):
    """Энкодер-блок: conv1d stride-4 + GLU (каналы x2 до GLU)."""
    def __init__(self, cin, cout, kernel=8, stride=4):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout * 2, kernel, stride=stride,
                             padding=kernel // 2)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        return self.glu(self.conv(x))


class Bottleneck(nn.Module):
    """Два dilated-conv слоя с GLU и residual-связью."""
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels * 2, kernel, padding=1)
        self.glu1 = nn.GLU(dim=1)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel, padding=2, dilation=2)
        self.glu2 = nn.GLU(dim=1)

    def forward(self, x):
        return x + self.glu2(self.conv2(self.glu1(self.conv1(x))))


class DecBlock(nn.Module):
    """Декодер-блок: ConvTranspose stride-4 + GLU + skip + conv 3x3.

    Skip подрезается/допадывается до длины апсемпла, чтобы уровни
    совпадали по времени при любой длине входа.
    """
    def __init__(self, cin, cout, kernel=8, stride=4):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(cin, cout * 2, kernel, stride=stride,
                                         padding=kernel // 2, output_padding=1)
        self.glu = nn.GLU(dim=1)
        self.conv = nn.Conv1d(cout, cout, 3, padding=1)

    def forward(self, x, skip=None):
        x = self.glu(self.deconv(x))
        if skip is not None:
            if x.shape[-1] > skip.shape[-1]:
                x = x[..., :skip.shape[-1]]
            elif x.shape[-1] < skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = x + skip
        return self.conv(x)


class WaveUNet(nn.Module):
    """1D waveform U-Net в стиле Demucs v2 (стерео).

    Вход: (B, 2, N) нормализованный микс; выход: (B, 4, 2, N) стемы
    (та же длина, что и вход). Стерео — это 2 канала входа, последний
    слой выдаёт n_sources * 2 каналов (пара на каждый стем).
    """

    def __init__(self, n_sources=N_SOURCES, channels=64, levels=5,
                 kernel=8, stride=4):
        super().__init__()
        self.stride = stride
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        cin = 2
        for i in range(levels):
            cout = channels * (2 ** i)
            self.encoders.append(EncBlock(cin, cout, kernel, stride))
            cin = cout
        # декодеры: выход каналов = 2**max(0, i-1) * channels (под skip уровней),
        # верхний (i=levels-1) принимает bottleneck; нижний (i=0) — апсемпл без skip
        for i in range(levels):
            cout = channels * (2 ** max(0, i - 1))
            cin = channels * (2 ** i) if i == levels - 1 else channels * (2 ** max(0, i))
            self.decoders.append(DecBlock(cin, cout, kernel, stride))
        self.bottleneck = Bottleneck(channels * (2 ** (levels - 1)))
        self.head = nn.Conv1d(channels, n_sources * 2, 1)
        self.dummies = nn.Parameter(torch.zeros(n_sources))  # глобальный residual

    def forward(self, x):
        mix0 = x
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        x = self.bottleneck(x)
        # скипы уровней: dec4 <- s3, ..., dec1 <- s0; нижний dec0 без skip
        for dec, skip in zip(self.decoders[::-1], skips[-2::-1]):
            x = dec(x, skip)
        x = self.decoders[0](x)
        if x.shape[-1] < mix0.shape[-1]:   # робастность к неделимым длинам
            x = F.pad(x, (0, mix0.shape[-1] - x.shape[-1]))
        x = self.head(x)[..., :mix0.shape[-1]]           # (B, S*2, N)
        x = x.view(x.shape[0], -1, 2, x.shape[-1])       # (B, S, 2, N)
        return x + (1 + self.dummies)[None, :, None, None] * mix0[:, None]


class SpecBlock(nn.Module):
    """2D conv-блок: conv3x3 -> GroupNorm -> SiLU -> conv3x3 -> GroupNorm + residual."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(cout, 8), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(cout, 8), cout)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + self.shortcut(x))

class BiLSTMBottleneck(nn.Module):
    """BiLSTM по оси времени в bottleneck 2D U-Net.

    1x1 conv сжимает каналы в `hidden`, частота пулится до `f_pool` фиксированных
    бинов, признаки кадра складываются в один вектор, BiLSTM идёт по оси
    времени; затем частота восстанавливается и 1x1 conv возвращает каналы
    обратно. Residual-связь.
    """

    def __init__(self, channels, hidden=32, f_pool=16):
        super().__init__()
        self.hidden = hidden
        self.f_pool = f_pool
        self.proj_in = nn.Conv2d(channels, hidden, 1)
        self.lstm = nn.LSTM(hidden * f_pool, hidden * f_pool // 2, batch_first=True,
                           bidirectional=True)
        self.proj_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        b, c, f, t = x.shape
        h = self.proj_in(x)                                  # (B, H, F, T)
        h = F.interpolate(h, size=(self.f_pool, t), mode="bilinear",
                          align_corners=False)               # (B, H, P, T)
        h = h.permute(0, 3, 1, 2).reshape(b, t, -1)          # (B, T, H*P)
        h, _ = self.lstm(h)                                  # (B, T, H)
        h = h.reshape(b, t, self.hidden, self.f_pool)
        h = h.permute(0, 2, 3, 1)                            # (B, H, P, T)
        h = F.interpolate(h, size=(f, t), mode="bilinear",
                          align_corners=False)               # (B, H, F, T)
        return self.proj_out(h) + x

class STFTBranch(nn.Module):
    """2D U-Net по log-магнитуде STFT микса -> 4 конкурентные маски.

    Вход: (B, 1, F, T); выход: (B, 4, F, T) маски в [0, 1], softmax по стемам
    на каждом T-F бине: сумма масок = 1, каждый бин делится между стемами
    «по-честному» (нет утечки other/vocals пополам).
    Частота и время даунсемплются в 2 раза на уровень.
    """
    def __init__(self, n_sources=N_SOURCES, base=16, levels=4):
        super().__init__()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels):
            cin = 1 if i == 0 else base * 2 ** (i - 1)
            cout = base * 2 ** i
            self.enc.append(SpecBlock(cin, cout))
        c = base * 2 ** (levels - 1)
        self.bottleneck = BiLSTMBottleneck(c)
        for i in range(levels - 1, -1, -1):
            cout = base * 2 ** i
            cin = c + c if i == levels - 1 else base * 2 ** i * 3
            self.dec.append(SpecBlock(cin, cout))
        self.head = nn.Conv2d(base, n_sources, 1)

    def forward(self, spec):
        skips = []
        x = spec
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bottleneck(x)
        for dec, skip in zip(self.dec, skips[::-1]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x).softmax(dim=1)


class HybridUNet(nn.Module):
    """Гибрид: 1D waveform U-Net + 2D спектральные маски, объединение гейтом.

    Вход: (B, 2, N) нормализованный микс; выход: (B, 4, 2, N) стемы.

    - wave-ветка предсказывает стемы напрямую в waveform-домене;
    - spec-ветка строит мягкие маски по log-магнитуде STFT микса
      (канал микса обрабатывается независимо с общими весами), стемы
      восстанавливаются маскированием комплексного STFT + ISTFT;
    - выход = g * wave + (1 - g) * spec, где g — обучаемый per-stem гейт
      (init 0.5/0.5, модель сама учится, кому верить).
    """
    def __init__(self, n_sources=N_SOURCES, channels=64, levels=5,
                 nfft=4096, hop=1024, spec_base=16, spec_levels=4):
        super().__init__()
        self.wave = WaveUNet(n_sources=n_sources, channels=channels, levels=levels)
        self.spec = STFTBranch(n_sources=n_sources, base=spec_base, levels=spec_levels)
        self.nfft = nfft
        self.hop = hop
        self.gate = nn.Parameter(torch.zeros(n_sources))

    def forward(self, x):
        wav = self.wave(x)                             # (B, 4, 2, N)
        xf = x.float()
        win = torch.hann_window(self.nfft, device=x.device)
        st = torch.stft(xf.reshape(-1, x.shape[-1]), self.nfft, self.hop,
                        window=win, return_complex=True)   # (B*2, F, T)
        masks = self.spec(st.abs().log1p().unsqueeze(1))   # (B*2, 4, F, T)
        B = x.shape[0]
        spec_stems = torch.istft(
            (masks * st.unsqueeze(1)).view(B * 2 * N_SOURCES, *st.shape[-2:]),
            self.nfft, self.hop, window=win, length=x.shape[-1],
        )
        spec_stems = spec_stems.view(B, 2, N_SOURCES, x.shape[-1]) \
                                 .permute(0, 2, 1, 3)       # (B, 4, 2, N)
        g = torch.sigmoid(self.gate)[None, :, None, None]
        return (g * wav + (1 - g) * spec_stems).to(wav.dtype)


model = HybridUNet(n_sources=N_SOURCES, channels=WAVE_CHANNELS,
                   levels=WAVE_LEVELS, nfft=SPEC_NFFT, hop=SPEC_HOP).to(DEVICE)

