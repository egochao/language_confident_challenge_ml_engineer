import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


def log_specgram(audio, sample_rate):
    """Wrapper of scipy.signal.spectrogram"""
    nperseg = int(round(20 * sample_rate / 1e3))
    noverlap = int(round(10 * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )
    return freqs, times, np.log(spec.T.astype(np.float32) + 1e-10)


def plot_audio(file_name):
    """Plot audio array and spectrogram"""
    sample_rate, samples = wavfile.read(file_name)
    freqs, times, spectrogram = log_specgram(samples, sample_rate)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title("Raw wave")
    ax1.set_ylabel("Amplitude")
    ax1.plot(samples)

    ax2 = fig.add_subplot(212)
    ax2.imshow(
        spectrogram.T,
        aspect="auto",
        origin="lower",
        extent=[times.min(), times.max(), freqs.min(), freqs.max()],
    )
    ax2.set_yticks(freqs[::16])
    ax2.set_xticks(times[::16])
    ax2.set_title("Spectrogram")
    ax2.set_ylabel("Freqs in Hz")
    ax2.set_xlabel("Seconds")


def plot_fft(file_name):
    sample_rate, samples = wavfile.read(file_name)
    xf, vals = custom_fft(samples, sample_rate)
    plt.figure(figsize=(12, 4))
    plt.title("FFT of recording sampled with " + str(sample_rate) + " Hz")
    plt.plot(xf, vals)
    plt.xlabel("Frequency")
    plt.grid()
    plt.show()


def custom_fft(y, fs):
    """Custom fft function for plotting"""
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    vals = (
        2.0 / N * np.abs(yf[0 : N // 2])
    )  # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    return xf, vals
