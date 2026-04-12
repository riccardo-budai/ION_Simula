"""
utils_funcIOM.py
utilities function of post processing of signals usable by differente simula modules
"""

import numpy as np
import scipy
from scipy.signal import butter, filtfilt, fftconvolve, welch, find_peaks
from scipy.interpolate import CubicSpline
# from numpy.lib.stride_tricks import sliding_window_view
from ecgdetectors import Detectors

########################################################################################################################
#   FILTERING
########################################################################################################################

def butter_bandpass(lowcut, highcut, fs, order=5):
    # Nyquist frequency
    nyq = 0.5 * fs
    # normalization 0.0 to 1.0
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0: low = 0.001
    if high >= 1: high = 0.99
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        return filtfilt(b, a, data)
    except Exception as e:
        print(f"Errore filtro: {e}")
        return data


def apply_spline_smoothing(data, time, smoothing_factor=5):
    n_points = len(data)
    sparse_indices = np.arange(0, n_points, smoothing_factor)
    if sparse_indices[-1] != n_points - 1:
        sparse_indices = np.append(sparse_indices, n_points - 1)
    cs = CubicSpline(time[sparse_indices], data[sparse_indices])
    return cs(time)


def apply_simple_smooth(data, window_len=5):
    if window_len < 3:
        return data
    s = np.r_[data[window_len-1:0:-1], data, data[-1:-window_len:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    # Trim to original size
    return y[int(window_len/2):-int(window_len/2)] if len(y) > len(data) else y[:len(data)]

########################################################################################################################
#   CORRELATION ANALYSIS
########################################################################################################################

def fast_normalized_cross_correlation(signal, template):
    """
    Calcola la cross-correlazione normalizzata usando FFT (Fast Fourier Transform).
    Molto più veloce O(N log N) rispetto alla sliding window O(N*M).
    """
    n = len(template)
    m = len(signal)
    if m < n: return np.array([])

    # 1. Preparazione Template (Normalizzazione Z-Score implicita nel calcolo)
    # Sottraiamo la media dal template
    template = template - np.mean(template)

    # Calcolo norma del template
    norm_template = np.sqrt(np.sum(template ** 2))
    if norm_template == 0: return np.zeros(m)

    # Normalizziamo il template
    template = template / norm_template

    # 2. Calcolo Numeratore (Dot Product veloce via FFT)
    # Convoluzione con template invertito = Cross-Correlazione
    numerator = fftconvolve(signal, template[::-1], mode='valid')

    # 3. Calcolo Denominatore (Norma locale del segnale)
    # La norma locale è sqrt(sum((x - mean)^2)) = sqrt(sum(x^2) - (sum(x)^2)/n)

    # Somma locale di x (via convoluzione con finestra di 1)
    ones = np.ones(n)
    sum_x = fftconvolve(signal, ones, mode='valid')

    # Somma locale di x^2
    sum_x2 = fftconvolve(signal ** 2, ones, mode='valid')

    # Calcolo varianza locale * n
    # (sum_x2 - (sum_x^2)/n) può talvolta essere leggermente < 0 per errori di virgola mobile
    val = sum_x2 - (sum_x ** 2) / n
    window_norm = np.sqrt(np.maximum(val, 0))

    # Evita divisione per zero
    window_norm[window_norm == 0] = 1.0

    # 4. Risultato finale
    correlation = numerator / window_norm

    # Padding per mantenere la lunghezza originale (allineamento a sinistra)
    pad_width = m - len(correlation)
    correlation = np.pad(correlation, (0, pad_width), mode='constant', constant_values=0)

    return correlation

########################################################################################################################
# PEAKS DETECTION
########################################################################################################################

def analyze_r_peaks(ecg_signal, fs):
    if np.max(ecg_signal) == np.min(ecg_signal):
        return {'peaks': [], 'bpm': 0, 'rr_intervals_ms': [], 'sdnn': 0, 'rmssd': 0}

    try:
        detectors = Detectors(fs)
        peaks_indices = detectors.hamilton_detector(ecg_signal)
        peaks_indices = np.array(peaks_indices).astype(int)

        if len(peaks_indices) < 2:
            return {'peaks': peaks_indices, 'bpm': 0, 'rr_intervals_ms': [], 'sdnn': 0, 'rmssd': 0}

        rr_samples = np.diff(peaks_indices)
        rr_sec = rr_samples / fs
        rr_ms = rr_sec * 1000.0

        if len(rr_sec) > 0:
            bpm = 60.0 / np.mean(rr_sec)
        else:
            bpm = 0

        sdnn = np.std(rr_ms)
        diff_rr = np.diff(rr_ms)
        if len(diff_rr) > 0:
            rmssd = np.sqrt(np.mean(diff_rr ** 2))
        else:
            rmssd = 0.0

        return {
            'peaks': peaks_indices,
            'bpm': round(bpm, 1),
            'rr_intervals_ms': rr_ms,
            'sdnn': round(sdnn, 2),
            'rmssd': round(rmssd, 2)
        }

    except Exception as e:
        print(f"Errore detector: {e}")
        return {'peaks': [], 'bpm': 0, 'rr_intervals_ms': [], 'sdnn': 0, 'rmssd': 0}


########################################################################################################################
# SPECTRAL ANALYSIS
########################################################################################################################

def apply_spectral_whitening(psd_db, freqs):
    """
    Applica la correzione 1/f sottraendo il fit lineare nello spazio log-log.
    Input:
        psd_db: (n_channels, n_freqs) in Decibel
        freqs: (n_freqs,) array delle frequenze
    Output:
        whitened_psd: stesso shape, componente aperiodica rimossa
    """
    # Evitiamo log(0) o log(numeri negativi)
    valid_mask = freqs > 0.5
    valid_freqs = freqs[valid_mask]

    # Asse X per il fit: Logaritmo della frequenza
    log_freqs = np.log10(valid_freqs)
    whitened_psd = psd_db.copy()
    n_channels = psd_db.shape[0]

    for i in range(n_channels):
        channel_psd_db = psd_db[i, valid_mask]

        # --- FIT LINEARE (y = mx + q) ---
        # y = dB power
        # x = log10(frequency)
        slope, intercept = np.polyfit(log_freqs, channel_psd_db, 1)

        # Calcoliamo il trend aperiodico (la linea retta inclinata)
        aperiodic_trend = (slope * log_freqs) + intercept

        # --- SOTTRAZIONE (WHITENING) ---
        whitened_psd[i, valid_mask] = channel_psd_db - aperiodic_trend

        # Opzionale: azzeriamo le frequenze scartate (< 0.5Hz) per pulizia
        whitened_psd[i, ~valid_mask] = 0

    return whitened_psd
