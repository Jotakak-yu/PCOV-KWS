import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fixed‑point scales
Q015_SCALE = 2 ** 15

QWIN_NUM = 0
QWIN_SCALE = 2 ** (15 - QWIN_NUM)

QFFT_NUM   = 4
QFFT_SCALE = 2 ** (15 - QFFT_NUM)

QMEL_NUM   = 6
QMEL_SCALE = 2 ** (15 - QMEL_NUM)

QLOG_NUM   = 4
QLOG_SCALE = 2 ** (15 - QLOG_NUM)

LOG_MEL_MEAN = -1.0
LOG_MEL_STD  =  2.0

def quantize_signed(x, scale, saturate=False):
    x_scaled = tf.round(x * scale)
    if saturate:
        x_scaled = tf.clip_by_value(x_scaled, -2**15, 2**15 - 1)
    return tf.cast(x_scaled, tf.int16)

def quantize_unsigned(x, scale, saturate=False):
    x_scaled = tf.round(x * scale)
    if saturate:
        x_scaled = tf.clip_by_value(x_scaled, 0.0, 2**16 - 1)
    return tf.cast(x_scaled, tf.uint16)

def optimized_preemphasis_q(signal, coef=0.97, saturate=False):
    sig_f = tf.cast(signal, tf.float32) / Q015_SCALE
    emphasized = tf.concat([
        sig_f[:1],
        sig_f[1:] - coef * sig_f[:-1]
    ], axis=0)
    return quantize_signed(emphasized, Q015_SCALE, saturate)

def optimized_frame_q(signal, frame_length, frame_step, saturate=False):
    sig_f  = tf.cast(signal, tf.float32) / Q015_SCALE
    frames = tf.signal.frame(sig_f, frame_length, frame_step)
    window = tf.cast(tf.signal.hamming_window(frame_length), tf.float32)
    framed = frames * window
    return quantize_signed(framed, QWIN_SCALE, saturate)

def optimized_stft_magnitude_q(frames_q, fft_length, saturate=False, unsigned=False):
    frm_f = tf.cast(frames_q, tf.float64) / QWIN_SCALE
    frm_p = tf.pad(frm_f, [[0,0],[0, fft_length - tf.shape(frm_f)[1]]])
    fft_r = tf.signal.rfft(frm_p)
    mag_f = tf.abs(fft_r)
    return quantize_unsigned(mag_f, QFFT_SCALE, saturate) \
        if unsigned else quantize_signed(mag_f, QFFT_SCALE, saturate)

def optimized_mel_filterbank_q(mag_q, sample_rate=16000, mel_bands=64, saturate=False, unsigned=False):
    mag_f = tf.cast(mag_q, tf.float32) / QFFT_SCALE
    num_bins = tf.shape(mag_f)[1]
    mel_mat = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_bands,
        num_spectrogram_bins=num_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=50,
        upper_edge_hertz=8000
    )
    mel_e = tf.matmul(mag_f, mel_mat)
    return quantize_unsigned(mel_e, QMEL_SCALE, saturate=True) \
        if unsigned else quantize_signed(mel_e, QMEL_SCALE, saturate=True)

@tf.function
def optimized_mel_calculation_graph(
    pdm_waveform_q015,
    use_norm=True,
    feature_type='logfbank',
    use_pre_emphasis=True,
    convert_to_float=False,
    saturate=False,
    unsigned=False
):
    if use_pre_emphasis:
        pdm_waveform_q015 = optimized_preemphasis_q(
            pdm_waveform_q015, saturate=saturate)

    frame_length, frame_step = 400, 160
    frames_q = optimized_frame_q(
        pdm_waveform_q015, frame_length, frame_step, saturate=saturate)

    fft_length = 512
    mag_q = optimized_stft_magnitude_q(
        frames_q, fft_length, saturate=saturate, unsigned=unsigned)

    MEL_BANDS = 64
    mel_q = optimized_mel_filterbank_q(
        mag_q, mel_bands=MEL_BANDS, saturate=saturate, unsigned=unsigned)

    if feature_type == 'logfbank':
        mel_f = tf.maximum(tf.cast(mel_q, tf.float32) / QMEL_SCALE, 1e-3)
        log_m = tf.math.log(mel_f)
        if use_norm:
            log_m = (log_m - LOG_MEL_MEAN) / LOG_MEL_STD
        feat_q = quantize_signed(log_m, QLOG_SCALE, saturate=saturate)
    elif feature_type == 'fbank':
        feat_q = mel_q
    elif feature_type == 'fft':
        feat_q = mag_q
    elif feature_type == 'window':
        feat_q = frames_q
    elif feature_type == 'pre':
        feat_q = pdm_waveform_q015
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    if convert_to_float:
        if feature_type == 'fft':
            feats = tf.cast(feat_q, tf.float32) / QFFT_SCALE
        elif feature_type == 'fbank':
            feats = tf.cast(feat_q, tf.float32) / QMEL_SCALE
        elif feature_type == 'logfbank':
            feats = tf.cast(feat_q, tf.float32) / QLOG_SCALE
        elif feature_type == 'window':
            feats = tf.cast(feat_q, tf.float32) / QWIN_SCALE
        else:
            feats = tf.cast(feat_q, tf.float32) / Q015_SCALE
    else:
        feats = feat_q

    return feats

if __name__ == "__main__":
    sr, dur, f0 = 16000, 1.0, 440
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wf = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    wf_q = np.int16(np.round(wf * Q015_SCALE))
    wf_tf = tf.constant(wf_q, dtype=tf.int16)

    feats = optimized_mel_calculation_graph(
        wf_tf,
        use_norm=True,
        feature_type='logfbank',
        use_pre_emphasis=True,
        convert_to_float=True,
        saturate=False,
        unsigned=True
    )
    feats_np = feats.numpy().squeeze().T

    print("Feature shape:", feats_np.shape)
    print(f"Range: [{feats_np.min():.3f}, {feats_np.max():.3f}]")

    plt.imshow(feats_np, origin='lower', aspect='auto',
               interpolation='nearest')
    plt.title("Log Mel Spectrogram")
    plt.xlabel("Frame")
    plt.ylabel("Mel Band")
    plt.colorbar(label='Amplitude')
    plt.show()
