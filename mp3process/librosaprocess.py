import librosa, numpy as np

def process(mp3file):
    y, sr = librosa.load(mp3file)
    data = dict()
    #collect all from data
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    data["log_S"] = log_S
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # What do the spectrograms look like?
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
    log_Sp = librosa.power_to_db(S_percussive, ref=np.max)

    data["log_Sh"] = log_Sh
    data["log_Sp"] = log_Sp

    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
    data["C"] = C
    # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
    mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    # Let's pad on the first and second deltas while we're at it
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    data["mfcc"] = mfcc
    data["delta_mfcc"] = delta_mfcc
    data["delta2_mfcc"] = delta2_mfcc

    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

    data["tempo"] = tempo
    data["beats"] = beats

    C_sync = librosa.util.sync(C, beats, aggregate=np.median)

    data["C_sync"] = C_sync
    return data
