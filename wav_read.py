import librosa
import numpy as np


def wav2mfcc(file_path, max_pad_len=150):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

if __name__ == '__main__':
    wav2mfcc("D:\\af2019-sr-devset-20190312\\data\\00df05c18b3ad92648119e8ad06c7fc7.wav")