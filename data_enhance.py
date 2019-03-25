import os

import acoustics
import numpy as np
from librosa import load
from pysndfx import AudioEffectsChain
from librosa.output import write_wav
from pydub import AudioSegment


fx = (
    AudioEffectsChain()
    .normalize()
)


def test_on_one_file():
    infile = 'D:\\2019af-sr-aishell2\\AISHELL-2\\iOS\\data\\data\\D1048\\ID1048W0001.wav'
    outfile = './t.wav'
    fx(infile, outfile)


def add_noise():
    infile = 'D:\\2019af-sr-aishell2\\AISHELL-2\\iOS\\data\\data\\D1048\\ID1048W0001.wav'
    outfile = './t.wav'
    noise_file = "D:\\_background_noise_\\pink_noise.wav"
    sound1 = AudioSegment.from_file(infile)
    sound2 = AudioSegment.from_file(noise_file) - 30

    combined = sound1.overlay(sound2)

    combined.export(outfile, format='wav')

    # fx(infile, outfile)


def enhance_all_training_data():
    infile = 'D:\\2019af-sr-aishell2\\AISHELL-2\\iOS\\data\\data'

    normalize = (
        AudioEffectsChain()
            .normalize()
    )

    dirs = os.listdir(infile)
    for dir in dirs:
        files = os.listdir(os.path.join(infile, dir))
        for file in files:
            echo = (
                AudioEffectsChain()
                    .delay(gain_in=np.random.random()*0.2+0.7,
                           gain_out=np.random.random()*0.2+0.5,
                           delays=list((np.random.random()*50+25, np.random.random()*100+100)),
                           decays=list((np.random.random()*0.1+0.25, np.random.random()*0.1+0.20)),
                           parallel=False)
                    .normalize()
            )

            # echo
            echo(os.path.join(infile, dir, file), os.path.join(infile, dir, "_echo_"+file))
            # noise
            noise_file = "D:\\_background_noise_\\pink_noise.wav"
            sound1 = AudioSegment.from_file(os.path.join(infile, dir, file)) + 10
            sound2 = AudioSegment.from_file(noise_file) - 5*np.random.random()-27.5

            combined = sound1.overlay(sound2)
            combined.export(os.path.join(infile, dir, "_noise_"+file), format='wav')
            # normalize(os.path.join(infile, dir, "_noise_"+file), os.path.join(infile, dir, "_noise_"+file))


enhance_all_training_data()
# infile = 'D:\\af2019-sr-devset-20190312\\data\\'
# outfile = 'D:\\af2019-sr-devset-20190312\\data1\\'
# # fx(infile, outfile)
#
# k = os.listdir(infile)
# for i in k:
#     fx(os.path.join(infile, i), os.path.join(outfile, i))
