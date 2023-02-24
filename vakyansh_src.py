''' Example file to test tts_infer after installing it. Refer to section 1.1 in README.md for steps of installation. '''

from tts_infer.tts import TextToMel, MelToWav
from tts_infer.transliterate import XlitEngine
from tts_infer.num_to_word_on_sent import normalize_nums

import re
import json
import numpy as np
import argparse
from scipy.io.wavfile import write

from mosestokenizer import *
from indicnlp.tokenize import sentence_tokenize

class vak_tts():
    def __init__(self, glow_path='./vakyansh-tts/glow_ckp', hifi_path='./vakyansh-tts/hifi_ckp', lang='hi'):
        self.text_to_mel = TextToMel(glow_model_dir=glow_path, device='cuda')
        self.mel_to_wav = MelToWav(hifi_model_dir=hifi_path, device='cuda')
        self.INDIC = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
        self.lang = lang
        self.engine = XlitEngine(self.lang)

    def split_sentences(self, paragraph):
        if self.lang == "en":
            with MosesSentenceSplitter(self.lang) as splitter:
                return splitter([paragraph])
        elif self.lang in self.INDIC:
            return sentence_tokenize.sentence_split(paragraph, lang=self.lang)

    def translit(self,text):
        reg = re.compile(r'[a-zA-Z]')
        words = [self.engine.translit_word(word, topk=1)[self.lang][0] if reg.match(word) else word for word in text.split()]
        updated_sent = ' '.join(words)
        return updated_sent
    
    def run_tts(self, text, length_scale=0.9):
        text = text.replace('।', '.') # only for hindi models
        text_num_to_word = normalize_nums(text, self.lang) # converting numbers to words in lang
        text_num_to_word_and_transliterated = self.translit(text_num_to_word) # transliterating english words to lang
        final_text = ' ' + text_num_to_word_and_transliterated

        mel = self.text_to_mel.generate_mel(final_text,length_scale=length_scale) #speedup, 0.75: 211 wpm , 1: 165 wpm, 1.25: 135 wpm
        # mel = text_to_mel.generate_mel(final_text)

        audio, sr = self.mel_to_wav.generate_wav(mel)
        write(filename='temp.wav', rate=sr, data=audio) # for saving wav file, if needed
        return (sr, audio)

    def read_txt(self, txt_file):
        with open(txt_file, "r") as fi:
            dat = fi.read()
        text = json.loads(dat)['translated']
        return text

    def run_tts_paragraph(self, transcript, outfile, ls=0.9):
        audio_list = []
        if transcript.endswith(".txt"):
            text = self.read_txt(transcript)
        else:
            text = transcript
        split_sentences_list = self.split_sentences(text)

        for i,sent in enumerate(split_sentences_list):
            print(f"This is the current sent: {sent}")
            if i%5==0:
                print("slowing down")
                sr, audio = self.run_tts(sent, ls+0.75)
            # if i%7==0:
            #     print("speeding up")
            #     sr, audio = self.run_tts(sent, ls-0.1)

            sr, audio = self.run_tts(sent, ls)
            audio_list.append(audio)

        concatenated_audio = np.concatenate([i for i in audio_list])
        write(filename=outfile, rate=sr, data=concatenated_audio)
        return (sr, concatenated_audio)

    def run_tts_sentence(self, transcript, outfile, ls):
        # check if transcript list has same elements as ls list
        if len(transcript) != len(ls):
            print("The transcript list and length_scale list has mismatched number of elements!!! Cannot proceed ahead and returning!")
            return
        audio_list = []

        for sen, ls_sen in zip(transcript, ls):
            print(f"this is the current sentence: {sen} \n with length_scale value: {ls_sen}")
            sr, audio = self.run_tts(sen, ls_sen)
            audio_list.append(audio)

        concatenated_audio = np.concatenate([i for i in audio_list])
        write(filename=outfile, rate=sr, data=concatenated_audio)
        return (sr, concatenated_audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text to speech generator')
    parser.add_argument('--outfile', required=False,
                        default='./outputs/temp_long.wav',
                        metavar="/path/to/wav/file/",
                        help='Path to generated .wav file')
    parser.add_argument('--transcript', required=False,
                        default='../outputs/translated_text.txt',
                        metavar="/path/to/read/translated/text/",
                        help='File to read transcript (translated) text from')
    parser.add_argument('--lang', required=False,
                        default='hi',
                        help='Target language for TTS')

    args = parser.parse_args()

    tts_model = vak_tts(lang=args.lang)
    _, audio_long = tts_model.run_tts_paragraph(args.transcript, args.outfile)
