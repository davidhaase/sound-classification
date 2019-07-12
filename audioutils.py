
import os
import re
import json

import pandas as pd
import numpy as np

import librosa

from pydub import AudioSegment



class AudioSplitter():
    def __init__(self, target_dir, test_split):
        self.target_dir = target_dir + 'Extracts_' + str(test_split) + '/'
        self.test_dir = target_dir + 'Test_' + str(test_split) + '/'
        self.test_split = test_split
        self.test_catalog = []
        self.train_catalog = []
        self.train_df = None
        self.res_type = None
        self.pickle_path = 'Pickles/'
        self.languages = {'ar':'Arabic', 'gk':'Greek', 'fa': 'Persian', 'ru':'Russian', 'tr': 'Turkish' }
        self.lang_files = { 'fa' : '/Volumes/LaCie Rose Gold 2TB/Datasets/Source/ASSIMIL Persian/ASSIMIL Persian/',
                            'tr' : '/Volumes/LaCie Rose Gold 2TB/Datasets/Source/ASSIMIL Turkish With Ease/ASSIMIL Turkish With Ease/',
                            'ru' : '/Volumes/LaCie Rose Gold 2TB/Datasets/Source/ASSIMIL Russian/ASSIMIL Russian/',
                            'gk' : '/Volumes/LaCie Rose Gold 2TB/Datasets/Source/ASSIMIL New Greek With Ease/ASSIMIL New Greek With Ease/',
                            'ar' : '/Volumes/LaCie Rose Gold 2TB/Datasets/Source/ASSIMIL Arabic/ASSIMIL Arabic/'}

    def process(self, seconds=4):
        for lang in self.lang_files:
            print('Processing {}'.format(lang))

            if (os.path.isdir(self.target_dir) is False):
                try:
                    os.makedirs(self.target_dir)
                except Exception as e:
                    print(e)
                    return

            if (os.path.isdir(self.test_dir) is False):
                try:
                    os.makedirs(self.test_dir)
                except Exception as e:
                    print(e)
                    return


            self.split(lang, self.lang_files[lang], seconds)

    def split(self, lang, source_dir, seconds):
        count = 1

        for file in os.listdir(source_dir):
            if 'mp3' not in file:
                continue

            file_name, ext = file.split('.')
            ext = '.' + ext
            file_path = source_dir + file

            # Opening file and extracting segment
            song = AudioSegment.from_mp3(file_path)
            print(len(song))

            # Time to miliseconds
            startTime = 0
            endTime = seconds*1000
            interval = seconds*1000

            while endTime < len(song):
                extract = song[startTime:endTime]
                startTime = endTime
                endTime += interval
                # Saving


                new_file = self.target_dir+ lang + str(count) + ext
                print(new_file)
                extract.export(new_file, format="mp3")

                if (count%self.test_split == 0):
                    test_file = self.test_dir + lang + str(count) + ext
                    os.rename(new_file, test_file)


                count += 1

    def build_train_catalog(self):
        pattern = re.compile(r'(\w{2})(.+)\.mp3')
        for file in os.listdir(self.target_dir):
            details = {}
            match = pattern.search(file)
            if(match):
                lang, ID = match.group(1), match.group(2)
                details['Filename'] = file
                details['Path'] = self.target_dir + file
                details['Lang_ID'] = lang
                details['File_ID'] = ID
                details['Language'] = self.languages[lang]
            else:
                print(file, 'not matched')
            self.train_catalog.append(details)

        filename = self.target_dir + 'train_catalog.json'
        with open(filename, 'w+') as f:
            json.dump(self.train_catalog, f)
        print(len(self.train_catalog))

    def build_test_catalog(self):
        pattern = re.compile(r'(\w{2})(.+)\.mp3')
        for file in os.listdir(self.test_dir):
            details = {}
            match = pattern.search(file)
            if(match):
                lang, ID = match.group(1), match.group(2)
                details['Filename'] = file
                details['Path'] = self.test_dir + file
                details['File_ID'] = ID
            else:
                print(file, 'not matched')
            self.test_catalog.append(details)

        filename = self.test_dir + 'test_catalog.json'
        with open(filename, 'w+') as f:
            json.dump(self.test_catalog, f)
        print(len(self.test_catalog))

    def extract_audio_features(self, num_features, res_type):
        self.res_type = res_type
        self.num_features = num_features
        file_path = self.pickle_path + 'Train_' + str(self.test_split) + 's_' + str(num_features) + 'f.pkl'
        if os.path.isfile(file_path):
            print('Loading from pickle, {}'.format(file_path))
            self.train_df = pd.read_pickle(file_path)
        else:
            train = pd.DataFrame.from_dict(self.train_catalog)

            self.train_df = train.apply(self.train_parser, axis=1)
            # self.train_df.rename(columns={0:'Features'}, inplace=True)
            # self.train_df['Label'] = self.train_df['Features'].map(lambda x: x[1])
            # self.train_df['Features'] = self.train_df['Features'].map(lambda x: x[0])
            # self.train_df.to_pickle(self.pickle_train)
            self.train_df.to_pickle(file_path)

        # target = self.train_df['Label']
        # features = self.train_df.drop('Label', axis=1)
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=self.test_size, random_state=self.random_state)
        # self.y_train = self.train_df['Label']
        # self.X_train = self.train_df.drop('Label', axis=1)
    def train_parser(self, row):
       # function to load files and extract features
        file_name = row.Path
       # handle exception to check if there isn't a file which is corrupted
        try:
    #       # here kaiser_fast is a technique used for faster extraction
            X, sample_rate = librosa.load(file_name, res_type=self.res_type)
    #       # we extract mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=self.num_features).T,axis=0)
        except Exception as e:
            print(e, "Error encountered while parsing file: ", row.Filename)
            return None

        features = mfccs
        label = row.Language

        return [features, label]
