
import os
import sys
sys.path.append('./mp3process')

import re
import json

import pandas as pd
import numpy as np

import librosa

from pydub import AudioSegment
from multifileprocess import multiprocess, data_label_write



class AudioSplitter():
    def __init__(self, target_dir, test_split):
        self.target_dir = target_dir + 'Extracts_' + str(test_split) + '/'
        self.test_dir = target_dir + 'Test_' + str(test_split) + '/'
        self.test_split = test_split

        # Catalogs are arrays that hold dictionaries for each extract-audio file
        self.test_catalog = []
        self.train_catalog = []

        # These attributes apply to the linear-feature extraction
        self.train_df = None
        self.res_type = None
        self.pickle_path = 'Pickles/'

        # Make a list of languages and their source files here
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
        self.train_catalog = []
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
                print('{}: invalid audio filename; skipping'.format(file))
            self.train_catalog.append(details)

        filename = self.target_dir + 'train_catalog.json'
        with open(filename, 'w+') as f:
            json.dump(self.train_catalog, f)
        print(len(self.train_catalog))

    def build_test_catalog(self):
        self.test_catalog = []
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
                print('{}: invalid audio filename; skipping'.format(file))
            self.test_catalog.append(details)

        filename = self.test_dir + 'test_catalog.json'
        with open(filename, 'w+') as f:
            json.dump(self.test_catalog, f)
        print(len(self.test_catalog))

    def extract_features(self, batch_size, train_set=True ):
        if train_set:
            target_dir = self.target_dir
            self.build_train_catalog()
            catalog = self.train_catalog
        else:
            target_dir = self.test_dir
            self.build_test_catalog()
            catalog = self.test_catalog

        file_list = []
        for file_details in catalog:
            if 'Path' in file_details.keys():
                file_list.append(file_details['Path'])
        start = 0
        step = batch_size
        count = 0
        while ((start + step) < len(file_list)):
            count += 1
            file_name = target_dir + 'Pickles/processed_batch' + str(count) + '.pkl'
            print(file_name)
            end = start + step
            all_data, labels = multiprocess(file_list[start:end])
            data_label_write(all_data, labels, file_name)

            start = end


        count += 1
        file_name = target_dir + 'Pickles/processed_batch' + str(count) + '.pkl'
        print(file_name)

        all_data, labels = multiprocess(file_list[start:])
        data_label_write(all_data, labels, file_name)


    def extract_linear_features(self, num_features, res_type):
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
