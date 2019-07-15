
import os
import sys
sys.path.append('./mp3process')

import re

import pandas as pd
import numpy as np

import librosa

from pydub import AudioSegment
from multifileprocess import multiprocess, data_label_write


arabic = {'Label': 'Arabic', 'Short':'ar', 'Path':'/Volumes/LaCie Rose Gold 2TB/Datasets/Audio/Source/ASSIMIL Arabic/ASSIMIL Arabic/'}
greek = {'Label': 'Greek', 'Short':'gk', 'Path':'/Volumes/LaCie Rose Gold 2TB/Datasets/Audio/Source/ASSIMIL New Greek With Ease/ASSIMIL New Greek With Ease/'}
persian = {'Label': 'Persian', 'Short':'fa', 'Path':'/Volumes/LaCie Rose Gold 2TB/Datasets/Audio/Source/ASSIMIL Persian/ASSIMIL Persian/'}
russian = {'Label': 'Russian', 'Short':'ru', 'Path':'/Volumes/LaCie Rose Gold 2TB/Datasets/Audio/Source/ASSIMIL Russian/ASSIMIL Russian/'}
turkish = {'Label': 'Turkish', 'Short':'tr', 'Path':'/Volumes/LaCie Rose Gold 2TB/Datasets/Audio/Source/ASSIMIL Turkish With Ease/ASSIMIL Turkish With Ease/'}

class AudioSplitter():
    def __init__(self, source_languages):
        # Make a list of languages and their source files here
        self.source_languages = source_languages
        self.language_map = {}
        for lang in source_languages:
            self.language_map[lang['Short']] = lang['Label']

    def split(self, write_dir, duration_s, test_split=0.2, rebuild=False):

        # test_split is the percentage of test files versus training
        # First confirm that test_split is a valid percentage
        if (test_split > 1.0 or test_split < 0.0):
            print('test_split value must be between 0 and 1. You enterered {}'.format(test_split))
            return


        # Build extracted audio files based on a duration in seconds
        # The target and test files are stored in folders with called 'Train_[the num of seconds]'
        train_dir = write_dir + str(duration_s) + '_Seconds/Train_' + str(duration_s) + '/'
        test_dir = write_dir + str(duration_s) + '_Seconds/Test_' + str(duration_s) + '/'

        # Make the target and test directories if theg don't already exist
        if (os.path.isdir(train_dir) is False):
            try:
                os.makedirs(train_dir)
                if (os.path.isdir(test_dir) is False):
                    os.makedirs(test_dir)
            except Exception as e:
                print(e)
                return

        # Avoid rebuilding if necessary, so check the file count
        # But also give them the chance to force a rebuild
        file_count = len(os.listdir(train_dir))
        if (file_count > 0) and (rebuild==False):
            print('{} files already found in {}'.format(file_count, train_dir))
            print('Not rebuilding segments. Pass rebuild=True to force rebuild')
            return train_dir, test_dir

        # Here is the main loop for slicing the files into extracts
        print('Processing files for {} languages'.format(len(self.source_languages)))
        for lang in self.source_languages:
            print('\t{}...'.format(lang['Label']))
            self.slice(lang['Short'], lang['Path'], train_dir, test_dir, test_split, duration_s)

        return train_dir, test_dir

    def slice(self, lang, source_dir, train_dir, test_dir, test_split, seconds):
        count = 1

        test_split = int(1/test_split)

        for file in os.listdir(source_dir):
            if 'mp3' not in file:
                continue

            file_name, ext = file.split('.')
            ext = '.' + ext
            file_path = source_dir + file

            # Opening file and extracting segment
            song = AudioSegment.from_mp3(file_path)

            # Time to miliseconds
            startTime = 0
            endTime = seconds*1000
            interval = seconds*1000

            while endTime < len(song):
                extract = song[startTime:endTime]
                startTime = endTime
                endTime += interval

                # Saving
                new_file = train_dir + lang + str(count) + ext
                print(new_file, end='\r')
                extract.export(new_file, format="mp3")

                if (count%test_split == 0):
                    test_file = test_dir + lang + str(count) + ext
                    os.rename(new_file, test_file)

                count += 1

    def get_details(self, source_dir):
        details = []
        pattern = re.compile(r'(\w{2})(.+)\.m4a')
        for file in os.listdir(source_dir):
            detail = {}
            match = pattern.search(file)
            if(match):
                lang, ID = match.group(1), match.group(2)
                detail['Filename'] = file
                detail['Path'] = source_dir + file
                detail['Lang_ID'] = lang
                detail['File_ID'] = ID
                detail['Language'] = self.language_map[lang]
            else:
                print('{}: invalid audio filename; skipping'.format(file))
            details.append(detail)
        return details

    def extract_features(self, train_dir, test_dir, write_dir, func):
        # First try to build an output directory
        dir_name = self.get_next_dir(write_dir)
        next_dir = write_dir + dir_name + '/'
        train_out = next_dir+ 'Train_Feature_Pickles/'
        test_out = next_dir + 'Test_Feature_Pickles/'

        try:
            os.makedirs(train_out)
            os.makedirs(test_out)
        except Exception as e:
            print(e)
            return

        # Second, now build the features based on the passed function
        # train_df = pd.DataFrame.from_dict(self.get_details(train_dir))
        test_df = pd.DataFrame.from_dict(self.get_details(test_dir))

        # train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        # train_df['Features'] = train_df['Path'].map(lambda x: func(x))
        # train_df.to_pickle(train_out + 'extracted.pkl')
        test_df['Features'] = test_df['Path'].map(lambda x: func(x))
        test_df.to_pickle(test_out + 'extracted.pkl')


        # Now write all the settings to a companion file in the output directory
        settings = {'train_dir':train_dir,
                    'test_dir':test_dir,
                    'write_dir':write_dir,
                    'train_out':train_out,
                    'test_out':test_out}
        settings_file = next_dir + dir_name + '_job_settings.txt'
        try:
            f = open(settings_file, 'w+')
            f.write(str(settings))
            f.close()
        except Exception as e:
            print(e)

    def get_next_dir(self, write_dir):
        if (os.path.isdir(write_dir) is False):
            try:
                os.makedirs(write_dir)
            except Exception as e:
                print(e)
                return

        pattern = re.compile(r'feat_(\d+)')

        max_dir = 0
        for dir_name in os.listdir(write_dir):
            match = pattern.search(dir_name)
            if match:
                count = int(match.group(1))
                if count > max_dir:
                    max_dir = count

        new_dir = 'feat_{0:03d}'.format(max_dir + 1)
        return new_dir

    # def extract_features(self, batch_size, train_set=True ):
    #     if train_set:
    #         target_dir = self.target_dir
    #         self.build_train_catalog()
    #         catalog = self.train_catalog
    #     else:
    #         target_dir = self.test_dir
    #         self.build_test_catalog()
    #         catalog = self.test_catalog
    #
    #     file_list = []
    #     for file_details in catalog:
    #         if 'Path' in file_details.keys():
    #             file_list.append(file_details['Path'])
    #     start = 0
    #     step = batch_size
    #     count = 0
    #     while ((start + step) < len(file_list)):
    #         count += 1
    #         file_name = target_dir + 'Pickles/processed_batch' + str(count) + '.pkl'
    #         print(file_name)
    #         end = start + step
    #         all_data, labels = multiprocess(file_list[start:end])
    #         data_label_write(all_data, labels, file_name)
    #
    #         start = end
    #
    #
    #     count += 1
    #     file_name = target_dir + 'Pickles/processed_batch' + str(count) + '.pkl'
    #     print(file_name)
    #
    #     all_data, labels = multiprocess(file_list[start:])
    #     data_label_write(all_data, labels, file_name)
