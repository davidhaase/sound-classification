
import os
from pydub import AudioSegment



class AudioSplitter():
    def __init__(self, target_dir, test_split):
        self.target_dir = target_dir + 'Extracts_' + str(test_split) + '/'
        self.test_dir = target_dir + 'Test_' + str(test_split) + '/'
        self.test_split = test_split
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
