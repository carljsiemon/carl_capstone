import os

class Globals:
    song_dir = os.path.dirname(os.path.realpath(__file__)) + '\\song_data'  
    csv_dir = os.path.dirname(os.path.realpath(__file__)) + '\\wave_form_data'    
    labels_dir = os.path.dirname(os.path.realpath(__file__)) + '\\GenreLabels' 
    @classmethod
    def getSongDir(cls):
        return cls.song_dir
    @classmethod
    def getCsvDir(cls):
        return cls.csv_dir
    @classmethod
    def getLabelsDir(cls):
        return cls.labels_dir
    
    
    
#cd C:\Users\carls\Google Drive\PythonStocks\src\root\nested