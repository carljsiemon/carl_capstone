import os

'''This class creates global variables containing various file locations'''
class Globals:
    '''song directory'''
    song_dir = os.path.dirname(os.path.realpath(__file__)) + '\\song_data'  
    '''feature data csv directory'''
    csv_dir = os.path.dirname(os.path.realpath(__file__)) + '\\wave_form_data'    
    '''song labels directory'''
    labels_dir = os.path.dirname(os.path.realpath(__file__)) + '\\GenreLabels' 
    
    '''use getters (encapsulation) here to retrieve directories'''
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