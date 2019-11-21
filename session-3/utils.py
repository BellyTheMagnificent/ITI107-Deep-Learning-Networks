import os
import shutil
import zipfile
import numpy as np


def prepare_data(data_path="./data", valid_size=0.2, seed=21, FORCED_DATA_REWRITE=False):

    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")

    if FORCED_DATA_REWRITE:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

    if not os.path.exists(data_path):
        
        if not os.path.exists("intel_emotions_dataset.zip"):
            os.system("wget https://sdaaidata.s3-ap-southeast-1.amazonaws.com/datasets/intel_emotions_dataset.zip")
        
        zip_ref = zipfile.ZipFile("intel_emotions_dataset.zip", "r")
        zip_ref.extractall("data_temp")
        zip_ref.close()
        
        os.rename("data_temp", data_path)
        #shutil.rmtree("data_temp")
        
        os.mkdir(train_path)
        os.mkdir(valid_path)

        for category in ["Negative", "Positive"]:
            train_emo_path = os.path.join(train_path, category)
            valid_emo_path = os.path.join(valid_path, category)
            os.mkdir(train_emo_path)
            os.mkdir(valid_emo_path)

            categoty_list = np.array(os.listdir(os.path.join(data_path, category)))
            np.random.seed(seed)
            np.random.shuffle(categoty_list)
            
            train_list = categoty_list[int(len(categoty_list) * valid_size):]
            valid_list = categoty_list[:int(len(categoty_list) * valid_size)]
            
            for filename in train_list:
                os.rename(os.path.join(data_path, category, filename), 
                          os.path.join(train_emo_path, filename.replace(" ", "")))
                
            for filename in valid_list:
                os.rename(os.path.join(data_path, category, filename), 
                          os.path.join(valid_emo_path, filename.replace(" ", "")))
                
            shutil.rmtree(os.path.join(data_path, category))

    return train_path, valid_path


def download_trained_model_and_history(model_path):
    if not os.path.exists(model_path):
        os.system("wget https://sdaaidata.s3-ap-southeast-1.amazonaws.com/pretrained-weights/iti107/session-3/baseline.model.h5")
        os.rename('baseline.model.h5',model_path)
    if not os.path.exists('baseline.history'):
        os.system("wget https://sdaaidata.s3-ap-southeast-1.amazonaws.com/pretrained-weights/iti107/session-3/baseline.history")

if __name__ == '__main__':
    #prepare_data('data', valid_size=0.2, FORCED_DATA_REWRITE=True)
    download_trained_model_and_history(os.path.join('models', 'baseline.model.h5'))

    #download_trained_model_and_history('models')

