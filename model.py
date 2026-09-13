import os
import pandas as pd 
from main_class import cls_frame

current = "data/current"
outputs = "data/output"
inputs = "data/input"

os.makedirs(current, exist_ok=True)
os.makedirs(outputs, exist_ok=True)
os.makedirs(inputs, exist_ok=True)

dico = {'folder_name':[],
        'video_id':[],
        "video_duration":[],
        'n_extracted_frames':[],
        'extraction_duration':[],
        'getting_coord_duration':[],
        'classification_duration':[],
        'n_b-mode':[],
        'n_doppler-mode':[],
        'n_measurement-mode':[],
        'n_split_frame':[],
        'n_non-usable':[],
        'n_pw-doppler_frame':[]
       }

print("Extracting frames ... ( can take a while, just wait ....) ")
cls_frame.extract_frames(input_folder=inputs, current_folder=current, outputs_folder=outputs)
print("Extracting frames is Done.!")
print("Extracting best coordiantions ...")
cls_frame.get_coords()
print("DONE.!")
print("Classifying ultrasound frames ...!")
cls_frame.classify(dico)
print("Classification is DONE.!")

df = pd.DataFrame(dico)
df.to_csv("data/results.csv", index=False)