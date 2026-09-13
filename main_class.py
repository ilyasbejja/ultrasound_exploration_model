import cv2
import os
from pathlib import Path
import numpy as np
import random
import subprocess
import json
from tqdm import tqdm
import time

class cls_frame():
    objects = []
    
    def __init__(self,current_folder:str = None, frames_folder: str=None, output_folder:str=None):
        
        self.frames_folder = Path(str(frames_folder))
        self.current_folder = Path(str(current_folder))
        self.output_folder = output_folder
        self.__class__.objects.append(self)
        self.dico = {}
        self.n_frames = None
        self.extraction_duration = None
        self.getting_coord_duration = None
        self.video_path = None



    
    def get_crop_coords(self, frame = None, folder:str= None, set_as_default:bool = None):
        
        if folder is None:
            roi = cv2.selectROI("Select Borders", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows
        else:
            folder = Path(str(folder))
            files = [f for f in folder.iterdir() if f.is_file()]
            file = random.choice(files)
            frame = cv2.imread(str(file))
            roi = cv2.selectROI("Select Borders", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows
            
        x , y, w, h = roi
        x_start, x_end = x, x+w
        y_start, y_end = y, y+h
        if set_as_default:
            self.x_c_t = (x_start, x_end)
            self.y_c_t = (y_start, y_end)
        return (x_start, x_end), (y_start, y_end)
    
    def set_crop_coors(self, x_t: float, y_t: float):
        self.x_c_t = x_t
        self.y_c_t = y_t
        
    def crop_frame(self, frame,x_tuple: tuple =None, y_tuple: tuple = None):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        
        return frame[y_tuple[0]:y_tuple[1], x_tuple[0]:x_tuple[1]]  
    
    
    def create_folders(self, output_dir:str = None):
        if output_dir is None:
            output_dir = self.output_folder
        folders = ["b-mode", "doppler-mode", "measurement-mode","split_frame", "pw-doppler", "non-usable"]
        for folder in folders:
            (output_dir / folder).mkdir(parents=True, exist_ok=True)
            
    
    def set_frames_folder(self, path:str):
        
        self.frame_folder = Path(path)
        count_files = sum([1 for file in self.frame_folder.iterdir()])
        print(f"Folder name {self.frame_folder.name} was set!")
        print(f"{count_files} file was detected!")
        
        
    def detect_measurement_mode(self, frame,x_tuple: tuple =None, y_tuple: tuple = None, threshold: float= 0.002)-> bool:
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t  
                    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = self.crop_frame(frame, x_tuple, y_tuple)
        
        lower_yellow = np.array([32,100,100])
        upper_yellow = np.array([35,255,255])
        
        tmp = cv2.inRange(frame, lower_yellow, upper_yellow)
        
        
        num_pixels = frame.shape[0] * frame.shape[1]
        
        tmp_meas = cv2.countNonZero(tmp)
        
        return (tmp_meas / num_pixels) >= threshold        
    def detect_doppler_mode(self, frame, x_tuple: tuple = None, y_tuple: tuple= None, threshold_red1=0.0002,  threshold_red2=0.0001, threshold_blue=0.1)-> bool:
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t      
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = self.crop_frame(frame, x_tuple, y_tuple)
        # red scale 
        lower_red1 = np.array([0,100,80])
        upper_red1 = np.array([10,255,255])
        
        lower_red2 = np.array([170,50,50])
        upper_red2 = np.array([180,255,255])
        # blue scale
        lower_blue = np.array([112,100,80])
        upper_blue = np.array([130,255,255])

        
        tmp_red1 = cv2.inRange(frame , lower_red1, upper_red1)
        tmp_red2 = cv2.inRange(frame , lower_red2, upper_red2)

        tmp_blue = cv2.inRange(frame , lower_blue, upper_blue)
        num_pixels = frame.shape[0] * frame.shape[1]
        
        tmp_red1 = cv2.countNonZero(tmp_red1)
        tmp_red2 = cv2.countNonZero(tmp_red2)
        tmp_blue = cv2.countNonZero(tmp_blue)
        
        if (tmp_blue/num_pixels)>=threshold_blue or (tmp_red1/num_pixels)>=threshold_red1 or (tmp_red2/num_pixels)>=threshold_red2:
            return True
        else:
            return False
               
    def save_measurement_frame(self,frame,x_tuple: tuple = None, y_tuple: tuple= None, frame_name:str = "frame_na", folder_path:str = None):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        frame = self.crop_frame(frame=frame, x_tuple=x_tuple,y_tuple=y_tuple)             
        if folder_path is None:
            folder_path = self.output_folder / "measurement-mode"
            
        folder_path = Path(str(folder_path))
        image_path = folder_path / f"{frame_name}"
        
        cv2.imwrite(str(image_path), frame)
        #print(f"{frame_name} was saved as {folder_path.name} frame")
        
    def save_b_mode_frame(self,frame, x_tuple: tuple = None, y_tuple: tuple= None,frame_name:str = "frame_na", folder_path:str = None):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        frame = self.crop_frame(frame=frame, x_tuple=x_tuple,y_tuple=y_tuple)   
        if folder_path is None:
            folder_path = self.output_folder / "b-mode"
            
        folder_path = Path(str(folder_path))
        image_path = folder_path / f"{frame_name}"
        
        cv2.imwrite(str(image_path), frame)
        #print(f"{frame_name} was saved as {folder_path.name} frame")
        
    def save_doppler_mode_frame(self,frame,x_tuple: tuple = None, y_tuple: tuple= None, frame_name:str = "frame_na", folder_path:str = None):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        frame = self.crop_frame(frame=frame, x_tuple=x_tuple,y_tuple=y_tuple)   
        if folder_path is None:
            folder_path = self.output_folder / "doppler-mode"
            
        folder_path = Path(str(folder_path))
        image_path = folder_path / f"{frame_name}"
        
        cv2.imwrite(str(image_path), frame)
        #print(f"{frame_name} was saved as {folder_path.name} frame")
    def timing(self, start, end):
        duration = end - start
        return duration    

        
    def detect_split_frame(self, frame, x_tuple: tuple = None, y_tuple: tuple= None, threshold: float = 0.97):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        
        x_avg = (x_tuple[0] + x_tuple[1]) //2
        pourc = int((x_tuple[1] - x_tuple[0]) * 0.05)
        x_tuple = (x_avg-pourc, x_avg+pourc)
        
        
        pourc = int((y_tuple[1] - y_tuple[0]) * 0.05)
        y_tuple_tuple = (y_tuple[0]+pourc, y_tuple[1]-pourc)
        
            
        
        liste = []
        
        
        frame_ = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
        cropped_frame = self.crop_frame(frame_, x_tuple, y_tuple)
        num_cropped_pixels = cropped_frame.shape[0] * cropped_frame.shape[1]
        
        tmp = cropped_frame <=8
        
        liste = np.mean(tmp, axis=0)
        return max(liste) >=threshold
        
    
    def save_split_frame(self,frame,x_tuple: tuple = None, y_tuple: tuple= None, frame_name:str = "frame", folder_path:str = None):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        frame = self.crop_frame(frame=frame, x_tuple=x_tuple,y_tuple=y_tuple)  
        if folder_path is None:
            folder_path = self.output_folder / "split_frame"
            
        folder_path = Path(str(folder_path))
        image_path = folder_path / f"{frame_name}"
        
        cv2.imwrite(str(image_path), frame)
        #print(f"{frame_name} was saved as {folder_path.name} frame")
        
    def count_folders_output(self, dico:dict):
        liste = []
        for folder in sorted(self.output_folder.iterdir()):
            
            n = sum([1 for file in folder.iterdir()])
            print(f"{n} detected as {folder.name}")
            liste = liste + [n]
            #string  = "n_" + str(folder.name)
            #dico[str(folder.name)].append(n)
        dico['n_b-mode'].append(liste[0])
        dico['n_doppler-mode'].append(liste[1])
        dico['n_measurement-mode'].append(liste[2])
        dico['n_non-usable'].append(liste[3])
        dico['n_split_frame'].append(liste[5])
        dico['n_pw-doppler_frame'].append(liste[4])
        
        
    def detect_doppler_threshold(self, frames_folder_:str = None, x_tuple:tuple = None, y_tuple:tuple=None):
        if frames_folder_ is None:
            frames_folder_ = self.frames_folder
        else:
            frames_folder_ = Path(str(frames_folder_))
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        
        # red scale 
        lower_red1 = np.array([0,50,50])
        upper_red1 = np.array([10,255,255])
        
        lower_red2 = np.array([170,50,50])
        upper_red2 = np.array([180,255,255])
        # blue scale
        lower_blue = np.array([90,50,50])
        upper_blue = np.array([130,255,255])

        for file in sorted(frames_folder_.iterdir()):
            frame = cv2.imread(str(file))
            frame = self.crop_frame(frame, x_tuple, y_tuple)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            tmp_red1 = cv2.inRange(frame , lower_red1, upper_red1)
            tmp_red2 = cv2.inRange(frame , lower_red2, upper_red2)

            tmp_blue = cv2.inRange(frame , lower_blue, upper_blue)
            num_pixels = frame.shape[0] * frame.shape[1]
            
            tmp_red1 = cv2.countNonZero(tmp_red1)
            tmp_red2 = cv2.countNonZero(tmp_red2)
            tmp_blue = cv2.countNonZero(tmp_blue)
            red1_ratio = round(tmp_red1 / num_pixels, 4)
            red2_ratio = round(tmp_red2 / num_pixels, 4)
            blue_ratio = round(tmp_blue / num_pixels, 4)
            print(f"{file.name} ratio red1:{red1_ratio}, red2:{red2_ratio}, blue:{blue_ratio}")
            
    def detect_measurement_threshold(self, frames_folder:str = None, x_tuple:tuple = None, y_tuple:tuple=None):
        
        if frames_folder is None:
            frames_folder = self.frames_folder
        else:
            frames_folder = Path(str(frames_folder))
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
            
        for file in sorted(frames_folder.iterdir()):
            frame = cv2.imread(str(file))
            frame = self.crop_frame(frame, x_tuple, y_tuple)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower_yellow = np.array([20,50,50])
            upper_yellow = np.array([32,255,255])
            
            tmp = cv2.inRange(frame, lower_yellow, upper_yellow)
            
            
            num_pixels = frame.shape[0] * frame.shape[1]
            
            tmp_meas = cv2.countNonZero(tmp)
            ratio = round(tmp_meas / num_pixels, 3)
            
            print(f"{file.name} ratio : yellow:{ratio}")
        
    def detect_non_usable_threshold(self, frames_folder:str = None, x_tuple: tuple = None, y_tuple: tuple = None):
        
        if frames_folder is None:
            frames_folder = self.frames_folder
        else:
            frames_folder = Path(str(frames_folder))
        
        for file in sorted(frames_folder.iterdir()):
            
            frame = cv2.imread(str(file))
            liste = []
        
            frame_ = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
            cropped_frame = self.crop_frame(frame_, x_tuple, y_tuple)
            num_cropped_pixels = cropped_frame.shape[0] * cropped_frame.shape[1]
            
            tmp = cropped_frame <=8
            
            liste = np.mean(tmp, axis=0)
            print(f"{file.name} ratio : dark: {round(max(liste), 4)}")
        
    def extract_vd_frames(self,video_path:str, set_frames_fold: bool= False):
        vid_path = Path(str(video_path))
        frames_path = self.current_folder / f"{vid_path.name}_frames"
        frames_path.mkdir(parents=True, exist_ok=True)

        self.current_folder = frames_path
        
        frame_ = str(frames_path) + f"/frame_%04d.png"
        
        cmd = ['ffmpeg', '-i', video_path, '-vf',"select='gt(scene,0.02)'", '-vsync', 'vfr', frame_]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if set_frames_fold:
            self.frames_folder = frames_path
        
        num_f = sum([1 for f in frames_path.iterdir()])
        print(f"{num_f} frame was successfully extracted from {str(vid_path.name)[:7]}...")
        #self.dico['video_id'].append(str(vid_path.name))
        #self.dico['n_extracted_frames'].append(num_f)
        
        return num_f
        
    
    def detect_non_usable_frame(self, frame, x_tuple: tuple = None, y_tuple: tuple= None, dark_threshold=10, ratio_threshold=0.95):

        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = self.crop_frame(frame, x_tuple, y_tuple)

        dark_pixels = np.sum(frame < dark_threshold)
        n_pixels = frame.shape[0] * frame.shape[1]
        ratio = dark_pixels / n_pixels

        return ratio >= ratio_threshold
    
    def save_non_usable_frame(self,frame,x_tuple: tuple = None, y_tuple: tuple= None, frame_name:str = "frame_na", folder_path:str = None):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        frame = self.crop_frame(frame=frame, x_tuple=x_tuple,y_tuple=y_tuple)   
        if folder_path is None:
            folder_path = self.output_folder / "non-usable"
            
        folder_path = Path(str(folder_path))
        image_path = folder_path / f"{frame_name}"
        
        cv2.imwrite(str(image_path), frame)
        #print(f"{frame_name} was saved as {folder_path.name} frame")

    def get_duration_fast(self):
        video_path = self.video_path


        command = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            str(video_path)
        ]
        
        try:
            # Run the command and capture the output
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Convert the string output (e.g., "12.345") to a float
            duration = float(result.stdout.strip())
            return duration
            
        except Exception as e:
            print(f"Error reading {video_path}: {e}")
            return 0.0
        
    def cal_brightness_score(self,image_path):
        frame = cv2.imread(str(image_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = frame.shape
        x_start , x_end = int(width * 0.25),  int(width * 0.75)
        y_start , y_end = int(height * 0.25),  int(height * 0.75)


        ones = np.ones_like(frame, dtype=bool)
        ones[y_start:y_end, x_start:x_end] = False
        tmp = frame[ones]
        darkness = np.mean(tmp)




        extracted_frame = frame[y_start:y_end, x_start:x_end]
        brightness = np.mean(extracted_frame)


        ratio = brightness / (darkness + 1e-6)
        return ratio

    def detect_brightest_frame(self,frames_path=None):
        if frames_path is None:
            frames_path = self.current_folder
        

        best_frame = None
        score_max = 0
        for frame in tqdm(sorted(frames_path.iterdir())):
            ratio = self.cal_brightness_score(frame)
            if ratio > score_max:
                score_max=ratio
                best_frame= frame

        return best_frame


    def detect_pw_doppler(self, frame,x_tuple: tuple = None, y_tuple: tuple= None, SPECTRAL_THRESHOLD=2000):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t

        frame = self.crop_frame(frame, x_tuple, y_tuple)

        height, width, _ = frame.shape
        
        # 1. Crop to the bottom half of the screen. 
        # We only care about the graph at the bottom, which prevents 
        # accidentally triggering on the 2D ultrasound image at the top.
        bottom_half = frame[int(height * 0.5):, :]
        
        # 2. Convert to HSV for accurate color detection
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        
        # 3. Define the HSV color range for the Golden/Orange waveform
        # In OpenCV, Hue 10 to 35 covers orange to golden-yellow.
        lower_gold = np.array([10, 80, 80])
        upper_gold = np.array([35, 255, 255])
        
        gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # 4. Count the matching pixels
        gold_pixels = cv2.countNonZero(gold_mask)
        
        # 5. Set a high threshold. 

        
        if gold_pixels > SPECTRAL_THRESHOLD:
            return True
        else:
            return False

    def save_pw_doppler_frame(self,frame,x_tuple: tuple = None, y_tuple: tuple= None, frame_name:str = "frame_na", folder_path:str = None):
        if x_tuple is None:
            x_tuple = self.x_c_t
        if y_tuple is None:
            y_tuple = self.y_c_t
        frame = self.crop_frame(frame=frame, x_tuple=x_tuple,y_tuple=y_tuple)   
        if folder_path is None:
            folder_path = self.output_folder / "pw-doppler"
            
        folder_path = Path(str(folder_path))
        image_path = folder_path / f"{frame_name}"
        
        cv2.imwrite(str(image_path), frame)
        #print(f"{frame_name} was saved as {folder_path.name} frame")
        
    def get_best_coords(self,frames_path=None, save_coords=True):
        if frames_path is None:
            frames_path = self.current_folder

        best_frame = self.detect_brightest_frame()

        frame = cv2.imread(str(best_frame))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        NULL, binary_frame = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((30,30), np.uint8)

        output = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel=kernel)

        contours, NULL = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        #padding = 20
        #x = max(0, x - padding)
        #y = max(0, y - padding)
        #w = min(frame.shape[1] - x, w + (padding * 2))
        #h = min(frame.shape[0] - y, h + (padding * 2))
        
        if save_coords:
            x_start , x_end = x, x+w
            y_start , y_end = y, y+h
            x_tuple = (x_start+10, x_end+20)
            y_tuple = (y_start, y_end)
            self.set_crop_coors(x_t=x_tuple, y_t= y_tuple)

        return (x_start, x_end), (y_start, y_end)
    
    @classmethod
    def extract_frames(cls, input_folder:str,current_folder:str, outputs_folder:str):
        for folder in tqdm(sorted(Path(input_folder).iterdir())):
            video = [f for f in folder.iterdir()][0]

            
                
            tmp = Path(outputs_folder) / f"{str(folder.name)}"
            tmp.mkdir(parents=True, exist_ok=True)
            o = cls_frame(current_folder=current_folder, output_folder=tmp)
            o.video_path = video
            
            start = time.perf_counter()
            num_f = o.extract_vd_frames(video_path=str(video), set_frames_fold=True)
            end = time.perf_counter()
            o.extraction_duration = o.timing(start=start, end=end)
            o.n_frames = num_f

            

            
    @classmethod
    def get_coords(cls,current_folder:str=None, outputs_folder:str=None, save_as_csv: bool= None):
        if current_folder is None:
            for o in cls.objects:
                o.create_folders()
                start = time.perf_counter()
                o.get_best_coords()
                end = time.perf_counter()
                o.getting_coord_duration = o.timing(start=start, end=end)
                
        else:
            for folder in Path(current_folder).iterdir():

                tmp = Path(outputs_folder) / f"{str(folder.name)}"
                tmp.mkdir(parents=True, exist_ok=True)
                o = cls_frame(frames_folder=str(folder), output_folder=tmp)
                o.create_folders()
                o.current_folder = folder
                start = time.perf_counter()
                o.get_best_coords()
                end = time.perf_counter()
                o.getting_coord_duration = o.timing(start=start, end=end)


        




    @classmethod
    def classify(cls, dico:dict):
        for o in tqdm(cls.objects):
            start = time.perf_counter()
            for file in sorted(o.current_folder.iterdir()):
                frame = cv2.imread(str(file))
                
                if o.detect_measurement_mode(frame):
                    o.save_measurement_frame(frame,frame_name= str(file.name))
                
                    
                elif o.detect_non_usable_frame(frame):
                    o.save_non_usable_frame(frame,frame_name=str(file.name))

                elif o.detect_split_frame(frame): 
                    o.save_split_frame(frame,frame_name=str(file.name))

                elif o.detect_pw_doppler(frame): 
                    o.save_pw_doppler_frame(frame,frame_name=str(file.name))

                elif o.detect_doppler_mode(frame):
                    o.save_doppler_mode_frame(frame, frame_name=str(file.name))

                else:
                    o.save_b_mode_frame(frame, frame_name=str(file.name))       
            o.count_folders_output(dico)
            end = time.perf_counter()
            o.classification_duration = o.timing(start=start, end=end)

            dico['folder_name'].append(o.current_folder.name)
            dico['video_id'].append(o.video_path.name)
            dico['n_extracted_frames'].append(o.n_frames)
            dico['extraction_duration'].append(o.extraction_duration)
            dico['getting_coord_duration'].append(o.getting_coord_duration)
            dico['classification_duration'].append(o.classification_duration)
            dico['video_duration'].append(o.get_duration_fast())
            

  
        
        
            
            
            

    
        
    
        