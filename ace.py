import weatherbench2
import xarray as xr
import numpy as np
from weatherbench2 import config
import cv2
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    
    image_normalized = (image - min_val) / (max_val - min_val)
    
    image_normalized = (image_normalized * 255).astype(np.uint8)
    
    return image_normalized

def ACE(t2, t1, gt):
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    t1 = normalize_image(t1)
    t2 = normalize_image(t2)
    gt = normalize_image(gt)
    flow1 = optical_flow.calc(t1, t2, None)
    flow2 = optical_flow.calc(t1, gt, None)
    AE = abs(flow1-flow2)
    AE = AE.mean()

    h, w = t1.shape[:2]
    map1, map2 = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    map1 = map1 + flow1[..., 0]
    map2 = map2 + flow1[..., 1]    

    remapped1 = cv2.remap(t2, map1.astype(np.float32), map2.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    h, w = t1.shape[:2]
    map11, map22 = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    map11 = map11 + flow2[..., 0]
    map22 = map22 + flow2[..., 1]   
    
    remapped2 = cv2.remap(gt, map11.astype(np.float32), map22.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    
    CE1 = abs(t1 - remapped1)
    CE2 = abs(t1 - remapped2)
    
    total_CE = abs(CE1-CE2)
    total_CE = ((total_CE).mean())/255.
    ACE = AE + total_CE/AE    
    return ACE

if __name__ == "__main__":
    obs_data = np.load(path1)
    forecast_data = np.load(path2)
    obs_data_target = np.load(path3)
    ace = ACE(np.array(forecast_data), np.array(obs_data), np.array(obs_data_target))
