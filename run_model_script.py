from pycaret.regression import *
import xarray as xr
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import shap
from glob import glob
from rasterio.enums import Resampling
import rioxarray
import scipy as sp
import scipy.ndimage

import sys
sys.path.insert(1, '/data/leuven/357/vsc35702/_7_run_model/')
from functions import *


                    ### ------ ML model version ------ #####
version = 'final_model_xg'

                    ### ------ FOLDERS ------ #####
s1_folder = '/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/S1_Alps_1km/'
s1_scale_folder = '/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/S1_1km_scaling_factors/'
sc_folder = '/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/snowcover/MODIS_IMS_cumulative_100m/'

                    ### ------ UNIVERSAL VARIABLES ------ #####
numeric_features = ['elevation','slope','aspect','fcf','tpi','DayOfSeason','vv_scaled','cr_scaled','lia', 'sc_percum', 'sc_per']
categorical_features = ['snowclass']
all_features = numeric_features.copy()
all_features.extend(categorical_features)

static_var = xr.open_dataset('/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/time_independent_var/all_static_var.nc').transpose('lat','lon')
snow_classes = xr.open_dataset('/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/time_independent_var/snowclass.nc').transpose('lat','lon')
landcover = xr.open_dataset('/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/time_independent_var/landcover_1.nc').transpose('lat','lon')
static_var['snow_class'] = snow_classes['class']
static_var['lc'] = landcover['lc']
grid = xr.open_dataset('/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/time_independent_var/grid_lowres.nc')
static_var.rio.write_crs("EPSG:4326", inplace = True)
glaciers = xr.open_dataset('/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/time_independent_var/glacier_raster.nc')

model = load_model(f'/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/models/{version}')

#inputt = sys.argv[1]
s1_files = glob(f'/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/S1_mosaic/*.nc')[1:3]
#print(s1_files)

output_folder = '/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/OUTPUT/SD_temp/'
finished_files = glob(f'{output_folder}*.nc')
print(finished_files)
                    ##### ------ CODE ------ #####
    
for f in s1_files:
    date = f.split('_')[-3]
    month = date[4:6]
    ym = date[0:6]
    orbit = f.split('_')[-2]  
    year = date[0:4]
    print(date, orbit,year)
    
    out_file = f'{output_folder}S1_ml_SD_{date}_{orbit}.nc'
    
    if out_file not in finished_files and orbit not in ['051','037'] and month not in ['06','07','08'] and ym not in ['201501', '201502', '201503', '201504', '201505']:
    
        print('prepping data')
        all_var, df_x = prep_data(date, orbit, static_var)

        df_nonan = df_x[all_features]
        df_nonan = df_nonan.dropna()#.reset_index(drop = True)

        print('running model')
        if len(df_nonan)>0:
            df_nonan['SD'] = model.predict(df_nonan)
            df_x.loc[df_nonan.index, 'SD'] = df_nonan['SD']
            df_x.loc[df_x.SD < 0, 'SD'] = 0

        SD = df_x.SD.values.reshape(all_var.cr_scaled.values.shape)
        SD[(all_var.lc==80)|(all_var.lc==200)] = np.nan
        all_var = all_var.assign(SD=(['lat', 'lon'],  SD))

        print('saving')
        sd = all_var.SD#.drop(['TPI','slope','aspect','dem','forest','grid','snow_class','lc'])
        sd = reproject_m(sd, glaciers)
        sd = sd.where(glaciers.glacier !=1).SD
        sd.to_netcdf(f'{out_file}')

        
        
        #all_var.to_netcdf(f'{out_file}') 
        #with open(f'{output_folder}p_{date}_{orbit}.p', 'wb') as fp:
            #pickle.dump(df_x, fp)
            
    else:
        print('Already done', date, orbit)
