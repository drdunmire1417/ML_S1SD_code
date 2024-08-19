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

s1_folder = '/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/S1_mosaic/'
s1_scale_folder = '/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/S1_scaling_factors/'
sc_folder = '/staging/leuven/stg_00024/OUTPUT/devond/S1_ML_project/DATA/snowcover/NEW/'

numeric_features = ['elevation','slope','aspect','fcf','tpi','DayOfSeason','vv_scaled','cr_scaled','lia', 'sc_percum', 'sc_per']
categorical_features = ['snowclass']
all_features = numeric_features.copy()
all_features.extend(categorical_features)



                    ### ------ FUNCTIONS ------ #####   
def crop_xr_nonan(df, var):
    data_crop = df[var].dropna(dim = 'lat', how = 'all').dropna(dim = 'lon', how = 'all')
    lat_crop = data_crop.lat.values
    lon_crop = data_crop.lon.values
    df = df.sel(lat = lat_crop, lon = lon_crop)
    return lat_crop, lon_crop, df

def reproject_m(df_src, df_dest):
    df_src = df_src.rename({'lat': 'y', 'lon': 'x'}).transpose('y', 'x')
    df_dest = df_dest.rename({'lat': 'y', 'lon': 'x'}).transpose('y', 'x')
    df_src.rio.write_crs("EPSG:4326", inplace = True)
    df_dest.rio.write_crs("EPSG:4326", inplace = True)
    df_out = df_src.rio.reproject_match(df_dest, resampling = Resampling.average)
    return df_out.rename({'y': 'lat', 'x': 'lon'})

s1_weights = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1],
                    [1, 3, 3, 3, 3, 3, 3, 3, 3, 3,1],
                    [1, 3, 5, 5, 5, 5, 5, 5, 5, 3,1],
                    [1, 3, 5, 7, 7, 7, 7, 7, 5, 3,1],
                    [1, 3, 5, 7, 9, 9, 9, 7, 5, 3,1],
                    [1, 3, 5, 7, 9, 15, 9, 7, 5, 3,1], 
                    [1, 3, 5, 7, 9, 9, 9, 7, 5, 3,1],
                    [1, 3, 5, 7, 7, 7, 7, 7, 5, 3,1],
                    [1, 3, 5, 5, 5, 5, 5, 5, 3, 3,1],
                    [1, 3, 3, 3, 3, 3, 3, 3, 3, 3,1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1]])

s1_weights = s1_weights / np.sum(s1_weights[:])

def inverse_weighted_rolling_mean(ds):
    vv_roll = sp.ndimage.filters.convolve(ds.vv_scaled.values, s1_weights, mode='constant')
    cr_roll = sp.ndimage.filters.convolve(ds.cr_scaled.values, s1_weights, mode='constant')
    lia_roll = sp.ndimage.filters.convolve(ds.lia.values, s1_weights, mode='constant')
        
    ds2 = xr.Dataset(
            data_vars=dict(
                vv_scaled=(["lat", "lon"], vv_roll),
                cr_scaled=(["lat", "lon"], cr_roll),
                lia=(["lat", "lon"], lia_roll),
            ),
            coords=dict(
                lon=("lon", ds.lon.values),
                lat=("lat", ds.lat.values),
            ),
        )
        
    return ds2

def get_shap(df_nonan, model):
    df_nozero = df_nonan[df_nonan.SD>0]
    
    explainer = shap.TreeExplainer(model.named_steps["actual_estimator"])
    data_pipe = model[:-1].transform(df_nozero[all_features])

    shap_vals = explainer(data_pipe)
    df = pd.DataFrame(columns = data_pipe.columns, data = shap_vals.values)
    df['snowclass'] = df['snowclass_1.0']+df['snowclass_3.0']+df['snowclass_5.0']+df['snowclass_6.0']+df['snowclass_7.0']+df['snowclass_2.0']+df['snowclass_4.0']
    df = df.drop(columns = ['snowclass_1.0','snowclass_3.0','snowclass_5.0','snowclass_6.0','snowclass_7.0','snowclass_2.0','snowclass_4.0'])
    df = df.rename(columns = {'elevation':'elevation_shap','slope':'slope_shap','aspect':'aspect_shap','fcf':'fcf_shap',\
                              'tpi':'tpi_shap','DayOfSeason':'dos_shap','sc_percum':'sc_percum_shap','vv_scaled':'vv_shap',\
                              'cr_scaled':'cr_shap','snowclass':'snowclass_shap', 'lia':'lia_shap', 'sc_per':'sc_perc_shap'})

    df = df.set_index(df_nozero.index)
    
    return df

def add_shap_to_xr(df_x, df, all_var):
    df_x.loc[df.index, 'cr_shap'] = df['cr_shap']
    df_x.loc[df.index, 'vv_shap'] = df['vv_shap']
    df_x.loc[df.index, 'elevation_shap'] = df['elevation_shap']
    df_x.loc[df.index, 'slope_shap'] = df['slope_shap']
    df_x.loc[df.index, 'aspect_shap'] = df['aspect_shap']
    df_x.loc[df.index, 'fcf_shap'] = df['fcf_shap']
    df_x.loc[df.index, 'tpi_shap'] = df['tpi_shap']
    df_x.loc[df.index, 'dos_shap'] = df['dos_shap']
    df_x.loc[df.index, 'tpi_shap'] = df['tpi_shap']
    df_x.loc[df.index, 'snowclass_shap'] = df['snowclass_shap']
    df_x.loc[df.index, 'lia_shap'] = df['lia_shap']
    df_x.loc[df.index, 'sc_perc_shap'] = df['sc_perc_shap']
    df_x.loc[df.index, 'sc_percum_shap'] = df['sc_percum_shap']
    
    elevation_shap = df_x.elevation_shap.values.reshape(all_var.cr_scaled.values.shape)
    elevation_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(elevation_shap=(['lat', 'lon'],  elevation_shap))

    slope_shap = df_x.slope_shap.values.reshape(all_var.cr_scaled.values.shape)
    slope_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(slope_shap=(['lat', 'lon'],  slope_shap))

    aspect_shap = df_x.aspect_shap.values.reshape(all_var.cr_scaled.values.shape)
    aspect_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(aspect_shap=(['lat', 'lon'],  aspect_shap))

    fcf_shap = df_x.fcf_shap.values.reshape(all_var.cr_scaled.values.shape)
    fcf_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(fcf_shap=(['lat', 'lon'],  fcf_shap))

    tpi_shap = df_x.tpi_shap.values.reshape(all_var.cr_scaled.values.shape)
    tpi_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(tpi_shap=(['lat', 'lon'],  tpi_shap))

    dos_shap = df_x.dos_shap.values.reshape(all_var.cr_scaled.values.shape)
    dos_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(dos_shap=(['lat', 'lon'],  dos_shap))

    snowclass_shap = df_x.snowclass_shap.values.reshape(all_var.cr_scaled.values.shape)
    snowclass_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(snowclass_shap=(['lat', 'lon'],  snowclass_shap))

    lia_shap = df_x.lia_shap.values.reshape(all_var.cr_scaled.values.shape)
    lia_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(lia_shap=(['lat', 'lon'],  lia_shap))

    sc_per_shap = df_x.sc_perc_shap.values.reshape(all_var.cr_scaled.values.shape)
    sc_per_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(sc_perc_shap=(['lat', 'lon'],  sc_per_shap))

    cr_shap = df_x.cr_shap.values.reshape(all_var.cr_scaled.values.shape)
    cr_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(cr_shap=(['lat', 'lon'],  cr_shap))

    vv_shap = df_x.vv_shap.values.reshape(all_var.cr_scaled.values.shape)
    vv_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(vv_shap=(['lat', 'lon'],  vv_shap))

    sc_percum_shap = df_x.sc_percum_shap.values.reshape(all_var.cr_scaled.values.shape)
    sc_percum_shap[(all_var.lc==80)|(all_var.lc==200)] = np.nan
    all_var = all_var.assign(sc_percum_shap=(['lat', 'lon'],  sc_percum_shap))

    return df_x, all_var

def prep_data(date, orbit, static_var):
    year = int(date[0:4])
    month = int(date[4:6])
    if month < 8: year = year - 1

    file = f'{s1_folder}S1mosaic_{date}_{orbit}_2.nc'

    s1 = xr.open_dataset(file)
    s1 = s1.mean(dim = 'time') #get rid of time dimension

    s1_scale = xr.open_dataset(f'{s1_scale_folder}S1_{year}_{orbit}_scale.nc')
    s1['vv_scaled'] = s1.g0vv - s1_scale.g0vv
    s1['cr_scaled'] = s1.g0vh - s1.g0vv - s1_scale.cr
    s1 = s1.drop(['g0vv','g0vh'])

    lat_crop, lon_crop, s1 = crop_xr_nonan(s1, 'vv_scaled') #crop out rows/cols of all nan
    static_crop = static_var.sel(lat = slice(lat_crop.max(), lat_crop.min()), lon = slice(lon_crop.min(), lon_crop.max())) #crop static vars to same area
    s1 = reproject_m(s1, static_crop) #reproject S1 to 100m
    s1 = inverse_weighted_rolling_mean(s1)
    #s1 = s1.rolling(lat = 10, lon = 10, center = True, min_periods = 1).mean() #moving mean of window 10 pixels

    #sccum_data = prep_variable(f'{sc_cum_folder}sc_perc_{date}_.nc', lon_crop.min()-1, lon_crop.max()+1, lat_crop.max()+1, lat_crop.min()-1, static_crop, 500)
    sc_data = xr.open_dataset(f'{sc_folder}snowcover_{date}_.nc')
    sc_data = sc_data.sel(lat = slice(lat_crop.max(), lat_crop.min()), lon = slice(lon_crop.min(), lon_crop.max())) #crop data

    all_var = xr.merge([s1, static_crop, sc_data]) #merge all datasets

    df_x = pd.DataFrame({'vv_scaled':all_var.vv_scaled.values.flatten(),'cr_scaled':all_var.cr_scaled.values.flatten(),'lia':all_var.lia.values.flatten(),\
                           'elevation':all_var.dem.values.flatten(),'slope':all_var.slope.values.flatten(),'aspect':all_var.aspect.values.flatten(),\
                           'fcf':all_var.forest.values.flatten(),'tpi':all_var.TPI.values.flatten(),'sc_percum':all_var.sc_percum.values.flatten(),\
                           'snowclass':all_var.snow_class.values.flatten(), 'sc_per':all_var.sc_per.values.flatten()})
    df_x['SD'] = np.nan

    df_x.loc[(df_x.sc_percum>1e20)] = np.nan
    df_x.loc[(df_x.sc_percum<0.25),'SD'] = 0
    df_x.loc[(df_x.SD==0),'lia'] = np.nan

    df_x['DayOfSeason'] = (pd.Timestamp(date) - pd.Timestamp(year, 8,1)).days
    
    return all_var, df_x
