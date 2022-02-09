# module.py
# ----------------------------------------------------------------------------------------------------------------------
# authors: Florian Hellwig & Henrik Schmidt
# date: 28.02.2022
# ----------------------------------------------------------------------------------------------------------------------
# Package for randomforest classification of Sentinel-1 radardata with Copernicus Wetland Highresolution product as reference
# Creates project folderstructure and imports the necessary  input data.
# Prepares the data for randomforest classification and splits the data in train and test samples.
# The users can choose between two different validation- and hyperparametertuning methods (random- and grid-search) with 10-folds Cross-Validation
# Automatically evaluates the results in a confusion matrix with helpful numerical-measures and exports the classification results as a map.
# ----------------------------------------------------------------------------------------------------------------------
# System
import re
import os
import sys
import time
import glob
import shutil
from progressbar import ProgressBar
from distutils.util import strtobool
# Geodata handling
import numpy as np
import pandas as pd
from osgeo import gdal,osr
from datetime import datetime
# Random forest classification
import joblib
from scipy import ndimage
from sklearn import metrics as metrics
from sklearn import ensemble as ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt


def user_yes_no_query(question):
    """
    Asks Yes/No Question to user and returns bool.

    Parameters
    ----------
    question: str
        dialog text for the asked question

    Returns
    -------
    bool
        true for y, false for n
    """
    # User has to answer the question with 'y' or 'n' in command line
    sys.stdout.write('%s [(y/1)/(n/0)]\n' % question)
    time.sleep(.2)  # Debugging timer to print the question before the response opportunities
    while True:
        try:
            # Convert string to boolian
            return strtobool(input().lower())
        except ValueError:
            # Invalid answers will trigger repeat response
            sys.stdout.write('Please respond with \'y/1\' or \'n/0\'.\n')


def create_project_structure(foldername, workspace):
    """
    Create the project folder structure.

    Parameters
    ----------
    foldername: str
        name of the parent project folder
    workspace: str
        path to the project workspace

    Return
    ------
    folders: list
        list of paths to the project subfolders in the workspace
    """
    # List of the filed folders
    dirlist = os.listdir(workspace)
    # Combine the workspace path with the target project foldername
    outfile = os.path.join(workspace, foldername)
    # Create subfolders for the different project products
    out_ref = os.path.join(outfile, '0_input_wetland')
    out_class = os.path.join(outfile, '1_input_sentinel1')
    out_ctrain = os.path.join(outfile, '2_class_train_test')
    out_cres = os.path.join(outfile, '3_class_results')
    # Create a list of all created paths
    folders = [outfile, out_ref, out_class, out_ctrain, out_cres]
    # Create the stored folder paths from folders, if they don't exist.
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    return folders


def check_project_structure(foldername, workspace):
    """
    Check if the projectfolder-structure already exists.
    If not create it or ask the user for another name or reuse the existing structure.

    Parameters
    ----------
        foldername: str
            name of the parent project folder
        workspace: str
            path to the project workspace

    Return
    ------
    folders: list
        list of paths to the project subfolders in the workspace
    """
    # List of subfolders
    dirlist = os.listdir(workspace)
    # Dialog text for Y/N Question
    question = '{} already exists. Do you want to create a new folder?(y) Or do you want to use the existing folder?(n)'.format(
        foldername)
    # If the project structure is not already existing, this variable will be set False
    skip_create_project_structure = True
    # If the Directory list is empty just create the project structure
    if not dirlist:
        folders = create_project_structure(foldername, workspace)
    # If not, ask the user:
    else:
        for directory in dirlist:
            if foldername == directory:
                if user_yes_no_query(question) == True:
                    if skip_create_project_structure: print('Creating new folder...')
                    # If the user wants to create a new folder ask for a new non-existing foldername.
                    while foldername in dirlist:
                        print('Please rename your folder {}. '.format(foldername))
                        foldername = input("New Foldername:")
                    # Create the folderstructure
                    folders = create_project_structure(foldername, workspace)
                # If the user wants to reuse the existing folderstructure, use the existing folderstructure.
                else:
                    if skip_create_project_structure: print('Using the already existing folder...')
                    folders = create_project_structure(foldername, workspace)
            # Foldername nonexisitng, create folderstructure.
            else:
                folders = create_project_structure(foldername, workspace)
            skip_create_project_structure = False

    return folders


def copy_files(filelist, folder_number, folders):
    """
    Iterate through filelist and copy the files if there are non-existing in the project-structure.

    Parameters
    ----------
    filelist: list
        list of input data
    folder_number: int
        index for subfolder in folders
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    destination_path_filename: str
        if condition met: path to copied QGISColour.txt
    changed: bool
        if condition not met: status info: true if the amount of input data has changed, false if vice versa
    """
    # Iterate through the filelist
    changed = False # If the number of input data has changed set True
    time.sleep(.5)  # Delay for the progressbar
    pbar = ProgressBar()
    for file in pbar(filelist):
        # Fetch the filenames, join them to the project-folder
        filename_source = os.path.basename(os.path.normpath(file))
        destination_path_filename = os.path.join(folders[folder_number], filename_source)
        # If the file is not existing copy them
        if not os.path.exists(destination_path_filename):
            shutil.copy2(file, folders[folder_number])
            if folder_number == 2:
                changed = True
        else:
            continue
    # Reference data:
    if folder_number == 1:
        return destination_path_filename
    # Sentinel-1 data:
    else:
        return changed


def copy_sentinel1_data(origin_paths, folders):
    """
    Copy Sentinel-1 input data to the project-folder.

    Parameters
    ----------
    origin_paths: list
        list containing paths to the sentinel-1 data (VH,VV)
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    s1_stack_changed: bool
        status info: true if there is a change in number of sentinel-1 files, false if vice versa
    """
    # If the stack is changed in number this variable will be set true
    s1_stack_changed = False  # If there is a change in number of sentinel-1 files set True
    s1_data_existing = False  # If there isn't sentinel-1 data set False
    # Iterate through VV and VH data paths
    for origin_path in origin_paths:
        print('copying files from {}'.format(origin_path))
        # List all files in one path
        if os.listdir(origin_path):
            # Fetch all tif-files in the specified path
            filelist = glob.glob(r'{}\*.tif'.format(origin_path))
            # Iterate through the filelist
            s1_stack_changed = copy_files(filelist, 2, folders)
            # Sentinel-1 files copied so set True
            s1_data_existing = True
        else:
            # Never copied Sentinel-1 files set False
            if s1_data_existing == False:
                s1_data_existing = False
    # Raise RuntimeError
    if not s1_data_existing:
        raise RuntimeError('There is no sentinel-1 data.')

    return s1_stack_changed


def copy_wetland(origin_path, folders):
    """
    Copy reference input data to the project-folder.

    Parameters
    ----------
    origin_path: str
        parent path to the Copernicus Wetland HRes reference data
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------

    """
    print('copying files from {}'.format(origin_path))
    # List all files in path
    if os.listdir(origin_path):
        # Fetch all tif-files in the specified path
        filelist = glob.glob(r'{}\*.tif'.format(origin_path))
        # Raise RuntimeError if there are no files in path
        if len(filelist) == 0:
            raise RuntimeError('There is no reference data.')
        copy_files(filelist, 1, folders)


def copy_wetland_color(origin_path, folders):
    """
    Copy reference input data to the project-folder.

    Parameters
    ----------
    origin_path: str
        path to the wetland reference tif data
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    destination_path_filename: str
        path to copied QGISColour.txt
    """
    print('copying files from {}'.format(origin_path))
    if os.listdir(origin_path):
        # Fetch specific text-files in the specified path
        filelist = glob.glob(r'{}\*QGISColour.txt'.format(origin_path))
        # Terminate the function, if there is no QGISColour.txt
        if len(filelist) == 0:
            raise RuntimeError('QGISColour.txt for reference data not found.')
        destination_path_filename = copy_files(filelist, 1, folders)

    return destination_path_filename


def sentinel_stack(folders):
    """
    Convert Sentinel-1 tifs to space-time-datacube.

    Parameters
    ----------
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    array_s1_stack: numpy.ndarray
        sentinel-1 data stack
    metadata_stack: list
        list with metadata informations
    """
    print('creating Sentinel-1 stack from {}'.format(folders[2]))
    # Fetch all the files to be processed
    files_to_process = glob.glob(os.path.join(folders[2], '*.tif'))
    metadata_stack = []  # Metadata
    array_s1_stack = []  # Sentinel-1 data (2-D array: tif pixel-values, time)
    # Create an empty pandas dataframe
    df = pd.DataFrame()
    time.sleep(.5)  # Delay for progressbar
    pbar = ProgressBar()
    for file in pbar(files_to_process):
        # Extract metadata from sentinel-1 tif and append them to metadata-stack
        filename = os.path.basename(file)
        acquisition_date = datetime.strptime(filename[12:20], '%Y%m%d').strftime('%Y-%m-%d')
        acquisition_time = datetime.strptime(filename[21:27], '%H%M%S').strftime('%H:%M:%S')
        polarisation = filename[28:30]
        orbit_direction = filename[10:11]
        metadata = [filename, acquisition_date, acquisition_time, polarisation, orbit_direction]
        metadata_stack.append(metadata)

        # Create a space-time cube
        temp_raster = gdal.Open(file, gdal.GA_ReadOnly)
        # Error if the tif has more or less then 1 band
        if temp_raster.RasterCount != 1:
            array_s1_stack = None
            metadata_stack = None
            raise RuntimeError('The Sentinel-1 data is not in the required format (bandnumber is not 1).')
        # Transfrom tif to array
        temp_array = np.array(temp_raster.GetRasterBand(1).ReadAsArray())
        # Set -99 values to NaN
        temp_array = np.where(temp_array == -99, np.nan, temp_array)
        # 2-D array to 1-D array and append to data-stack
        array_s1_stack.append(temp_array.flatten())
        # Clear memory
        temp_raster = None
        temp_array = None

    return array_s1_stack, metadata_stack


def merge(ds_list, ds_merge_name):
    """
    Merges every dataset in ds_list to one dataset and save it.

    Parameters
    ----------
        ds_list: list
            list of paths to the datasets
        ds_merge_name: str
            name of the merged dataset

    Returns
    -------
    """
    # Create vrt from ds_list
    vrt = gdal.BuildVRT('merge.vrt', ds_list)
    # Save vrt to tif with compression
    merge_vrt = gdal.Translate('{}'.format(ds_merge_name), vrt, creationOptions=["COMPRESS=LZW"])
    # Clear memory
    vrt = None


def intersect(ds_list, ds_ref, folders, nan):
    """
    Checks for all datasets from ds_list whether they intersect with ds_ref and returns the names of those that do.

    Parameters
    ----------
        ds_list: list
            list of paths to the datasets
        ds_ref: object
            reference dataset (gdal object)
        folders: list
            list of paths to the project subfolders in the workspace
        nan: int
            explicite value for NaN, e.g. 255

    Returns
    -------
        ds_list_intersect: list
            list of paths to the datasets that intersect with ds_ref
    """
    # Create empty list
    ds_list_intersect = []

    # Get epsg from ds_ref
    proj = osr.SpatialReference(wkt=ds_ref.GetProjection())
    proj.AutoIdentifyEPSG()
    epsg_ref = proj.GetAttrValue('AUTHORITY', 1)

    # Get extent and spatial resolution from ds_ref
    gt = ds_ref.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * ds_ref.RasterXSize
    miny = maxy + gt[5] * ds_ref.RasterYSize
    projWin = [minx, miny, maxx, maxy]

    # Check for intersection
    for ds_name in ds_list:
        ds = gdal.Open(ds_name)
        try:
            # Georeferencing files to reference file
            ds_out = gdal.Warp("ds.tif", ds, dstSRS="EPSG:{}".format(epsg_ref),
                               resampleAlg="mode", outputBounds=projWin)
            ds_out_array = ds_out.GetRasterBand(1).ReadAsArray()
            # Set nan to np.nan
            ds_out_nparray = np.where(ds_out_array == nan, np.nan, ds_out_array)
            # If dataset does only consist of np.nan: continue
            if np.isnan(ds_out_nparray).all():
                continue
            # Clear memory
            ds_out = None
        # Raise RuntimeError
        except:
            raise RuntimeError('Could not load data in {}.'.format(folders[1]))

        # Write to intersection list
        ds_list_intersect += [ds_name]

    # Terminate the function, if ds_list_intersect is empty
    if len(ds_list_intersect) == 0:
        raise RuntimeError('There is no intersection between the reference data and the sentinel-1 data.')

    return ds_list_intersect


def reproject_resample(ds_in, ds_ref, ds_out_name):
    """
    Adjusts the projection from ds_in to ds_ref and resamples to equal resolution.

    Parameters
    ----------
        ds_in: str
            path to merged reference image (not reprojected)
        ds_ref: object
            reference dataset (gdal object)
        ds_out_name: str
            name of the reprojected dataset

    Returns
    -------
        ds_out: object
            reprojected image (gdal object)
    """

    # Get epsg from ds_ref
    proj = osr.SpatialReference(wkt=ds_ref.GetProjection())
    proj.AutoIdentifyEPSG()
    epsg_ref = proj.GetAttrValue('AUTHORITY', 1)

    # Get extent and spatial resolution from ds_ref
    gt = ds_ref.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * ds_ref.RasterXSize
    miny = maxy + gt[5] * ds_ref.RasterYSize
    projWin = [minx, miny, maxx, maxy]
    resolution_hor = gt[1]
    resolution_vert = gt[5]

    # Reproject and resample ds_in to ds_ref
    ds_out = gdal.Warp("{}".format(ds_out_name), ds_in, dstSRS="EPSG:{}".format(epsg_ref), xRes=resolution_hor,
                       yRes=resolution_vert,
                       resampleAlg="mode", outputBounds=projWin)
    # Clear memory
    os.remove(ds_in)

    return ds_out


def ref_preprocessing(folders):
    """
    Reference resampling to Sentinel-1 boundary and reprojecting to equal EPSGs.

    Parameters
    ----------
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    ds_processed: object
        processed reference image (gdal object)
    """

    # Outputpath for processed reference tif
    ds_processed_name = folders[1] + '\wetland_merge_reproj_resamp.tif'
    if not os.path.exists(ds_processed_name):
        # Reference dataset
        s1_list = glob.glob(folders[2] + '\*.tif')
        # Terminate the programm, if there is no sentinel-1 data
        if len(s1_list) == 0:
            raise RuntimeError('There is no sentinel-1 data.')
        ds_ref = gdal.Open(s1_list[0])
        # Directory with tif-files
        ds_list = glob.glob(folders[1] + '\*.tif')
        # Intersection test,
        ds_list_intersect = intersect(ds_list, ds_ref, folders, nan=255)
        # Merge all tif-files that intersect with ds_ref
        ref_merge_name = folders[1] + '\wetland_merge.tif'
        merge(ds_list_intersect, ref_merge_name)
        # Reproject and resample merged tif-file using ds_ref and clear input
        ds_processed = reproject_resample(ref_merge_name, ds_ref, ds_processed_name)
        # Clear memory
        ds_ref = None
    # If the file already exists, load it
    else:
        ds_processed = gdal.Open(ds_processed_name)

    return ds_processed



def ds_s1_ref_to_df(s1_stack, metadata_stack, ds_wetland):
    """
    Create a single dataframe from sentinel-1 und reference input datasets. Adds pixelcoordinates to the dataframe.

    Parameters
    ----------
    s1_stack: numpy.ndarray
        sentinel-1 data stack
    metadata_stack: list
        list with metadata informations
    ds_wetland: object
        processed reference image (gdal object)

    Returns
    -------
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame
    """
    # Progressbar
    print('[' + '#' * 17 + '_' * 53 + ']')
    print('creating dataframe...')
    # Create the coloumn-header
    col_name = [str(metadata_stack[i][1] + '_' + metadata_stack[i][3] + '_' + metadata_stack[i][4]) for i in
                range(len(metadata_stack))]
    # Create an array from sentinel-1 dataframe
    s1_nparray = np.array(s1_stack)
    # Create an array from reference dataframe
    wetland_array = ds_wetland.GetRasterBand(1).ReadAsArray()
    wetland_nparray = np.array(wetland_array)
    # Clear memory
    wetland_array = None
    len_row_ref = len(wetland_nparray[0])  # Variable for the length of the referene-array
    df = pd.DataFrame(s1_nparray.T, columns=col_name)  # Sentinel-1 dataframe
    nan_rows = df.isnull().values.all(axis=1)
    # Insert new coloumn with reference values
    df.insert(0, "wetland_ref", wetland_nparray.flatten())
    df.loc[nan_rows, "wetland_ref"] = 255  # Set the rows with NaN-values to class 255
    # Insert a column with the pixel-coordinates (row_column) in the scene
    pixels = len(wetland_nparray.flatten())  # Variable with the pixel-count of the scene
    list_coord = []
    for pixel in range(pixels):
        row = int(pixel / len_row_ref)
        column = int(pixel - (len_row_ref * row))
        list_coord.append(str(row) + "_" + str(column))  # Append to the array
    df.insert(0, "row_col", list_coord)  # Insert the array as a coordinate column

    return df


def datastack_creation(folders):
    """
    Function to preprocess reference data and create combined DataFrame with reference and Sentinel-1 data.

    Parameters
    ----------
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame
    """
    # Progressbar
    print('[' + '#' * 13 + '_' * 57 + ']')
    print("Preprocessing s1_stack...")
    print('-' * 60)
    # Create the sentinel-1 stack
    s1_stack, metadata_stack = sentinel_stack(folders)
    # Reference preprocessing (merge, resample, reproject)
    ds_wetland = ref_preprocessing(folders)
    # Merge Sentinel-1 stack and reference in one dataframe
    df = ds_s1_ref_to_df(s1_stack, metadata_stack, ds_wetland)

    return df


def ref_erosion(df, classes):
    """
    Function to filter (erode 2 pixels (40m) to the inside of the area ( =3x3 eroded )) the edge pixels of the different classes.
    Adds extra column for the remaining class pixels.

    Parameters
    ----------
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixel-coordinates as a pandas DataFrame
    classes: list
        list contains the reference class-numbers as int

    Returns
    -------
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixel-coordinates as a pandas DataFrame, with eroded reference class-pixels
    """
    print("Eroding ref_classes...")
    df["mask"] = 0  # Insert a mask column (needs to be initialised before the for-loop starts otherwise pandas-error)
    # Readout the columns and rows
    row, col = df.iloc[-1]['row_col'].split('_')
    # Add eroded reference column
    df.insert(2, "wetland_ref_eroded", np.nan)
    pbar = ProgressBar()
    # Iterate through all classes
    for class_num in pbar(classes):
        # Fill the mask column with 0 and for the iterated class with 1
        df["mask"] = 0
        df.loc[df['wetland_ref'] == class_num, 'mask'] = 1
        # Transfrom mask column to array and resize it to reference extent
        ref_nparray = df['mask'].to_numpy().flatten()
        ref_nparray.resize(int(row) + 1, int(col) + 1)
        # Erode the borders of the class and reattach the array to the dataframe
        ref_nparray = ndimage.binary_erosion(ref_nparray, structure=np.ones((3, 3)), iterations=1).astype(
            ref_nparray.dtype)
        # --------------------------------------------------------------------------------------------------------------------
        # OPTIONAL: No erosion
        # ref_nparray = ndimage.binary_erosion(ref_nparray, structure=np.ones((1,1)),iterations = 1).astype(ref_nparray.dtype)
        # --------------------------------------------------------------------------------------------------------------------
        ref_nparray = ref_nparray.flatten()
        df.insert(0, "wetland_ref_masked_{}".format(str(class_num)), ref_nparray)
        # Set the cell value for the eroded cells to the class-value
        df["wetland_ref_eroded"].loc[df["wetland_ref_masked_{}".format(str(class_num))] == 1] = int(class_num)
        df.drop(["wetland_ref_masked_{}".format(str(class_num))], axis=1, inplace=True)
    # Clear dataframe
    df.drop(["mask"], axis=1, inplace=True)

    return df


def handle_nan(df):
    """
    Drop sentinel-1 scene (column of df), if more then 90% nan-values to avoid problems with randomforest classification

    Parameters
    ----------
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame, with eroded reference class-pixels

    Returns
    -------
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame, with eroded reference class-pixels, without NaNs
    """
    # Drop column, if more then 90% nan-values (required for non unisized images)
    df = df[df.columns[df.isnull().mean() < 0.9]]

    return df


def test_train_split(classes, df, folders):
    """
    Function to split the stack in eroded train- and test-stacks (split = train 80% & test 20%).
    Saves the stacks as feather files.

    Parameters
    ----------
    classes: list
        list contains the reference classnumbers as int
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame, with eroded reference class-pixels, without NaNs
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame, with eroded reference class-pixels, without NaNs
    df_train: pandas.DataFrame
        train-dataframe containing 80% of df
    df_test: pandas.DataFrame
        test-dataframe containing 20% of df
    """
    print('Splitting stack...')
    # Insert column to store the split-information
    df.insert(3, 'wetland_ref_eroded_train', np.nan)
    time.sleep(.5)  # Delay for progressbar
    pbar = ProgressBar()
    for class_number in pbar(classes):
        # Fetch the indices of the eroded sample-pixels (split = train 80% & test 20%)
        indices = list(df.loc[df['wetland_ref_eroded'] == class_number].sample(frac=0.8, random_state=42).index.values)
        df.loc[indices, 'wetland_ref_eroded_train'] = class_number
    # Split in train dataframe
    filt_train = (df['wetland_ref_eroded'].notna()) & (df['wetland_ref_eroded_train'].notna()) & (
            df['wetland_ref_eroded_train'] != -99)
    df_train = df.loc[filt_train]
    # Assign all remaining eroded pixels to test-samples
    filt_test = (df['wetland_ref_eroded'].notna()) & (df['wetland_ref_eroded_train'].isna())
    df.loc[filt_test, 'wetland_ref_eroded_train'] = -99
    df_test = df.loc[filt_test]
    # Progressbar
    print('[' + '#' * 27 + '_' * 43 + ']')
    print('Saving s1_stack...')
    # Save the s1_stack
    datastack = r'{}\df_stack.ftr'.format(folders[3])
    df.to_feather(datastack)
    # Save the splitted data-stacks as feather file
    df_test = df_test.reset_index()
    df_train = df_train.reset_index()
    df_test.drop(["index"], axis=1, inplace=True)
    df_train.drop(["index"], axis=1, inplace=True)
    df_test_data = r'{}\df_test_data.ftr'.format(folders[3])
    df_train_data = r'{}\df_train_data.ftr'.format(folders[3])
    print('Saving test-data...')
    df_test.to_feather(df_test_data)
    print('Saving train-data...')
    df_train.to_feather(df_train_data)

    return df, df_train, df_test


def create_geo_tiff(image_array, outputTif, folders):
    """
    Function to create a GeoTiff-file from an array

    Parameters
    ----------
    image_array: list
        list containing pixel-values for image
    outputTif: str
        outputpath to created image
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------

    """
    # Read tif to pick target geometrys and projection
    inputTif = gdal.Open(glob.glob(folders[2] + '\*.tif')[0])
    x_pix = inputTif.RasterXSize  # Pixelcount x
    y_pix = inputTif.RasterYSize  # Pixelcount y
    pix_resolution = inputTif.GetGeoTransform()[1]  # Size of the pixels
    x_min = inputTif.GetGeoTransform()[0]
    y_max = inputTif.GetGeoTransform()[3]  # X_min & y_max are like the "top left" corner.
    projection = inputTif.GetProjectionRef()
    # Create the output-dataset with the loaded geometrys and projection
    driver = gdal.GetDriverByName('GTiff')
    outDataset = driver.Create(outputTif, x_pix, y_pix, 1, gdal.GDT_Byte, )
    outDataset.SetGeoTransform((x_min, pix_resolution, 0, y_max, 0, -pix_resolution))  # Set the pixel-dimensions
    outDataset.SetProjection(projection)  # Set the projection
    outDataset.GetRasterBand(1).WriteArray(image_array)  # Add the data
    outDataset.FlushCache()  # Write to disk.
    # Clear memory
    del inputTif, outDataset


def color_change(outputTif, color_list):
    """
    Change default colors from outputTif to reference matching colors

    Parameters
    ----------
    outputTif: str
        path to the input-image
    color_list: list
        list containing the RGBA-values from reference

    Returns
    -------

    """
    ds = gdal.Open(outputTif, 1)
    outDataset = ds.GetRasterBand(1)
    # Create color table
    colors = gdal.ColorTable()
    # Set color for each value
    for color in color_list:
        colors.SetColorEntry(int(color[0]), (int(color[1]), int(color[2]), int(color[3])))
    # Set color table and color interpretation in tif
    outDataset.SetRasterColorTable(colors)
    outDataset.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    # Close and save file
    outDataset.FlushCache()
    # Clear memory
    del outDataset, ds


def colorpicker(destination_path_filename):
    """
    Function to readout the specified class-colors.

    Parameters
    ----------
    destination_path_filename: str
        path to copied QGISColour.txt

    Returns
    -------
    classes: list
        list contains the reference class-numbers as int
    color_list: list
        list containing the RGBA-values from reference
    classes_label: list
        list containing the class-labels to the matching colors
    """
    # Read the QGISColour.txt file and ignore the classes "unclassifiable" and "outside area"
    df = pd.read_csv(destination_path_filename, header=None)
    df_color = df.loc[~((df[5] == 'unclassifiable') | (df[5] == 'outside area'))]
    # Extract the class-values, RGBA-values and the class-labels to seperate arrays
    classes = df_color[0].to_numpy()
    color_list = df_color.iloc[:, 0:4].to_numpy()
    classes_label = df_color[5].to_numpy()

    return classes, color_list, classes_label


def user_validation_method_query():
    """
    Query to ask the user for the validation method.

    Returns
    -------
    validation_method: int
        number of validation- hyperparametertuning method (1 = grid-search, 2 = random-search)
    """
    # Progressbar
    print('[' + '#' * 35 + '_' * 35 + ']')
    sys.stdout.write('Initialyse randomforest classification...\n')
    print('-' * 60)
    # System prints to define the validation-methods
    sys.stdout.write('Basis for the two validation- and hyperparametertuning-method:\n')
    sys.stdout.write('- Tests definded parameter-setting for randomforest classification\n')
    sys.stdout.write('- Each validation contains a 10-fold CV\n')
    sys.stdout.write('- Only the best parameter-setting will be applied\n\n')
    sys.stdout.write('Hyperparameters:\n')
    sys.stdout.write('- Depth: Number of decision-layers\n')
    sys.stdout.write('- N-Estimators: Number of decision-trees\n\n')
    sys.stdout.write('Grid-Search: Specific list of parameter-settings for depth and n-estimators.\n')
    sys.stdout.write('  Default: depth: [2]   | n-estimators: [10]\n')
    sys.stdout.write('  Example: depth: [2,3] | n-estimators: [50,100,225]\n')
    sys.stdout.write(
        'Random-Search: Minimum and Maximum range splitted into n-equal steps for depth and n-estimators.\n')
    sys.stdout.write(
        '  Default: depth: [2-3], n-equal: 1 | n-estimators: [100-150], n-equal: 2\n -> depth: [2]   | n-estimators: [100,150].\n')
    sys.stdout.write(
        '  Example: depth: [2-3], n-equal: 2 | n-estimators: [100-300], n-equal: 3\n -> depth: [2,3] | n-estimators: [100,200,300]\n')
    sys.stdout.write('-------------------------------------------------\n')
    sys.stdout.write('    Methodname        |(estimated processingtime)\n')
    sys.stdout.write('-------------------------------------------------\n')
    sys.stdout.write('1) Grid-Search       |(short to extra long)\n')
    sys.stdout.write('2) Random-Search     |(long)\n')
    sys.stdout.write('-------------------------------------------------\n')
    sys.stdout.write('Choose your validation- and hyperparametertuning-method:\n')

    validation_method = 0
    time.sleep(.5)  # Debugging delay, otherwise the methods will be printed AFTER the question
    # Ask the user which method should be used
    while ((validation_method != 1) and (validation_method != 2)):
        try:
            validation_method = int(input())
        except ValueError:
            # Invalid answers will trigger repeat response
            sys.stdout.write('Please respond with an integer.\n')
        if ((validation_method != 1) and (validation_method != 2)):
            print('Please choose your validation-method (1 to 2)')

    return validation_method


def user_single_parameter(parameter, rf_model_list):
    """
    Query to ask the user for a single custom hyperparameter-value.

    Parameters
    ----------
    parameter: str
        name of hyperparameter to be set or the randomforest model-number to be loaded
    rf_model_list: list
        list contains the names of existing randomforest models

    Returns
    -------
    parameter_value: list
        list contains the custom hyperparameter-value
    """
    # random-search
    if parameter == 'min_depth':
        print('Defines the lower range boundary for the depth of the decisiontree.')
        parameter_min = 2  # < 2 runs into an error for f-scores
        parameter_max = 101
    if parameter == 'max_depth':
        print('Defines the upper range boundary for the depth of the decisiontree.')
        parameter_min = 2  # < 2 runs into an error for f-scores
        parameter_max = 101
    if parameter == 'num_depth':
        print('Devides the specified range into {} parts. '.format(parameter))
        parameter_min = 1
        parameter_max = 1001
    if parameter == 'min_estimators':
        print('Defines the lower range boundary for estimators.')
        parameter_min = 10  # needs to be 10, cause of the 10-folds CV
        parameter_max = 1001
    if parameter == 'max_estimators':
        print('Defines the upper range boundary for estimators.')
        parameter_min = 10  # needs to be 10, cause of the 10-folds CV
        parameter_max = 1001
    if parameter == 'num_estimators':
        print('Devides the specified range into {} parts. '.format(parameter))
        parameter_min = 2  # needs to be >2, cause of the iterations in randomsearch
        parameter_max = 1001
    # If the user wants to load an existing model
    if parameter == 'rf_model_number':
        parameter_min = 0
        parameter_max = len(rf_model_list)

    # Ask the user which value should be used
    sys.stdout.write(
        'Choose your value for {} in between {} and {} :\n'.format(parameter, parameter_min, parameter_max - 1))
    parameter_value = -1  # Predefined value to activate the while-loop
    while not parameter_value in range(parameter_min, parameter_max):
        try:
            parameter_value = int(input())
        # Invalid answers will trigger repeat response
        except ValueError:
            sys.stdout.write('Please respond with an integer.\n')
        if not parameter_value in range(parameter_min, parameter_max):
            print('Please choose your value in between {} and {} :'.format(parameter_min, parameter_max - 1))

    return [parameter_value]


def user_multi_parameter(parameter):
    """
    Query to ask the user for hyperparameters in grid-search. Appends them to a list.

    Parameters
    ----------
    parameter: str
        name of hyperparameter to be set or the randomforest model-number to be loaded

    Returns
    -------
    value_list: list
        list contains the custom hyperparameter-values
    """
    if parameter == 'n_depth':
        print('Defines the depth of the decisiontree.')
        parameter_min = 2  # < 2 runs into an error for f-scores
        parameter_max = 101
    if parameter == 'n_estimators':
        print('Defines the estimators.')
        parameter_min = 10  # needs to be 10, cause of the 10-folds CV
        parameter_max = 1001

    # Ask the user which value should be used
    print('Choose your integer value to append for {} in between {} and {} :\n'.format(parameter, parameter_min,
                                                                                       parameter_max - 1))
    print('(After you are finished just enter a non-integer value to break the query)')
    parameter_value = 0  # Predefined value to activate the while-loop
    value_list = []  # Empty list, will be filled with hyperparameter
    while True:
        try:
            parameter_value = int(input())
            if parameter_value in range(parameter_min, parameter_max):
                value_list.append(parameter_value)
                print('{} attached to list..'.format(parameter_value))
            else:
                print('{} could not be appended to the list.\n'.format(parameter_value))
        # Invalid answers will trigger repeat response
        except ValueError:
            if len(value_list) == 0:
                print("Warning your hyperparameter-list is empty.")
            else:
                print("Completed...")
                break
        print('list: ', value_list)  # Print the current list for visualisation
        # Value out of min/max:
        if not parameter_value in range(parameter_min, parameter_max):
            print('Please choose your integer value in between {} and {} :'.format(parameter_min, parameter_max - 1))

    return value_list


def rf_gridsearch_validation(x_train, x_test, y_train, y_test, n_depth=[2], n_estimators=[10]):
    """
    Function to run a grid-search randomforest model

    Parameters
    ----------
    x_train: numpy.ndarray
        train-dataset containing only the sentinel-1-values
    x_test: numpy.ndarray
        test-dataset containing only the sentinel-1-values
    y_train: numpy.ndarray
        train-dataset containing only the class-values
    y_test: numpy.ndarray
        test-dataset containing only the class-values
    n_depth: list
        list containing the values for the hyperparameter depth
    n_estimators: list
        list containing the values for the hyperparameter n_estimators

    Returns
    -------
    randomforest_model: sklearn.model_selection.GridSearchCV
        randomforest classifier
    """
    # Hyperparameter-settings as dict
    param_grid = {'max_depth': n_depth,
                  'n_estimators': n_estimators}
    # Progressbar
    print('[' + '#' * 40 + '_' * 30 + ']')
    print('Initialising randomforest-model...')
    print('-' * 60)
    # User information
    print('Hyperparameters: max_depth:', n_depth,
          ', n_estimators:', n_estimators)
    # Initialise rf-classifier
    base_model = ensemble.RandomForestClassifier(class_weight='balanced',
                                                 random_state=42)  # Instantiate the grid search model
    randomforest_model = GridSearchCV(estimator=base_model,
                                      param_grid=param_grid,
                                      cv=10,
                                      n_jobs=-2,
                                      verbose=4)
    pbar = ProgressBar()
    for i in pbar(range(1)):
        randomforest_model = randomforest_model.fit(x_train, y_train.ravel())  # Fit model to train dataset

    return randomforest_model


def rf_randomsearch_validation(x_train, x_test, y_train, y_test, min_depth=[2], max_depth=[3], num_depth=[1],
                               min_estimators=[100], max_estimators=[150], num_estimators=[2]):
    """
    Function to run a random-search randomforest model

    Parameters
    ----------
    x_train: numpy.ndarray
        train-dataset containing only the sentinel-1-values
    x_test: numpy.ndarray
        test-dataset containing only the sentinel-1-values
    y_train: numpy.ndarray
        train-dataset containing only the class-values
    y_test: numpy.ndarray
        test-dataset containing only the class-values
    min_depth: list
        list containing the value for the lower value for hyperparameter range depth
    max_depth: list
        list containing the value for the upper value for hyperparameter range depth
    num_depth: list
        list containing the value for the n-equal steps for hyperparameter range depth
    min_estimators: list
        list containing the value for the lower value for hyperparameter range estimators
    max_estimators: list
        list containing the value for the upper value for hyperparameter range estimators
    num_estimators: list
        list containing the value for the n-equal steps for hyperparameter range estimators

    Returns
    -------
    randomforest_model: sklearn.model_selection.RandomizedSearchCV
        randomforest classifier
    """

    # Split n-estimator range in n-equal steps
    n_estimators = [int(x) for x in np.linspace(min(min_estimators, max_estimators),
                                                max(min_estimators, max_estimators),
                                                num=num_estimators[0])]
    # Split depth range in n-equal steps
    max_depth = [int(x) for x in np.linspace(min(min_depth, max_depth),
                                             max(min_depth, max_depth),
                                             num=num_depth[0])]
    # Hyperparameter-settings as dict
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth}
    # Progressbar
    print('[' + '#' * 40 + '_' * 30 + ']')
    print('Initialising randomforest-model...')
    print('-' * 60)
    print('Hyperparameters: max_depth:', max_depth,
          ', n_estimators:', n_estimators)
    # Initialise rf-classifier
    rf = ensemble.RandomForestClassifier(class_weight='balanced')
    randomforest_model = RandomizedSearchCV(estimator=rf,
                                            param_distributions=random_grid,
                                            n_iter=2,
                                            cv=10,
                                            verbose=4,
                                            random_state=42,
                                            n_jobs=-2)
    pbar = ProgressBar()
    for i in pbar(range(1)):
        randomforest_model.fit(x_train, y_train.ravel())  # Fit model to train dataset

    return randomforest_model


def load_randomforest(folders):
    """
    Function to load a presaved randomforest model.

    Parameters
    ----------
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    loaded_rf_model: sklearn.ensemble.RandomForestClassifier
        saved randomforest classifier (grid- or random-search)
    method_number: str
        name of the validation- and hyperparametertuning method
    rf_folder: str
        path to the selected model
    rf_model_list: list
        list contains all names of saved randomforest models
    """
    # Progressbar
    print('[' + '#' * 40 + '_' * 30 + ']')
    print('Loading RF_Model...')
    print('-' * 60)
    # List all the presaved models
    rf_model_dirlist = glob.glob('{}/*/*.joblib'.format(folders[4]))
    rf_model_list = []  # Create empty list for the presaved models
    for model in rf_model_dirlist:
        rf_model_list.append(os.path.basename(model))
    # Ask the user which model should be loaded and load the model
    print('Please choose one of the preexisting models with the shown numbers:\n')
    for choice, model in enumerate(rf_model_list):
        print(str(choice) + ')', model)
    rf_model_number = user_single_parameter('rf_model_number', rf_model_list)[0]
    # Load the selected model
    loaded_rf_model = joblib.load(rf_model_dirlist[rf_model_number])
    method_number = rf_model_dirlist[rf_model_number].split('rf_model_')[-1].split('_')[
        0]  # Needed for the names of the outputdata
    # Fetch the path to the selected model
    rf_folder = os.path.normpath(rf_model_dirlist[rf_model_number] + os.sep + os.pardir)

    return loaded_rf_model, method_number, rf_folder, rf_model_list


def export_randomforest(randomforest_model, method_number, folders):
    """
    Function to create a model specific folder and dumb the model.

    Parameters
    ----------
    randomforest_model: sklearn.ensemble.RandomForestClassifier
        randomforest classifier (grid- or random-search) to be exported
    method_number: int
        number of validation- hyperparametertuning method (1 = grid-search, 2 = random-search)
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    rf_folder: str
        path to the exported model
    """
    # Progressbar
    print('[' + '#' * 45 + '_' * 25 + ']')
    print('Exporting RF_Model...')
    # Extract the hyperparameters
    best_parameters = randomforest_model.best_params_
    # Transform method_number to method
    validation_method = method_int_to_str(method_number)
    # Create the name of the specific randomforest model
    rf_model_name = "rf_model_{}_depth_{}_estim_{}".format(validation_method, best_parameters['max_depth'],
                                                           best_parameters['n_estimators'])
    rf_folder = r'{}/{}'.format(folders[4], rf_model_name)
    # Delete old randomforest modelsaves with same name to allow overwriting in different OS
    shutil.rmtree(rf_folder, ignore_errors=True)
    os.makedirs(rf_folder)
    # Dumb the model and append the folderpath
    joblib.dump(randomforest_model, "{}/{}.joblib".format(rf_folder, rf_model_name))

    return rf_folder


def predict_image(df, randomforest_model, color_list, method_number, rf_folder, folders):
    """
    Function to use the model and predict an image on the full stack (incl. eroded pixels).

    Parameters
    ----------
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame, with eroded reference class-pixels, without NaNs
    randomforest_model: sklearn.ensemble.RandomForestClassifier
        randomforest classifier (grid- or random-search)
    color_list: list
        list containing the RGBA-values from reference
    method_number: int
        number of validation- hyperparametertuning method (1 = grid-search, 2 = random-search)
    rf_folder: str
        path to the selected model
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------

    """
    # Copy the stack
    df_prediction = df.copy(deep=True)
    # -----------------------------------------------------------------------------
    # IF YOU WANT TO FILTER THE POLARISATION USE:
    # df_prediction=df.filter(regex='(?=.*VH)|(?=.*col)|(?=.*ref)').copy(deep=True)
    # df_prediction=df.filter(regex='(?=.*VV)|(?=.*col)|(?=.*ref)').copy(deep=True)
    # -----------------------------------------------------------------------------
    # Drop all NaN-values in sentinel-1 data
    df_prediction.dropna(subset=df_prediction.columns[4:], axis=0, inplace=True)
    create_predicted_image(randomforest_model, df_prediction, df, color_list, method_number, rf_folder, folders, masked=False)


def predict_image_masked(df_train, df_test, df, randomforest_model, color_list, method_number, rf_folder, folders):
    """
    Function to use the model and predict an masked image on the eroded stack (excl. eroded pixels).

    Parameters
    ----------
    df_train: pandas.DataFrame
        train-dataframe containing 80% of df
    df_test: pandas.DataFrame
        test-dataframe containing 20% of df
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame, with eroded reference class-pixels, without NaNs
    randomforest_model: sklearn.ensemble.RandomForestClassifier
        randomforest classifier (grid- or random-search)
    color_list: list
        list containing the RGBA-values from reference
    method_number: int
        number of validation- hyperparametertuning method (1 = grid-search, 2 = random-search)
    rf_folder: str
        path to the selected model
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------

    """
    # Remerge the train and test-dataset
    frames = [df_train, df_test]
    df_train_test = pd.concat(frames)
    create_predicted_image(randomforest_model, df_train_test, df, color_list, method_number, rf_folder, folders, masked=True)


def create_predicted_image(randomforest_model, df_prediction, df, color_list, method_number, rf_folder, folders, masked):
    """
    Function to create predicted image based on randomforest model with and without eroded pixels.

    Parameters
    ----------
    randomforest_model: sklearn.ensemble.RandomForestClassifier
        randomforest classifier (grid- or random-search)
    df_prediction: pandas.DataFrame
        input for predictions, only contains the sentinel-1 values
    df: pandas.DataFrame
        merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame, with eroded reference class-pixels, without NaNs
    color_list: list
        list containing the RGBA-values from reference
    method_number: int
        number of validation- hyperparametertuning method (1 = grid-search, 2 = random-search)
    rf_folder: str
        path to the selected model
    folders: list
        list of paths to the project subfolders in the workspace
    masked:
        status info: true if eroded pixel should be masked, false if vice versa

    Returns
    -------

    """
    # Use the model to predict the image
    randomforest_prediction = randomforest_model.predict(df_prediction.iloc[:, 4:].values)
    # Extract hyperparameters
    best_parameters = randomforest_model.best_params_
    # Insert the predicted values to the image-dataframe and transform it to an array
    df_prediction.insert(4, "classified", randomforest_prediction)
    df_image = df.merge(df_prediction[['row_col', 'classified']], how='left', on="row_col")
    predicted_image = df_image['classified'].to_numpy()
    row, col = df.iloc[-1]['row_col'].split('_')
    predicted_image.resize(int(row) + 1, int(col) + 1)
    predicted_image = np.nan_to_num(predicted_image, nan=255)
    # Transform method_number to method
    validation_method = method_int_to_str(method_number)
    if masked:
        outputTif = r'{}\class_result_{}_depth_{}_estim_{}_masked.tif'.format(rf_folder, validation_method,
                                                                              best_parameters['max_depth'],
                                                                              best_parameters['n_estimators'])
    else:
        outputTif = r'{}\class_result_{}_depth_{}_estim_{}.tif'.format(rf_folder, validation_method,
                                                                       best_parameters['max_depth'],
                                                                       best_parameters['n_estimators'])
    create_geo_tiff(predicted_image, outputTif, folders)
    # Change the default-colors to the same colors as the reference map
    color_change(outputTif, color_list)
    # User information
    if masked:
        print('Saved the masked image in {}'.format(rf_folder))
    else:
        print('Saved the image in {}'.format(rf_folder))


def check_df_train_test(classes, folders, s1_stack_changed):
    """
    Function to check if train/test data already exists. If not call the necessary functions to create them, else load the saved files.
    Check also for changes between input-data and existing stacks.

    Parameters
    ----------
    classes: list
        list contains the reference class-numbers as int
    folders: list
        list of paths to the project subfolders in the workspace
    s1_stack_changed: bool
        status info: true if there is a change in number of sentinel-1 files, false if vice versa

    Returns
    -------
    df: pandas.DataFrame
        loaded merged sentinel-1, reference data and pixelcoordinates as a pandas DataFrame, with eroded reference class-pixels, without NaNs
    df_train: pandas.DataFrame
        loaded train-dataframe containing 80% of df
    df_test: pandas.DataFrame
        loaded test-dataframe containing 20% of df
    """
    # Define the relevant paths for the datastacks
    df_stack = r'{}\df_stack.ftr'.format(folders[3])
    df_test_data = r'{}\df_test_data.ftr'.format(folders[3])
    df_train_data = r'{}\df_train_data.ftr'.format(folders[3])
    # Check for existing stacks
    # Not existing
    if not (os.path.exists(df_stack) and os.path.exists(df_test_data) and os.path.exists(
            df_train_data) and not s1_stack_changed):
        df_creation = True  # If true, the stack needs to be created
    # Existing
    else:
        # Compare the existing stack and the unprocessed input data for equal classes
        df = pd.read_feather(df_stack, columns=['wetland_ref'])
        df_classes = df.loc[df['wetland_ref'] != 255]
        stack_classes_nparray = np.sort(df_classes['wetland_ref'].unique())
        # No Changes
        if np.array_equal(stack_classes_nparray, classes):
            df_creation = False  # No creation needed
        # Changed
        else:
            df_creation = True  # Creation needs to be done
        # Clear memory
        del df_classes, stack_classes_nparray

    if df_creation:
        # Create the df, df_test and df_train
        df = datastack_creation(folders)
        # Progressbar
        print('[' + '#' * 20 + '_' * 50 + ']')
        df = ref_erosion(df, classes)
        df = handle_nan(df)
        # Progressbar
        print('[' + '#' * 23 + '_' * 47 + ']')
        df, df_train, df_test = test_train_split(classes, df, folders)
    else:
        # Load stored df, df_test and df_train
        # Progressbar
        print('[' + '#' * 20 + '_' * 50 + ']')
        print('Loading stored data...')
        df = pd.read_feather(df_stack)
        df_test = pd.read_feather(df_test_data)
        df_train = pd.read_feather(df_train_data)

    return df, df_train, df_test


def method_int_to_str(method_number):
    """
    Function to convert validation- hyperparametertuning method number to string.

    Parameters
    ----------
    method_number: int
        number of validation- hyperparametertuning method (1 = grid-search, 2 = random-search)
    Returns
    -------
    validation_method: str
        name of the validation- and hyperparametertuning method
    """
    validation_method = ""  # Empty validation-method name as string
    if method_number == 1:
        validation_method = "Grid-Search"
    elif method_number == 2:
        validation_method = "Random-Search"
    # For presaved models, the validation-method name will be extraced out of the filename as string
    else:
        validation_method = method_number

    return validation_method


def model_call(df_train, df_test, classes_label, folders):
    """
    Splits the dataframe into train and test-data. Asks for the hyperparametertuning and validation-method.

    Parameters
    ----------
    df_train: pandas.DataFrame
        train-dataframe containing 80% of df
    df_test: pandas.DataFrame
        test-dataframe containing 20% of df
    classes_label: list
        list containing the class-labels to the matching colors
    folders: list
        list of paths to the project subfolders in the workspace

    Returns
    -------
    randomforest_model: sklearn.ensemble.RandomForestClassifier
        randomforest classifier (grid- or random-search)
    method_number: int
        number of validation- hyperparametertuning method (1 = grid-search, 2 = random-search)
    rf_folder: str
        path to the selected model
    """
    # Split into train and test data
    x_train = df_train.iloc[:, 4:].values
    x_test = df_test.iloc[:, 4:].values
    # Split into train and test classes
    y_train = df_train.iloc[:, 2:3].values
    y_test = df_test.iloc[:, 2:3].values
    # ----------------------------------------------------------------------------------
    # OPTIONAL: Polarisation-filtering
    # x_train = df_train.iloc[:,4:].filter(regex='(?=.*VH)|(?=.*col)|(?=.*ref)').values
    # x_test = df_test.iloc[:,4:].filter(regex='(?=.*VH)|(?=.*col)|(?=.*ref)').values
    # x_train = df_train.iloc[:,4:].filter(regex='(?=.*VV)|(?=.*col)|(?=.*ref)').values
    # x_test = df_test.iloc[:,4:].filter(regex='(?=.*VV)|(?=.*col)|(?=.*ref)').values
    # ----------------------------------------------------------------------------------
    # Progressbar
    print('[' + '#' * 30 + '_' * 40 + ']')
    print('Starting classification...')
    print('-' * 60)
    # Ask the user to run an existing model or create a new one:
    if len(glob.glob('{}/*/*.joblib'.format(folders[4]))) > 0:  # Checks for existing models
        question = 'Do you want to use an already existing randomforest-model [y] or create a new randomforest-model [n]?'
        use_exist_model = user_yes_no_query(question)
    else:  # No existing models
        use_exist_model = False

    if use_exist_model == True:  # Load existing model
        randomforest_model, method_number, rf_folder, rf_model_list = load_randomforest(folders)
    else:  # Asks the user for the hyperparametertuning and validation-method
        method_number = user_validation_method_query()
        question = 'Do you want to use the default model-parameters [y] or custome model-parameters [n]?'
        if user_yes_no_query(question) == True:  # Ask the user to use the default or customized parameters
            # Default parameters
            # Grid-search
            if method_number == 1:
                print('Using Grid-Search with default parameters')
                randomforest_model = rf_gridsearch_validation(x_train, x_test, y_train, y_test)
            # Random-search
            elif method_number == 2:
                print('Using Random-Search with default parameters')
                randomforest_model = rf_randomsearch_validation(x_train, x_test, y_train, y_test)
        # Customized parameters
        else:
            # Grid-search
            if method_number == 1:
                print('Using Grid-Search with custom parameters')
                # Ask for hyperparameter-settings
                n_depth = user_multi_parameter('n_depth')  # Ask for the depth as array
                n_estimators = user_multi_parameter('n_estimators')  # Ask for the estimators as array
                randomforest_model = rf_gridsearch_validation(x_train, x_test, y_train, y_test, n_depth, n_estimators)
            # Random-search
            elif method_number == 2:
                print('Using Random-Search with custom parameters')
                # Ask for hyperparameter-settings
                min_depth = user_single_parameter('min_depth', None)  # Min range depth
                max_depth = user_single_parameter('max_depth', None)  # Max range depth
                num_depth = user_single_parameter('num_depth', None)  # N-equal steps depth
                min_estimators = user_single_parameter('min_estimators', None)  # Min range n-estimators
                max_estimators = user_single_parameter('max_estimators', None)  # Max range n-estimators
                num_estimators = user_single_parameter('num_estimators', None)  # N-equal steps n-estimators
                randomforest_model = rf_randomsearch_validation(x_train, x_test, y_train, y_test, min_depth,
                                                                max_depth, num_depth, min_estimators,
                                                                max_estimators, num_estimators)
        # Save the randomforest model
        rf_folder = export_randomforest(randomforest_model, method_number, folders)
    # Create confusion matrix
    accuracy_accessment(randomforest_model, x_test, y_test, classes_label, method_number, rf_folder)

    return randomforest_model, method_number, rf_folder


def accuracy_accessment(randomforest_model, x_test, y_test, classes_label, method_number, rf_folder):
    """
    Function to create and plot the confusion matrix with numerical-measures.

    Parameters
    ----------
    randomforest_model: sklearn.ensemble.RandomForestClassifier
        randomforest classifier (grid- or random-search)
    x_test: numpy.ndarray
        test-dataset containing only the sentinel-1-values
    y_test: numpy.ndarray
        test-dataset containing only the class-values
    classes_label: list
        list containing the class-labels to the matching colors
    method_number: int
        number of validation- hyperparametertuning method (1 = grid-search, 2 = random-search)
    rf_folder: str
        path to the selected model

    Returns
    -------

    """
    # Progressbar
    print('[' + '#' * 50 + '_' * 20 + ']')
    print('Evaluating randomforest-model...')
    randomforest_prediction = randomforest_model.predict(x_test)  # Use the fitted model on the test-dataset
    # Fetch numerical-measures and calculate confusion matrix
    validation_method = method_int_to_str(method_number)
    best_parameters = randomforest_model.best_params_  # Hyperparameter-setting
    randomforest_accuracy = metrics.accuracy_score(y_test, randomforest_prediction)  # Model accuracy
    kappa = metrics.cohen_kappa_score(y_test, randomforest_prediction)  # Kappa coef.
    best_std = randomforest_model.cv_results_['std_test_score'][randomforest_model.best_index_]  # Standard deviation
    best_score = randomforest_model.cv_results_['mean_test_score'][randomforest_model.best_index_]  # Model score

    # To plot relative and absolute cell-values, absolute and relative confusion matrix is needed
    # Confusion matrix absolute
    randomforest_confusion_matrix_abs = metrics.confusion_matrix(y_test, randomforest_prediction)
    # Confusion matrix relative
    randomforest_confusion_matrix_rel = metrics.confusion_matrix(y_test, randomforest_prediction, normalize='pred')
    # Create relative and absolute dataframes
    rel_data = pd.DataFrame(randomforest_confusion_matrix_rel, columns=classes_label, index=classes_label)
    abs_data = pd.DataFrame(randomforest_confusion_matrix_abs, columns=classes_label, index=classes_label)

    # Add the sums of rows and columns
    abs_data['sum_predicted'] = abs_data.sum(axis=1)
    abs_data.loc['sum_observed'] = abs_data.sum(numeric_only=True)
    # Diagonal line
    diagional_values = np.diag(abs_data)
    abs_data['diagonal_sum'] = diagional_values
    diagional_values = np.append(diagional_values, diagional_values[-1])
    abs_data.loc['diagonal_sum'] = diagional_values
    # Calculate the producers and users acc.
    rel_data['Producers accuracy'] = (abs_data['diagonal_sum'] / abs_data['sum_predicted'])
    rel_data.loc['Users accuracy'] = (abs_data.loc['diagonal_sum'] / abs_data.loc['sum_observed'])
    rel_data.loc['Users accuracy']['Producers accuracy'] = randomforest_accuracy
    rel_data.loc['Users accuracy'] = rel_data.loc['Users accuracy']
    rel_data = round(rel_data * 100, 2)
    abs_data.drop('diagonal_sum', inplace=True)
    abs_data.drop('diagonal_sum', axis=1, inplace=True)

    # Start plotting
    fig, ax = plt.subplots(figsize=(10, 15), edgecolor='k')
    # Create one frame for the plot of the relative and absolute confusion matrix, so both can be plotted in one turn
    labels = (np.asarray(["{} %\n{}".format(rel_data, abs_data) for abs_data, rel_data in
                          zip(abs_data.to_numpy().flatten(), rel_data.to_numpy().flatten())]))
    labels = labels.reshape((len(classes_label) + 1), (len(classes_label) + 1))
    # Plot the OAA first
    ax_oaa = sns.heatmap(rel_data, mask=labels != labels[-1][-1], annot=labels, annot_kws={"size": 15, 'color': 'red'},
                         linewidths=1, linecolor='black', fmt='', cmap=plt.cm.Blues)
    ax_oaa.figure.axes[-1].set_visible(False)  # Set Colorbar to not visible
    # Plot the remaining data
    ax_heatmap = sns.heatmap(rel_data, mask=labels == labels[-1][-1], annot=labels, annot_kws={"size": 14},
                             linewidths=1, linecolor='black', fmt='', cmap=plt.cm.Blues,
                             cbar_kws={'label': 'Accuracy in %', 'orientation': 'horizontal', 'location': 'bottom',
                                       'pad': 0.2})

    # Formatting the plot:
    ax_heatmap.figure.axes[-1].xaxis.label.set_size(13)  # Set the colorbar font size to 13
    ax_heatmap.figure.axes[-1].tick_params(labelsize=13)  # Set the colorbartick font size to 13
    plt.ylabel('Observed (Copernicus HRL Wetlands)', size=16, fontweight='bold')  # Add the Y-axis Label
    plt.xlabel('Predicted (Sentinel-1 Classification)', size=16, fontweight='bold')  # Add the X-axis Label
    plt.title('Confusion matrix', fontdict=None, loc='center', pad=20, size=25, fontweight='bold')  # Add the title
    plt.yticks(rotation=45)  # Rotate the Y-axis tick-label to 45 degree
    plt.xticks(rotation=45)  # Rotate the X-axis tick-label to 45 degree
    ax.set_xticklabels(rel_data.columns, size=14)  # Specify the X-axis labels
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_yticklabels(rel_data.index, size=14)  # Specify the Y-axis labels
    ax.get_yticklabels()[-1].set_fontweight("bold")

    # Add extra text for numerical-measures
    x_offset = len(classes_label) + 1.2
    ax.text(x_offset, 0.2, 'Hyperparameters:', size=16, fontweight='bold')
    ax.text(x_offset, 0.5, '- method: {}'.format(validation_method), size=16)
    ax.text(x_offset, 0.7, '- max depth: {}'.format(best_parameters['max_depth']), size=16)
    ax.text(x_offset, 0.9, '- n_estimators: {}'.format(best_parameters['n_estimators']), size=16)
    ax.text(x_offset, 1.5, 'Validation:', size=16, fontweight='bold')
    ax.text(x_offset, 1.8, '- method: 10-fold Cross-Validation', size=16)
    ax.text(x_offset, 2, '- best score: {}'.format(round(best_score, 2)), size=16)
    ax.text(x_offset, 2.2, '- standard deviation: {}'.format(round(best_std, 2)), size=16)
    ax.text(x_offset, 2.4, '- overall accuracy: {}'.format(round(randomforest_accuracy, 2)), size=16)
    ax.text(x_offset, 2.6, '- cohen`s kappa coefficient: {}'.format(round(kappa, 2)), size=16)
    # Add borders to the plot
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Export plot
    plt.savefig(
        r'{}\conf_matrix_{}_depth_{}_estim_{}.jpg'.format(rf_folder, validation_method, best_parameters['max_depth'],
                                                          best_parameters['n_estimators']), bbox_inches='tight',dpi=400)


def wetCopRF(foldername, workspace, origin_wetland, origin_wetland_color, origin_sentinel):
    """
    Main function for randomforest classification of Sentinel-1 radardata with Copernicus Wetland Highresolution product as reference.

    Parameters
    ----------
    foldername: str
        name of the parent project folder
    workspace: str
        path of working directory (project folder)
    origin_wetland: str
        path to the parent wetland reference data (...\WAW_2018_010m_ee_03035_v020\DATA)
    origin_wetland_color: str
        path to the wetland reference symbology folder (...\WAW_2018_010m_ee_03035_v020\Symbology)
    origin_sentinel: list
        list containing paths to the sentinel-1 data (VH,VV)

    Returns
    -------

    """
    # ----------------------------------------------------------------------------
    # Project structure
    folders = check_project_structure(foldername, workspace)
    # ----------------------------------------------------------------------------
    # Copy files to project structure
    # Progressbar
    print('[' + '_' * 70 + ']')
    copy_wetland(origin_wetland, folders)
    color_file = copy_wetland_color(origin_wetland_color, folders)
    # Progressbar
    print('[' + '#' * 10 + '_' * 60 + ']')
    s1_stack_changed = copy_sentinel1_data(origin_sentinel, folders)
    classes, color_list, classes_label = colorpicker(color_file)
    # ----------------------------------------------------------------------------
    # Dataprocessing
    df, df_train, df_test = check_df_train_test(classes, folders, s1_stack_changed)
    # ----------------------------------------------------------------------------
    # randomforest classification
    question = 'Are you satisfied with the results and want to proceed?'
    repeat_classification = False
    while repeat_classification == False:
        randomforest_model, method_number,rf_folder = model_call(df_train, df_test, classes_label,folders)
        # Progressbar
        print('[' + '#' * 60 + '_' * 10 + ']')
        print('Exporting predicted image...')
        print('-' * 70)
        # ----------------------------------n------------------------------------------
        # Save classification results
        predict_image(df, randomforest_model, color_list, method_number, rf_folder, folders)
        predict_image_masked(df_train, df_test, df, randomforest_model, color_list, method_number, rf_folder, folders)
        # Progressbar
        print('[' + '#' * 65 + '_' * 5 + ']')
        # ----------------------------------------------------------------------------
        # End the programm or repeat the classification
        repeat_classification = user_yes_no_query(question)
    # Progressbar
    print('Done.')
    print('[' + '#' * 70 + ']')
