# wetCopRF

* Package for randomforest classification of Sentinel-1 radardata with Copernicus Wetland Highresolution product as reference. 
* Creates project folderstructure and imports the necessary  input data. (Keep in mind: The workspace requires approximatly twice as much free space on the harddrive as the input data.)
* Prepares the data for randomforest classification and splits the data in train and test samples.
* The users can choose between two different validation- and hyperparametertuning methods (random- and grid-search) with 10-folds Cross-Validation. 
* Automatically evaluates the results in a confusion matrix with helpful numerical-measures and exports the classification results as a map.

### Package installation 
* via Anaconda [environment.yml] file
```
conda env create -f environment.yml
```
### Input-data
* [Copernicus HighResolution Layers Wetland] extract downloaded zip
* Sentinel-1 VH,VV radardata in processing level x as tif

### Demonstration
* The [Jupyter Notebook] can be used for demonstration purposes. <br>*Note: There is an issue with notebooks internal links on github so they can only be used locally.*

### Documentation:
* The Documentation can be found [here].


[environment.yml]: https://github.com/Henno-hash/WetCopRF/blob/master/environment.yml

[Jupyter Notebook]: https://github.com/Henno-hash/wetCopRF/blob/master/wetCopRF/Showcase.ipynb

[here]: http://wetcoprf.readthedocs.io/

[Copernicus HighResolution Layers Wetland]: https://land.copernicus.eu/pan-european/high-resolution-layers/water-wetness/status-maps

