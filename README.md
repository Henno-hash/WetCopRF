# wetCopRF

* Package for randomforest classification of Sentinel-1 radardata with Copernicus Wetland Highresolution product as reference. 
* Creates project folderstructure and imports the necessary  input data. 
* Prepares the data for randomforest classification and splits the data in train and test samples.
* The users can choose between two different validation- and hyperparametertuning methods (random- and grid-search) with 10-folds Cross-Validation. 
* Automatically evaluates the results in a confusion matrix with helpful numerical-measures and exports the classification results as a map.

### Package installation 
* via Anaconda [environment.yml] file
```
conda env create -f environment.yml
```


### Demonstration
* The [Jupyter Notebook] can be used for demonstration purposes. <br>Note: There is an [issue] with notebooks internal links 


### Documentation:
* The Documentation can be found [here].


[environment.yml]: https://github.com/Henno-hash/WetCopRF/blob/master/environment.yml

[Jupyter Notebook]: https://github.com/Henno-hash/wetCopRF/blob/53eef00c7309aecb383e306b71d2d8af0a5925ac/wetCopRF/Showcase.ipynb

[here]: http://wetcoprf.readthedocs.io/

[issue]: https://github.com/microsoft/vscode-jupyter/issues/1330#issuecomment-789150166
