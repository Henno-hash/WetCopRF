import utils
if __name__ == "__main__":
    # Client
    # Please set your workspace location (Requires at least twice the free space of the input data)
    workspace = r"...."
    # Please set your foldername for this classification run
    foldername = '....'
    # Please set your path to the origin Sentinel-1 radardata (VH,VV)
    origin_sentinel = [r"....\VH",
                       r"....\VV"]
    # Please set your path to the origin Wetland HRES Copernicus data
    origin_wetland = r"....\DATA"
    # Please set your path to the symbology folder of the origin Wetland HRES Copernicus data
    origin_wetland_color = r"....\Symbology"
    # Call the main function
    utils.wetCopRF(foldername, workspace, origin_wetland, origin_wetland_color, origin_sentinel)