from wetCopRF.py import *
if __name__ == "__main__":
    # Client
    # Please set your workspace location (Requires at least twice the free space of the input data)
    workspace = r"F:\UNI\GEO419b_Abschlussprojekt\2_Testrun"
    # Please set your foldername for this classification run
    foldername = 'test4'
    # Please set your path to the origin Sentinel-1 radardata (VH,VV)
    origin_sentinel = [r"F:\UNI\GEO419b_Abschlussprojekt\0_Basedata\SWOS_Estonia_Matsalu\VH",
                       r"F:\UNI\GEO419b_Abschlussprojekt\0_Basedata\SWOS_Estonia_Matsalu\VV"]
    # Please set your path to the origin Wetland HRES Copernicus data
    origin_wetland = r"F:\UNI\GEO419b_Abschlussprojekt\0_Basedata\WAW_2018_010m_ee_03035_v020-20211105T103255Z-001\WAW_2018_010m_ee_03035_v020\DATA"
    # Please set your path to the symbology folder of the origin Wetland HRES Copernicus data
    origin_wetland_color = r"F:\UNI\GEO419b_Abschlussprojekt\0_Basedata\WAW_2018_010m_ee_03035_v020-20211105T103255Z-001\WAW_2018_010m_ee_03035_v020\Symbology"
    # Call the main function
    wetCopRF(foldername, workspace, origin_wetland, origin_wetland_color, origin_sentinel)
