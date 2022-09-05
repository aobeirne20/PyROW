import arcpy
import pandas as pd
import numpy as np

elevationRas = arcpy.Raster(r'C:\Users\augus\Documents\ArcGIS\Projects\PyROW\PyROW_CA.gdb\elevation')
elevation_array = arcpy.RasterToNumPyArray(elevationRas, nodata_to_value=-9999).flatten()

rows_width = elevationRas.width
columns_height = elevationRas.height


print(elevation_array.shape)

bad_values = np.asarray(np.where(elevation_array == -9999), dtype='int').flatten()
print(bad_values)
print(bad_values.shape)

np.save('novalue_indices', bad_values)

no_value_indices = np.load('novalue_indices.npy')

