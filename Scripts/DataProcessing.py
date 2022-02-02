import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import geopandas as gpd


def data_process(params):

    lat = np.array([])
    lon = np.array([])

    for filename in os.listdir(params.DataFolder):
        if filename.startswith('BSLocations'):
            bs_file = pd.read_csv(params.DataFolder + filename)
            lat = np.concatenate([lat, np.array(bs_file['lat'])])
            lon = np.concatenate([lon, np.array(bs_file['lon'])])

    # Remove Repeated Entry
    lat, unique_index = np.unique(lat, return_index=True)
    lon = lon[unique_index]
    lon, unique_index = np.unique(lon, return_index=True)
    lat = lat[unique_index]

    if params.visualise:
        fig, ax = plt.subplots(figsize=(25, 25))
        map_file = gpd.read_file(params.DataFolder + 'Geography.shp')
        map_file.plot(ax=ax, color='grey', alpha=0.8)
        plt.scatter(lon, lat, 1, c='black')
        plt.ylabel('Latitude', fontsize=40)
        plt.xlabel('Longitude', fontsize=40)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.title(params.Country, fontsize=50)
        plt.savefig(params.ResultsFolder + 'DataVisualization.png')

    min_lat = np.min(lat)
    min_lon = np.min(lon)

    lat_div = params.lat_div
    lon_div = (180*params.xLength)/(3.1415*params.EarthR*np.cos(lat*3.1415/180))

    # Each BS is assigned a box number according to its location
    box_number_lat = ((lat - min_lat) / lat_div).astype(int)   # At most 3 digits
    box_number_lon = ((lon - min_lon) / lon_div).astype(int)   # At most 3 digits
    box_numbers = box_number_lat*1000 + box_number_lon         # Bijective map

    t = np.argsort(box_numbers)
    box_numbers = box_numbers[t]         # Sorted Box indices
    box_number_lat = box_number_lat[t]
    box_number_lon = box_number_lon[t]
    lat = lat[t]                  # Sorted (according to box numbers) Latitude
    lon = lon[t]                  # Sorted (according to box numbers) Longitude
    lon_div = lon_div[t]          # Sorted

    # Relative position of BSs with respect to their box boundaries
    rel_y = (lat - (box_number_lat * lat_div + min_lat)) * (3.1415 * params.EarthR/180)
    rel_x = (lon - (box_number_lon * lon_div + min_lon)) * np.cos(lat*3.1415/180) * (3.1415 * params.EarthR/180)

    # Counts the number of BSs in each box, with ascending box numbers
    BoxCardinality = np.array(pd.Series(box_numbers).value_counts().sort_index())
    NumBox = np.size(BoxCardinality)          # Number of Boxes
    BoxIndex = np.array(range(NumBox))
    EndBSIndex = np.cumsum(BoxCardinality)    # Index of last BS in each Box

    if params.visualise:
        _, ax = plt.subplots(figsize=(6, 5))
        pmf = np.array(pd.Series(BoxCardinality).value_counts().sort_index()) / len(BoxCardinality)
        ax.bar(1 + np.array(range(len(pmf))), pmf)
        plt.xlabel("Number of BSs")
        plt.ylabel("Normalized Frequency")
        plt.ylim([0, 0.15])
        plt.xlim([0, 50])
        plt.savefig(params.ResultsFolder + 'NumBSDistribution.png')

    # Remove underpopulated and overpopulated boxes
    BoxIndex = BoxIndex[(BoxCardinality <= params.MaxBS)*(BoxCardinality >= params.MinBS)]
    AvgBS = np.mean(BoxCardinality[BoxIndex])

    np.save(params.ModelsFolder + 'rel_x.npy', rel_x)
    np.save(params.ModelsFolder + 'rel_y.npy', rel_y)
    np.save(params.ModelsFolder + 'AvgBS.npy', AvgBS)
    np.save(params.ModelsFolder + 'BoxIndex.npy', BoxIndex)
    np.save(params.ModelsFolder + 'NumBox.npy', np.size(BoxIndex))
    np.save(params.ModelsFolder + 'BoxCardinality.npy', BoxCardinality)
    np.save(params.ModelsFolder + 'EndBSIndex.npy', EndBSIndex)
