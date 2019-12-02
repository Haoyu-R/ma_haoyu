import os
import glob
import pandas as pd
import simplekml

def files_to_kml(folder_path):
    if os.path.isdir(folder_path):
        files = glob.glob(folder_path+'\*.csv')
        for file in files:
            with open(file) as traj_file:
                df_temp = pd.read_csv(traj_file, delimiter=',')
                kml_file = extract_kml(df_temp)
                kml_file.save(os.path.splitext(file)[0]+'.kml')
    else:
        print("does not found any data")


def extract_kml(dataframe):
    lat_column = [col for col in dataframe.columns if 'NP_LatDegree' in col]
    lon_column = [col for col in dataframe.columns if 'NP_LongDegree' in col]
    latitudes = dataframe[lat_column[0]]
    latitudes = latitudes.interpolate(method='linear', limit_direction='both').ffill().bfill()
    longitudes = dataframe[lon_column[0]]
    longitudes = longitudes.interpolate(method='linear', limit_direction='both').ffill().bfill()
    # Write ego kml File
    kml_file = simplekml.Kml()
    colorArray = ['50143CFF','5014B4FF','5014F0FF','5078FF00','5078FFB4','50FF7800','50FF7878','50FF78F0','5000FF14']
    folder = kml_file.newfolder(name='folder%s' % str(1))

    for i in range(0, len(latitudes)-51, 50):
        coordTmp = [latitudes[i], longitudes[i]]
        print(coordTmp)
        coordTmpFuture = [latitudes[i+50],longitudes[i+50]]
        pathway = folder.newlinestring(name='EgoTrace', description='EgoTrace')
        pathway.coords = [(coordTmp[1], coordTmp[0], 0), (coordTmpFuture[1], coordTmpFuture[0], 0)]
        pathway.style.linestyle.color = colorArray[0]
        pathway.style.linestyle.width = 5
    return kml_file


if __name__ == "__main__":
    folder_path = r'..\kml\potential_csv_1'
    files_to_kml(folder_path)


