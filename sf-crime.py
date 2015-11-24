import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as pl
import seaborn as sns

# Initialise variables
wk_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
we_days = ['Friday', 'Saturday', 'Sunday']

# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
mapdata = np.loadtxt("./input/sf_map_copyright_openstreetmap_contributors.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366], [37.699, 37.8299]]

z = zipfile.ZipFile('./input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'))

train['Xok'] = train[train.X < -121].X
train['Yok'] = train[train.Y < 40].Y
train = train.dropna()
train_d = train[train.Category == 'DRUNKENNESS']

sorter = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
train_d.DayOfWeek = pd.Categorical(train_d.DayOfWeek, sorter)
train_d.sort_values('DayOfWeek', ascending=True)

# Seaborn FacetGrid, split by crime Category
g = sns.FacetGrid(train_d, col="DayOfWeek", col_wrap=3, size=5, aspect=1/asp)

# Show the background map
for ax in g.axes:
    ax.imshow(mapdata,
              cmap=pl.get_cmap('gray'),
              extent=lon_lat_box,
              aspect=asp)
# Kernel Density Estimate plot
g.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)

pl.savefig('drunkenness_weekday_map.png')
