"""
This produces a smoothed a0,a1 and fcc prior from the MODIS
active fire record using a Gaussian Process
"""
#!gdalwarp -t_srs "+proj=longlat +ellps=WGS84" -te -180 -90 180 90 -r near -dstnodata -999 means.tif means_wgs84.tif
#!gdalwarp -t_srs "+proj=longlat +ellps=WGS84" -te -180 -90 180 90 -r near -dstnodata -999 covs.tif covs_wgs84.tif
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import gdal
import numpy as np
#import smoothn

means = gdal.Open("means_wgs84.tif").ReadAsArray()
cov = gdal.Open("covs_wgs84.tif").ReadAsArray()
cov = cov.reshape((3,3, 294, 589))

"""
Get projection info
"""
geo = gdal.Open("means_wgs84.tif").GetGeoTransform()
proj = gdal.Open("covs_wgs84.tif").GetProjection()

# clean up
means[means==-999]=np.nan
cov[cov==-999]=np.nan
#good = np.logical_and(means[1]>0, means[1]<0.15)
#means[1][~good]=np.nan
#good = np.logical_and(means[2]>0, means[2]<0.35)
#means[2][~good]=np.nan
# this works okay
# how to get uncs?
#a0 = smoothn.smoothn(means[1], isrobust=True, sd=np.sqrt(cov[1,1]))
#a1 = smoothn.smoothn(means[2], isrobust=False, sd=np.sqrt(cov[2,2]))
#x = smoothn.smoothn(means[0], isrobust=False, sd=np.sqrt(cov[0,0]))


"""
Just run over a small area
first as its quite slow
"""
ymin = 160
ymax = 220
xmin = 480
xmax = 550

#means = means[:, ymin:ymax, xmin:xmax]
#cov = cov[:, :, ymin:ymax, xmin:xmax]

"""
Make x-y grid
"""
ny, nx = cov.shape[-2], cov.shape[-1]
xy = np.mgrid[0:ny:1, 0:nx:1]
xy2 = np.mgrid[0:ny:1, 0:nx:1] # and prediction grid

# this will do for now
sparse = 5


"""
*--a0 --*
"""
a0_m = means[1]
a0_s = np.sqrt(cov[1,1])
good = np.logical_and(a0_m>0, a0_m<0.15)
mask = good
X = np.stack([xy[0, mask], xy[1, mask]]).T
Y = a0_m[mask]
std = a0_s[mask]
# remove rubbish
std[~np.isfinite(std)]=1e6
"""
using the Matern kernel
"""
kernel = Matern(nu=1.5, length_scale=(50, 50), length_scale_bounds=(1e1, 1e2)) #+ WhiteKernel(noise_level=std.mean())
gpr = GaussianProcessRegressor(kernel=kernel,
         random_state=0, alpha=std[::sparse]**2, n_restarts_optimizer=1, normalize_y=True).fit(X[::sparse], Y[::sparse])
# Do prediction
Xpre = np.stack([xy2[0].flatten(), xy2[1].flatten()]).T
Ypre = list(gpr.predict(Xpre, return_std=True))
Ypre[0] = Ypre[0].reshape((ny, nx))
Ypre[1] = Ypre[1].reshape((ny, nx))
a0 = Ypre


"""
*--a1 --*
"""
a1_m = means[2]
a1_s = np.sqrt(cov[2,2])
good = np.logical_and(a1_m>0, a1_m<0.35)
mask = good
X = np.stack([xy[0, mask], xy[1, mask]]).T
Y = a1_m[mask]
std = a1_s[mask]
# remove rubbish
std[~np.isfinite(std)]=1e6
"""
using the Matern kernel
"""
kernel = Matern(nu=1.5, length_scale=(50, 50), length_scale_bounds=(1e1, 1e2)) #+ WhiteKernel(noise_level=std.mean())
gpr = GaussianProcessRegressor(kernel=kernel,
         random_state=0, alpha=std[::sparse]**2, n_restarts_optimizer=1, normalize_y=True).fit(X[::sparse], Y[::sparse])
# Do prediction
Xpre = np.stack([xy2[0].flatten(), xy2[1].flatten()]).T
Ypre = list(gpr.predict(Xpre, return_std=True))
Ypre[0] = Ypre[0].reshape((ny, nx))
Ypre[1] = Ypre[1].reshape((ny, nx))
a1 = Ypre

"""
*-- fcc --*
"""
fcc_m = means[0]
fcc_s = np.sqrt(cov[0,0])
good = np.logical_and(fcc_m>0, fcc_m<1.2)
mask = good
X = np.stack([xy[0, mask], xy[1, mask]]).T
Y = fcc_m[mask]
std = fcc_s[mask]
# remove rubbish
std[~np.isfinite(std)]=1e6
"""
using the Matern kernel
"""
kernel = Matern(nu=1.5, length_scale=(50, 50), length_scale_bounds=(1e1, 1e2)) #+ WhiteKernel(noise_level=std.mean())
gpr = GaussianProcessRegressor(kernel=kernel,
         random_state=0, alpha=std[::sparse]**2, n_restarts_optimizer=1, normalize_y=True).fit(X[::sparse]**2, Y[::sparse]**2)
# Do prediction
Xpre = np.stack([xy2[0].flatten(), xy2[1].flatten()]).T
Ypre = list(gpr.predict(Xpre, return_std=True))
Ypre[0] = Ypre[0].reshape((ny, nx))
Ypre[1] = Ypre[1].reshape((ny, nx))
fcc = Ypre


"""
Get a land-water mask from GFED
"""
#!gdalwarp -ts 589 294 -t_srs "+proj=longlat +ellps=WGS84" -te -180 -90 180 90 -r near -dstnodata -999 /gws/nopw/j04/nceo_generic/users/jbrennan01/TColBA/GFED_regions.tif GFED_regions.tif
LW = gdal.Open("GFED_regions.tif").ReadAsArray()
L = LW>0
#a0[0][~L]=-999
#a0[1][~L]=-999
#a1[0][~L]=-999
#a1[1][~L]=-999
# Need a higher res mask i think!

"""
Write to geo-tiffs
"""


out = './'
dst_ds = gdal.GetDriverByName('GTiff').Create(out+'_prior_mean.tif', nx, ny, 3, gdal.GDT_Float32)
dst_ds.SetGeoTransform(geo)    # specify coords
dst_ds.SetProjection(proj) # export coords to file
dst_ds.GetRasterBand(1).WriteArray(a0[0])   # write r-band to the raster
dst_ds.GetRasterBand(2).WriteArray(a1[0])   # write g-band to the raster
dst_ds.GetRasterBand(3).WriteArray(fcc[0])   # write b-band to the raster
[dst_ds.GetRasterBand(b).SetNoDataValue(-999) for b in range(1,4)]
dst_ds.FlushCache()                     # write to disk
dst_ds = None

dst_ds = gdal.GetDriverByName('GTiff').Create(out+'_prior_std.tif', nx, ny, 3, gdal.GDT_Float32)
dst_ds.SetGeoTransform(geo)    # specify coords
dst_ds.SetProjection(proj) # export coords to file
dst_ds.GetRasterBand(1).WriteArray(a0[1])   # write r-band to the raster
dst_ds.GetRasterBand(2).WriteArray(a1[1])   # write g-band to the raster
dst_ds.GetRasterBand(3).WriteArray(fcc[1])   # write b-band to the raster
dst_ds.FlushCache()                     # write to disk
[dst_ds.GetRasterBand(b).SetNoDataValue(-999) for b in range(1,4)]
dst_ds = None

"""
up-scale them with gdal
"""

#!gdalwarp -ts 7200 3600 -overwrite -t_srs "+proj=longlat +ellps=WGS84" -te -180 -90 180 90 -r near -dstnodata -999 _prior_mean.tif prior_mean.tif
#!gdalwarp -ts 7200 3600 -overwrite -t_srs "+proj=longlat +ellps=WGS84" -te -180 -90 180 90 -r near -dstnodata -999 _prior_std.tif prior_std.tif
