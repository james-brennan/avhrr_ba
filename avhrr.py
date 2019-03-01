import gdal
import glob

files = glob.glob("/data/store05/phd/data/zcfah19/PhD/avhrr/*hdf")
files.sort()
from kernels import *


"""
BRDF use the modis 5km brdf parameters...

and maybe even the isotropic long-term?
"""





"""
    QA functions
"""
def bitMask(qa, bitStart, bitLength, bitString="00"):
    """
    makes mask for a particular part of the modis bit string
    "inspired" from pymasker
    """
    lenstr = ''
    for i in range(bitLength):
        lenstr += '1'
    bitlen = int(lenstr, 2)

    if type(bitString) == str:
        value = int(bitString, 2)

    posValue = bitlen << bitStart
    conValue = value << bitStart
    mask = (qa & posValue) == conValue
    return mask

def apply_QA(qa, refl):
    """
    make a QA mask
    """
    notNight = bitMask(qa, 6, 1, bitString="0")
    no_sunglint = bitMask(qa, 4, 1, bitString="0")
    land_pixel = bitMask(qa, 3, 1, bitString="0")
    no_cloud = bitMask(qa, 1, 1, bitString="0")
    no_cloud_shdow = bitMask(qa, 1, 1, bitString="0")
    no_brdf_iss = bitMask(qa, 14, 1, bitString="0")
    qa_mask = notNight & no_sunglint & land_pixel & no_cloud & no_cloud_shdow
    #qa_mask = notNight & no_sunglint & land_pixel & no_cloud
    dark_dense_veg = bitMask(qa, 5, 1, bitString="1")
    """
    check refl is sensible
    """
    refl_mask = np.logical_and(refl>0, refl<1)
    mask = qa_mask & refl_mask
    return mask



#veg = bitMask(qa, 5, 1, bitString="1")





# Define some constants
nT = len(files)

# Australia
ymin = 2000
ymax = 2600
xmin = 5800
xmax = 6700

# aus subset
ymin=2135
xmin=6317
xmax =xmin + 200
ymax=ymin+200

# Africa
ymin = 1850
ymax = 2200
xmin = 3900
xmax = 4300


# Boreal area
ymin = 600
ymax = 900
xmin = 5000
xmax = 5800

ymin = 2000
ymax = 2600
xmin = 5800
xmax = 6700

# Australia
ymin = 2000
ymax = 2600
xmin = 5800
xmax = 6700


# Boreal area
ymin = 600
ymax = 900
xmin = 5000
xmax = 5800

ysize = ymax - ymin
xsize = xmax - xmin


# make some storage
B1 = np.zeros((nT, ysize, xsize))
B2 =  np.zeros((nT, ysize, xsize))
QA =  np.zeros((nT, ysize, xsize))
SZA = np.zeros((nT, ysize, xsize))
VZA =  np.zeros((nT, ysize, xsize))
RAA =  np.zeros((nT, ysize, xsize))





for f in files:
    """
    load b1 and b2
    """
    tmpl_b1  = 'HDF4_EOS:EOS_GRID:"%s":Grid:SREFL_CH1' %f
    tmpl_b2  = 'HDF4_EOS:EOS_GRID:"%s":Grid:SREFL_CH2' %f
    tmpl_qa  = 'HDF4_EOS:EOS_GRID:"%s":Grid:QA' %f
    tmpl_sza = 'HDF4_EOS:EOS_GRID:"%s":Grid:SZEN' %f
    tmpl_vza = 'HDF4_EOS:EOS_GRID:"%s":Grid:VZEN' %f
    tmpl_raa = 'HDF4_EOS:EOS_GRID:"%s":Grid:RELAZ' %f

    t = int(f.split("/")[-1].split(".")[1][-3:])


    try:
        b1 = gdal.Open(tmpl_b1).ReadAsArray(yoff=ymin, xoff=xmin, xsize=xsize, ysize=ysize)
        b2 = gdal.Open(tmpl_b2).ReadAsArray(yoff=ymin, xoff=xmin, xsize=xsize, ysize=ysize)
        qa = gdal.Open(tmpl_qa).ReadAsArray(yoff=ymin, xoff=xmin, xsize=xsize, ysize=ysize)
        sza = gdal.Open(tmpl_sza).ReadAsArray(yoff=ymin, xoff=xmin, xsize=xsize, ysize=ysize)
        vza = gdal.Open(tmpl_vza).ReadAsArray(yoff=ymin, xoff=xmin, xsize=xsize, ysize=ysize)
        _raa = gdal.Open(tmpl_raa).ReadAsArray(yoff=ymin, xoff=xmin, xsize=xsize, ysize=ysize)
        scaleref = 1e-4
        b1 = b1.astype(float) * scaleref
        b2 = b2.astype(float) * scaleref
        scale_ang = 0.01
        sza = sza.astype(float) * scale_ang
        vza = vza.astype(float) * scale_ang
        _raa = _raa.astype(float) * scale_ang
        """
        nan fills
        """
        sza[sza==-9999]=np.nan
        vza[vza==-9999]=np.nan
        _raa[_raa==-9999]=np.nan
        # correct raa
        _r = np.deg2rad(_raa)
        SIN_REL_AZ = np.sin(_r)
        COS_REL_AZ = np.cos(_r)
        raa = np.rad2deg(np.arctan2(SIN_REL_AZ, COS_REL_AZ))
        """
        do qa
        """
        mask = apply_QA(qa, b1)
        # save them
        B1[t] = b1
        B2[t]= b2
        QA[t]= mask
        VZA[t]= vza
        SZA[t]= sza
        RAA[t]= raa
        print(f)
    except:
        print("failed", f)



QA = np.array(QA).astype(bool)
B1 = np.ma.array(data=B1, mask=~QA)
B2 = np.ma.array(data=B2, mask=~QA)
refl = np.ma.stack([B1, B2], axis=1)
SZA = np.ma.array(data=SZA, mask=~QA)
VZA = np.ma.array(data=VZA, mask=~QA)
RAA = np.ma.array(data=RAA, mask=~QA)

# free up some storage
del B1
del B2
del QA


"""
try a simple brdf correction?
"""
for y in range(ysize):
    for x in range(xsize):
        vza = VZA[:, y,x]
        sza = SZA[:, y,x]
        raa = RAA[:, y,x]
        m = ~vza.mask
        if m.sum() > 20:
            kerns =  Kernels(vza, sza, raa,
                            LiType='Sparse', doIntegrals=False,
                            normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                            RossType='Thick',nbar=0.0)
            K = np.stack((kerns.Isotropic, kerns.Ross, kerns.Li)).T
            # mask
            K =K[m]
            # solve it
            xx = np.linalg.lstsq(K, refl[m, :, y, x])[0]
            # subtract brdf...
            brdf = K[:, 1:].dot(xx[1:, ])
            iso = refl[m, :, y, x] - brdf
            # put back into the refl
            refl[m, :, y, x]=iso
    print(y, x)



"""
Try fcc with fixed burn signal
"""
wavelengths = np.array([630., 858.5])
loff = 400.
lmax = 2000.
ll =  wavelengths - loff
llmax = lmax-loff
lk = (2.0 / llmax) * (ll - ll*ll/(2.0*llmax))
# fix a0 and a1
a0 = 0.01
a1 = 0.15
#make the burn signal
burn_signal = np.ones(2) * a0 + a1 * lk

"""
make pre and post windows from 8 days
"""
nT = len(refl)
kT = 10
pre = np.zeros((nT, 2, ysize, xsize))
post = np.zeros((nT, 2, ysize, xsize))
# fill them
for t in range(kT, nT-kT):
    # get where a pre
    pre[t] = np.ma.mean(refl[t-kT:t], axis=0)
    post[t] = np.ma.mean(refl[t:t+kT], axis=0)

pre = np.ma.masked_where(pre==0, pre)
post = np.ma.masked_where(post==0, post)

# calculate fcc
fcc = (post - pre) / (burn_signal[None, :, None, None] - pre)

"""
Try some simple threshold tests first...

both bands are within good range?

eg both fcc should be greater than 0.3
"""

mask = np.logical_and(fcc[:, 0]>0.5, fcc[:, 1]>0.5)
fcc[:, 1][~mask]=0
#mask = np.logical_and(fcc[:, 0]<1.5, fcc[:, 1]<1.5)
#fcc[:, 1][~mask]=0

# check fcc within 0.3 of eachother?


mask = np.abs(fcc[:, 0]-fcc[:, 1]) <0.1
fcc[:, 1][~mask]=0


cc = np.ma.max(fcc[:, 1], axis=0)
dob = np.ma.argmax(fcc[:, 1], axis=0)

"""
is a good pixel:
y,x = 62, 17

looks like a fire

interesting is when the two values are similar...
    how do we use these as an indicator of rmse etc?

another great one is 20 13 which has a definite fire

                        35, 25 tooo
"""



y,x = 35, 45
y,x = 15, 10


for t in range(kT, nT-kT):
    # get where a pre
    _pre = pre[t, :, y, x]
    _post = post[t, :, y, x]
    # try jose's least squares idea
    obs = np.matrix(_post - _pre).T
    K = np.matrix(burn_signal - _pre ).T
    _fcc = np.linalg.lstsq(K, obs, rcond=None)[0]
    fcc[t, y, x] = _fcc




"""
solve it all big style

We can use the normal equations to solve everything
at once... with matrix transformations!
#

eg want to replicate
(1/K.T.dot(K)).dot(K.T).dot(obs)
"""
K = burn_signal[:, None] - pre.reshape((2, -1))
obs = post.reshape((2, -1)) - pre.reshape((2, -1))

"""
Solve it
"""
KTK = (K.T * K.T).sum(axis=1)
Inv = 1/KTK
fcc= (Inv * K * obs).sum(axis=0)

fcc = fcc.reshape((nT, ysize, xsize))



#iso_sm, unc, w_b, d = solve_banded_fast(refl[:, 1, y, x], alpha=alpha, band=1, do_edges=False)
