import gdal
import glob
from scipy import signal
from kernels import *
import numpy as np
import argparse
import textwrap as _textwrap
from regularisation import *
import logging
gdal.SetCacheMax(4000000000)
"""
TODO

  *  Figure out the broband conversions for the two sensors so that the kernels match up!
        laing paper or something?

  *  Just noticed get negative reflectance when not enough obs for kernels...
"""

global ysize
global xsize

__author__ = "James Brennan"
__copyright__ = "Copyright 2019 James Brennan"
__version__ = "0.1 (06.03.2019)"
__email__ = "james.brennan.11@ucl.ac.uk"

class MultilineFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        paragraphs = text.split('|n ')
        multiline_text = ''
        for paragraph in paragraphs:
            formatted_paragraph = _textwrap.fill(paragraph, width, initial_indent=indent,
                                                    subsequent_indent=indent) + '\n\n'
            multiline_text = multiline_text + formatted_paragraph
        return multiline_text


def mkdate(datestr):
    return datetime.datetime.strptime(datestr, '%Y-%m-%d')



"""
*-- QA functions --*
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

def apply_QA_AVHRR(qa, b1_refl, b2_refl):
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
    qa_mask2 = notNight * no_sunglint * land_pixel * no_cloud * no_cloud_shdow
    #qa_mask = notNight & no_sunglint & land_pixel & no_cloud
    dark_dense_veg = bitMask(qa, 5, 1, bitString="1")
    """
    check refl is sensible
    """
    refl_mask = np.logical_and(b1_refl>0, b1_refl<1)
    # check NDVI is sensible eg 0.1-1
    ndvi = (b2_refl-b1_refl)/(b2_refl+b1_refl)
    ndvi_mask = np.logical_and(ndvi>0.05, ndvi<0.99)
    mask = qa_mask * refl_mask * ndvi_mask
    return mask


"""
*-- Broadband calculations --*
    These are taken from Liang et al. (2001)
"""
def avhrr_broadband(refl):
    """
    Convert AVHRR to broadband vis and NIR
    """
    alpha1, alpha2= refl[:, 0], refl[:, 1]
    vis = 0.0074 + 0.5975 * alpha1 + 0.4410 * alpha1**2
    nir =  -1.4759 *alpha2**2 - 0.6536 * alpha2**2 + 1.8591 * alpha1 * alpha2 + 1.063 * alpha2
    # overwrite
    refl[:, 0]=vis
    refl[:, 1]=nir
    del vis
    del nir

def modis_broadband(iso, vol, geo):
    """
    Produce MODIS broadband vis and NIR kernels...
    """
    a1, a2,a3, a4, a5, a6, a7 = iso
    iso_vis = 0.331*a1 + 0.424*a3 + 0.246*a4
    iso_nir = 0.039*a1 + 0.504*a2 - 0.071*a3 + 0.105*a4 \
              + 0.252*a5 + 0.069*a6 + 0.101*a7
    a1, a2,a3, a4, a5, a6, a7 = vol
    vol_vis = 0.331*a1 + 0.424*a3 + 0.246*a4
    vol_nir = 0.039*a1 + 0.504*a2 - 0.071*a3 + 0.105*a4 \
              + 0.252*a5 + 0.069*a6 + 0.101*a7
    a1, a2,a3, a4, a5, a6, a7 = geo
    geo_vis = 0.331*a1 + 0.424*a3 + 0.246*a4
    geo_nir = 0.039*a1 + 0.504*a2 - 0.071*a3 + 0.105*a4 \
              + 0.252*a5 + 0.069*a6 + 0.101*a7
    # stack into a useful form
    K = np.stack([(iso_vis, vol_vis, geo_vis), (iso_nir, vol_nir, geo_nir)])
    return K


"""
*-- IO functions --*
"""
def load_MODIS_prior(ymin, ymax, xmin, xmax):
    """
    BRDF use the modis 5km brdf parameters...

    and maybe even the isotropic long-term?
    """
    # find files
    files = glob.glob("/data/store05/phd/data/zcfah19/PhD/avhrr/mcd43c3/*/*hdf")
    files.sort()
    # Define some constants
    nT = len(files)
    ysize = ymax - ymin
    xsize = xmax - xmin
    # make some storage
    B1 = np.zeros((nT, 3, ysize, xsize), dtype=np.float32)
    B2 =  np.zeros((nT,3, ysize, xsize), dtype=np.float32)
    QA =  np.zeros((nT, ysize, xsize)).astype(bool)
    for f in files:
        t = int(f.split("/")[-1].split(".")[1][-3:])
        try:
            """
            Want to try the narrow -> broadband calculation
            """
            data = []
            for param in range(1, 4):
                for band in range(1,8):
                    tmpl = 'HDF4_EOS:EOS_GRID:"%s":MCD_CMG_BRDF_0.05Deg:BRDF_Albedo_Parameter%i_Band%i'
                    arr = gdal.Open(tmpl % (f, param, band)).ReadAsArray(yoff=ymin, xoff=xmin, xsize=xsize, ysize=ysize)
                    data.append(arr.astype(np.float32) * 0.001)
            data = np.array(data)
            iso, vol, geo = data[:7], data[7:14], data[14:]
            """
            do narrowband -> broadband
            """
            vis, nir = modis_broadband(iso, vol, geo)
            B1[t]=vis
            B2[t]=nir
        except:
            pass
    # remove errors
    """
    Now these seem quite noisy??
    So let's do a little "averaging" with some low-pass filtering for now...
    eg make 16-day smoothed product

    this seems to work ok
    signal.savgol_filter(B2[:, 1], 101, 1, axis=0)
    """
    # fill missing values for now...
    # overwrite them
    for p in range(3):
        B1[:, p] = signal.savgol_filter(B1[:, p], 151, 3, axis=0)
    for p in range(3):
        B2[:, p] = signal.savgol_filter(B2[:, p], 151, 3, axis=0)
    kPriors = np.swapaxes(np.stack((B1, B2)), 0, 1)
    return kPriors


def load_avhrr(ymin, ymax, xmin, xmax, ucl=False):
    """
    """
    # find files
    if ucl:
        files = glob.glob("./data/*hdf")
    else:
        files = glob.glob("/work/scratch/jbrennan01/data/*hdf")
    files.sort()
    # Define some constants
    nT = len(files)
    # make some storage
    B1 = np.zeros((nT, ysize, xsize), dtype=np.float32)
    B2 =  np.zeros((nT, ysize, xsize), dtype=np.float32)
    QA =  np.zeros((nT, ysize, xsize)).astype(bool)
    SZA = np.zeros((nT, ysize, xsize), dtype=np.float32)
    VZA =  np.zeros((nT, ysize, xsize), dtype=np.float32)
    RAA =  np.zeros((nT, ysize, xsize), dtype=np.float32)
    # loop over files
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
        # get doy
        t = int(f.split("/")[-1].split(".")[1][-3:])
        try:
            b1 = gdal.Open(tmpl_b1).ReadAsArray(yoff=int(ymin), xoff=int(xmin), xsize=int(xsize), ysize=int(ysize))
            b2 = gdal.Open(tmpl_b2).ReadAsArray(yoff=int(ymin), xoff=int(xmin), xsize=int(xsize), ysize=int(ysize))
            qa = gdal.Open(tmpl_qa).ReadAsArray(yoff=int(ymin), xoff=int(xmin), xsize=int(xsize), ysize=int(ysize))
            sza = gdal.Open(tmpl_sza).ReadAsArray(yoff=int(ymin), xoff=int(xmin), xsize=int(xsize), ysize=int(ysize))
            vza = gdal.Open(tmpl_vza).ReadAsArray(yoff=int(ymin), xoff=int(xmin), xsize=int(xsize), ysize=int(ysize))
            _raa = gdal.Open(tmpl_raa).ReadAsArray(yoff=int(ymin), xoff=int(xmin), xsize=int(xsize), ysize=int(ysize))
            scaleref = 1e-4
            b1 = b1.astype(np.float32) * scaleref
            b2 = b2.astype(np.float32) * scaleref
            scale_ang = 0.01
            sza = sza.astype(np.float32) * scale_ang
            vza = vza.astype(np.float32) * scale_ang
            _raa = _raa.astype(np.float32) * scale_ang
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
            mask = apply_QA_AVHRR(qa, b1, b2)
            # save them
            B1[t] = b1
            B2[t]= b2
            QA[t]= mask
            VZA[t]= vza
            SZA[t]= sza
            RAA[t]= raa
            #logger.logger.handlers[0].flush()
        except:
            pass
    # make storage into masked arrays
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
    # return what we need
    return refl, SZA, VZA, RAA


def load_spectral_prior(file_, ymin, ymax, xmin, xmax):
    arr = gdal.Open(file_).ReadAsArray(yoff=int(ymin), xoff=int(xmin), xsize=int(xsize), ysize=int(ysize))
    return arr


"""
*-- BRDF functions --*
"""
def modBRDF():
    """
    Use the Modis geo and vol for the correction
    ISSUE: arrays not perfectly aligned atm...
    """
    for band in range(2):
        _vol = kPriors[m[:-1], band, 1, y, x]
        _geo = kPriors[m[:-1], band, 2, y, x]
        ke = np.zeros((m.sum(), 2)).T
        ke[:, :-1]=np.stack((_vol, _geo))
        ke[:, -1]=(_vol[-1], _geo[-1])
        # subtract brdf...
        brdf2 = (K[:, 1:] * ke.T).sum(axis=1)
        iso = refl[m, band, y, x] - brdf2
        # put back into the refl

def do_BRDF_correction(refl, vza, sza, raa):
    """
    So version 1 idea:
    Use the MODIS priors to correct the brdf of AVHRR
    with a simple brdf subtraction scheme...
    """
    ysize = vza.shape[1]
    xsize = vza.shape[2]
    for y in range(ysize):
        for x in range(xsize):
            _vza = vza[:, y,x]
            _sza = sza[:, y,x]
            _raa = raa[:, y,x]
            m = ~_vza.mask
            if m.sum() > 20:
                kerns =  Kernels(_vza, _sza, _raa,
                                LiType='Sparse', doIntegrals=False,
                                normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                                RossType='Thick')
                K = np.stack((kerns.Isotropic, kerns.Ross, kerns.Li)).T
                # mask
                K =K[m]
                # solve it
                xx = np.linalg.lstsq(K, refl[m, :, y, x])[0]
                brdf = (K[:, 1:]).dot(xx[1:, :])
                iso = refl[m, :, y, x] - brdf
                refl[m, :, y, x]=iso
    return None



def sort_logging(extra):
    """
    Sort out logging file
    """
    logger = logging.getLogger(__name__)
    syslog = logging.StreamHandler()
    formatter = logging.Formatter('%(extra)s %(asctime)s - %(message)s')
    #syslog.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    #logger.addHandler(syslog)
    # create a file handler
    #handler = logging.FileHandler('log1234.log')
    handler = logging.FileHandler('/work/scratch/jbrennan01/log1234.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    # extra info into the logger
    logger = logging.LoggerAdapter(logger, extra)
    return logger


#print logoTxt

helpTxt = """
AVHRR processor version 0.1 \n
----------------
This a first implementation of the algorithm for uncertainty characterised
BA retrieval for AVHRR
"""


if __name__ == "__main__":


    """
    get options
    """
    parser = argparse.ArgumentParser(description=helpTxt,  formatter_class=MultilineFormatter)
    parser.add_argument('ymin',  type=int,
                        help='ymin of the processing')
    parser.add_argument('xmin', type=int,
                        help='xmin of the processing')
    parser.add_argument('ymax',  type=int, default=None,
                        help='ymax of the processing')
    parser.add_argument('xmax',  type=int, default=None,
                        help='xmax of the processing')
    parser.add_argument('--outdir', dest='outdir',
                        default="/home/users/jbrennan01/DATA2/avhrr/outputs/",
                        help="""specify a custom output directory for the files.
                             Otherwise a sub-directory determined by the tile. """)
    options = parser.parse_args()


    global logger
    extra = {'extra':'[AVHRR_PROCESSOR_V1]'}
    logger = sort_logging(extra)
    logger.info("Commence processing")

    # Australia
    ymin = 2000
    ymax = 2600
    xmin = 5800
    xmax = 6700

    # Africa
    ymin = 1850
    ymax = 2200
    xmin = 3900
    xmax = 4300

    # Boreal area
    ymin = 600
    ymax = 900
    xmin = 5000
    xmax = 5400

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

    # Africa 2
    ymin = 1518
    ymax = ymin + 200
    xmin = 4100
    xmax = xmin + 200

    # Africa 3
    ymin = 1930
    ymax = ymin + 200
    xmin = 3880
    xmax = xmin + 200

    # north america /alaska
    ymin = 640
    ymax = ymin + 200
    xmin = 1240
    xmax = xmin + 200

    # china
    ymin = 810
    ymax = ymin + 200
    xmin = 6060
    xmax = xmin + 200



    # get parsed extent
    ymin = options.ymin
    xmin = options.xmin
    xmax = options.xmax
    ymax = options.ymax
    # force limits just incase
    xmax = np.minimum(xmax, 7200)
    ymax = np.minimum(ymax, 3600)
    outdir = options.outdir


    # us2
    ymin = 970
    ymax = ymin + 200
    xmin = 1740
    xmax = xmin +200


    # alaska
    ymin = 500
    ymax = ymin + 200
    xmin = 1350
    xmax = 1350 + 200

    # get size
    import builtins

    ysize = ymax - ymin
    xsize = xmax - xmin
    builtins.ysize = ysize
    builtins.xsize = xsize

    # phenology NA
    name = 'us_pheno'
    ymin = 940
    ymax = ymin +200
    xmin = 1700
    xmax = xmin + 200
    # get size
    ysize = ymax - ymin
    xsize = xmax - xmin
    builtins.ysize = ysize
    builtins.xsize = xsize



    """
    Load AVHRR
    """
    refl, sza, vza, raa = load_avhrr(ymin, ymax, xmin, xmax, ucl=True)


    # load africa fires
    dd = np.load("africa_fires_3950_1580.npz")
    refl = dd['refl']
    sza = dd['sza']
    vza = dd['vza']
    raa = dd['raa']
    mask = dd['mask']


    sza = np.ma.array(mask=mask, data=sza)
    vza = np.ma.array(mask=mask, data=vza)
    raa = np.ma.array(mask=mask, data=raa)
    refl = np.ma.array(mask=np.swapaxes(np.stack([mask, mask]), 0,1), data=refl)



    """
    Load MODIS kernel priors
    """
    #kPriors = load_MODIS_prior(ymin, ymax, xmin, xmax)


    """
    Do BRDF correction...
    """
    kWeights = do_BRDF_correction(refl, vza, sza, raa)
    #logger.info("Done BRDF correction")
    del sza
    del raa
    """
    Run the edge preserving
    """
    # pre-make the D matrix for speed
    nT = len(refl)
    Dmat = make_D_matrix(nT)
    filled = np.zeros((nT, 2, ysize, xsize), dtype=np.float32)
    filled_unc = np.zeros((nT, 2, ysize, xsize))
    for y in range(0, ysize):
        for x in range(xsize):
            m = (~refl[:, 1, y,x].mask).sum()
            if m > 10:
                # use about 500
                sol,unc,w = regularisation(~refl[:, 1,y, x].mask, refl[:, :, y, x], alpha=1, _D=Dmat)
                filled[:, :, y, x]=sol
                filled_unc[:,:, y,x]=unc
    del vza

    """
    Check ndvi again...
    remove where ndvi is nonse
    eg otuside 0.1-1
    """
    ndvi = (filled[:,1]-filled[:, 0])/(filled[:, 1]+filled[:, 0])
    bad_ndvi = np.logical_or(ndvi<0.1, ndvi>1.1)
    # mask filled with this
    filled = np.ma.array(data=filled, mask = np.stack((bad_ndvi, bad_ndvi)).swapaxes(0,1))
    filled_unc = np.ma.array(data=filled_unc, mask = np.stack((bad_ndvi, bad_ndvi)).swapaxes(0,1))
    """
    Load and produce spectral priors
    """
    #fcc_prior, a0_prior, a1_prior = load_spectral_prior("priors.tif", ymin, ymax, xmin, xmax)
    a0_prior, a1_prior, fcc_prior = load_spectral_prior("prior_mean.tif", ymin, ymax, xmin, xmax)
    a0s_prior, a1s_prior, fcc_prior = load_spectral_prior("prior_std.tif", ymin, ymax, xmin, xmax)
    """
    Try fcc with fixed burn signal
    """
    wavelengths = np.array([640., 860])
    loff = 400.
    lmax = 2000.
    ll =  wavelengths - loff
    llmax = lmax-loff
    lk = (2.0 / llmax) * (ll - ll*ll/(2.0*llmax))
    # fix a0 and a1
    a0 = 0.02
    a1 = 0.25
    #make the burn signal
    #burn_signal = np.ones(2) * a0 + a1 * lk

    """
    Make the spatial burn signals
    """
    burn_signal = np.ones((ysize, xsize)) * a0_prior + a1_prior * lk[:, None, None]


    """
    solve it all big style

    We can use the normal equations to solve everything
    at once... with matrix transformations!
    #

    eg want to replicate
    (1/K.T.dot(K)).dot(K.T).dot(obs)
    """
    fcc = np.zeros((nT, ysize, xsize))
    fcc_unc = np.zeros((nT, ysize, xsize))
    rmse = np.zeros((nT, ysize, xsize))
    # add a little border

    """
    We want to weight the signal more
    towards the change in the NIR
    bigger fire signal
    so W matrix too
    W = 0.33, 0.66
    """
    _w = np.array([.1, .9])
    W = np.ones((2, ysize*xsize))
    W.T[:]=_w

    for t in range(1, nT-1):
        pre = filled[t-1]
        post = filled[t]
        pre_unc = filled_unc[t-1]
        post_unc = filled_unc[t]
        K = burn_signal.reshape((2, -1)) - pre.reshape((2, -1))
        obs = post.reshape((2, -1)) - pre.reshape((2, -1))
        C = 1/(pre_unc**2 + post_unc**2).reshape((2, -1))
        #C = C*W
        """
        Solve it
        without band weighting for now
        """
        #KTK = (K.T *C.T * K.T).sum(axis=1)
        KTK = (K.T*K.T).sum(axis=1)
        Inv = 1/KTK
        #_fcc= (Inv * K * C * obs).sum(axis=0)
        _fcc = (Inv * K * obs).sum(axis=0)
        err = np.sqrt((((K * _fcc) - obs)**2).sum(axis=0))
        fcc[t]=_fcc.reshape((ysize, xsize))
        rmse[t]= err.reshape((ysize, xsize))
        """
        get the uncertainty in fcc
        Given the use of the weights matrix
        its different
        https://en.wikipedia.org/wiki/Weighted_least_squares
        M^\beta = (X^{\rm T} W X)^{-1} X^{\rm T}
                    W M W^{\rm T} X (X^{\rm T} W^{\rm T} X)^{-1}
        """
        #a = (1/(K.T*W.T*K.T))
        #b = (K.T * W.T * C.T *W.T *K.T)
        #_fcc_unc = a * b *a
        # Do the inverse again
        KTK = (K.T *C.T * K.T).sum(axis=1)
        Inv = 1/KTK
        fcc_unc[t]=np.sqrt(Inv.reshape((ysize, xsize)))

    #logger.info("Done fcc calculation")
    """
    Record maximum fc
    """
    # remove dodgy
    mask = np.logical_or(fcc<0, fcc>1.2)
    fcc[mask]=-1
    cc = np.ma.max(fcc, axis=0)
    dob = np.ma.argmax(fcc, axis=0)
    #error = rmse[dob]
    error = np.take_along_axis(rmse, dob.reshape((1, ysize, xsize)), axis=0)[0]
    unc = np.take_along_axis(fcc_unc, dob.reshape((1, ysize, xsize)), axis=0)[0]

    """
    Produce a proto product

    - want to save cc max, rmse, dob into a file
    """
    #logger.info("Writing files")
    outdir ='./outputs/'
    outfile = outdir + f'BA_AVHRR_2001_{ymin}_{xmin}.npz'
    np.savez(outfile, fcc=cc, dob=dob, rmse=error, unc=unc)
    logger.info("Written files. ")
    logger.info("Finished. ")
