import numpy as np
from kernels import *
import scipy.linalg
import glob
import datetime
import numba as nb
import matplotlib.pyplot as plt
import time
np.seterr(all='ignore')
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import bandmat as bm # nice package!


global C_obs
# This is now true variance...
C_obs = 5*np.array([0.015, 0.015])**2


global GTSV
GTSV = scipy.linalg.get_lapack_funcs("gtsv")



def make_D_matrix(nT):
    """
    a function to make a sparse D matrix
    """
    I = np.eye(nT) #np.diag (np.ones(nx))
    D = (I - np.roll ( I, -1)).T
    #D = D.T.dot(D)
    D1A =np.zeros((nT, 2))
    D1A[1:, 0]=np.diag(D, 1)
    D1A[:, 1]=np.diag(D, 0)
    # convert to banded matrices
    D =  bm.BandMat(0,1, D1A.T, transposed=False)
    return D



def solve_banded_fast(y, alpha, band=1, do_edges=True, W=None, rmse=0.1, _D=None):
    """
    Speed up to actually solve the thing really fast
    using
        scipy.linalg.solve_banded(L.T, y)
    """
    # do some prep work
    nT = y.shape[0]
    q = np.zeros(nT)
    eidx = y>0
    q[eidx]=1
    Iobs = q
    # unc
    #Iobs /= C_obs[band]
    #y /= C_obs[band]
    nObs = eidx.sum()
    """
    *-- Alpha value --*
    This needs to be normalised by
    the number of observations
    OR DO I ??
    """
    """
    make D matrix
    in flattened form
    faster to pre-compute as is
    it's a standard length
    """
    #_D = None
    if _D ==None:
        D = make_D_matrix(nT)
    else:
        D = _D
    """
    get a smooth solution
    """
    # weighting for outliers
    Wr = np.ones(nT)

    scale = (nObs )
    # re-scale alpha

    """
    re-scale alpha

    So trying something a little
    different...

    Basically allow this to vary across
    the time-series too based on the number of samples 
    in a local window. I think this should improve the
    scaling of alpha so we can keep edges where there
    are lots of samples....
    """
    #winN = np.ones(20)
    #nScale = np.convolve(eidx, winN, mode='same')
    #alpha = 0.5*alpha * nScale + 0.5*alpha * scale
    alpha = alpha * scale
    alpha = np.minimum(alpha, 200)
    alpha = np.maximum(alpha, 70)
    D2 = bm.dot_mm(D.T, D)
    dl = alpha * D2.data[2][:-1]
    di =  alpha * D2.data[1]  + Iobs * Wr
    du = alpha * D2.data[0][1:]
    ret = GTSV(dl, di, du, y, False, False, False, False)
    x0 = ret[-2]
    #import pdb; pdb.set_trace()
    """
    *-- now do edge preserving --*
    """
    cost =[]
    tx = np.arange(len(y))[eidx]
    if do_edges:
        # Define some stuff
        converged=False
        convT = 1e-5
        itera =0
        MAXITER =10
        MINITER=8
        T = 0.01 # threshold in the change needed...
        # Define the weights matrix
        _wg = np.ones(nT)
        W = bm.diag(_wg)
        # run until convergence
        x = x0
        R = []
        co0 = 1e3
        var_C = C_obs[band]
        sig_C = np.sqrt(var_C)
        obs = y[eidx]
        _w = np.ones(nT-1)
        _w0 = np.ones(nT-1)*100
        while not converged and itera<MAXITER:
            """
            Robust smoothing...
            using the method from the garcia
            dct paper
            """
            r = x[eidx]-y[eidx]
            """
            h is the leverage from the
            hat matrix ege the inverse matrix
            """
            h = 0.0009
            """
            Robust estimate of the standard
            deviation of residuals
            """
            #sig_rob = 1.4826 * np.median(r - np.median(r))
            ui = r/(sig_C)
            #Wr[eidx] = (1-(ui/4.685)**2)**2
            # mask bad
            #bad = np.abs(ui/4.685)>2
            #Wr[eidx][bad]=0
            """
            *- edge preserving --*

            So this should now be fixed.
            The functional is

            w = T / sqrt( T**2 +  (dx * alpha**2)**2  )
            """
            """
            The derivative has to be scale by
            alpha
            """
            #dx = np.diff(x) * alpha**2
            #ww[1:] = T/(np.sqrt(T**2 +  dx**2))
            dx = np.diff(x) * alpha**2
            # only want drop in NIR
            dx[dx>0]=0
            """
            Thereshol T has to be scaled relative
                    to alpha (alpha actually works well)
            """
            T = 0.7 * alpha 
            # calulcate the edge preserving functional w
            _w = T / np.sqrt(T**2 + dx**2)
            _w = np.exp(-dx**2/T**2)
            """
            Scaling from CERC
            """
            wScale = nObs/np.sum(_w)
            _w = _w * wScale
            #_w = scipy.ndimage.minimum_filter(_w, 8)
            co1 = np.sqrt(np.sum((x0[eidx] - obs)**2/var_C)/nObs)
            # faster version
            cost.append(co1)
            # remove nan
            if itera > 2:
                """
                Allow for some initial robust
                outlier rejection
                """
                np.place(_w, np.isnan(_w), 1)
                W.data[0, :-1]=_w
            else:
                _w[:]=1
            """
            Remove edge effects
            -- force every outside
                in the edge windows back to 1
            """
            W.data[0, :5]=1
            W.data[0, -5:]=1
            """
            I cant figure out the flat formulat to do below
            """
            reg = alpha * bm.dot_mm(bm.dot_mm(D.T, W), D)
            dl =  reg.data[2][:-1]
            di=   reg.data[1] + Iobs * Wr
            du =  reg.data[0][1:]
            ret = GTSV(dl, di, du, y, False, False, False, False)
            x = ret[-2]
            co = np.sum((x-x0)**2) / np.sum(x0**2) + 1e-6
            cow = np.sum((_w-_w0)**2) / np.sum(_w0**2) + 1e-6
            #if co < convT:
            #    converged=True
            if cow < 1e5:
                converged=True
            if np.abs(co1-co0) < 0.01:
                converged=True
            if itera < MINITER:
                converged=False
            if _w.min() < 0.1:
                converged=True
            x0 = x
            _w0 = _w
            co0 = co1
            itera+=1
        """

        Do some checking

        So some times the algorithm freaks out
        and over does the edge preserving...

        Let's do some checks

        1. Want one strong area with a small w



        If w meets certain conditions optimise it a bit

        - Sparsify it..
            eg take the min and keep one min
                place the min at the end of its window
        """
        if np.any(_w<0.3):
            """
            sufficent edge process
            to check whats going on
            """

            """
            Check 1.

            What percentage of timesteps are below
            a threhsold --> eg is w all over the place?
            and we've added to many edges
            """
            reset=False
            frac = (_w<0.3).sum()/float(_w.shape[0])
            # want no more than 5%
            if frac > 0.1:
                # re-set edge process
                _w.fill(1)
                reset=True
            idx = np.nanargmin(_w)
            val = _w[idx]
            """
            Want to check a couple of things

            1. If min w aligns with an observation
                how far off the prediction are we?

                --> this helps remove cases where
                    we have shot noise from clouds
                    or cloud shadows
            """
            if eidx[idx]: # have an observ
                if np.abs((y[idx]-x[idx]))/np.sqrt(C_obs[band])>2:
                    # it's a nonse break remove
                    _w.fill(1)
                    reset=True
            else:
                # check the surrounding pixels
                # to find some obs to check
                low = np.maximum(idx-40, 0)
                upp = np.minimum(idx+40, 363)
                mask = eidx[low:upp]
                yy = y[low:upp][mask]
                xx = x[low:upp][mask]
                _ww = _w[low:upp][mask]
                zz = np.abs((yy-xx))/np.sqrt(C_obs[band])
                cond = np.logical_and(_ww<0.3, zz>2).sum()
                if cond > 4:
                    _w.fill(1)
                    reset=True
            """
            Resolve it...
            """
            if not reset:
                """
                So this is probably a step
                fix it up a bit
                """
                #_w[:]=1
                _w[idx]=0.01
                pass
            W.data[0, :-1]=_w
            """
            I cant figure out the flat formulat to do below
            """
            reg = alpha * bm.dot_mm(bm.dot_mm(D.T, W), D)
            dl = reg.data[2][:-1]
            di=  reg.data[1] + Iobs
            du = reg.data[0][1:]
            ret = GTSV(dl, di, du, y, False, False, False, False)
            x = ret[-2]
    else:
        # use the smooth solution
        x = x0
    if type(W)!=None and do_edges==False:
        """
        Weights matrix supplied already...
        use this for edges...
        """
        # make W into a bandmat matrix
        W = bm.diag(W)
        reg = alpha * bm.dot_mm(bm.dot_mm(D.T, W), D)
        dl = reg.data[2][:-1]
        di=  reg.data[1] + Iobs
        du = reg.data[0][1:]
        ret = GTSV(dl, di, du, y, False, False,
                                       False, False)
        x = ret[-2]
    """
    Add the MSE (eg obs unc estimate) to inflat unc...
    """
    C = GTSV(dl, di, du, np.eye(nT), False, False,False, False)[-2]
    unc =  np.sqrt(C_obs[band] * np.diag(C))
    return x, unc, W.data.flatten(), D


@nb.njit
def prepare_iso(doy, iso, Z):
    # Get a optimal 1 ob per day record...
    # not ideal of course...
    uni_doys = np.unique(doy)
    iso_ret = np.zeros((uni_doys.shape[0], 7))
    for idoy, _doy in enumerate(uni_doys):
        idx = np.where(doy==_doy)[0]
        # take the mean? better for SNR
        ii =  iso[idx]
        #_z = Z[idx]
        for band in range(7):
            r = np.nanmean(iso[:, band][idx])
            iso_ret[idoy, band]=r
    return iso_ret



#@nb.jit(nopython=True)
def regularisation(qa, refl, alpha=10, _D=None):
    """
    A simple inmplemenation to see if
    an archetype and edge preserving on
    isotropic is good enough
    """
    # Some constants...
    nT = qa.shape[0]
    #alpha = .1e2
    """
    1. perform an archetype brdf correction for each band
    """
    """
    2. Now perform edge preserving smoothing on this...
    """
    solutions = np.zeros((nT, 2))
    uncs = np.zeros((nT, 2))
    """
    Do edge preserving on both bands
    """
    iso = np.copy(refl)
    iso[~qa]=0
    iso_b2, unc_b2, w_b2, d = solve_banded_fast(iso[:, 1], alpha=alpha, band=1, do_edges=True, W=None, _D=_D)
    iso_b1, unc_b1, w_b1, d = solve_banded_fast(iso[:, 0], alpha=alpha, band=0, do_edges=True, W=None, _D=_D)
    # save them
    solutions[:, 0]=iso_b1
    solutions[:, 1]=iso_b2
    uncs[:, 0]=unc_b1
    uncs[:, 1]=unc_b2
    return solutions, uncs, w_b2


