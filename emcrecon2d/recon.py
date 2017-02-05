# julien lhermitte 2017
"""
This is the module for putting image reconstruction tools.
These should be interesting compositions of existing
tools, not just straight wrapping of np/scipy/scikit images.
"""
import numpy as np
from skbeam.core.accumulators.binned_statistic import RPhiBinnedStatistic


def removenans(data):
    w = np.where(np.isnan(data))
    if len(w[0]) > 0:
        data[w] = 0


class EMCRecon2D:
    def __init__(self, shape, mask=None, origin=None, bins=(100, 100),
                 bgimg=None, Niter=100, Nrot=100):
        ''' Initialize the reconstruction using the EMC algorithm.

            Parameters
            ----------

            shape : the shape of the images

            mask : 2d array
                array of 0's (masked) and 1's (not masked)

            origin : (float, float), optional
                the beam center
                defaults to the center of the images

            bins : (int, int)
                the number of q and phi to bin

            bgimg : optional, the background image to subtract from
                set to zero by default

            Niter : int, optional
                the number of iterations, defaults to 100

            Nrot : int, optional
                the number of rotations to sample, defaults to 100
        '''
        self.shape = shape
        self.bins = bins
        self.Niter = Niter
        self.Nrot = Nrot

        if mask is None:
            self.mask = np.ones(self.shape)
        else:
            self.mask = mask

        if origin is None:
            self.origin = np.array(self.mask.shape)//2
        else:
            self.origin = origin

        # should be replaced with a better way later
        if bgimg is None:
            self.bgimg = np.zeros_like(self.mask)
        else:
            self.bgimg = bgimg
            removenans(self.rphibg)

        # first transform into a qphi map
        self.rphibinstat = RPhiBinnedStatistic(self.shape, mask=self.mask,
                                               bins=self.bins,
                                               origin=self.origin)

        self.rvals = self.rphibinstat.bin_centers[0]
        self.phivals = self.rphibinstat.bin_centers[1]

        self.rphibg = self.rphibinstat(self.bgimg)

        # set up mask
        self.rphimask = self.rphibinstat(self.mask)
        # rphibinstat puts nans where there is nothing, remove that
        removenans(self.rphimask)

        self.noqs = self.bins[0]
        self.nophis = self.bins[1]

        # partitioning
        self.qind = np.arange(self.noqs)[:, np.newaxis]
        self.qind = np.tile(self.qind, (1, self.nophis))
        self.pind = np.arange(self.nophis)[np.newaxis, :]
        self.pind = np.tile(self.pind, (self.noqs, 1))

        # select where data valid
        self.wsel = np.where(self.rphimask.ravel() > .1)
        # number of pixels
        self.Npix = len(self.wsel[0])

        self.qind_data = self.qind.ravel()[self.wsel]
        self.pind_data = self.pind.ravel()[self.wsel]

    def __call__(self, imgs, bgimg=None, return_rotations=False):
        ''' The EMC reconstruction algorithm for 2D samples, no distortions assumed
                along angles.
            The samples are assumed to be rotated only in plane.

            This one will assume images of same dimensions as mask.

            The model will be the full image, binned by bins.
            The data will be the non masked pixels

            Make the mask bigger to exclude pixels.

            Parameters
            ----------

            imgs : series of images to analyze

            bgimg : supply a background image. The images are subtracted by
            this background and the probability distribution compensates for
            this.

            return_rotations : bool, optional
                if True, return the rotation probability matrix.
                Default is False

        '''
        if bgimg is not None:
            # reset bgimg
            self.bgimg = bgimg
            self.rphibg = self.rphibinstat(self.bgimg)
            removenans(self.rphibg)

        imgs = np.array(imgs)
        if imgs.ndim < 3:
            imgs = imgs[np.newaxis, :, :]

        # number of measurements
        Nm = len(imgs)

        # initialize the data from the images
        self.rphis = np.zeros((Nm, self.noqs, self.nophis))

        for i in range(Nm):
            self.rphis[i] = self.rphibinstat(imgs[i]-self.bgimg)

        removenans(self.rphis)

        # this is K_ik
        K_ik = np.zeros((self.Npix, Nm))
        for k in range(Nm):
            K_ik[:, k] = self.rphis[k, self.qind_data, self.pind_data]

        # estimate average count rate
        mu = np.sum(K_ik)/Nm/self.Npix

        # Initialization step
        # initialize random model
        W_model = np.random.random((self.noqs, self.nophis))*mu
        W_model_bg = self.rphibg
        # W_model[:, 50:60] = mu*1000
        W_model_cnt = np.ones_like(W_model)

        for l in range(self.Niter):
            print("EMC: Step {} of {}".format(l, self.Niter))
            # ----------- Expansion step --------------
            W_ij = np.zeros((self.Npix, self.Nrot))
            # not implemented yet
            # W_ij_bg = np.zeros((self.Npix, self.Nrot))
            for j in range(self.Nrot):
                W_ij[:, j] = np.roll(W_model, j, axis=1)[self.qind_data,
                                                         self.pind_data]
                # don't rotate bg
                W_ij[:, j] += W_model_bg[self.qind_data, self.pind_data]

            # ------- maximize ------------
            # make sure to minimize over flow, subtract some constant factor in
            # norm
            ''' basically
                e^{V}/sum(e^{v})
                do
                V = V' + C
                e^{V'}/sum(e^{V'})
                choose C to be maximum of Vs. Basically, the highest val should
                factor in more than the lowest vals. We shift from high values
                yielding infinity to low values yielding zero...
            '''
            R_jk = np.sum(np.log(W_ij[:, :, np.newaxis]) *
                          K_ik[:, np.newaxis, :] - W_ij[:, :, np.newaxis],
                          axis=0)
            # if l == 0:
            #    raise ValueError
            consts = np.max(R_jk, axis=0)[np.newaxis, :]
            R_jk -= consts
            R_jk = np.exp(R_jk)
            # raise ValueError
            R_jk /= np.sum(R_jk, axis=0)[np.newaxis, :]
            removenans(R_jk)

            W_ijp = np.sum(R_jk[np.newaxis, :, :]*K_ik[:, np.newaxis, :],
                           axis=2)/np.sum(R_jk, axis=1)[np.newaxis, :]
            removenans(W_ijp)
            # --------- compress ----------

            # compress
            W_model *= 0
            W_model_cnt *= 0
            for j in range(self.Nrot):
                W_model[self.qind_data, (self.pind_data-j) % self.nophis] +=\
                        W_ijp[:, j] - W_model_bg[self.qind_data,
                                                 self.pind_data]
                W_model_cnt[self.qind_data, (self.pind_data-j) %
                            self.nophis] += 1

            w = np.where(W_model_cnt != 0)
            W_model[w] /= W_model_cnt[w]

            W_model_mask = W_model_cnt >= 1

            # debugging
            # if l == 0:
            # raise ValueError

            # save final results in object (overwriting previous)
            self.W_model = W_model
            self.W_model_mask = W_model_mask
            self.R_jk = R_jk

        if return_rotations:
            return self.rvals, self.phivals, self.W_model, \
                   self.W_model_mask, R_jk
        else:
            return self.rvals, self.phivals, self.W_model, self.W_model_mask
