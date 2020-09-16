# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society

from time import time
import nifty7 as ift
import numpy as np


def myassert(cond):
    if not cond:
        raise RuntimeError


class NFFT(ift.LinearOperator):
    def __init__(self, domain, fourier_sampling_points):
        self._domain = ift.DomainTuple.make(domain)  # signal space
        myassert(len(self._domain.shape) == 1)
        myassert(isinstance(self._domain[0], ift.RGSpace))
        # signal space is not harmonic because the output of CorrelatedField
        # will be an RGSpace which is non-harmonic
        myassert(not self._domain[0].harmonic)
        myassert(isinstance(fourier_sampling_points, np.ndarray))
        myassert(fourier_sampling_points.ndim == 1)
        mi = np.min(fourier_sampling_points)
        ma = np.max(fourier_sampling_points)
        nyquist = 1/2/self._domain[0].distances[0]
        myassert(mi > -nyquist)
        myassert(ma <= nyquist)
        myassert(self._domain.size % 2 == 0)
        tgt = ift.UnstructuredDomain(len(fourier_sampling_points))
        self._ks = fourier_sampling_points
        self._target = ift.DomainTuple.make(tgt)  # data space
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        dom = self._domain[0]
        tgt = self._target[0]

        x = x/tgt.size
        phis = (np.arange(dom.size)-dom.shape[0]//2)*dom.distances[0]

        if mode == self.TIMES:
            res = np.empty(tgt.shape, dtype=np.complex128)
            for ii, k in enumerate(self._ks):
                res[ii] = np.sum(x.val * np.exp(2j * phis * k))
        else:
            res = np.empty(dom.shape, dtype=np.complex128)
            for ii, phi in enumerate(phis):
                res[ii] = np.sum(x.val * np.exp(-2j * phi * self._ks))
        return ift.makeField(self._tgt(mode), res)


def main():
    phi_max = 10000
    npix = 2048
    ndata = 1100
    lamsq = np.exp(ift.random.current_rng().random((ndata)))*1e-3

    dst = 2*phi_max/npix
    s_space = ift.RGSpace(npix, dst)
    R = NFFT(s_space, lamsq)

    fld = ift.full(R.domain, 1.+1j)
    t0 = time()
    nn = 10
    for _ in range(nn):
        R(fld)
    print(f'One response application takes {(time()-t0)/nn:.2f} s')

    # The next line checks < y , R (x) > = < R.adjoint(y), x > for random x, y
    ift.extra.check_linear_operator(R, np.complex128, np.complex128)

    # ift.single_plot(R.adjoint(d))
    # ift.single_plot((R.adjoint @ N.inverse)(d))


if __name__ == '__main__':
    main()
