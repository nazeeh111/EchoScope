import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftn, ifftn

def getPSF(X, Y, Z, S, slope_x, slope_y):
    x = np.linspace(-1, 1, 2*X)
    y = np.linspace(-1, 1, 2*Y)
    z = np.linspace(0,2,2*S*Z)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')
    psf = np.abs(slope_x**2 * grid_x**2 + slope_y**2 * grid_y**2 - grid_z)

    psf = psf == np.tile(np.min(psf, axis=0, keepdims=True), (2*S*Z, 1, 1)) 
    psf = psf.astype(np.float32)
    psf = psf / np.sum(psf)

    psf = np.roll(psf, X, axis=2)
    psf = np.roll(psf, Y, axis=1)
    fpsf = fftn(psf)

    return psf, fpsf

def interpMtx(M, N, start, stop):
    x = np.linspace(start, stop, M)# - start
    x = x ** 2
    val = np.eye(M)
    xq = np.linspace(start**2, stop**2, M*N)# - start**2

    mtx = np.zeros((M*N, M))
    for i in range(mtx.shape[1]):
        mtx[:, i] = np.interp(xq, x, val[:, i])

    mtx = mtx / np.tile(np.sum(mtx, axis=0, keepdims=True), (mtx.shape[0], 1))
    mtx[np.isnan(mtx)] = 0
    mtxi = mtx.T
    mtxi = mtxi / np.tile(np.sum(mtxi, axis=0, keepdims=True), (mtxi.shape[0], 1))
    mtxi[np.isnan(mtxi)] = 0

    return mtx, mtxi

def reconstruct(meas, S, slope_x, slope_y, start, stop, snr):
    """Supported LCT core on [range, y, x] samples, using NumPy only.

    Retains the research pipeline's interpolation, regularization and one-bin
    crop offset. Odd spatial grids receive the extra pad sample on the right.
    This computes a numerical volume; it does not validate a physical scene.
    """
    meas = np.asarray(meas)
    if meas.ndim != 3 or any(n == 0 for n in meas.shape):
        raise ValueError('meas must be a nonempty [range, y, x] volume')
    if meas.dtype.kind not in 'buifc' or not np.isfinite(meas).all():
        raise ValueError('meas must contain finite numeric samples')
    if not isinstance(S, (int, np.integer)) or isinstance(S, (bool, np.bool_)) or S < 1:
        raise ValueError('S must be a positive integer')
    parameters = np.asarray([slope_x, slope_y, start, stop, snr], dtype=float)
    if not np.isfinite(parameters).all() or min(slope_x, slope_y, start) < 0 or stop <= start or snr <= 0:
        raise ValueError('Require finite nonnegative slopes, 0 <= start < stop and snr > 0')
    Z, Y, X = meas.shape
    _, fpsf = getPSF(X, Y, Z, S, slope_x, slope_y)
    mtx, mtxi = interpMtx(Z, S, start, stop)
    invpsf = np.conj(fpsf) / (abs(fpsf)**2 + 1 / snr)
    tdata = np.matmul(mtx, meas.reshape((Z, -1))).reshape((S*Z, Y, X))
    widths = tuple((n//2, n-n//2) for n in tdata.shape)
    fdata = fftn(np.pad(tdata, widths, 'constant'))
    starts = (S*(Z//2)+1, Y//2+1, X//2+1)
    slices = tuple(slice(a, a+n) for a, n in zip(starts, tdata.shape))
    tvol = abs(ifftn(fdata * invpsf)[slices])
    out = np.matmul(mtxi, tvol.reshape((S*Z, -1))).reshape((Z, Y, X))
    if not np.isfinite(out).all():
        raise ValueError('LCT produced nonfinite output; check input scale and parameters')
    return out


def lct(x1, y1, t1, v, vol, snr):
    X = len(x1)
    Y = len(y1)
    Z = len(t1)
    S = 2
    slope = np.max(x1) / (np.max(t1) * v/2)
    slope = np.max(y1) / (np.max(t1) * v/2)
    psf, fpsf = getPSF(X, Y, Z, S, slope)
    mtx, mtxi = interpMtx(Z, S, 0, np.max(t1)*v)

    def pad_array(x, S, Z, X):
        return np.pad(x, ((S*Z//2, S*Z//2), (X//2, X//2), (Y//2, Y//2)), 'constant')

    def trim_array(x, S, Z, X):
        return x[S*(Z//2)+1:-S*(Z//2)+1, X//2+1:-X//2+1, Y//2+1:-Y//2+1]

    invpsf = np.conj(fpsf) / (abs(fpsf)**2 + 1 / snr)
    tdata = np.matmul(mtx, vol)
    fdata = fftn(pad_array(tdata, S, Z, X))
    tvol = abs(trim_array(ifftn(fdata * invpsf), S, Z, X))
    vol = np.matmul(mtxi, tvol)
    return vol
