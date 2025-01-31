import numpy as np
from scipy import interpolate, constants, integrate
from numba import jit


def find_parmax(x, y, coeff=False):

    """Fit a parabola to the maximum of a curve"""
    
    index = np.argmax(y)

    x1 = x[index]
    f1 = y[index]
    
    r0 = f1 - y[index-1]
    r1 = y[index+1] - f1

    p1 = x1 - x[index-1]
    p0 = (x1 + x[index-1]) * p1
    p3 = x[index+1] - x1
    p2 = (x[index+1] + x1) * p3

    det = p0*p3 - p1*p2
    a   = (r0*p3 - r1*p1) / det
    b   = (p0*r1 - p2*r0) / det

    xmax = (-b) / (2.0*a)

    if coeff:
        ymax = f1 - a * (x1 - xmax)**2
        c    = f1 - x1 * (a*x1 + b)
        
        return xmax, ymax, [a, b, c]
    else:
        return xmax
        

def find_parmin(x, y, coeff=False):

    """Fit a parabola to the minimum of a curve"""
    
    index = np.argmin(y)

    x1 = x[index]
    f1 = y[index]
    
    r0 = f1 - y[index-1]
    r1 = y[index+1] - f1

    p1 = x1 - x[index-1]
    p0 = (x1 + x[index-1]) * p1
    p3 = x[index+1] - x1
    p2 = (x[index+1] + x1) * p3

    det = p0*p3 - p1*p2
    a   = (r0*p3 - r1*p1) / det
    b   = (p0*r1 - p2*r0) / det

    xmin = (-b) / (2.0*a)

    if coeff:
        ymin = f1 - a  * (x1 - xmin)**2
        c    = f1 - x1 * (a*x1 + b)
        
        return xmin, ymin, [a, b, c]
    else:
        return xmin

def get_2D_extreme(f):

    fx  = 0.5 * (f[1, 2] - f[1, 0])
    fy  = 0.5 * (f[2, 1] - f[0, 1])
    fxx = f[1, 2] + f[1, 0] - 2.0*f[1, 1]
    fyy = f[2, 1] + f[0, 1] - 2.0*f[1, 1]
    fxy = f[0, 0] + f[2, 2] - (f[0, 2] + f[2, 0])

    det = 1.0 / (fxx*fyy - fxy*fxy)
    x   = (fy*fxy - fx*fyy) * det
    y   = (fx*fxy - fy*fxx) * det

    return x, y


def monotonic_increasing(x):
    dx = np.diff(x)
    return np.all(dx > 0)


def table_invert(table, values, mode=None):
    
    if np.ndim(table) == 0:
        print("Table cannot be a scalar!")
        return table
        
    t_array = np.array(table)
    
    if np.size(t_array) <= 1:
        print("Table cannot have one element!")
        return t_array
    
    if monotonic_increasing(t_array) == False:
        print("Table has to be monotonically increasing!")
        return None
    
    if np.ndim(values) == 0:
        v_array = np.array([values])
    else:  
        v_array = np.array(values)
    
    lookup = []
    for v in v_array:
        if v <= t_array[0]:
            lookup.append({"index": 0, "value": t_array[0], "eff_index": 0})
        elif v >= t_array[-1]:
            lookup.append({"index": len(t_array)-1, \
                           "value": t_array[-1], "eff_index": len(t_array)-1})
        else:
            index = np.argmin(np.abs(t_array - v))

            if v > t_array[index]:
                eff_index = index + \
                    (v - t_array[index]) / (t_array[index+1] - t_array[index])
            else:
                eff_index = index - 1 + \
                (v - t_array[index-1]) / (t_array[index] - t_array[index-1])
            
            lookup.append({"index": index, \
                           "value": t_array[index], "eff_index": eff_index})
    
    if mode == "effective":
        return np.array([dict["eff_index"] for dict in lookup])
    elif mode == "index":
        return np.array([dict["index"] for dict in lookup])
    else:
        return lookup


def LimbDark(wavelength, mu):
    
  ## Calculate limb-darkening coefficient
  ##  (I_wavelength(mu) / I_wavelength(mu=1))
  ## as function of wavelength wavelength and cosine of viewing angle mu

  ## Reference: Allen's Astrophysical Quantities, 4th edition,
  ##             Arthur N. Cox, ed., Springer, p. 355, table 14.17

  ## Input:   wavelength  -- Wavelength [nm]
  ##              mu  -- Cosine of viewing angle (can be an array)

  MICRON_TO_NM = 1.0E3
    
  ## Coefficients from table 14.17, section 14.7

  clv = [{"wavelength": 0.20,  "u2":  0.12,  "v2":  0.33}, \
      
         {"wavelength": 0.22,  "u2": -1.30,  "v2":  1.60}, \
         {"wavelength": 0.245, "u2": -0.1,   "v2":  0.85}, \
         {"wavelength": 0.265, "u2": -0.1,   "v2":  0.90}, \
         {"wavelength": 0.28,  "u2":  0.38,  "v2":  0.57}, \
         {"wavelength": 0.30,  "u2":  0.74,  "v2":  0.20}, \
         {"wavelength": 0.32,  "u2":  0.88,  "v2":  0.03}, \
         {"wavelength": 0.35,  "u2":  0.98,  "v2": -0.10}, \
         {"wavelength": 0.37,  "u2":  1.03,  "v2": -0.16}, \
         {"wavelength": 0.38,  "u2":  0.92,  "v2": -0.05}, \
         {"wavelength": 0.40,  "u2":  0.91,  "v2": -0.05}, \
         {"wavelength": 0.45,  "u2":  0.99,  "v2": -0.17}, \
         {"wavelength": 0.50,  "u2":  0.97,  "v2": -0.22}, \
         {"wavelength": 0.55,  "u2":  0.93,  "v2": -0.23}, \
         {"wavelength": 0.60,  "u2":  0.88,  "v2": -0.23}, \
         {"wavelength": 0.80,  "u2":  0.73,  "v2": -0.22}, \
         {"wavelength": 1.0,   "u2":  0.64,  "v2": -0.20}, \
         {"wavelength": 1.5,   "u2":  0.57,  "v2": -0.21}, \
         {"wavelength": 2.0,   "u2":  0.48,  "v2": -0.18}, \
         {"wavelength": 3.0,   "u2":  0.35,  "v2": -0.12}, \
         {"wavelength": 5.0,   "u2":  0.22,  "v2": -0.07}, \
         {"wavelength": 10.0,  "u2":  0.15,  "v2": -0.07}]

  clv_waves = np.array([dict["wavelength"] for dict in clv]) * MICRON_TO_NM
  clv_u2    = np.array([dict["u2"] for dict in clv])  
  clv_v2    = np.array([dict["v2"] for dict in clv]) 

  fu2 = interpolate.interp1d(clv_waves, clv_u2)                      
  fv2 = interpolate.interp1d(clv_waves, clv_v2)     

  u2 = fu2(wavelength)
  v2 = fv2(wavelength)

  return 1.0 - u2 - v2 + mu * (u2 + mu*v2)


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


class hairline:

    def __init__(self, region):
        
        self.region   = region
        self.position = None


class hairlineset:

    def __init__(self, reference_img):

        self.reference_img = reference_img
        self.hairlines    = self.find()
        

    def find(self):

        HAIR_CONTRAST_TRESHOLD = 0.5
        HAIR_MARGIN            = 5

        (Npix, Nscan) = np.shape(self.reference_img)

        ## Spatial average over the scan direction in the continuum intensity

        avg_cont_slit  = np.mean(self.reference_img, axis=1)
        avg_cont_slit /= np.mean(avg_cont_slit)
        
        hairmask      = np.where(avg_cont_slit < HAIR_CONTRAST_TRESHOLD)
        hair_regions  = consecutive(hairmask[0])
        
        hairlines = []
        for region in hair_regions:
            hl = hairline([region[0] - HAIR_MARGIN, \
                            region[-1] + HAIR_MARGIN])

            hl.position = \
                find_parmin(np.arange(hl.region[0], hl.region[1], 1),\
                            avg_cont_slit[hl.region[0]:hl.region[1]])

            hairlines.append(hl)

        return hairlines


    def remove(self, spectrum):

        (Npix, Nscan, Nwave, Nstokes) = np.shape(spectrum)

        for hl in self.hairlines:
            for n in range(Nstokes):
                for m in range(Nwave):
                    for i in range(Nscan):
                        j0, j1 = hl.region
                        for j in range(j0+1, j1-1, 1):
                            frac = (j1 - j) / (j1 - j0)
                            spectrum[j, i, m, n] = \
                                (1.0 - frac) * spectrum[j0, i, m, n] + \
                                frac * spectrum[j1, i, m, n]
                    

def psf_broad(wavelength, spectrum, FWHM, mode="Gaussian"):

  dwave_min = np.min(np.diff(wavelength))
  Nwave     = (wavelength[-1] - wavelength[0]) / dwave_min

  Nequid = 1
  while Nequid < Nwave:
      Nequid *= 2

  wave_equid = np.linspace(wavelength[0], wavelength[-1], \
                           num=Nequid, endpoint=True, dtype=float)
  lambda0    = (wave_equid[0] + wave_equid[-1]) / 2.0
    
  f1 = interpolate.interp1d(wavelength, spectrum)
  spec_equid = f1(wave_equid)

  if mode == "Gaussian":
      sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

      delta_wave = (wave_equid - lambda0) / (np.sqrt(2.0) * sigma)
      kernel = [np.exp(-delta_wave[n]**2) if \
                np.abs(delta_wave[n]) < 7.0 else 0.0 for n in range(Nequid)]
  elif mode == "Lorentzian":
      gamma = FWHM / 2.0
      
      delta_wave = (wave_equid - lambda0) / gamma
      kernel = 1.0 / (1.0 + delta_wave**2)

  conv = np.reshape(np.abs(np.fft.ifft(np.fft.fft(spec_equid) * \
                                       np.fft.fft(kernel))), Nequid)
  print(np.shape(kernel), np.shape(conv))
  conv_roll = np.roll(conv, Nequid//2)

  f2 = interpolate.interp1d(wave_equid, conv_roll)
  conv_spec = f2(wavelength) / np.sum(kernel)

  return conv_spec


def ViSP_arm_makeup(axs, title, atlas, lines, telluric):
    
    normatlas = atlas[1] / atlas[2]
    axs.plot(atlas[0], normatlas, label='atlas')
    
    axs.set(title=title, xlabel='wavelength [nm]', ylabel='relative intensity')
    trans = mtransforms.blended_transform_factory(axs.transData, axs.transAxes)

    for key,value in lines.items():
        axs.axvline(key, color='c', linestyle='--')
        axs.text(key, 1.0, value, fontfamily='serif', fontsize=8, color='c')
    
    for pair in telluric:
        axs.fill_between (pair, 0, 1, facecolor='m', alpha=0.2, transform=trans)
        for wave in pair:
            axs.axvline(wave, color='m', linestyle=':', linewidth=0.8)

    axs.legend()



class DeSIRe_line:
    
    def __init__(self, lineID, element, ion, lambda0, \
                 Ei, loggf, mi, oi, Ji, mj, oj, Jj):

        orbits = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}
        
        self.ID      = lineID
        self.element = element
        self.ion     = ion
        self.lambda0 = lambda0
        self.Ei      = Ei
        self.loggf   = loggf
        self.Si      = (mi - 1) / 2
        self.Li      = orbits[oi]
        self.Ji      = Ji
        self.Sj      = (mj - 1) / 2 
        self.Lj      = orbits[oj]
        self.Ji      = Jj

        self.indices = None

    @classmethod
    def get_list(cls):

        list = []
        list.append(DeSIRe_line(1,  'CA', 2, 396.8469,   0.0000, \
                                -0.16596,  2, 'S', 0.5, 2, 'P', 0.5))
        list.append(DeSIRe_line(2,  'CA', 2, 399.3663,   0.0000, \
                                -0.13399,  2, 'S', 0.5, 2, 'P', 1.5))
        list.append(DeSIRe_line(3,  'CA', 2, 849.8023,   1.6924, \
                                -1.31194,  2, 'D', 1.5, 2, 'P', 1.5))
        list.append(DeSIRe_line(4,  'CA', 2, 854.2091,   1.7000, \
                                -0.36199,  2, 'D', 2.5, 2, 'P', 1.5))
        list.append(DeSIRe_line(5,  'CA', 2, 866.2141,   1.6924, \
                                -0.62299,  2, 'D', 1.5, 2, 'P', 0.5))
        list.append(DeSIRe_line(6,  'MG', 1, 518.36042,  2.7166, \
                                -0.164309, 3, 'P', 2.0, 3, 'S', 1.0))
        list.append(DeSIRe_line(7,  'MG', 1, 517.26843,  2.7166, \
                                -0.38616,  3, 'P', 1.0, 3, 'S', 1.0))
        list.append(DeSIRe_line(8,  'MG', 1, 516.73219,  2.7166, \
                                -0.863279, 3, 'P', 0.0, 3, 'S', 1.0))
        list.append(DeSIRe_line(9,  'NA', 1, 588.995095, 0.0000, \
                                 0.10106,  2, 'S', 0.5, 2, 'P', 1.5))
        list.append(DeSIRe_line(10, 'NA', 1, 589.592424, 0.0000, \
                                -0.18402,  2, 'S', 0.5, 2, 'P', 0.5))
        list.append(DeSIRe_line(23, 'FE', 1, 630.15012, 3.654, \
                                -0.718,    5, 'P', 2.0, 5, 'D', 2.0))
        list.append(DeSIRe_line(24, 'FE', 1, 630.24936, 3.686, \
                                -1.131,    5, 'P', 1.0, 5, 'D', 0.0))
        return list


def match_spectra_pixel(spectrum1, spectrum2, show_corr=False):

    ## Returns the shift in pixels between two spectra, 
    ## by finding the maximum of the cross correlation between them.
    
    length1, = spectrum1.shape
    length2, = spectrum2.shape

    ## Pad the shortest array to the length of the longest
    
    if length2 > length1:
        spectrum1 = np.pad(spectrum1, (0, length2-length1), mode="edge")
    elif length1 > length2:
        spectrum2 = np.pad(spectrum2, (0, length1-length2), mode="edge")

    length = max([length1, length2])

    ## Calculate the cross correlations with FFTs
    
    xcorr = np.abs(np.fft.ifft(np.fft.fft(spectrum2) * \
                               np.conj(np.fft.fft(spectrum1))))
    
    ## Find the maximum with parabolic interpolation, taking care of 
    ## the possibility the maximum might be on the edge of the domain,
    ## using the periodic nature of the convolution.
    
    xval  = np.argmax(xcorr) + [-2, -1, 0, 1, 2]
    shift = find_parmax(xval, xcorr[(xval + length) % length])
        
    if show_corr:
        plt.figure(figsize=(14, 7))
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs.plot(xcorr)
        axs.axvline(shift, linestyle=':', linewidth=1.0)
        plt.show()
        
    if shift > length/2.0:
        shift -= length

    return shift

def vacuum_to_air(lambda_vac):

    #-# Source:
    #-#    [1] http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    #-# Original reference:
    #-#    Donald Morton (2000, ApJ. Suppl., 130, 403)
    
    
    NM_TO_ANGSTROM = 10.0
    
    s2 = (1.0E4 / (NM_TO_ANGSTROM * lambda_vac))**2
    n  = 1.0 + 8.34254E-5 + 2.406147E-2 / (130.0 - s2) + 1.5998E-4 / (38.9 - s2)

    lambda_air = lambda_vac / n
    return lambda_air

def air_to_vacuum(lambda_air):

    #-# Source:
    #-#    [1] http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    #-# Original reference:
    #-#    Donald Morton (2000, ApJ. Suppl., 130, 403)
    
    NM_TO_ANGSTROM = 10.0
    
    s2 = (1.0E4 / (NM_TO_ANGSTROM * lambda_air))**2
    n  = 1.0 + 8.336624212083E-5 + 2.408926869968E-2 / (130.1065924522 - s2) + \
        1.599740894897E-4 / (38.92568793293 - s2)
    
    lambda_vac = lambda_air * n
    return lambda_vac


##  --- Routines for interpolation by cubic convolution.
##
##      Author:        Han Uitenbroek  (huitenbroek@nso.edu)
##       Last modified: Fri Jan 31 09:19:29 2025 --
##
##  See: R.G. Keys, 1981, in IEEE Trans. Acoustics, Speech,
##        and Signal Processing, Vol. 29, pp. 1153-1160.
##       --                                              -------------- */


@jit
def cc_kernel(s, u):
 
    s2   = s * s
    s3   = s2 * s
    u[0] = -0.5*(s3 + s) + s2
    u[1] =  1.5*s3 - 2.5*s2 + 1.0
    u[2] = -1.5*s3 + 2.0*s2 + 0.5*s
    u[3] =  0.5*(s3 - s2)

@jit
def cubeconvol(image, x, y):

    (Nx, Ny) = np.shape(image)
    
    Ncc   = 4
    dtype = np.float64
    c     = np.zeros((Ncc, Ncc), dtype=dtype)

    ux = np.zeros(Ncc, dtype=dtype)
    uy = np.zeros(Ncc, dtype=dtype)
    
    g  = 0.0;

    
    if x <= 0.0:
        i = 0
        cc_kernel(0.0, ux)
    elif x >= Nx-1:
        i = Nx - 2
        cc_kernel(1.0, ux)
    else:
        ir, xp = divmod(x, 1)
        i = int(ir)
        cc_kernel(xp, ux)

    if y <= 0.0:
        j = 0
        cc_kernel(0.0, uy)
    elif y >= Ny-1:
        j = Ny - 2
        cc_kernel(1.0, uy)
    else:
        jr, yp = divmod(y, 1)
        j = int(jr)
        cc_kernel(yp, uy)


    if j == 0:
        if i == 0:
            c[1:4, 1:4] = image[0:3, 0:3]
            c[0, 1:4]   = 3.0 * (c[1, 1:4] - c[2, 1:4]) + c[3, 1:4]

        elif i == Nx-2:
            c[0:3, 1:4] = image[Nx-3:Nx, 0:3]
            c[3, 1:4]   = 3.0 * (c[2, 1:4] - c[1, 1:4]) + c[0, 1:4]

        else:
            c[:, 1:4] = image[i-1:i+3, 0:3]

        c[:, 0] = 3.0 * (c[:,1]  - c[:, 2]) + c[:, 3]

      
    elif j == Ny-2:
        if i == 0:
            c[1:4, 0:3] = image[0:3, Ny-3:Ny]
            c[0, 0:3]   = 3.0 * (c[1, 0:3] - c[2, 0:3]) + c[3, 0:3]

        elif i == Nx-2:
            c[0:3, 0:3] = image[Nx-3:Nx, Ny-3:Ny]
            c[3, 0:3]   = 3.0 * (c[2, 0:3] - c[1, 0:3]) + c[0, 0:3]

        else:
            c[:, 0:3] = image[i-1:i+3, Ny-3:Ny]

        c[:, 3] = 3.0 * (c[:, 2] - c[:, 1]) + c[:, 0]
      

    else:
        if i == 0:
            c[1:4, :] = image[0:3, j-1:j+3]
            c[0, :]   = 3.0 * (c[1, :] - c[2, :]) + c[3, :]

        elif i == Nx-2:
            c[0:3, :] = image[Nx-3:Nx, j-1:j+3]
            c[3, :]   = 3.0 * (c[2, :] - c[1, :]) + c[0, :]

        else:
            for m in range(Ncc):
                for n in range(Ncc):
                    g +=  image[i-1+n, j-1+m] * ux[n] * uy[m]

            return g


    for m in range(Ncc):
        for n in range(Ncc):
            g += c[n, m] * ux[n] * uy[m]

    return g


def ViSP_scaleimage(oldimage, Nx_new, Ny_new):

    (Nx_old, Ny_old) = np.shape(oldimage)
    newimage = np.zeros((Nx_new, Ny_new), dtype=np.float64)

    dx = (Nx_old - 1) / (Nx_new - 1)
    dy = (Ny_old - 1) / (Ny_new - 1)

    for m in range(Ny_new):
        y = m * dy
        for n in range(Nx_new):
            x = n * dx
            newimage[n, m] = cubeconvol(oldimage, x, y)

    return newimage


def ViSP_remap_image(oldimage, x_ind, y_ind):

    Nx_new,  = x_ind.shape
    Ny_new,  = y_ind.shape
    newimage = np.zeros((Nx_new, Ny_new), dtype=np.float32)

    for m in range(Ny_new):
        for n in range(Nx_new):
            newimage[n, m] = cubeconvol(oldimage, x_ind[n], y_ind[m])

    return newimage


def FilterCurve(lambda0, waves, FWHM=1.0, cavity=2.0):

    filter_curve = 0.0
    if cavity > 0.0:
        sigma = 0.5 * FWHM
        filter_curve = 1.0 / (1.0 + (np.abs(waves - lambda0) /
                                     sigma) ** (2.0 * cavity))
    elif cavity == 0.0:
        sigma = FWHM / (2.0 * np.sqrt(np.log(2.0)))
        filter_curve = np.exp(-(((wave - lambda0) / sigma) ** 2))
    else:
        return filter_curve
    
    return filter_curve / integrate.trapezoid(filter_curve, waves)

