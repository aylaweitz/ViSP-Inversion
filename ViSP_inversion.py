import glob
import os
import numpy as np
import pandas as pd
from numba import jit
import multiprocessing as mp
from astropy.io import fits
from astropy.coordinates import SkyCoord, get_sun
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as units
from scipy.ndimage import shift
from scipy import constants, interpolate
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import dkist
import ViSP_tools as vt


class ViSP_arm:

    def __init__(self, dataset_path):

        if os.path.isdir(dataset_path):
            self.dataset_path = dataset_path
        else:
            print("Data directory {} does not exist. Exiting".format(dataset_path))
            exit()

        home_dir = os.path.expanduser("~")
        self.aux_data_dir = os.path.join(home_dir, "Source/Python/DKIST/ViSP_invert/Aux_data")

        self.ViSP_analyze_asdf()


    def ViSP_analyze_asdf(self):

        self.asdf_file   = glob.glob(os.path.join(self.dataset_path, '*.asdf'))[0]
        self.dataset     = dkist.load_dataset(self.asdf_file)
        self.transp_data = np.transpose(self.dataset.data, axes=(0, 2, 1, 3))

        self.datasetID  = self.dataset.headers[0]["DSETID"]
        self.spectrumID = self.dataset.headers[0]["WAVEBAND"]
        self.armID      = self.dataset.headers[0]["VSPARMID"]
        self.slit_width = self.dataset.headers[0]["VSPWID"]
        self.slit_step  = self.dataset.headers[0]["CDELT3"]
        self.wcs_names  = self.dataset.wcs.pixel_axis_names

        match self.wcs_names.index('dispersion axis'):
            case 0:

                ### Initial Level-1 case, OCP1, in error
                
                dlambda  = self.dataset.headers[0]["CDELT1"]
                lambda_0 = self.dataset.headers[0]["CRVAL1"]


                ## Use estimate from ViSP/NSO webpage:
                ##  (https://nso.edu/telescopes/dkist/instruments/visp/)
                
                match self.armID:
                    case 1:
                        self.slit_sample = 0.02958
                    case 2:
                        self.slit_sample = 0.02393
                    case 3:
                        self.slit_sample = 0.01937

            case 1:
                dlambda  = self.dataset.headers[0]["CDELT2"]
                lambda_0 = self.dataset.headers[0]["CRVAL2"]
                
                self.slit_sample = self.dataset.headers[0]["CDELT1"]
            case _:
                pass

        self.Nlambda     = self.dataset.headers[0]["DNAXIS2"]
        DeSIRe_line_list = vt.DeSIRe_line.get_list()

        match self.spectrumID:
             case "Fe I (630.25 nm)":
                 self.DeSIRe_line = DeSIRe_line_list[10]
                 self.blends      = [DeSIRe_line_list[11]]
                 clv_file = "FeI_6302_clv.fits"

             case "Na I D1 (589.59 nm)":
                 self.DeSIRe_line = DeSIRe_line_list[8]
                 self.blends      = [DeSIRe_line_list[9], DeSIRe_line_list[17], DeSIRe_line_list[18]]
                 clv_file = "NaI_5896_clv.fits"
                 
             case "Ca II (854.21 nm)":
                 self.DeSIRe_line = DeSIRe_line_list[3]
                 self.blends      = DeSIRe_line_list[12:17]
                 clv_file = "CaII_8542_clv.fits"

        clv_file_path = os.path.join(self.aux_data_dir, clv_file)
        self.clv_interp = self.ViSP_read_clv(clv_file_path)


    def ViSP_find_wavelength_solution(self):

        ATL_RANGE     = 2.0
        DELTA_LAM     = 0.01
        POL_THRESHOLD = 0.002
        
        stokes_I = self.transp_data[0, :, :, :].compute()
        Nwave    = stokes_I.shape[0]

        self.avg_spectrum    = np.mean(stokes_I, axis=(1, 2))
        self.continuum_index = np.argmax(self.avg_spectrum)
        norm_spectrum        = self.avg_spectrum / np.max(self.avg_spectrum)
        ref_index            = np.argmin(self.avg_spectrum)
        ref_lambda           = self.DeSIRe_line.lambda0

        fts = vt.satlas_ds(self.aux_data_dir)
        self.lambda_atlas, intensity, continuum = fts.nmsiatlas(ref_lambda - ATL_RANGE,\
                                                                ref_lambda + ATL_RANGE)
        self.norm_atlas = intensity / continuum   

        match self.spectrumID:
             case "Fe I (630.25 nm)":
                 dispersion = 1.281E-03

                 lines    = {629.77927: 'Fe I', 629.9582: 'Zr I', \
                             630.068: 'Hf I', 630.15012: 'Fe I', \
                             630.24936: 'Fe I', 630.3755: 'Ti I', \
                             630.4325: 'Zr I', 630.5275: 'Fe II'}
                 
                 self.telluric = [[629.833, 629.854], [629.913, 629.930], \
                                  [630.189, 630.209], [630.266, 630.286], \
                                  [630.568, 630.588], [630.643, 630.666]]

                 self.lambda_blu = 630.085
                 self.lambda_red = 630.330
                 self.continuum_interval = [630.300, self.lambda_red]

             case "Na I D1 (589.59 nm)":
                 dispersion   = 1.4E-3

                 lines    = {588.9973: 'Na I', 589.1175: 'Fe I', \
                             589.2883: 'Ni I', 589.5940: 'Na I'}
                 
                 self.telluric = [[589.157, 589.176], [589.231, 589.250]]

                 self.lambda_blu = 588.653
                 self.lambda_red = 589.886
                 self.continuum_interval = [589.360, 589.420]
                 
             case "Ca II (854.21 nm)":
                 dispersion   = 1.873E-03

                 lines    = {853.6163: 'Si I', 853.80147: 'Fe I', 854.21: 'Ca II',\
                             854.8079: 'Ti I'}
                 
                 self.telluric = [[853.462, 853.493], [854.068, 854.092], [854.605, 854.632]]

                 self.lambda_blu = 853.232
                 self.lambda_red = 855.193
                 self.continuum_interval = [self.lambda_blu, 853.300]


        Nlines          = len(lines)
        wave_positions  = np.zeros(Nlines, dtype=np.float64)
        index_positions = np.zeros(Nlines, dtype=np.float64)
        waves_guess     = ref_lambda + dispersion * (np.arange(Nwave, dtype=np.float64) - ref_index)


        n = 0
        for key, label in lines.items():
            values  = np.array([key - DELTA_LAM, key + DELTA_LAM])
            indices = vt.table_invert(self.lambda_atlas, values, mode="index")

            wave_positions[n] = vt.find_parmin(self.lambda_atlas[indices[0]:indices[1]],
                                               self.norm_atlas[indices[0]:indices[1]])
            n += 1

        n = 0
        for key, label in lines.items():
            values  = np.array([key - DELTA_LAM, key + DELTA_LAM])
            indices = vt.table_invert(waves_guess, values, mode="index")

            wave_min = vt.find_parmin(waves_guess[indices[0]:indices[1]],
                                      norm_spectrum[indices[0]:indices[1]])
            index_positions[n] = vt.table_invert(waves_guess, wave_min, mode="effective")[0]

            n += 1
     
        coefficients = np.polyfit(index_positions, wave_positions, 2)
        poly         = np.poly1d(coefficients)
        
        #-# Store the calibrated wavelengths for the current arm
        
        self.calib_waves = poly(np.arange(Nwave))


    def ViSP_broadened_atlas(self, waves, FWHM, wave_offset):

        f = interpolate.interp1d(self.lambda_atlas + wave_offset, self.norm_atlas)
        atlas_int = f(waves)

        Nwave = len(atlas_int)
        mu_wave_grid = np.zeros((Nwave, 2), dtype=np.float32)
        mu_wave_grid[:, 0] = self.mu
        mu_wave_grid[:, 1] = waves
        atlas_int *= (vt.LimbDark(self.DeSIRe_line.lambda0, self.mu) * self.clv_interp(mu_wave_grid))
        
        return vt.psf_broad(waves, atlas_int, FWHM, mode="Gaussian")


    def ViSP_find_PSF(self, pol_map):

        POL_THRESHOLD = 0.005

        quiet = np.where(pol_map < POL_THRESHOLD)
        avg_quiet_spectrum = np.mean(self.spectrum[0, :, quiet[0], quiet[1]], axis=0)
        
        for limits in self.telluric:
            j0, j1 = vt.table_invert(self.lambda_atlas, limits, mode="index")

            Nwave_atlas = len(self.lambda_atlas)
            stokes_I = np.reshape(self.norm_atlas, shape=(Nwave_atlas, 1, 1))

            vt.remove_telluric_pix(stokes_I, j0, j1)
        
        popt, pcov = curve_fit(self.ViSP_broadened_atlas, self.calib_waves, \
                               avg_quiet_spectrum, p0=[0.004, 0.0]) 

        self.FWHM = popt[0]
        self.calib_waves -= popt[1]

        
    def ViSP_read_arm_data(self):

        limits = vt.table_invert(self.calib_waves, np.array([self.lambda_blu, self.lambda_red]), \
                                 mode="index")
        
        self.spectrum = self.transp_data[:, limits[0]:limits[1], 100:175, :].compute()

        self.calib_waves     = self.calib_waves[limits[0]:limits[1]]
        self.avg_spectrum    = self.avg_spectrum[limits[0]:limits[1]]
        self.continuum_index = np.argmax(self.avg_spectrum)
 

    def ViSP_remap_data(self, fiducial_arm, dummy):

        x1 = fiducial_arm.hairlineset.hairlines[0].position
        x2 = fiducial_arm.hairlineset.hairlines[1].position
        Nscan_fid, Npix_fid = fiducial_arm.spectrum.shape[2:4]

        y1 = self.hairlineset.hairlines[0].position
        y2 = self.hairlineset.hairlines[1].position

        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        x_fid = np.arange(Nscan_fid, dtype=np.float64)
        y_fid = slope * np.arange(Npix_fid, dtype=np.float64) + intercept
        reference_img = self.spectrum[0, self.continuum_index, :, :].copy()
        arm_ref_remap = vt.ViSP_remap_image(reference_img, x_fid, y_fid)

        fid_image = fiducial_arm.spectrum[0, fiducial_arm.continuum_index, :, :].copy()
        fid_image = fid_image / np.mean(fid_image) - 1.0
        arm_image = arm_ref_remap / np.mean(arm_ref_remap) - 1.0

        cross_corr = np.abs(np.fft.ifft2(np.fft.fft2(arm_image) * \
                                         np.conj(np.fft.fft2(fid_image))))
        icx, icy = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        max_area = np.zeros((3, 3), dtype=np.float32)

        for k in range(3):
            jcy = (icy - 1 + k + Npix_fid) % Npix_fid
            for l in range(3):
                jcx = (icx -1 + l + Nscan_fid) % Nscan_fid
                max_area[k, l] = cross_corr[jcx, jcy]

        dx, dy   = vt.get_2D_extreme(max_area)
        
        x_offset = icx + dx
        if x_offset > (Nscan_fid / 2): y_offset -= Nscan_fid
        y_offset = icy + dy
        if y_offset > (Npix_fid / 2): y_offset -= Npix_fid

        print("Arm {0:1d}, line ID {1} has offsets ({2:1.5f}, {3:1.5f})".format(self.armID, \
                                                                                self.spectrumID, \
                                                                                x_offset, y_offset))

        Nstokes, Nwave = self.spectrum.shape[0:2]
        spectrum_remap = np.zeros((Nstokes, Nwave, Nscan_fid, Npix_fid), dtype=np.float32)

        for n in range(Nstokes):
            for m in range(Nwave):
                spectrum_remap[n, m, :, :] = vt.ViSP_remap_image(self.spectrum[n, m, :, :], \
                                                                 x_fid + x_offset, \
                                                                 y_fid + y_offset)

        return spectrum_remap

    
    def ViSP_read_clv(self, clv_file_path):

        hdul = fits.open(clv_file_path)

        clv   = hdul[0].data
        xmu   = hdul[1].data
        waves = hdul[2].data
        hdul.close()

        clv_interp = interpolate.RegularGridInterpolator((xmu, waves), clv,\
                                                         bounds_error=False, fill_value=None)
        
        return clv_interp

        
    def ViSP_calibrate_intensity(self, pol_map):

        POL_THRESHOLD = 0.005
        
        quiet = np.where(pol_map < POL_THRESHOLD)

        avg_quiet_spectrum = np.mean(self.spectrum[0, :, quiet[0], quiet[1]], axis=0)
        continuum_index    = np.argmax(avg_quiet_spectrum)
        quiet_continuum    = avg_quiet_spectrum[continuum_index]
        lambda_continuum   = self.calib_waves[continuum_index]

        f = interpolate.interp1d(self.lambda_atlas, self.norm_atlas)
        atlas_continuum = f(lambda_continuum)
        
        limbdark   = vt.LimbDark(self.DeSIRe_line.lambda0, self.mu)
        clv_factor = self.clv_interp([self.mu, lambda_continuum])[0]
        
        normalization  = (limbdark * clv_factor) * (atlas_continuum / quiet_continuum)
        self.spectrum *= normalization


    def ViSP_write_data_fits(self, fits_directory):

        filename  = "ViSP_" + self.datasetID + "_" + self.spectrumID + ".fits"
        file_path = os.path.join(fits_directory, filename)

        hdu  = fits.PrimaryHDU(self.spectrum)
        hduw = fits.ImageHDU(self.calib_waves)
        hdul = fits.HDUList([hdu, hduw])
        hdul.writeto(file_path, overwrite=True)


    def ViSP_get_rebin_params(self, fiducial_arm, slit_width):
        
        self.Nscan_fid, self.Npix_fid = fiducial_arm.spectrum.shape[2:4]
        
        self.Npix_avg   = int(np.floor(slit_width / fiducial_arm.slit_sample))
        self.Npix_rebin = self.Npix_fid // self.Npix_avg
        self.last_pixel = self.Npix_avg * self.Npix_rebin
        
        print("Npix_avg: {0}, Npix_rebin: {1}".format(self.Npix_avg, self.Npix_rebin))

        
    def ViSP_rebin(self):

        Nstokes, Nwave = self.spectrum.shape[0:2]
        save_spectrum  = np.zeros((Nstokes, Nwave, self.Nscan_fid, self.Npix_rebin), \
                                  dtype=np.float32)
            
        save_spectrum[:, :, :, :] = np.mean(np.reshape(self.spectrum[:, :, :, 0:self.last_pixel], \
                                                       (Nstokes, Nwave, self.Nscan_fid, \
                                                        self.Npix_rebin, self.Npix_avg)), axis=4)

        del self.spectrum
        self.spectrum = save_spectrum


    def ViSP_remove_crosstalk(self, mode="Sanchez_Kuhn"):

        match mode:
            
            case "Sanchez_Kuhn":
                
                c0, c1 = vt.table_invert(self.calib_waves, self.continuum_interval, mode="index")
        
                DELTA_CORE = 5
                DELTA_WING = 20
                P_TRESHOLD = 0.05
        
                index0   = vt.table_invert(self.calib_waves, self.DeSIRe_line.lambda0, mode="index")[0]
                l0c, l1c = index0 - DELTA_CORE, index0 + DELTA_CORE
                l0w, l1w = index0 - DELTA_WING, index0 + DELTA_WING
        
                polmap = np.max(np.sqrt(np.sum(self.spectrum[1:, l0w:l1w, :, :]**2, axis=0)) / \
                                self.spectrum[0, l0w:l1w, :, :], axis=0)

                pstrong = np.argwhere(polmap >= P_TRESHOLD)
                py = pstrong[:, 0]
                px = pstrong[:, 1]
        
                (Nstokes, Nwave, Nscan, Npix) = self.spectrum.shape
                spectrum_shft = np.zeros((Nstokes, Nwave, Nscan, Npix), dtype=np.float32)

                Mij    = np.zeros(Nstokes, dtype=np.float64)
                Mij[0] = 1.0
                for i in range(1, Nstokes):
                    Mij[i] = np.mean(self.spectrum[i, c0:c1, :, :] / self.spectrum[0, c0:c1, :, :])
                    spectrum_shft[i, :, :, :] = self.spectrum[i, :, :, :] - Mij[i] * self.spectrum[0, :, :, :]
            

                lindex = np.arange(Nwave).reshape([Nwave, 1, 1])
                    
                lcen = np.sum(np.sqrt(np.sum(spectrum_shft[1:, l0c:l1c, :, :]**2, axis=0)) * \
                              lindex[l0c:l1c, :, :], axis=0) / \
                              np.sum(np.sqrt(np.sum(spectrum_shft[1:, l0c:l1c, :, :]**2, axis=0)), axis=0)
                lshift = int(np.median(lcen)) - lcen

                spectrum_shift = np.zeros(Nwave, dtype=np.float32)
                for k in range(Npix):
                    for l in range(Nscan):
                        for m in range(Nstokes):
                            spectrum_shift = shift(spectrum_shft[m, :, l, k], lshift[l, k], \
                                                   order=3, mode="nearest")
                            spectrum_shft[m, :, l, k] = spectrum_shift


                Qtmp = np.sum(spectrum_shft[1, l0c:l1c, py, px], axis=1)
                Utmp = np.sum(spectrum_shft[2, l0c:l1c, py, px], axis=1)
                Vtmp = np.sum(spectrum_shft[3, l0c:l1c, py, px], axis=1)

                data_frame = pd.DataFrame({'SQ':Qtmp, 'SU':Utmp, 'SV':Vtmp}, \
                                          columns=['SQ', 'SU', 'SV'])

                regression = linear_model.LinearRegression()
                regression.fit(data_frame[['SQ','SU']], data_frame['SV'])
                                
                a = regression.coef_[0]
                b = regression.coef_[1]

                # Make a corrected Stokes V for the determination of V->Q,U
                # Use all profiles with strong polarization signal
        
                Qtmp = spectrum_shft[1, l0w:l1w, py, px]
                Utmp = spectrum_shft[2, l0w:l1w, py, px]
                Vtmp = spectrum_shft[3, l0w:l1w, py, px] - a*Qtmp - b*Utmp

                # Determine the V->QU parameters
                                
                c = np.nanmedian(np.nanmean(Qtmp*Vtmp, axis=1) / np.nanmean(Vtmp**2, axis=1) )
                d = np.nanmedian(np.nanmean(Utmp*Vtmp, axis=1) / np.nanmean(Vtmp**2, axis=1) )


                # Reconstruct Sanchez Almeida & Lites (1992) I->QUV cross talk matrix
                                
                iMM1b = np.array( [[ Mij[0], 0., 0., 0.],
                                   [-Mij[1], 1., 0., 0.],
                                   [-Mij[2], 0., 1., 0.],
                                   [-Mij[3], 0., 0., 1.]], dtype=np.float64)
                MM1b = np.linalg.inv(iMM1b)

                # Reconstruct the Kuhn et al (1994) QU<->V cross talk matrix

                iMM2b = np.array([[1.,     0.,     0., 0.],
                                  [0., 1.+a*c,    c*b, -c],
                                  [0.,    a*d, 1.+b*d, -d],
                                  [0.,     -a,     -b, 1.]], dtype=np.float64)
                MM2b = np.linalg.inv(iMM2b)

                del spectrum_shft, Qtmp, Utmp, Vtmp
        
                # Apply the final sign correction to match the original data

                iMM3b = np.array([[1., 0.,  0., 0.],
                                  [0., 1.,  0., 0.],
                                  [0., 0.,  1., 0.],
                                  [0., 0.,  0., 1.]], dtype=np.float64)
                MM3b = np.linalg.inv(iMM3b)
                                
                MMb  = MM1b@MM2b@MM3b
                iMMb = iMM3b@iMM2b@iMM1b
                                
                #make reconstructed data

                spectrum_cal = np.zeros((Nstokes, Nwave, Nscan, Npix), dtype=np.float32)
                spectrum_cal[:, :, :, :] = np.einsum('ij, jabc->iabc', iMMb, self.spectrum)

                del self.spectrum
                self.spectrum = spectrum_cal
        
            case "I->QUV-only":
                pass


    def ViSP_remove_telluric(self):

        for limits in self.telluric:
            j0, j1 = vt.table_invert(self.calib_waves, limits, mode="index")

            stokes_I = self.spectrum[0, :, :, :]
            vt.remove_telluric_pix(stokes_I, j0, j1)


    def ViSP_get_drift(self):
        pass

            
class ViSP_inversion:

    def __init__(self, dataset_root, fits_directory, fiducial_arm_ID=3, \
                 fiducial_pol_ID=1, Blanca_nodes=48):

        if os.path.isdir(dataset_root):
            self.dataset_root = dataset_root
        else:
            print("Data directory {} does not exist. Exiting".format(dataset_root))
            exit()

        if not os.path.isdir(fits_directory):
            os.mkdir(fits_directory)

        self.fits_directory = fits_directory


        dataset_dirs = glob.glob(os.path.join(dataset_root, "*"))
        self.visp_arms = []
        for dataset_path in dataset_dirs:
             self.visp_arms.append(ViSP_arm(dataset_path))

        for arm in self.visp_arms:
            if arm.armID == fiducial_arm_ID:
                self.fiducial_arm = arm
            if arm.armID == fiducial_pol_ID:
                self.fiducial_pol_arm = arm

        self.home_dir     = os.path.expanduser("~") 
        self.Blanca_nodes = Blanca_nodes


    def ViSP_show_arms(self):
        
        print("Found data sets for {:2d} arms:\n".format(len(self.visp_arms)))
        
        for arm in sorted(self.visp_arms, key=lambda arm: arm.armID):
            print(" Arm {0:1d}: {1:s}\n".format(arm.armID, arm.spectrumID) +
                  " Spatial sampling: {0:1.5f}\n   ".format(arm.slit_sample) +
                  " Number of wavelengths: {0:d}\n".format(arm.Nlambda))

        print("Slit step size is {:1.5f} [arcsec]\n".format(self.slit_step) + 
              "Slit width is {:1.5f} [arcsec]".format(self.slit_width))

        
        fig, ax = plt.subplots(ncols=1, nrows=len(self.visp_arms), figsize=(10, 4),\
                               constrained_layout=True)
        for i in range(len(self.visp_arms)):
            arm = self.visp_arms[i]

            norm_obs = arm.avg_spectrum / np.max(arm.avg_spectrum)
            ax[i].plot(arm.calib_waves, norm_obs, label='average spectrum')
            ax[i].plot(arm.lambda_atlas, arm.norm_atlas, label='atlas')
            ax[i].set(title='arm ID: {0}'.format(arm.armID))

        plt.legend()
        plt.savefig("wavelength_solution.pdf", format="pdf")


        fig, ax = plt.subplots(ncols=1, nrows=len(self.visp_arms), figsize=(7,10))
        
        for i in range(len(self.visp_arms)):
            arm = self.visp_arms[i]

            (Nstokes, Nwave, Nscan, Npix) = np.shape(self.fiducial_arm.spectrum)
            yarcsec = self.slit_step * np.arange(0, Nscan)
            xarcsec = self.fiducial_arm.slit_sample * np.arange(0, Npix) * arm.Npix_avg

            reference_img = arm.spectrum[0, arm.continuum_index, :, :]
                
            im = ax[i].imshow(reference_img, origin='lower', cmap="gray", 
                              vmin=0.55, vmax=1.2, \
                              extent=[xarcsec[0], xarcsec[-1], yarcsec[0], yarcsec[-1]])
            ax[i].set(ylabel='scan direction [arcsec]', xlabel='along slit [arcsec]')
            fig.colorbar(im, label='continuum ' + arm.spectrumID, location='top', \
                         aspect=30, shrink=0.8)
        
        plt.savefig("arms_remapped.pdf", format="pdf")

        
    def ViSP_slit_properties(self):

        slit_widths = [arm.slit_width for arm in self.visp_arms]
        slit_steps  = [arm.slit_step for arm in self.visp_arms]

        ## A little sanity check:

        if len(set(slit_widths)) > 1:
            print("Slit widths not unique between arms.")
        else:
            self.slit_width = slit_widths[0]

        if len(set(slit_steps)) > 1:
            print("\n Warning: Slit steps not unique between arms. Using mean value.\n")

        self.slit_step = np.mean(slit_steps)


    def ViSP_solar_location(self):

        (Nstokes, Nscan, Nwave, Npix) = self.fiducial_arm.dataset.shape

        midscan_index = Nstokes * (Nscan // 2)
        
        wcs_midscan  = WCS(self.fiducial_arm.dataset.headers[midscan_index])
        latitude     = (wcs_midscan.pixel_to_world_values(range(0, Npix), 0, 0)[0] * \
                        units.deg).to(units.arcsec)
        longitude    = (wcs_midscan.pixel_to_world_values(0, 0, range(0, Nscan))[2] * \
                        units.deg).to(units.arcsec)

        time_midscan = Time(self.fiducial_arm.dataset.headers[midscan_index]["DATE-AVG"])
        sun_midscan  = get_sun(time_midscan)
        sun_distance = sun_midscan.distance.to(units.km)
        sun_coordin  = SkyCoord(0*units.arcsec, 0*units.arcsec, frame="helioprojective",
                                observer="earth")
        sun_radius   = sun_coordin.rsun
        sun_app_rad  = np.atan(sun_radius / sun_distance).to(units.arcsec)

        self.mu_fiducial = np.sqrt(1.0 - (latitude[Npix//2]**2 + \
                                          longitude[Nscan//2]**2) / sun_app_rad**2)
        print("Solar viewing angle: mu = {0:5.3f}".format(self.mu_fiducial))

        for arm in self.visp_arms:
            arm.mu = self.mu_fiducial


    def ViSP_align_arms(self):

        for arm in self.visp_arms:
            
            reference_img  = arm.spectrum[0, arm.continuum_index, :, :].copy()
            reference_img /= np.mean(reference_img)
            
            arm.hairlineset = vt.hairlineset(reference_img)
            arm.hairlineset.remove(arm.spectrum)

        nonfid_arms = [arm for arm in self.visp_arms if arm.armID != self.fiducial_arm.armID]

        arm_pool = mp.Pool(processes=len(self.visp_arms) - 1)
        
        results = [arm_pool.apply_async(arm.ViSP_remap_data, \
                                        args=(self.fiducial_arm, 0)) \
                   for arm in nonfid_arms]


        for arm, arm_No in zip(nonfid_arms, range(len(results))):
            spectrum_remap = results[arm_No].get()
            
            del arm.spectrum
            arm.spectrum = spectrum_remap
            
        arm_pool.close()

        for arm in self.visp_arms:
            arm.ViSP_get_rebin_params(self.fiducial_arm, self.slit_width)

        
    def ViSP_write_wavegrid(self, fits_directory):

        NM_TO_ANGSTROM = 10
        MILLI          = 1.0E-3

        GRID_FORMAT = '{:s}     : {:15.9f}, {:5.9f}, {:15.9f}\n'
        SEPARATOR   = '----------------------------------------------------------------------------\n'

        PREAMBLE    = "IMPORTANT: a) All items must be separated by commas.\n" + \
                      "b) The first six characters of the last line\n"  + \
                      "in the header (if any) must contain the symbol ---\n\n" + \
                      "Line and blends indices :   Initial lambda     Step       Final lambda\n" + \
                      "(in this order)                  (mA)          (mA)          (mA)\n"

        
        wavegrid_filename = "wave"
        for arm in self.visp_arms:
            wavegrid_filename += "_" + arm.datasetID
        wavegrid_filename += ".grid"

        wavegrid_filepath = os.path.join(fits_directory, wavegrid_filename)

        data = []
        data.append(PREAMBLE)
        data.append(SEPARATOR)

        for arm in sorted(self.visp_arms, key=lambda arm: arm.DeSIRe_line.lambda0):
 
            Nwave = len(arm.calib_waves)
            
            lineID_string = '{:d}'.format(arm.DeSIRe_line.ID)
            for blend in arm.blends:
                lineID_string += ',{:d}'.format(blend.ID)

            wave_step  = np.mean(np.diff(arm.calib_waves)) * \
                            NM_TO_ANGSTROM / MILLI
            wave_init  = (arm.calib_waves[0] - arm.DeSIRe_line.lambda0) * \
                            NM_TO_ANGSTROM / MILLI
            wave_final = wave_init + (Nwave - 1)* wave_step

            data.append(GRID_FORMAT.format(lineID_string, wave_init, wave_step, \
                                           wave_final))


        grid_file = open(wavegrid_filepath, 'w')
        for line in data:
            grid_file.write(line)
        grid_file.close()

        
    def ViSP_write_PSF(self, fits_directory):

        NM_TO_ANGSTROM = 10
        MILLI          = 1.0E-3

        PSF_FORMAT  = '    {: 2d}   {:9.8g}' + 4 * '  {:13.5E}' + "\n"
        MAX_SIGMA   = 5.0
        SAMPL_SIGMA = 25

        PSF_filename = "PSF"
        for arm in self.visp_arms:
            PSF_filename += "_" + arm.datasetID
        PSF_filename += ".per"
        
        PSF_filepath = os.path.join(fits_directory, PSF_filename)
        
        data = []
        for arm in sorted(self.visp_arms, key=lambda arm: arm.DeSIRe_line.lambda0):
            sigma     = arm.FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            PSF_wave  = np.arange(-MAX_SIGMA * sigma, MAX_SIGMA * sigma, \
                                  sigma/SAMPL_SIGMA, dtype=np.float64)
            PSF_value = np.exp(-(PSF_wave / sigma)**2 / 2.0)

            for la in range(len(PSF_wave)):
                 data.append(PSF_FORMAT.format(arm.DeSIRe_line.ID, \
                                               PSF_wave[la] * NM_TO_ANGSTROM / MILLI, \
                                               PSF_value[la], 0.0, 0.0, 0.0))

        psf_file = open(PSF_filepath, 'w')
        for line in data:
            psf_file.write(line)
        psf_file.close()

        
    def ViSP_write_aux_files(self, fits_directory):

        self.ViSP_write_wavegrid(fits_directory)
        self.ViSP_write_PSF(fits_directory)


    def ViSP_get_polarization_map(self):

        stokes_I = self.fiducial_pol_arm.spectrum[0, :, :, :]
        stokes_V = self.fiducial_pol_arm.spectrum[3, :, :, :]
        Nwave    = stokes_I.shape[0]

        self.fiducial_pol_map = np.sum(np.abs(stokes_V)/stokes_I, axis=0) / Nwave

        
def main():

    
    dataset_root   = '/scratch/alpine/hui9576/Data/DKIST/id.156511.623969/'
    fits_directory = '/scratch/alpine/hui9576/Data/DKIST/Fits_dir/'

    fiducial_arm_ID = 3
    fiducial_pol_ID = 1
    
    inv = ViSP_inversion(dataset_root, fits_directory, \
                         fiducial_arm_ID=fiducial_arm_ID, \
                         fiducial_pol_ID=fiducial_pol_ID)
    
    inv.ViSP_slit_properties()
    inv.ViSP_solar_location()
    
    for arm in inv.visp_arms:
        arm.ViSP_find_wavelength_solution()
        arm.ViSP_read_arm_data()

    inv.ViSP_align_arms()

    for arm in inv.visp_arms:
        arm.ViSP_rebin()
        arm.ViSP_get_drift()
        arm.ViSP_remove_telluric()
        
    inv.ViSP_get_polarization_map()
        
    for arm in inv.visp_arms:
        arm.ViSP_calibrate_intensity(inv.fiducial_pol_map)
        arm.ViSP_remove_crosstalk()
        arm.ViSP_find_PSF(inv.fiducial_pol_map)
        arm.ViSP_write_data_fits(fits_directory)
        
    inv.ViSP_write_aux_files(fits_directory)
    
    inv.ViSP_show_arms()

    
if __name__ == "__main__":
    main()
