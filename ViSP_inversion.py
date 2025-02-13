import glob
import os
import numpy as np
from scipy import constants, interpolate
import matplotlib.pyplot as plt
from numba import jit
import multiprocessing as mp
from astropy.io import fits
from astropy.coordinates import SkyCoord, get_sun
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as units

import dkist
from rhanalyze.satlas import satlas
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
        self.ViSP_find_wavelength_solution()
        self.ViSP_read_arm_data()


    def ViSP_analyze_asdf(self):

        self.asdf_file = glob.glob(os.path.join(self.dataset_path, '*.asdf'))[0]
        self.dataset   = dkist.load_dataset(self.asdf_file)
        
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
                 clv_file = "FeI_6302_clv.fits"

             case "Na I D1 (589.59 nm)":
                 self.DeSIRe_line = DeSIRe_line_list[8]
                 clv_file = "NaI_5896_clv.fits"
                 
             case "Ca II (854.21 nm)":
                 self.DeSIRe_line = DeSIRe_line_list[3]
                 clv_file = "CaII_8542_clv.fits"

        self.clv_file_path = os.path.join(self.aux_data_dir, clv_file)


    def ViSP_find_wavelength_solution(self):

        self.avg_spectrum    = np.mean(self.dataset.data[0, :, :, :].compute(), axis=(0, 2))
        self.continuum_index = np.argmax(self.avg_spectrum)

        norm_spectrum = self.avg_spectrum / np.max(self.avg_spectrum)
        ref_index     = np.argmin(self.avg_spectrum)
        Nlambda       = self.Nlambda
        ref_lambda    = self.DeSIRe_line.lambda0

        match self.spectrumID:
             case "Fe I (630.25 nm)":
                 dispersion = 1.281E-03

                 lines_arm    = {629.77927: 'Fe I', 629.9582: 'Zr I', \
                                  630.068: 'Hf I', 630.15012: 'Fe I', \
                                  630.24936: 'Fe I', 630.3755: 'Ti I', \
                                  630.4325: 'Zr I', 630.5275: 'Fe II'}
                 telluric_arm = [[629.833, 629.854], [629.913, 629.930], \
                                  [630.189, 630.209], [630.266, 630.286], \
                                  [630.568, 630.588], [630.643, 630.666]]

                 self.lambda_blu = 630.085
                 self.lambda_red = 630.306

             case "Na I D1 (589.59 nm)":
                 dispersion   = 1.4E-3

                 lines_arm    = {588.9973: 'Na I', 589.1175: 'Fe I', \
                                 589.2883: 'Ni I', 589.5940: 'Na I'}
                 telluric_arm = [[589.157, 589.176], [589.231, 589.250]]

                 self.lambda_blu = 588.653
                 self.lambda_red = 589.886
                 
             case "Ca II (854.21 nm)":
                 dispersion   = 1.873E-03

                 lines_arm    = {853.6163: 'Si I', 853.80147: 'Fe I', 854.21: 'Ca II',\
                                 854.8079: 'Ti I'}
                 telluric_arm = [[853.462, 853.493], [854.068, 854.092], [854.605, 854.632]]

                 self.lambda_blu = 853.232
                 self.lambda_red = 855.193

                 
        wave_corr = ref_lambda + dispersion * \
            (np.arange(Nlambda, dtype=np.float64) - ref_index)

        ATL_RANGE = 2.0
        DELTA_LAM = 0.01
        
        fts = satlas()
        self.lambda_atlas, intensity, continuum = fts.nmsiatlas(ref_lambda - ATL_RANGE,\
                                                                ref_lambda + ATL_RANGE)
        self.norm_atlas = intensity / continuum   
        Nlines          = len(lines_arm)
        wave_positions  = np.zeros(Nlines, dtype=np.float64)
        index_positions = np.zeros(Nlines, dtype=np.float64)

        n = 0
        for key, value in lines_arm.items():
            values  = np.array([key - DELTA_LAM, key + DELTA_LAM])
            indices = vt.table_invert(self.lambda_atlas, values, mode="index")

            wave_positions[n] = vt.find_parmin(self.lambda_atlas[indices[0]:indices[1]],
                                               self.norm_atlas[indices[0]:indices[1]])
            n += 1

        n = 0
        for key, value in lines_arm.items():
            values  = np.array([key - DELTA_LAM, key + DELTA_LAM])
            indices = vt.table_invert(wave_corr, values, mode="index")

            wave_min = vt.find_parmin(wave_corr[indices[0]:indices[1]],
                                      norm_spectrum[indices[0]:indices[1]])
            index_positions[n] = vt.table_invert(wave_corr, wave_min, mode="effective")[0]

            n += 1
     
        coefficients = np.polyfit(index_positions, wave_positions, 2)
        poly         = np.poly1d(coefficients)

        #-# Store the calibrated wavelengths for the current arm
        
        self.calib_waves = poly(np.arange(Nlambda))
        

    def ViSP_read_arm_data(self):

        limits = vt.table_invert(self.calib_waves, np.array([self.lambda_blu, self.lambda_red]), \
                                 mode="index")
        
        transposed_set = np.transpose(self.dataset.data, axes=(0, 2, 1, 3))
        self.spectrum  = transposed_set[:, limits[0]:limits[1], 100:175, :].compute()

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

    
    def ViSP_find_clv(self, mu, cont_index):

        hdul = fits.open(self.clv_file_path)

        clv   = hdul[0].data
        xmu   = hdul[1].data
        waves = hdul[2].data
        hdul.close()

        clv_interp = interpolate.RegularGridInterpolator((xmu, waves), clv)
        
        return clv_interp([mu, cont_index])

        
    def ViSP_calibrate(self, mu=1.0):

        (Nstokes, Nwave, Nscan, Npix) = self.spectrum.shape

        limbdark = vt.LimbDark(self.DeSIRe_line.lambda0, mu)
        
        lambda_cont      = self.calib_waves[self.continuum_index]
        index_cont_atlas = vt.table_invert(self.lambda_atlas, lambda_cont, mode="index")
        cont_atlas       = self.norm_atlas[index_cont_atlas]

        avg_continuum = np.mean(self.spectrum[0, self.continuum_index, :, :], axis=(0, 1))

        clv_factor = self.ViSP_find_clv(mu, lambda_cont)
        
        normalization  = (limbdark * cont_atlas * clv_factor) / avg_continuum
        self.spectrum *= normalization
        

class ViSP_inversion:

    def __init__(self, dataset_root, fits_directory, fiducial_arm_ID=3, Blanca_nodes=48):

        if os.path.isdir(dataset_root):
            self.dataset_root = dataset_root
        else:
            print("Data directory {} does not exist. Exiting".format(dataset_root))
            exit()

        if !os.path.isdir(fits_directory):
            os.mkdir(fits_directory)

        self.fits_directory = fits_directory


        dataset_dirs = glob.glob(os.path.join(dataset_root, "*"))
        self.visp_arms = []
        for dataset_path in dataset_dirs:
             self.visp_arms.append(ViSP_arm(dataset_path))

        for arm in self.visp_arms:
            if arm.armID == fiducial_arm_ID:
                self.fiducial_arm = arm

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


        fig, ax = plt.subplots(ncols=1, nrows=len(self.visp_arms), \
                               figsize=(7,10), tight_layout=True)
        
        for i in range(len(self.visp_arms)):
            arm = self.visp_arms[i]

            (Nstokes, Nwave, Nscan, Npix) = np.shape(self.fiducial_arm.spectrum)
            xarcsec = self.slit_step * np.arange(0, Nscan)
            yarcsec = self.fiducial_arm.slit_sample * np.arange(0, Npix)

            reference_img = arm.spectrum[0, arm.continuum_index, :, :]
                
            im = ax[i].imshow(reference_img, origin='lower', cmap="gray", 
                              vmin=0.55, vmax=1.2, \
                              extent=[yarcsec[0], yarcsec[-1], xarcsec[0], xarcsec[-1]])
            ax[i].set(ylabel='scan direction [arcsec]', xlabel='along slit [arcsec]', \
                      title=arm.spectrumID)
            fig.colorbar(im, label='continuum intensity', location='top', \
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
        
        
    def ViSP_write_data_fits(self):

        Nscan_fid, Npix_fid = self.fiducial_arm.spectrum.shape[2:4]
        
        Npix_avg   = int(np.floor(self.slit_width / self.fiducial_arm.slit_sample))
        Npix_rebin = Npix_fid // Npix_avg
        last_pixel = Npix_avg * Npix_rebin
        print("Npix_avg: {0}, Npix_rebin: {1}".format(Npix_avg, Npix_rebin))

        for arm in self.visp_arms:

            filename  = "ViSP_" + arm.datasetID + "_" + arm.spectrumID + ".fits"
            file_path = os.path.join(self.fits_directory, filename)
            
            Nstokes, Nwave = arm.spectrum.shape[0:2]
            save_spectrum  = np.zeros((Nstokes, Nwave, Nscan_fid, Npix_rebin), dtype=np.float32)
            
            save_spectrum[:, :, :, :] = np.mean(np.reshape(arm.spectrum[:, :, :, 0:last_pixel], \
                                                           (Nstokes, Nwave, Nscan_fid, \
                                                            Npix_rebin, Npix_avg)), axis=4)

            del arm.spectrum
            arm.spectrum = save_spectrum
            
            arm.ViSP_calibrate(mu=self.mu_fiducial)
            
            hdu  = fits.PrimaryHDU(arm.spectrum)
            hdul = fits.HDUList([hdu])
            hdul.writeto(file_path, overwrite=True)


    def ViSP_write_wavegrid(self):
        pass
    def ViSP_write_PSF(self):
        pass
        
    def ViSP_write(self):

        self.ViSP_write_data_fits()
        self.ViSP_write_wavegrid()
        self.ViSP_write_PSF()
        
         
def main():

##    dataset_root = '/Users/han/Data/DKIST/id.136838.353289/'
    dataset_root = '/Users/han/Data/DKIST/id.136838.527585/'
    
    inv = ViSP_inversion(dataset_root, fits_directory, fiducial_arm_ID=3)
    
    inv.ViSP_slit_properties()
    inv.ViSP_solar_location()
    inv.ViSP_align_arms()
    inv.ViSP_show_arms()
    
    inv.ViSP_write()

    
if __name__ == "__main__":
    main()
