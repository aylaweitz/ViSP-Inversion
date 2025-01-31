import glob
import os
import numpy as np
from scipy import constants, interpolate
import matplotlib.pyplot as plt
from numba import jit
import multiprocessing as mp

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
             case "Na I D1 (589.59 nm)":
                 self.DeSIRe_line = DeSIRe_line_list[8]
             case "Ca II (854.21 nm)":
                 self.DeSIRe_line = DeSIRe_line_list[3]


    def ViSP_read_arm_data(self):

        transposed_set = np.transpose(self.dataset.data, axes=(3, 1, 2, 0))
        self.spectrum  = transposed_set[:, 50:125, :, :].compute()

        
    def ViSP_find_wavelength_solution(self):

        self.avg_spectrum    = np.mean(self.dataset.data[0, :, :, :].compute(), axis=(0, 2))
        self.continuum_index = np.argmax(self.avg_spectrum)

        norm_spectrum = self.avg_spectrum/np.max(self.avg_spectrum)
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

             case "Na I D1 (589.59 nm)":
                 dispersion   = 1.4E-3

                 lines_arm    = {588.9973: 'Na I', 589.1175: 'Fe I', \
                                 589.2883: 'Ni I', 589.5940: 'Na I'}
                 telluric_arm = [[589.157, 589.176], [589.231, 589.250]]
                 
             case "Ca II (854.21 nm)":
                 dispersion   = 1.873E-03

                 lines_arm    = {853.6163: 'Si I', 853.80147: 'Fe I', 854.21: 'Ca II',\
                                 854.8079: 'Ti I'}
                 telluric_arm = [[853.462, 853.493], [854.068, 854.092], [854.605, 854.632]]


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
        

    def ViSP_remap_spectrum(self, fiducial_arm, dummy):

        y1 = fiducial_arm.hairlineset.hairlines[0].position
        y2 = fiducial_arm.hairlineset.hairlines[1].position
        Npix_fid, Nscan_fid, Nwave_fid, Nstokes = np.shape(fiducial_arm.spectrum)

        x1 = self.hairlineset.hairlines[0].position
        x2 = self.hairlineset.hairlines[1].position

        slope     = (x2 - x1) / (y2 - y1)
        intercept = x1 - slope * y1

        x_fid = slope * np.arange(Npix_fid, dtype=np.float64) + intercept
        y_fid = np.arange(Nscan_fid, dtype=np.float64)
        arm_ref_remap = vt.ViSP_remap_image(self.reference_img, x_fid, y_fid)

        fid_image = fiducial_arm.reference_img - 1.0
        arm_image = arm_ref_remap / np.mean(arm_ref_remap) - 1.0

        cross_corr = np.abs(np.fft.ifft2(np.fft.fft2(arm_image) * \
                                         np.conj(np.fft.fft2(fid_image))))
        icx, icy = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        max_area = np.zeros((3, 3), dtype=np.float32)

        for k in range(3):
            jcx = (icx - 1 + k + Npix_fid) % Npix_fid
            for l in range(3):
                jcy = (icy -1 + l + Nscan_fid) % Nscan_fid
                max_area[k, l] = cross_corr[jcx, jcy]

        dx, dy = vt.get_2D_extreme(max_area)
        x_offset = icx + dx
        if x_offset > (Npix_fid / 2): x_offset -= Npix_fid
        y_offset = icy + dy
        if y_offset > (Nscan_fid / 2): y_offset -= Nscan_fid

        spectrum_remap = vt.ViSP_remap_image(self.reference_img, x_fid + x_offset, \
                                             y_fid + y_offset)

        print("Arm {0:1d}, line ID {1} has offsets ({2:1.5f}, {3:1.5f})".format(self.armID, \
                                                                                self.spectrumID, \
                                                                                x_offset, y_offset))
        
        return spectrum_remap

    def ViSP_write(self):
        pass
        
        
class ViSP_inversion:

    def __init__(self, dataset_root, Blanca_nodes=48):

        if os.path.isdir(dataset_root):
            self.dataset_root = dataset_root
        else:
            print("Data directory {} does not exist. Exiting".format(dataset_root))
            exit()

        dataset_dirs = glob.glob(os.path.join(dataset_root, "*"))
        self.visp_arms = []
        for dataset_path in dataset_dirs:
             self.visp_arms.append(ViSP_arm(dataset_path))

        self.Blanca_nodes = Blanca_nodes


    def show_arms(self):
        
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
        plt.show()


        fig, ax = plt.subplots(nrows=1, ncols=len(self.visp_arms), figsize=(6,6), tight_layout=True)
        for i in range(len(self.visp_arms)):
            arm = self.visp_arms[i]

            (Npix, Nscan, Nwave, Nstokes) = np.shape(arm.spectrum)
            xarcsec = arm.slit_sample * np.arange(0, Npix)
            yarcsec = self.slit_step * np.arange(0, Nscan)
            
            im = ax[i].imshow(arm.reference_img, origin='lower', cmap="gray", 
                              vmin=0.55, vmax=1.2, \
                              extent=[yarcsec[0], yarcsec[-1], xarcsec[0], xarcsec[-1]])
            ax[i].set(xlabel='scan direction [arcsec]', ylabel='along slit [arcsec]', \
                      title=arm.spectrumID)
            fig.colorbar(im, label='continuum intensity', location='top', aspect=30, shrink=0.8)
        
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=len(self.visp_arms), figsize=(6,6), tight_layout=True)
        for i in range(len(self.visp_arms)):
            arm = self.visp_arms[i]

            (Npix, Nscan, Nwave, Nstokes) = np.shape(self.fiducial_arm.spectrum)
            xarcsec = self.fiducial_arm.slit_sample * np.arange(0, Npix)
            yarcsec = self.slit_step * np.arange(0, Nscan)

            if arm.armID == self.fiducial_arm.armID:
                image = arm.reference_img
            else:
                image = arm.spectrum_remap
                
            im = ax[i].imshow(image, origin='lower', cmap="gray", 
                              vmin=0.55, vmax=1.2, \
                              extent=[yarcsec[0], yarcsec[-1], xarcsec[0], xarcsec[-1]])
            ax[i].set(xlabel='scan direction [arcsec]', ylabel='along slit [arcsec]', \
                      title=arm.spectrumID)
            fig.colorbar(im, label='continuum intensity', location='top', aspect=30, shrink=0.8)
        
        plt.show()

        
    def slit_properties(self):

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

            
    def ViSP_align_arms(self, fiducial_arm_ID=3):

        for arm in self.visp_arms:
            
            arm.reference_img = arm.spectrum[:, :, arm.continuum_index, 0]
            arm.reference_img /= np.mean(arm.reference_img)
            
            arm.hairlineset = vt.hairlineset(arm.reference_img)
            arm.hairlineset.remove(arm.spectrum)

            if arm.armID == fiducial_arm_ID:
                fiducial_arm = arm

        self.fiducial_arm = fiducial_arm
        nonfid_arms       = [arm for arm in self.visp_arms if arm.armID != fiducial_arm.armID]
        
        arm_pool = mp.Pool(processes=len(self.visp_arms) - 1)
        
        results = [arm_pool.apply_async(arm.ViSP_remap_spectrum, \
                                        args=(fiducial_arm, 0)) \
                   for arm in nonfid_arms]


        for arm, arm_No in zip(nonfid_arms, range(len(results))):
            arm.spectrum_remap = results[arm_No].get()
                 
        arm_pool.close()

        
    def ViSP_write_fits(self):

        pass
        
def main():

    dataset_root = '/Users/han/Data/DKIST/id.136838.353289/'
    
    inv = ViSP_inversion(dataset_root)
    
    inv.slit_properties()
    inv.ViSP_align_arms(fiducial_arm_ID=3)
    inv.show_arms()

if __name__ == "__main__":
    main()
