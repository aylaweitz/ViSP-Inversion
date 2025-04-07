import sys
import numpy as np

import matplotlib.pyplot as plt
from   matplotlib.backend_bases import MouseButton
from   matplotlib.widgets import Cursor, Slider, Button

from astropy.io import fits


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
        return list


class DeSIRe_inverted_map:

    def __init__(self, base_dir='.', scanID='scan1738'):

        self.scanID   = scanID
        self.base_dir = base_dir
        self.scan_dir = base_dir + scanID + '/'
        self.fig_dir  = base_dir + 'Figures/'

        self.fig = None
        self.axs = None
        
        self.fig_prof = None
        self.axs_prof = None
        self.fig_phys = None
        self.axs_phys = None

        self.inversion  = None
        self.wavelength = None
        self.observed   = None
        self.model      = None
        self.IDs        = None

        self.lines      = None


    def read_map(self):

        NM_TO_ANGSTROM = 1
        MILLI          = 1.0E-3
        
        inversion_file = self.scan_dir + 'inv_res_pre.fits'
        observed_file  = self.scan_dir + 'per_ori.fits'
        models_file    = self.scan_dir + 'inv_res_mod.fits'

        hdul = fits.open(inversion_file)      
        self.inversion  = hdul[0].data
        self.wavelength = np.float64(hdul[1].data * MILLI / NM_TO_ANGSTROM)
        self.IDs        = np.intc(hdul[2].data)
        hdul.close()

        hdul = fits.open(observed_file)
        self.observed = hdul[0].data
        hdul.close()

        hdul = fits.open(models_file)
        self.model = hdul[0].data
        hdul.close()
        
        IDs = np.unique(self.IDs)
        DeSIRe_line_list = DeSIRe_line.get_list()
        self.lines = [line for line in DeSIRe_line_list if line.ID in IDs]

        for line in self.lines:
            line.indices, = np.where(self.IDs == line.ID)


    def display_map(self):

        CURSOR_LINE_WIDTH = 0.5
        
        STOKES_I = 0
        STOKES_V = 3
        L_CONT   = 0
        L_LOBE_V = 26
        
        self.fig, self.axs = plt.subplots(nrows=1, ncols=2, \
                                          sharey=True, figsize=(8, 8))
        self.fig.suptitle(self.scanID)

        im0 = self.axs[0].imshow(self.observed[STOKES_I, L_CONT, :, :], \
                                 cmap="gist_gray", \
                                 origin='lower')
        self.axs[0].set(ylabel = 'scan direction [steps]', \
                        xlabel='along slit [rebinned pixels]')
        plt.colorbar(im0, label='Stokes Ic/<Ic>', location='top', \
                     fraction=0.10, shrink=1.0)

        im1 = self.axs[1].imshow(self.observed[STOKES_V, L_LOBE_V, :, :], \
                                 cmap="bone", \
                                 origin='lower')
        self.axs[1].set(xlabel='along slit [rebinned pixels]')
        plt.colorbar(im1, label='Stokes V/<Ic>', location='top', \
                     fraction=0.10, shrink=1.0)

        cursor0 = Cursor(self.axs[0], useblit=True, color='cyan',
                         linewidth=CURSOR_LINE_WIDTH)
        cursor1 = Cursor(self.axs[1], useblit=True, color='yellow',
                         linewidth=CURSOR_LINE_WIDTH)

        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', \
                                                self.onclick)
        self.in_axes = False
        self.cid2 = self.fig.canvas.mpl_connect('axes_leave_event',
                                                self.axs_leave)
        self.cid3 = self.fig.canvas.mpl_connect('axes_enter_event',
                                                self.axs_enter)

        (Ny, Nx) = self.inversion[0, 0, :, :].shape
        self.x_position = Nx//2
        self.y_position = Ny//2

        self.display_profiles()
        self.display_physical_quantities()
        
        plt.show()

        
    def axs_leave(self, event):
        self.in_axes = False
        
    def axs_enter(self, event):
        self.in_axes = True   
        
    def onclick(self, event):
        if self.in_axes:
            self.button     = event.button
            self.x_position = np.round(event.xdata).astype(int)
            self.y_position = np.round(event.ydata).astype(int)
            
            print('attempt redraw at [%d, %d]' % \
                  (self.x_position, self.y_position))
            
            if (event.button == 1 or event.button == 3):
                self.draw_profiles(update=True)
                self.draw_physical_quantities(update=True)

                
    def display_profiles(self):
        
        Ncol = len(self.lines)
        self.fig_prof, self.axs_prof = plt.subplots(nrows=4, ncols=Ncol,
                                                    constrained_layout=True,
                                                    figsize=(10, 10))

        self.draw_profiles()
        self.fig_prof.show()


    def draw_profiles(self, update=False):
        
        ix = self.x_position
        iy = self.y_position

        N_STOKES = 4        
        ylabels  = ['relative intensity', 'Stokes $U/I_c$', \
                    'Stokes $Q/I_c$', 'Stokes $V/I_c$']
        

        if update:
            (Npx, Npy) = self.axs_prof.shape
            for i in range(Npx):
                for j in range(Npy):
                    self.axs_prof[i][j].clear()

        for k in range(N_STOKES):
            for l in range(len(self.lines)):
                
                lidx  = self.lines[l].indices
                waves = self.wavelength[lidx]
                
                self.axs_prof[k][l].plot(waves, \
                                         self.observed[k, lidx, iy, ix], \
                                         ".", label='data')
                self.axs_prof[k][l].plot(waves, \
                                         self.inversion[k, lidx, iy, ix], \
                                         label='fit')
                if l == 0:
                    self.axs_prof[k][0].set(ylabel=ylabels[k])

                if k == 0:
                    self.axs_prof[0][l].set(ylim=[0.0, 1.1])
                    title_text = 'Line:  %6.6g nm' % \
                                  (self.lines[l].lambda0)
                    self.axs_prof[0][l].title.set_text(title_text)

                if k == 0  and  l == 0:
                    self.axs_prof[0][0].legend(loc='lower left')
                    self.axs_prof[0][0].annotate('x = %d, y = %d' % \
                                                 (ix, iy), \
                                                 (0.65, 0.05), \
                                                 xycoords='axes fraction')
                    self.axs_prof[0][0].annotate(self.scanID, (0.4, 0.05), \
                                                 xycoords='axes fraction')
                    self.axs_prof[0][0].legend(loc='lower left')

                if k == 3:
                    self.axs_prof[k][l].set(xlabel='$\Delta\lambda$ [nm]')
                    
        self.fig_prof.canvas.draw_idle()
        self.fig_prof.canvas.flush_events()

        if update:
            if self.button == 3:
                location_str = '_%d_%d' % (ix, iy)
                pdf_file = self.fig_dir + 'FeI_CaII_' + self.scanID + \
                              location_str + '.pdf'
                self.fig_prof.savefig(pdf_file, format="pdf", \
                                      bbox_inches="tight")


    def display_physical_quantities(self):

        NCOL = 2
        NROW = 2
        
        self.fig_phys, self.axs_phys = plt.subplots(nrows=NROW, \
                                                    ncols=NCOL,
                                                    constrained_layout=True,
                                                    figsize=(8, 6))

        self.draw_physical_quantities()
        self.fig_phys.show()

        
    def draw_physical_quantities(self, update=False):

        CM_TO_KM     = 1.0E-5
        ARCSEC_TO_KM = 725.0
        KILO         = 1.0E3

        ix = self.x_position
        iy = self.y_position

        log_tau500  = self.model[0, :, iy, ix]
        temperature = self.model[1, :, iy, ix]
##        P_electron  = self.model[2, :, iy, ix]
##        micro_turb  = self.model[3, :, iy, ix] * CM_TO_KM
        B_strength  = self.model[4, :, iy, ix]
        v_los       = self.model[5, :, iy, ix] * CM_TO_KM
        B_inclin    = self.model[6, :, iy, ix]
##        B_azimuth   = self.model[7, :, iy, ix]
##        z_geometric = self.model[8, :, iy, ix] * CM_TO_KM
##        P_gas       = self.model[9, :, iy, ix]
##        density     = self.model[10, :, iy, ix]

        model_units = ['', 'K', 'dyn cm^-2', 'km s^-1', 'G', \
                       'km s^-1', 'degree', 'degree', \
                       'km', 'dyn cm^-2', 'g cm^-3']


        if update:
            (Npx, Npy) = self.axs_phys.shape
            for i in range(Npx):
                for j in range(Npy):
                    self.axs_phys[i][j].clear()

        self.axs_phys[0][0].plot(log_tau500, temperature)
        self.axs_phys[0][0].set(ylabel='temperature ['+model_units[5]+']')


        self.axs_phys[0][0].annotate('x = %d, y = %d' % \
                                     (ix, iy), \
                                     (0.50, 0.90), \
                                     xycoords='axes fraction')
        self.axs_phys[0][0].annotate(self.scanID, (0.1, 0.90), \
                                     xycoords='axes fraction')

        self.axs_phys[0][1].plot(log_tau500, v_los)
        self.axs_phys[0][1].set(ylabel='v_los ['+model_units[5]+']')

        self.axs_phys[1][0].plot(log_tau500, B_strength)
        self.axs_phys[1][0].set(xlabel='log tau_500', \
                                ylabel='B ['+model_units[4]+']')

        self.axs_phys[1][1].plot(log_tau500, B_inclin)
        self.axs_phys[1][1].set(xlabel='log tau_500', \
                                ylabel='gamma ['+model_units[6]+']')


        self.fig_phys.canvas.draw_idle()
        self.fig_phys.canvas.flush_events()
    

    def physical_maps(self):

        CM_TO_KM     = 1.0E-5
        ARCSEC_TO_KM = 725.0
        KILO         = 1.0E3

        model_units = ['', 'K', 'dyn cm^-2', 'km s^-1', 'kG', \
                       'km s^-1', 'degree', 'degree', \
                       'km', 'dyn cm^-2', 'g cm^-3']
        phys  = [5, 4, 6, 1]
        scale = [1.0/CM_TO_KM, KILO, 1.0, 1.0]

        fig_maps, axs_maps = \
            plt.subplots(nrows=1, ncols=4, \
                         sharey=True, figsize=(14, 6))
        fig_maps.suptitle(self.scanID)

        (Ntau, Ny, Nx) = self.model[0, :, :, :].shape
        tau_index0 = Ntau // 2

        ims = []
        ims.append(axs_maps[0].imshow(self.model[phys[0], tau_index0, :, :] / \
                                      scale[0], \
                                      origin='lower', cmap='bwr', \
                                      vmin=-5.0, vmax=5.0))
        plt.colorbar(ims[0], label=r'$v_{LOS}$ [' + model_units[phys[0]]+']', \
                     ax=axs_maps[0], shrink=0.84)
        axs_maps[0].set(ylabel = 'scan direction [steps]', \
                        xlabel='along slit [rebinned pixels]')


        ims.append(axs_maps[1].imshow(self.model[phys[1], tau_index0, :, :] / \
                                      scale[1], \
                                      origin='lower', \
                                      vmin=0.0, vmax=1.0))
        plt.colorbar(ims[1], \
                     label=r'field strength ['+model_units[phys[1]]+']', \
                     ax=axs_maps[1], shrink=0.84)
        axs_maps[1].set(xlabel='along slit [rebinned pixels]')

        
        ims.append(axs_maps[2].imshow(self.model[phys[2], tau_index0, :, :] / \
                                      scale[2], \
                                      origin='lower', cmap='bwr', \
                                      vmin=0.0, vmax=180.0))
        plt.colorbar(ims[2], label=r'inclination ['+model_units[phys[2]]+']', \
                     ax=axs_maps[2], shrink=0.84)
        axs_maps[2].set(xlabel='along slit [rebinned pixels]')

        
        ims.append(axs_maps[3].imshow(self.model[phys[3], tau_index0, :, :] / \
                                      scale[3], \
                                      origin='lower', cmap='hot', \
                                      vmin=3500))
        plt.colorbar(ims[3], label=r'temperature ['+model_units[phys[3]]+']', \
                     ax=axs_maps[3], shrink=0.84)
        axs_maps[3].set(xlabel='along slit [rebinned pixels]')

        logtau      = self.model[0, :, 0, 0]
        logtau_max  = logtau[0]
        logtau_min  = logtau[-1]
        logtau_init = logtau[tau_index0]
        
        axtau      = fig_maps.add_axes([0.15, 0.01, 0.65, 0.03])
        tau_slider = Slider(axtau, r'log $\tau_{500}$', logtau_min, \
                            logtau_max, valinit=logtau_init)
        
        def slider_update(val):
            tau_value = tau_slider.val
            indices,  = np.where(logtau <= val)
            tau_index = indices[0]
            
            for i in range(len(ims)):
                ims[i].set_data(self.model[phys[i], tau_index, :, :] / scale[i])
            fig_maps.canvas.draw_idle()
            
        tau_slider.on_changed(slider_update)
      
        plt.tight_layout()
        plt.show()


def main():
    

    base_dir = '/Users/han/Data/DKIST/Results/'

    if len(sys.argv) > 1:
        scanID = sys.argv[1]
    else:
        scanID = 'scan1738'

    map = DeSIRe_inverted_map(base_dir=base_dir, scanID=scanID)
    map.read_map()

    map.display_map()
##    map.physical_maps()
        

if __name__ == "__main__":
    main()
