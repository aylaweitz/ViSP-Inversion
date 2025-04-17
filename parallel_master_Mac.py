import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import astropy.io.fits as fits
from subprocess import call
import subprocess
import datetime
import multiprocessing
import ctypes
from os.path import expanduser
from scipy.stats import chisquare,chi2_contingency,stats
import glob
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp2d
import copy
import fnmatch
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
plt.switch_backend('agg')
plt.ioff()








#Define all functions

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

def make_dir(folder_name_def,dest=' .'):
    mdir=subprocess.Popen('mkdir '+str(folder_name_def)+dest, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = mdir.communicate()
    return(out, err)

def rm_dir(folder_name_def,dest=' .'):
    rmdir=subprocess.Popen('rm -rf '+str(folder_name_def)+dest, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = rmdir.communicate()
    return(out, err)


def rm_file(folder_name_def,dest=' .'):
    rmdir=subprocess.Popen('rm '+str(folder_name_def)+dest, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = rmdir.communicate()
    return(out, err)


def cp_file(file_name_def,dest=' .'):
    cpfi=subprocess.Popen('cp  '+str(file_name_def)+' '+dest, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cpfi.communicate()
    return(out, err)

def cp_folder(file_name_def,dest=' .'):
    cpfi=subprocess.Popen('cp  -r '+str(file_name_def)+dest, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cpfi.communicate()
    return(out, err)

def save_fits(array,file_name):
    '''
    Save array to a fits file.

    Parameters
    ----------
    first: array that you want to save
    second : Name of the fits file. You can add the path before name
    '''
    hdu = fits.PrimaryHDU(np.float32(array))
    hdul = fits.HDUList([hdu])
    hdul.writeto(file_name,overwrite=True)

def save_fits_p(array,arrayw,arrayind,file_name):
    '''
    Save array to a fits file.

    Parameters
    ----------
    first: array that you want to save
    second : Name of the fits file. You can add the path before name
    '''
    hdu1 = fits.PrimaryHDU(np.float32(array))
    hdu2 = fits.ImageHDU(arrayw)
    hdu3 = fits.ImageHDU(arrayind)
    new_hdul = fits.HDUList([hdu1, hdu2,hdu3])
    new_hdul.writeto(file_name,overwrite=True)

def fix_path(path):
    path = repr(path)
    path = path.replace(")", "\)")
    path = path.replace("(", "\(")
    path = path.replace(" ", "\ ")
    path = os.path.abspath(path).split("'")[1]
    return path




def plot_maps(dat,dat2,dat3,fullp_chi2,output_opt):

    '''
    Plot and save the temperature and velocity maps on a pdf file

    Parameters
    ----------
    first: array that you want to save
    second : Name of the fits file. You can add the path before name
    '''
    pdf_pages = PdfPages('results_maps.pdf')
    fig = plt.figure(figsize=(18, 6))
    plt.figure(1)
    plt.subplot(131)
    intensity=dat3[0,0,:,:]
    if output_opt==1 or output_opt==2:
        intensity_fitted=dat2[0,0,:,:]
    if output_opt==3:
        intensity_fitted=dat2[-1,0,0,:,:]
    Z=intensity
    X,Y=np.meshgrid(range(Z.shape[1]),range(Z.shape[0]))
    plt.pcolormesh(X,Y,Z,cmap='gray',vmin=intensity.min(),vmax=intensity.max(),rasterized=True)
    cb = plt.colorbar()
    cb.set_label(label='Cont',fontsize=10)
    plt.subplot(132)
    Z=intensity_fitted
    X,Y=np.meshgrid(range(Z.shape[1]),range(Z.shape[0]))
    plt.pcolormesh(X,Y,Z,cmap='gray',vmin=intensity.min(),vmax=intensity.max(),rasterized=True)
    cb = plt.colorbar()
    cb.set_label(label='Fitted_cont_int',fontsize=10)
    plt.subplot(133)
    if output_opt==1 or output_opt==2:
        if nlte_f>=10 or run_source==0:
            Z=fullp_chi2[0,:,:]
        if nlte_f<10:
            Z=fullp_chi2[0,:,:]
    if output_opt==3:
        if nlte_f>=10 or run_source==0:
            Z=fullp_chi2[-1,:,:]
        if nlte_f<10:
            Z=fullp_chi2[-1,0,:,:]

    X,Y=np.meshgrid(range(Z.shape[1]),range(Z.shape[0]))
    plt.pcolormesh(X,Y,Z,vmin=0,cmap='gray',rasterized=True)
    cb = plt.colorbar()
    cb.set_label(label='I_chi2',fontsize=10)
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close()
    if output_opt==1 or output_opt==2:
        rang=len(dat[4,:,0,0])
    if output_opt==3:
        rang=len(dat[-1,4,:,0,0])
    for lt in range(0,rang,5):
        fig = plt.figure(figsize=(18, 6))
        op_dep_pos=lt
        if output_opt==1 or output_opt==2:
            magneticfield=dat[4,op_dep_pos,:,:]
        if output_opt==3:
            magneticfield=dat[-1,4,op_dep_pos,:,:]
        magneticfield[magneticfield==0] = np.nan
        magrma=np.nanmean(magneticfield)+3*np.nanstd(magneticfield)
        if magrma > np.max(magneticfield):
            magrma=np.max(magneticfield)
        if output_opt==1 or output_opt==2:
            temperature=dat[1,op_dep_pos,:,:]
        if output_opt==3:
            temperature=dat[-1,1,op_dep_pos,:,:]
        temperature[temperature==0] = np.nan
        temprma=np.nanmean(temperature)+3*np.nanstd(temperature)
        temprmi=np.nanmean(temperature)-3*np.nanstd(temperature)
        if temprma > np.max(temperature):
            temprma=np.max(temperature)
        if temprmi < np.min(temperature):
            temprmi=np.min(temperature)
        if output_opt==1 or output_opt==2:
            velocity=dat[5,op_dep_pos,:,:]/100000.
        if output_opt==3:
            velocity=dat[-1,5,op_dep_pos,:,:]/100000.
        velocity[velocity==0] = np.nan
        stdv=np.nanstd(velocity)
        plt.subplot(131)
        Z=temperature
        plt.pcolormesh(X,Y,Z, cmap='hot',rasterized=True,vmin=temprmi,vmax=temprma)
        cb = plt.colorbar()
        cb.set_label(label='K',fontsize=10)
        if output_opt==1 or output_opt==2:
            plt.title('log tau '+str(dat[0,lt,0,0]))
        if output_opt==3:
            plt.title('log tau '+str(dat[-1,0,lt,0,0]))
        plt.subplot(132)
        Z=velocity
        plt.pcolormesh(X,Y,Z,vmin=-1*4*stdv-2,vmax=4*stdv+2,cmap='bwr',rasterized=True)
        cb = plt.colorbar()
        cb.set_label(label='Km/s',fontsize=10)
        plt.subplot(133)
        Z=magneticfield
        plt.pcolormesh(X,Y,Z, cmap='bone',vmin=0,vmax=magrma,rasterized=True)
        cb = plt.colorbar()
        cb.set_label(label='G',fontsize=10)
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close()

    firstPage = plt.figure(figsize=(11,10))
    firstPage.clf()
    ffi=open(run_file_folder+control_file,'r')
    txt = ffi.read()
    ffi.close()
    firstPage.text(0.05,0.95,txt, transform=firstPage.transFigure, size=12, va="top", ha="left")
    pdf_pages.savefig()
    plt.close()
    if run_source==1:
        firstPage = plt.figure(figsize=(11,48))
        firstPage.clf()
        ffi=open(run_file_folder+'keyword.input','r')
        txt = ffi.read()
        ffi.close()
        firstPage.text(0.05,0.97,txt, transform=firstPage.transFigure, size=12, va="top", ha="left")
        pdf_pages.savefig()
    plt.close()
    pdf_pages.close()


def create_coor_file(arr,x,y):
    '''
    Create the .coor file

    Parameters
    ----------
    first: array that you want to save
    second : Name of the fits file. You can add the path before name
    '''
    filc=open('pos.coor','w')
    if type(arr[0]) == float:
        filc.write(str(arr[0])+','+str(arr[1])+','+str(arr[2])+','+str(arr[3]))
    if type(arr) == np.ndarray:
        filc.write(str(arr[0,x,y])+','+str(arr[1,x,y])+','+str(arr[2,x,y])+','+str(arr[3,x,y]))
    filc.close()

def create_folder(q):
    for i in range(q):
        os.chdir(sir_location)
        make_dir(intern_folder_name+str(i+1))
        os.chdir(intern_folder_name+str(i+1))
        cp_file(run_file_folder+'*.* ',sir_location+intern_folder_name+str(i+1)+'/')
        cp_file(run_file_folder+'* ',sir_location+intern_folder_name+str(i+1)+'/')

        cp_file(run_file_folder+'input_files/* ',sir_location+intern_folder_name+str(i+1)+'/')
        cp_file(run_file_folder+'input_files/*.* ',sir_location+intern_folder_name+str(i+1)+'/')
        if len(init_atmos_file_name_2) >0:
            cp_file(run_file_folder+'atmos/'+init_atmos_file_name_2,sir_location+intern_folder_name+str(i+1)+'/')

        if clustering =='single' or clustering =='ne':
            cp_file(run_file_folder+'atmos/'+init_atmos_file_name,sir_location+intern_folder_name+str(i+1)+'/')

def create_folder_model(q):
    for i in range(q):
        os.chdir(sir_location)
        make_dir(intern_folder_name+str(i+1))
        os.chdir(intern_folder_name+str(i+1))
        cp_file(run_file_folder+'*', sir_location+intern_folder_name+str(i+1)+'/')
        cp_file(run_file_folder+'*.*', sir_location+intern_folder_name+str(i+1)+'/')

        cp_file(run_file_folder+'input_files/*', sir_location+intern_folder_name+str(i+1)+'/')
        cp_file(run_file_folder+'input_files/*.*', sir_location+intern_folder_name+str(i+1)+'/')


def create_folder_synt(q):
    for i in range(q):
        os.chdir(sir_location)
        make_dir(intern_folder_name+str(i+1))
        os.chdir(intern_folder_name+str(i+1))
        cp_file(run_file_folder+'* ', sir_location+intern_folder_name+str(i+1)+'/')
        cp_file(run_file_folder+'*.* ', sir_location+intern_folder_name+str(i+1)+'/')
        cp_file(run_file_folder+'input_files/* ', sir_location+intern_folder_name+str(i+1)+'/')
        cp_file(run_file_folder+'input_files/*.* ', sir_location+intern_folder_name+str(i+1)+'/')



def convert_profile(ll,data,x,y,wave_leng):
    '''
    Create the .per files

    Parameters
    ----------
    first: array that you want to save
    second : Name of the fits file. You can add the path before name
    '''
    try:
        fil=open(profile_file_name,'w')
        for kk,k in enumerate(ll):
            wheigth_line=wave_leng[k]
            line_indx_fin=str(k)
            prof_data=data[k]
            st=1
            if len(prof_data.shape)==3:st=0
            for n,i in enumerate(wheigth_line):
                if st==1:
                    fil.write(' '+line_indx_fin+'{:>15.3f}'.format(i)+'{:>15.5e}'.format(prof_data[0,n,y,x])+'{:>15.5e}'.format(prof_data[1,n,y,x])+'{:>15.5e}'.format(prof_data[2,n,y,x])+'{:>15.5e}'.format(prof_data[3,n,y,x])+'\n')
                if st==0:
                    fil.write(' '+line_indx_fin+'{:>15.3f}'.format(i)+'{:>15.5e}'.format(prof_data[n,y,x])+'{:>15.5e}'.format(0)+'{:>15.5e}'.format(0)+'{:>15.5e}'.format(0)+'\n')
        fil.close()
    except:
        None


def c_ar_qd(d1,d2,d3,d4):
        arr_base = multiprocessing.Array(ctypes.c_float, (1+d3)*(1+d4)*d2*d1)
        arr = np.ctypeslib.as_array(arr_base.get_obj())
        arr = arr.reshape((d1,d2,d3+1,d4+1))
        return(arr)

def c_ar_td(d1,d2,d3):
        arr_base = multiprocessing.Array(ctypes.c_float, (1+d2)*(1+d3)*d1)
        arr = np.ctypeslib.as_array(arr_base.get_obj())
        arr = arr.reshape((d1,d2+1,d3+1))
        return(arr)

def c_ar_cd(d5,d1,d2,d3,d4):
        arr_base = multiprocessing.Array(ctypes.c_float, d5*(1+d3)*(1+d4)*d2*d1)
        arr = np.ctypeslib.as_array(arr_base.get_obj())
        arr = arr.reshape((d5,d1,d2,d3+1,d4+1))
        return(arr)


def qui2sir(qq,qqq):
    qq[qq<0]=np.nan
    qqq=qqq[~np.isnan(qq)]
    qq=qq[~np.isnan(qq)]
    chi2all =r2_score(qq,qqq)
    return(chi2all)

def invertint(t):
    #invert(t)
    try:
        invert(t)
    except:
        None
    return()

def invert(t):
    pop_val_c=0
    chi2all=0
    mensagem='No convergence'
    folder_name=sir_location+intern_folder_name+str(multiprocessing.current_process()._identity[0])
    start2= time.time()
    k=t[0]
    i=t[1]
    os.chdir(folder_name)
    convert_profile(nindexa,dataa,k,i,wave_leng)
    create_coor_file(disk_pos,k,i)
    oriprof=np.loadtxt(profile_file_name, skiprows=0,unpack=True)
    per_ori[:,:,i-run_range_y_min,k-run_range_x_min]=oriprof[2:,:]
    n_atmos_run=0
    if clustering.isnumeric():
        chi2=10000003.
        chi2p=0.
        chi2ps=10000003.
        chi_lim=-100003.
        uatmos=-1
        for ati in range(int(clustering)):
            if chi_lim<0:
                n_atmos_run=n_atmos_run+1
                cp_file(run_file_folder+'atmos/atmcl'+str(ati+1)+'.mod ',folder_name+'/'+init_atmos_file_name)
                if run_source==0:
                    run_sir=subprocess.Popen('echo '+control_file+' | '+sir_location+'sir.x', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = run_sir.communicate()
                if run_source==1:
                    run_sir=subprocess.Popen('../../bin/desire -v '+control_file, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = run_sir.communicate()
                if os.path.exists(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.per'):
                    fitted_prof=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.per', skiprows=0,unpack=True)
                    try:
                        chi2all=qui2sir(oriprof[2,:],fitted_prof[2,:])
                        if chi2all> clustering_chi:
                            chi_lim=10
                        chi2int=float(open(control_file.split('.')[0]+'.chi','r').readlines()[-1].split()[1])
                        if chi2int==0.0:chi2int=100000

                    except:
                        chi2int=1000000001.
                        chi2intp=1000000001.
                else:
                    chi2int=1000000002.
                    chi2intp=1000000002.
                if chi2int<chi2:
                    chi2=chi2int
                    chi_val=float(open(control_file.split('.')[0]+'.chi','r').readlines()[-1].split()[1])
                    rmdir=subprocess.Popen('rm '+control_file.split('.')[0]+'.chi', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = rmdir.communicate()
                    if chi2all>chi2p:uatmos=ati
                    if chi2all>chi2p:
                        chi2ps=chi_val
                        atmcl[i-run_range_y_min,k-run_range_x_min]=ati+1
                        if output_opt==1 or output_opt==2:
                            mod_file=open(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.mod','r')
                            macro_temp=mod_file.readline()
                            macro_temp=macro_temp.split()
                            inv_res_macro_mod[:,i-run_range_y_min,k-run_range_x_min]=macro_temp
                            if len(init_atmos_file_name_2) >0:
                                inv_res_array_mod_2[:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod_2+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)
                            mod_file.close()
                            inv_res_array_mod[:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)[:,:]
                            inv_res_array_per[:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.per', skiprows=0,unpack=True)[2:,:]
                        if output_opt==2:
                            inv_error_mod[:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.err', skiprows=1,unpack=True)[:,:]
                        if output_opt==3:
                            for ii in range(n_cycles2):
                                mod_file=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.mod', skiprows=1,unpack=True)
                                inv_res_array_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=mod_file
                                if len(init_atmos_file_name_2) >0:
                                    inv_res_array_mod_2[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod_2+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)
                                mod_file_fit=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.per', skiprows=0,unpack=True)
                                inv_res_array_per[ii,:,:,i-run_range_y_min,k-run_range_x_min]=mod_file_fit[2:,:]
                                mod_file3=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.mod', max_rows=1,unpack=True)
                                inv_res_macro_mod[ii,:,i-run_range_y_min,k-run_range_x_min]=mod_file3
                                inv_error_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.err', skiprows=1,unpack=True)[:,:]
                                try:
                                    chi2all=qui2sir(oriprof[2,:],mod_file_fit[2,:])
                                    fullp_chi2[ii,0,i-run_range_y_min,k-run_range_x_min]=chi2all
                                    try:
                                        chi_val=float(open(control_file.split('.')[0]+'.chi','r').readlines()[-1].split()[1])
                                        fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=chi_val
                                    except:
                                        None
                                except:
                                    fullp_chi2[i-run_range_y_min,k-run_range_x_min]=np.nan
                                    inv_res_array_per[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                                    inv_res_array_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                                    if len(init_atmos_file_name_2) >0:
                                        inv_res_array_mod_2[:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                                if nlte_f<10:
                                    try:
                                        pop_val_c=float([s for s in out.decode("utf-8").split('\n') if "delta" in s][-1].split('=')[1].replace('(accelerated)',''))
                                        fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=pop_val_c
                                    except:
                                        None
                                if output_opt==3:
                                    if chi2all>chi2p:chi2p=chi2all
                                    fullp_chi2[ii,0,i-run_range_y_min,k-run_range_x_min]=chi2p
                                    fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=chi2ps
                    if output_opt==1 or output_opt==2:
                        if chi2all>chi2p:chi2p=chi2all
                        fullp_chi2[0,i-run_range_y_min,k-run_range_x_min]=chi2p
                        fullp_chi2[1,i-run_range_y_min,k-run_range_x_min]=chi2ps

                mensagem='R2='+'{:>2.5f}'.format(chi2p)+'  with atmos  '+'{:>2.0f}'.format(uatmos+1)
                #rmdir=subprocess.Popen('rm '+init_atmos_file_name_mo_mod+'_'+n_cycles3+'.per', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #out, err = rmdir.communicate()
        if debug==1:
            make_dir(dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
            cp_file('*',dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))

    else:
        if clustering =='yes':
            clust_ind=int(clust_ind_arr[k,i])
            call('cp '+run_file_folder+'atmos/atmcl'+str(clust_ind)+'.mod '+folder_name+'/'+init_atmos_file_name, shell=True)
        if clustering =='atmos':
            np.savetxt(init_atmos_file_name,ml_atmos[:,:,i,k].transpose(),fmt='%1.5e', delimiter=' ', newline='\n', header=atmos_header[0],comments='')
        if clustering =='nn':
            np.savetxt(init_atmos_file_name,ml_atmos[:,:,i,k].transpose(),fmt='%1.5e', delimiter=' ', newline='\n', header=atmos_header[0],comments='')

        if run_source==0:
            run_sir=subprocess.Popen('echo '+control_file+' | '+sir_location+'sir.x', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = run_sir.communicate()
        if run_source==1:
            run_sir=subprocess.Popen('../../bin/desire -v '+control_file, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = run_sir.communicate()
        #if k==2 and i==2: rm_file(init_atmos_file_name_mo_mod+'_'+n_cycles+'.per')
        if os.path.exists(init_atmos_file_name_mo_mod+'_'+n_cycles+'.per'):
            if output_opt==1 or output_opt==2:
                mod_file=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)
                inv_res_array_mod[:,:,i-run_range_y_min,k-run_range_x_min]=mod_file
                if len(init_atmos_file_name_2) >0:
                    inv_res_array_mod_2[:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod_2+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)
                mod_file_fit=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.per', skiprows=0,unpack=True)
                inv_res_array_per[:,:,i-run_range_y_min,k-run_range_x_min]=mod_file_fit[2:,:]
                mod_file3=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.mod', max_rows=1,unpack=True)
                inv_res_macro_mod[:,i-run_range_y_min,k-run_range_x_min]=mod_file3
                try:
                    chi2all=qui2sir(oriprof[2,:],mod_file_fit[2,:])
                    fullp_chi2[0,i-run_range_y_min,k-run_range_x_min]=chi2all
                    try:
                        chi_val=float(open(control_file.split('.')[0]+'.chi','r').readlines()[-1].split()[1])
                        fullp_chi2[1,i-run_range_y_min,k-run_range_x_min]=chi_val
                    except:
                        None
                except:
                    fullp_chi2[:,i-run_range_y_min,k-run_range_x_min]=np.nan
                    inv_res_array_per[:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                    inv_res_array_mod[:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                if nlte_f<10:
                    try:
                        pop_val_c=float([s for s in out.decode("utf-8").split('\n') if "delta" in s][-1].split('=')[1].replace('(accelerated)',''))
                        fullp_chi2[1,i-run_range_y_min,k-run_range_x_min]=pop_val_c
                    except:
                        None

            if output_opt==2:
                inv_error_mod[:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.err', skiprows=1,unpack=True)
            if output_opt==3:
                for ii in range(n_cycles2):
                    mod_file=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.mod', skiprows=1,unpack=True)
                    inv_res_array_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=mod_file
                    if len(init_atmos_file_name_2) >0:
                        inv_res_array_mod_2[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod_2+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)

                    mod_file_fit=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.per', skiprows=0,unpack=True)
                    inv_res_array_per[ii,:,:,i-run_range_y_min,k-run_range_x_min]=mod_file_fit[2:,:]
                    mod_file3=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.mod', max_rows=1,unpack=True)
                    inv_res_macro_mod[ii,:,i-run_range_y_min,k-run_range_x_min]=mod_file3
                    inv_error_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.err', skiprows=1,unpack=True)[:,:]
                    try:
                        chi2all=qui2sir(oriprof[2,:],mod_file_fit[2,:])
                        fullp_chi2[ii,0,i-run_range_y_min,k-run_range_x_min]=chi2all
                        try:
                            chi_val=float(open(control_file.split('.')[0]+'.chi','r').readlines()[-1].split()[1])
                            fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=chi_val
                        except:
                            None
                    except:
                        fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=np.nan
                        inv_res_array_per[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                        inv_res_array_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                    if nlte_f<10:
                        try:
                            pop_val_c=float([s for s in out.decode("utf-8").split('\n') if "delta" in s][-1].split('=')[1].replace('(accelerated)',''))
                            fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=pop_val_c
                        except:
                            None
            if run_source==0:
                mensagem='R2='+'{:>2.5f}'.format(chi2all)
            if run_source==1:
                if nlte_f<10:mensagem='R2='+'{:>2.3f}'.format(chi2all)+'  rh_delta='+'{:>2.1e}'.format(pop_val_c)
                if nlte_f>=10:mensagem='R2='+'{:>2.3f}'.format(chi2all)
        elif clustering =='ne':
            crt=0
            for nex in [-1,0]:
                for ney in [-1,0]:
                    if crt==0:
                        if output_opt==1 or output_opt==2:
                            if inv_res_array_mod[0,0,i+ney,k+nex]>0:
                                new_atmos=np.zeros((11,num_lines_mod))
                                new_atmos[:,:]=inv_res_array_mod[:,:,i+ney,k+nex]
                                crt=1
                        if output_opt==3:
                            if inv_res_array_mod[-1,0,0,i+ney,k+nex]>0:
                                new_atmos=np.zeros((11,num_lines_mod))
                                new_atmos[:,:]=inv_res_array_mod[-1,:,:,i+ney,k+nex]
                                crt=1
            np.savetxt(init_atmos_file_name,new_atmos.transpose(),fmt='%1.5e', delimiter=' ', newline='\n', header=atmos_header[0],comments='')
            if run_source==0:
                run_sir=subprocess.Popen('echo '+control_file+' | '+sir_location+'sir.x', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = run_sir.communicate()
            if run_source==1:
                run_sir=subprocess.Popen('../../bin/desire -v '+control_file, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = run_sir.communicate()
            if k==2 and i==2: rm_file(init_atmos_file_name_mo_mod+'_'+n_cycles+'.per')
            if os.path.exists(init_atmos_file_name_mo_mod+'_'+n_cycles+'.per'):
                if output_opt==1 or output_opt==2:
                    mod_file=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)
                    inv_res_array_mod[:,:,i-run_range_y_min,k-run_range_x_min]=mod_file
                    if len(init_atmos_file_name_2) >0:
                        inv_res_array_mod_2[:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod_2+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)
                    mod_file_fit=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.per', skiprows=0,unpack=True)
                    inv_res_array_per[:,:,i-run_range_y_min,k-run_range_x_min]=mod_file_fit[2:,:]
                    mod_file3=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.mod', max_rows=1,unpack=True)
                    inv_res_macro_mod[:,i-run_range_y_min,k-run_range_x_min]=mod_file3
                    try:
                        chi2all=qui2sir(oriprof[2,:],mod_file_fit[2,:])
                        fullp_chi2[0,i-run_range_y_min,k-run_range_x_min]=chi2all
                        try:
                            chi_val=float(open(control_file.split('.')[0]+'.chi','r').readlines()[-1].split()[1])
                            fullp_chi2[1,i-run_range_y_min,k-run_range_x_min]=chi_val
                        except:
                            None
                    except:
                        fullp_chi2[:,i-run_range_y_min,k-run_range_x_min]=np.nan
                        inv_res_array_per[:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                        inv_res_array_mod[:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                    if nlte_f<10:
                        try:
                            pop_val_c=float([s for s in out.decode("utf-8").split('\n') if "delta" in s][-1].split('=')[1].replace('(accelerated)',''))
                            fullp_chi2[1,i-run_range_y_min,k-run_range_x_min]=pop_val_c
                        except:
                            None

                if output_opt==2:
                    inv_error_mod[:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.err', skiprows=1,unpack=True)
                if output_opt==3:
                    for ii in range(n_cycles2):
                        mod_file=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.mod', skiprows=1,unpack=True)
                        inv_res_array_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=mod_file
                        if len(init_atmos_file_name_2) >0:
                            inv_res_array_mod_2[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod_2+'_'+n_cycles3+'.mod', skiprows=1,unpack=True)

                        mod_file_fit=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.per', skiprows=0,unpack=True)
                        inv_res_array_per[ii,:,:,i-run_range_y_min,k-run_range_x_min]=mod_file_fit[2:,:]
                        mod_file3=np.loadtxt(init_atmos_file_name_mo_mod+'_'+str(ii+1)+'.mod', max_rows=1,unpack=True)
                        inv_res_macro_mod[ii,:,i-run_range_y_min,k-run_range_x_min]=mod_file3
                        inv_error_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.loadtxt(init_atmos_file_name_mo_mod+'_'+n_cycles3+'.err', skiprows=1,unpack=True)[:,:]
                        try:
                            chi2all=qui2sir(oriprof[2,:],mod_file_fit[2,:])
                            fullp_chi2[ii,0,i-run_range_y_min,k-run_range_x_min]=chi2all
                            try:
                                chi_val=float(open(control_file.split('.')[0]+'.chi','r').readlines()[-1].split()[1])
                                fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=chi_val
                            except:
                                None
                        except:
                            fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=np.nan
                            inv_res_array_per[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                            inv_res_array_mod[ii,:,:,i-run_range_y_min,k-run_range_x_min]=np.nan
                        if nlte_f<10:
                            try:
                                pop_val_c=float([s for s in out.decode("utf-8").split('\n') if "delta" in s][-1].split('=')[1].replace('(accelerated)',''))
                                fullp_chi2[ii,1,i-run_range_y_min,k-run_range_x_min]=pop_val_c
                            except:
                                None
                if run_source==0:
                    mensagem='R2='+'{:>2.5f}'.format(chi2all)
                if run_source==1:
                    if nlte_f<10:mensagem='R2='+'{:>2.3f}'.format(chi2all)+'  rh_delta='+'{:>2.1e}'.format(pop_val_c)
                    if nlte_f>=10:mensagem='R2='+'{:>2.3f}'.format(chi2all)













        else:
            mensagem='No convergence'
            file = open(dir_path+'/results/errors/pixel_'+str(k)+'_'+str(i)+'.txt', 'w')
            file.write(out.decode("utf-8"))
            file.write('\n')
            file.write('----------------------------------------------------------')
            file.write('\n')
            file.write(err.decode("utf-8"))
            file.close()
        if debug==1:
            make_dir(dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
            cp_file('*',dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
    return()





def sint(t):
    folder_name=sir_location+intern_folder_name+str(multiprocessing.current_process()._identity[0])
    os.chdir(folder_name)
    start2= time.time()
    k=t[0]
    i=t[1]
    create_coor_file(disk_pos,k,i)
    np.savetxt(init_atmos_file_name,dataa[nindexa[0]][:,:,i,k].transpose(),fmt='%1.5e', delimiter=' ', newline='\n', header=atmos_header[0],comments='')
    if run_source==0:
        run_sir=subprocess.Popen('echo '+control_file+' | '+sir_location+'sir.x', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = run_sir.communicate()
    if run_source==1:
        run_sir=subprocess.Popen('../../bin/desire -v '+control_file, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = run_sir.communicate()
    if os.path.exists(profile_file_name):
        mod_file_fit=np.loadtxt(profile_file_name, skiprows=0,unpack=True)
        inv_res_array_per[:,:,i-run_range_y_min,k-run_range_x_min]=mod_file_fit[2:,:]
    else:
        None
    if os.path.exists(profile_file_name):
        mensagem='Converged'
    else:
        mensagem='No convergence'
        file = open(dir_path+'/results/errors/pixel_'+str(i)+'_'+str(k)+'.txt', 'w')
        file.write(out.decode("utf-8"))
        file.write('\n')
        file.write('----------------------------------------------------------')
        file.write('\n')
        file.write(err.decode("utf-8"))
        file.close()
    if debug==1:
        make_dir(dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
        cp_file('*',dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
    rm_file(init_atmos_file_name)
    return()





def rffp(t):
    try:
        os.chdir(sir_location)
        start2= time.time()
        k=t[0]
        i=t[1]
        make_dir('dr_'+str(k)+'_'+str(i))
        os.chdir('dr_'+str(k)+'_'+str(i))
        cp_file(run_file_folder+'*.* ',sir_location+'dr_'+str(k)+'_'+str(i)+'/')
        cp_file(run_file_folder+'* ',sir_location+'dr_'+str(k)+'_'+str(i)+'/')

        cp_file(run_file_folder+'input_files/* ',sir_location+'dr_'+str(k)+'_'+str(i)+'/')
        shp = subprocess.Popen('rm '+init_atmos_file_name, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outs, errs = shp.communicate()
        np.savetxt(init_atmos_file_name,dataa[nindexa[0]][:,:,i,k].transpose(),fmt='%1.3e', delimiter=' ', newline='\n', header=atmos_header[0],comments='')
        run_sir=subprocess.Popen('echo inver.trol | '+sir_location+'sir.x', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = run_sir.communicate()

        if os.path.exists(glob.glob(init_atmos_file_name_mo_mod+'.r*')[0]):
            mod_file2=open(glob.glob(init_atmos_file_name_mo_mod+'.r*')[0],'r')
            kl=0
            nada=mod_file2.readline()
            param_rf=nada.split()
            mod_file2.close()
            rf_temp=np.loadtxt(glob.glob(init_atmos_file_name_mo_mod+'.r*')[0],skiprows=1)
            if num_lines_per==int(param_rf[1]):
                rf2=rf_temp.reshape(int(param_rf[0]),int(int(param_rf[1])),order='C')
                rf[0,:,:,i-run_range_y_min,k-run_range_x_min]=rf2[:,:]
            else:
                rf2=rf_temp.reshape(int(param_rf[0]),int(int(param_rf[1])),order='C')
                rf[0,:,:,i-run_range_y_min,k-run_range_x_min]=rf2[:,:num_lines_per]
                rf[1,:,:,i-run_range_y_min,k-run_range_x_min]=rf2[:,num_lines_per*1:num_lines_per*2]
                rf[2,:,:,i-run_range_y_min,k-run_range_x_min]=rf2[:,num_lines_per*2:num_lines_per*3]
                rf[3,:,:,i-run_range_y_min,k-run_range_x_min]=rf2[:,num_lines_per*3:num_lines_per*4]
        else:
            None

        if os.path.exists(glob.glob(init_atmos_file_name_mo_mod+'.r*')[0]):
            mensagem='Converged'
        else:
            mensagem='No convergence'
            file = open(dir_path+'/results/errors/pixel_'+str(k)+'_'+str(i)+'.txt', 'w')
            file.write(out.decode("utf-8"))
            file.write('\n')
            file.write('----------------------------------------------------------')
            file.write('\n')
            file.write(err.decode("utf-8"))
            file.close()
    except:
        None
    if debug==1:
        make_dir(dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
        cp_file('*',dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
    call('rm -r ../dr_'+str(k)+'_'+str(i), shell=True)
    return()


























def modelador(t):

    folder_name=sir_location+intern_folder_name+str(multiprocessing.current_process()._identity[0])
    os.chdir(folder_name)

    xx=t[0]
    yy=t[1]
    start2= time.time()
    np.savetxt('input.mod',dataa[:,:,yy,xx].T,fmt='%1.5e', delimiter=' ', newline='\n', header=atmos_header[0],comments='')
    run_sir=subprocess.Popen('echo tau.mtrol | '+sir_location+'modelador*.x', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    run_sir.wait()
    if os.path.exists('output.mod'):
        mod_file=open('output.mod','r')
        mod_file.readline()
        kl=0
        for tline in mod_file:
            inv_res_array_mod[:,kl,yy-run_range_y_min,xx-run_range_x_min]=tline.split()
            kl=kl+1
        mod_file.close()
    if debug==1:
        make_dir(dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
        cp_file('*',dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
    return()


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



def frh(t):
    os.chdir(rh_location)
    start2= time.time()
    k=t[0]
    i=t[1]
    call('mkdir prh_'+str(k)+'_'+str(i), shell=True)
    os.chdir('prh_'+str(k)+'_'+str(i))
    call('cp '+run_file_folder+'*.* '+rh_location+'prh_'+str(k)+'_'+str(i)+'/', shell=True)
    if os.path.exists(run_file_folder+'input_files/'):
        call('cp '+run_file_folder+'input_files/* '+rh_location+'prh_'+str(k)+'_'+str(i)+'/', shell=True)
    save_atmos(atmos_dat,atmos_points,k,i,depth_s_t)
    run_rh=subprocess.Popen('../rhf1d', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = run_rh.communicate()
    run_solveray=subprocess.Popen('../solveray', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = run_solveray.communicate()
    if debug==1:
        make_dir(dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
        cp_file('*',dir_path+'/results/run_folders/pixel_'+str(k)+'_'+str(i))
    if os.path.exists('spectrum.out'):
        rhd1 = rhout()
        if n_p_pol==2:
            rh_spect[0,:,i-run_range_y_min,k-run_range_x_min]=rhd1.rays[0].I
        if n_p_pol==5:
            rh_spect[0,:,i-run_range_y_min,k-run_range_x_min]=rhd1.rays[0].I
            rh_spect[1,:,i-run_range_y_min,k-run_range_x_min]=rhd1.rays[0].Q
            rh_spect[2,:,i-run_range_y_min,k-run_range_x_min]=rhd1.rays[0].U
            rh_spect[3,:,i-run_range_y_min,k-run_range_x_min]=rhd1.rays[0].V
        if output_t==2:
            rh_geo[0,:,i-run_range_y_min,k-run_range_x_min]=np.log10(rhd1.geometry.tau500)
            rh_geo[1,:,i-run_range_y_min,k-run_range_x_min]=rhd1.geometry.height/1.0E3
            rh_geo[2,:,i-run_range_y_min,k-run_range_x_min]=rhd1.geometry.cmass

    if os.path.exists('spectrum.out'):
        message='Converged'
    else:
        message='Not converging'
    os.chdir(rh_location)
    dellf=subprocess.Popen('rm -r prh_'+str(k)+'_'+str(i), shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = dellf.communicate()
    return()

def save_atmos(dataa,nump,x,y,depth_s_t):
    '''
    Create the .atmos file

    Parameters
    ----------
    first: data x position
    second : number of atmospheric points
    '''
    fatmos=open('model.atmos','w')
    if depth_s_t=='m':
        fatmos.write('atmos.mod\n')
        fatmos.write('*\n')
        fatmos.write('Mass scale\n')
        fatmos.write('   4.437970\n')
        fatmos.write('          '+str(nump)+'\n')
        for i in range(nump):
            fatmos.write('{:>15.5e}'.format(dataa[i,0,y,x])+'{:>15.5e}'.format(dataa[i,1,y,x])+'{:>15.5e}'.format(dataa[i,2,y,x])+'{:>15.5e}'.format(dataa[i,3,y,x])+'{:>15.5e}'.format(dataa[i,4,y,x])+'\n')
        fatmos.close()

    if depth_s_t=='h':
        fatmos.write('atmos.mod\n')
        fatmos.write('*\n')
        fatmos.write('Height\n')
        fatmos.write('   4.437970\n')
        fatmos.write('          '+str(nump)+'\n')
        for i in range(nump):
            fatmos.write('{:>15.5e}'.format(dataa[i,0,y,x])+'{:>15.5e}'.format(dataa[i,1,y,x])+'{:>15.5e}'.format(dataa[i,2,y,x])+'{:>15.5e}'.format(dataa[i,3,y,x])+'{:>15.5e}'.format(dataa[i,4,y,x])+'\n')
        for i in range(nump):
            fatmos.write('{:>15.5e}'.format(dataa[i,8,y,x])+'{:>15.5e}'.format(dataa[i,9,y,x])+'{:>15.5e}'.format(dataa[i,10,y,x])+'{:>15.5e}'.format(dataa[i,11,y,x])+'{:>15.5e}'.format(dataa[i,12,y,x])+'{:>15.5e}'.format(dataa[i,13,y,x])+'\n')

        fatmos.close()

    if depth_s_t=='t':
        fatmos.write('atmos.mod\n')
        fatmos.write('*\n')
        fatmos.write('Tau scale\n')
        fatmos.write('   4.437970\n')
        fatmos.write('       '+str(nump)+'\n')
        for i in range(nump):
            fatmos.write('{:>15.5e}'.format(dataa[i,0,y,x])+'{:>15.5e}'.format(dataa[i,1,y,x])+'{:>15.5e}'.format(dataa[i,2,y,x])+'{:>15.5e}'.format(dataa[i,3,x,y])+'{:>15.5e}'.format(dataa[i,4,y,x])+'\n')
        for i in range(nump):
            fatmos.write('{:>15.5e}'.format(dataa[i,8,y,x])+'{:>15.5e}'.format(dataa[i,9,y,x])+'{:>15.5e}'.format(dataa[i,10,y,x])+'{:>15.5e}'.format(dataa[i,11,y,x])+'{:>15.5e}'.format(dataa[i,12,y,x])+'{:>15.5e}'.format(dataa[i,13,y,x])+'\n')
    if len(dataa[0,:,0,0])>6:
#        for i in range(nump):
#            fatmag.write('{:>15.5e}'.format(dataa[i,5,x,y])+'{:>15.5e}'.format(dataa[i,6,x,y])+'{:>15.5e}'.format(dataa[i,7,x,y])+'\n')
        field_xdr=np.zeros(nump*3)

        for i in range(nump):
            field_xdr[i]=dataa[i,5,y,x]
            field_xdr[i+nump]=dataa[i,6,y,x]
            field_xdr[i+2*nump]=dataa[i,7,y,x]


        pout = xdrlib.Packer()
        pout.pack_farray(atmos_points*3,field_xdr,pout.pack_double)

        ff=open('field.B','wb')
        ff.write(pout.get_buffer())
        ff.close()

def change_keyword(n):
    if n>0:
        file_ke=open('run_files/keyword.input','r')
        Input_file_temp=file_ke.readlines()
        file_ke.close()
        Input_file_kk=[]

        for i in range(len(Input_file_temp)):
            if Input_file_temp[i].find('#')<0:
                 Input_file_kk.append(Input_file_temp[i].replace("\n",''))

        for j in range(len(Input_file_kk)):
            if Input_file_kk[j].find('XDR_ENDIAN') >=0:
                Input_file_kk[j]='XDR_ENDIAN = TRUE'

        for j in range(len(Input_file_kk)):
            if Input_file_kk[j].find('STOKES_INPUT') >=0:
                Input_file_kk[j]='STOKES_INPUT = field.B'

        with open('run_files/keyword.input', 'w') as fk:
            for itemke in Input_file_kk:
                fk.write("%s\n" % itemke)
    return()


def vac_air_wave(wav):
    wav2=wav*wav*1.0
    nwave=wav
    if wav >=200:
        fact = 1.0 + 2.735182e-4 + (1.314182e0 + 2.76249e+4/wav2) / wav2
        fact2 = wav/fact
        return(fact2)
    else:
        return(nwave)




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#Han python scripts to read RH output


class rhout:

    def __init__(self, rhdir='.'):

        self.rhdir = rhdir
        self.inputs = inputs('{0}/input.out'.format(rhdir))

        self.geometry = geometry('{0}/geometry.out'.format(rhdir))
        self.atmos = atmos(self.geometry, '{0}/atmos.out'.format(rhdir))

        self.spectrum = spectrum(self.inputs,\
                                            self.geometry,\
                                            self.atmos,\
                                            '{0}/spectrum.out'.format(rhdir))

        self.rays = {}
        Nray = 0
        for file in os.listdir(rhdir):
            if fnmatch.fnmatch(file, 'spectrum_?.??*'):
                self.rays[Nray] =\
                    rays(self.inputs,\
                                    self.geometry,\
                                    self.spectrum,\
                                    filename=('{0}/'+file).format(rhdir))
                Nray += 1
        self.Nray = Nray

        self.atoms = {}
        Natom = 0
        for file in os.listdir(rhdir):
            if fnmatch.fnmatch(file, 'atom.*.out'):
                self.atoms[Natom] =\
                    atoms(self.geometry,\
                                path=('{0}/'+file).format(rhdir))
                Natom += 1
        self.Natom = Natom


class inputs:

    def __init__(self, filename='input.out'):
        self.filename = filename
        self.read()

    def read(self):

        stokes_mode = ["NO_STOKES", "FIELD_FREE", "FULL_STOKES"]

        f  = open(self.filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        self.magneto_optical = up.unpack_int()
        self.PRD_angle_dep   = up.unpack_int()
        self.XRD             = up.unpack_int()
        self.start_solution  = up.unpack_int()

        sm = up.unpack_int()
        self.stokes_mode     = stokes_mode[sm]

        self.metallicity     = float(up.unpack_double())
        self.backgr_pol      = up.unpack_int()
        self.big_endian      = up.unpack_int()

        up.done()

class atoms:

    def __init__(self, geometry, path='./atom.H.out'):
        self.filename = path
        self.read(geometry)

        (dirname, filename) = os.path.split(path)

        popsfile = '{0}/pops.{1}.out'.format(dirname, self.atomID)
        if os.path.exists(popsfile):
            self.readpops(geometry, popsfile)

        ratesfile = '{0}/radrate.{1}.out'.format(dirname, self.atomID)
        if os.path.exists(ratesfile):
            self.readrates(geometry, ratesfile)

        dampingfile = '{0}/damping.{1}.out'.format(dirname, self.atomID)
        if os.path.exists(dampingfile):
            self.readdamping(geometry, dampingfile)

        collisionfile = '{0}/collrate.{1}.out'.format(dirname, self.atomID)
        if os.path.exists(collisionfile):
            self.readcollisions(geometry, collisionfile)

    def read(self, geometry):

        f  = open(self.filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        self.active = up.unpack_int()
        self.Nlevel = up.unpack_int()
        self.Nline  = up.unpack_int()
        self.Ncont  = up.unpack_int()
        self.Nfixed = up.unpack_int()

        self.abund  = up.unpack_double()
        self.weight = up.unpack_double()

        self.labels = {}
        for i in range(self.Nlevel):
            self.labels[i] = read_string(up)

        self.atomID = self.labels[0][0:2].strip()

        self.g     = read_farray([self.Nlevel], up, "double")
        self.E     = read_farray([self.Nlevel], up, "double")
        self.stage = read_farray([self.Nlevel], up, "int")

        Nrad = self.Nline + self.Ncont

        self.transition = {}
        for kr in range(Nrad):
            self.transition[kr] = atoms.transition(up)

        for kr in range(self.Nline, Nrad):
            if self.transition[kr].shape == "HYDROGENIC":
                self.transition[kr].waves = up.unpack_double()
            else:
                self.transition[kr].waves =\
                read_farray([self.transition[kr].Nwave], up, "double")
                self.transition[kr].alpha =\
                read_farray([self.transition[kr].Nwave], up, "double")

        self.fixed = {}
        for kr in range(self.Nfixed):
            self.transition[kr] = atoms.fixed(up)

        up.done()


    def readpops(self, geometry, popsfile):

        f  = open(popsfile, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        atmosID     = read_string(up)
        Nlevel      = up.unpack_int()
        Nspace      = up.unpack_int()

        if geometry.type == "ONE_D_PLANE":
            dim = [geometry.Ndep, self.Nlevel]
        elif geometry.type == 'SPHERICAL_SYMMETRIC':
            dim = [geometry.Nradius, self.Nlevel]
        elif geometry.type == 'TWO_D_PLANE':
            dim = [geometry.Nx, geometry.Nz, self.Nlevel]
        elif geometry.type == 'THREE_D_PLANE':
            dim = [geometry.Nx, geometry.Ny, geometry.Nz, self.Nlevel]

        self.n     = read_farray(dim, up, "double")
        self.nstar = read_farray(dim, up, "double")

        up.done()

    def readrates(self, geometry, filename):

        f  = open(filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        Nrad = self.Nline + self.Ncont

        if geometry.type == "ONE_D_PLANE":
            dim = [geometry.Ndep]
        elif geometry.type == 'SPHERICAL_SYMMETRIC':
            dim = [geometry.Nradius]
        elif geometry.type == 'TWO_D_PLANE':
            dim = [geometry.Nx, geometry.Nz]
        elif geometry.type == 'THREE_D_PLANE':
            dim = [geometry.Nx, geometry.Ny, geometry.Nz]

        for kr in range(Nrad):
            self.transition[kr].Rij = read_farray(dim, up, "double")
            self.transition[kr].Rji = read_farray(dim, up, "double")
        up.done()

    def readdamping(self, geometry, filename):

        f  = open(filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        if geometry.type == "ONE_D_PLANE":
            dim = [geometry.Ndep]
        elif geometry.type == 'SPHERICAL_SYMMETRIC':
            dim = [geometry.Nradius]
        elif geometry.type == 'TWO_D_PLANE':
            dim = [geometry.Nx, geometry.Nz]
        elif geometry.type == 'THREE_D_PLANE':
            dim = [geometry.Nx, geometry.Ny, geometry.Nz]

        self.vbroad = read_farray(dim, up, "double")

        for kr in range(self.Nline):
            self.transition[kr].adamp =\
                read_farray(dim, up, "double")
        up.done()

    def readcollisions(self, geometry, filename):

        f  = open(filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        if geometry.type == "ONE_D_PLANE":
            dim = [geometry.Ndep, self.Nlevel, self.Nlevel]
        elif geometry.type == 'SPHERICAL_SYMMETRIC':
            dim = [geometry.Nradius, self.Nlevel, self.Nlevel]
        elif geometry.type == 'TWO_D_PLANE':
            dim = [geometry.Nx, geometry.Nz, self.Nlevel, self.Nlevel]
        elif geometry.type == 'THREE_D_PLANE':
            dim = [geometry.Nx, geometry.Ny, geometry.Nz,\
                   self.Nlevel, self.Nlevel]

        self.Cij = read_farray(dim, up, "double")

        up.done()

    class transition:

        def __init__(self, up):
            self.read(up)

        def read(self, up):

            shapes = {0:"GAUSS", 1: "VOIGT", 2: "PRD",\
                      3: "HYDROGENIC", 4: "EXPLICIT"}

            types  = {0: "ATOMIC_LINE", 1: "ATOMIC_CONTINUUM"}

            self.type     = types[up.unpack_int()]
            self.i        = up.unpack_int()
            self.j        = up.unpack_int()
            self.Nwave    = up.unpack_int()
            self.Nblue    = up.unpack_int()
            self.lambda0  = up.unpack_double()
            self.shape    = shapes[up.unpack_int()]
            self.strength = up.unpack_double()


    class fixed:
        def __init__(self, up):
            self.read(up)

        def read(self, up):
            self.type     = up.unpack_int()
            self.option   = up.unpack_int()
            self.i        = up.unpack_int()
            self.j        = up.unpack_int()
            self.lambda0  = up.unpack_double()
            self.strength = up.unpack_double()
            self.Trad     = up.unpack_double()

class element:

    def __init__(self, up):
        self.read_element(up)

    def read_element(self, up):
        self.ID     = read_string(up)
        self.weight = float(up.unpack_double())
        self.abund  = float(up.unpack_double())

class atmos:

    def __init__(self, geometry, filename='atmos.out'):
        self.filename = filename
        self.read(geometry)

    def read(self, geometry):

        f  = open(self.filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        self.NHydr  = up.unpack_int()
        self.Nelem  = up.unpack_int()
        self.moving = up.unpack_int()

        if geometry.type == "ONE_D_PLANE":
            dim1 = [geometry.Ndep]
            dim2 = [geometry.Ndep, self.NHydr]

        elif geometry.type == 'SPHERICAL_SYMMETRIC':
            dim1 = [geometry.Nradius]
            dim2 = [geometry.Nradius, self.NHydr]

        elif geometry.type == 'TWO_D_PLANE':
            dim1 = [geometry.Nx, geometry.Nz]
            dim2 = [geometry.Nx, geometry.Nz, self.NHydr]

        elif geometry.type == 'THREE_D_PLANE':
            dim1 = [geometry.Nx, geometry.Ny, geometry.Nz]
            dim2 = [geometry.Nx, geometry.Ny, geometry.Nz, self.NHydr]


        self.T      = read_farray(dim1, up, "double")
        self.n_elec = read_farray(dim1, up, "double")
        self.vturb  = read_farray(dim1, up, "double")

        self.nH = read_farray(dim2, up, "double")
        self.ID = read_string(up)


        self.elements = {}
        for n in range(self.Nelem):
            self.elements[n] = element(up)

        if geometry.type != 'SPHERICAL_SYMMETRIC':
            try:
                stokes = up.unpack_int()
            except EOFError or IOError:
                self.stokes = False
                return
            else:
                self.stokes = True

                self.B       = read_farray(dim1, up, "double")
                self.gamma_B = read_farray(dim1, up, "double")
                self.chi_B   = read_farray(dim1, up, "double")

        up.done()

class input_atmos:
    def __init__(self, geometrytype, atmosfile, Bfile=None):
        self.read(geometrytype, atmosfile, Bfile)

    def read(self, geometrytype, atmosfile, Bfile):
        self.type = geometrytype

        if self.type == "ONE_D_PLANE" or self.type == "SPHERICAL_SYMMETRIC":

            CM_TO_M = 1.0E-2
            G_TO_KG = 1.0E-3

            data = []
            with open(atmosfile, 'r') as file:
                for line in file:
                    if line.startswith('*'):
                        continue
                    data.append(line.strip())

            self.ID    = data[0]
            scale      = data[1][0]

            if self.type == "ONE_D_PLANE":
                self.grav = float(data[2])
                self.Ndep = int(data[3])
                Nd = self.Ndep
            else:
                self.grav, self.radius = [float(x) for x in data[2].split()]
                self.Nradius, self.Ncore, self.Ninter = \
                    [int(x) for x in data[3].split()]
                Nd = self.Nradius

            self.grav  = np.power(10.0, self.grav) * CM_TO_M
            self.NHydr = 6

            hscale      = np.array(range(Nd), dtype="float")
            self.T      = np.array(range(Nd), dtype="float")
            self.n_elec = np.array(range(Nd), dtype="float")
            self.v      = np.array(range(Nd), dtype="float")
            self.vturb  = np.array(range(Nd), dtype="float")

            for n in range(Nd):
                hscale[n], self.T[n],\
                    self.n_elec[n], self.v[n], self.vturb[n] =\
                        [float(x) for x in data[n+4].split()]

            if scale == 'M':
                self.scale  = 'MASS_SCALE'
                self.cmass  = np.power(10.0, hscale)
                self.cmass *= G_TO_KG / CM_TO_M**2

            elif scale == 'T':
                self.scale  = 'TAU500_SCALE'
                self.tau500 = np.power(10.0, hscale)
            elif scale == 'H':
                self.scale  = 'TAU500_SCALE'
                self.height = hscale

            if len(data) > (4 + Nd):
                self.HLTE = False
                self.nH = np.array(range(Nd * self.NHydr),\
                                   dtype="float").reshape([Nd,\
                                                           self.NHydr],\
                                                          order='F')
                for n in range(Nd):
                    self.nH[n,:] =\
                        [float(x) for x in data[n+4+Nd].split()]
            else:
                self.HLTE = True

            self.nH     /= CM_TO_M**3
            self.n_elec /= CM_TO_M**3

            dim1 = [Nd]

        elif self.type == "TWO_D_PLANE" or self.type == "THREE_D_PLANE":

            f = open(atmosfile, 'rb')
            up = xdrlib.Unpacker(f.read())
            f.close()

            if self.type == "TWO_D_PLANE":
                self.Nx = up.unpack_int()
                self.Nz = up.unpack_int()
                self.NHydr = up.unpack_int()

                self.boundary = read_farray([3], up, "int")

                self.dx = read_farray([self.Nx], up, "double")
                self.z  = read_farray([self.Nz], up, "double")

                dim1 = [self.Nx, self.Nz]
                dim2 = [self.Nx, self.Nz, self.NHydr]

            elif self.type == "THREE_D_PLANE":
                self.Nx = up.unpack_int()
                self.Ny = up.unpack_int()
                self.Nz = up.unpack_int()
                self.NHydr = up.unpack_int()

                self.boundary = read_farray([2], up, "int")

                self.dx = up.unpack_double()
                self.dy = up.unpack_double()
                self.z  = read_farray([self.Nz], up, "double")

                dim1 = [self.Nx, self.Ny, self.Nz]
                dim2 = [self.Nx, self.Ny, self.Nz, self.NHydr]

            self.T      = read_farray(dim1, up, "double")
            self.n_elec = read_farray(dim1, up, "double")
            self.vturb  = read_farray(dim1, up, "double")
            self.vx     = read_farray(dim1, up, "double")

            if self.type == "THREE_D_PLANE":
                self.vy     = read_farray(dim1, up, "double")

            self.vz     = read_farray(dim1, up, "double")

            self.nH     = read_farray(dim2, up, "double")

            up.done()

        else:
            print("Not a valid input atmosphere type: {0}".format(self.type))
            return

        if Bfile != None:

            f = open(Bfile, 'rb')
            up = xdrlib.Unpacker(f.read())
            f.close()

            self.B     = read_farray(dim1, up, "double")
            self.gamma = read_farray(dim1, up, "double")
            self.chi   = read_farray(dim1, up, "double")

            up.done()

    def write(self, outfile, Bfile=None):

        if self.type == "ONE_D_PLANE" or self.type == "SPHERICAL_SYMMETRIC":

            CM_TO_M = 1.0E-2
            G_TO_KG = 1.0E-3

            self.nH     *= CM_TO_M**3
            self.n_elec *= CM_TO_M**3

            data = []

            data.append("* Model atmosphere written by " \
                        "rhatmos.input_atmos.write()\n")
            data.append("*\n")
            data.append("  {0}\n".format(self.ID))

            if self.scale == "MASS_SCALE":
                hscale = np.log10(self.cmass / (G_TO_KG / CM_TO_M**2))
                data.append("  Mass scale\n")
            elif self.scale == "TAU500_SCALE":
                hscale = np.log10(self.tau500)
                data.append("  Tau500 scale\n")
            elif self.scale == "GEOMETRIC_SCALE":
                hscale = self.height
                data.append("  Height scale\n")

            data.append('*\n')

            grav = np.log10(self.grav / CM_TO_M)

            if self.type == "ONE_D_PLANE":
                data.append("* lg g [cm s^-2]\n")
                data.append('     {:5.2f}\n'.format(grav))
                data.append("* Ndep\n")
                data.append('   {:4d}\n'.format(self.Ndep))

                Nd = self.Ndep
            else:
                data.append("* lg g [cm s^-2]      Radius [km]\n")
                data.append('     {:5.2f}          '\
                            '{:7.2E}\n'.format(grav, self.radius))
                data.append("* Nradius   Ncore   Ninter\n")
                fmt = 3 * '    {:4d}' + "\n"
                data.append(fmt.format(self.Nradius, self.Ncore, self.Ninter))

                Nd = self.Nradius

            data.append("*\n")
            data.append("*  lg column Mass   Temperature    "\
                        "Ne             V              Vturb\n")

            fmt = '  {: 12.8E}' + 4 * '  {: 10.6E}' + "\n"
            for k in range(Nd):
                data.append(fmt.format(hscale[k], self.T[k], self.n_elec[k],\
                                       self.v[k], self.vturb[k]))

            data.append("*\n")

            if not self.HLTE:
                data.append("* NLTE Hydrogen populations\n")
                data.append("*  nh[1]        nh[2]        nh[3]        "\
                            "nh[4]        nh[5]        np\n")

                fmt = self.NHydr * '   {:8.4E}' + "\n"
                for k in range(Nd):
                    data.append(fmt.format(*self.nH[k, :]))

            f = open(outfile, 'w')
            for line in data:
                f.write(line)
            f.close()


        elif self.type == "TWO_D_PLANE" or self.type == "THREE_D_PLANE":

            pck = xdrlib.Packer()

            if self.type == "TWO_D_PLANE":
                write_farray(np.array([self.Nx, self.Nz, self.NHydr]),\
                             pck, "int")
                write_farray(np.array(self.boundary), pck, "int")
                write_farray(self.dx, pck, "double")

            elif self.type == "THREE_D_PLANE":
                write_farray(np.array([self.Nx, self.Ny,\
                                       self.Nz, self.NHydr]),\
                             pck, "int")
                write_farray(np.array(self.boundary), pck, "int")
                pck.pack_double(self.dx)
                pck.pack_double(self.dy)

            write_farray(self.z, pck, "double")
            write_farray(self.T, pck, "double")
            write_farray(self.n_elec, pck, "double")
            write_farray(self.vturb, pck, "double")
            write_farray(self.vx, pck, "double")

            if self.type == "THREE_D_PLANE":
                write_farray(self.vy, pck, "double")

            write_farray(self.vz, pck, "double")
            write_farray(self.nH, pck, "double")

            f = open(outfile, 'wb')
            f.write(pck.get_buffer())
            f.close()
            pck.reset()

        else:
            print("Not a valid input atmosphere type: {0}".format(self.type))
            return

        if Bfile != None:

            pck = xdrlib.Packer()

            write_farray(self.B, pck, "double")
            write_farray(self.gamma, pck, "double")
            write_farray(self.chi, pck, "double")

            f = open(Bfile, 'wb')
            f.write(pck.get_buffer())
            f.close()
            pck.reset()



def read_farray(dim, up, dtype="double"):

    if dtype == "float":
        func = up.unpack_float
        dt   = "float"
    elif dtype == "double":
        func = up.unpack_double
        dt   = "float"
    elif dtype == "int":
        func = up.unpack_int
        dt   = "int"

    N = np.prod(dim)
    farray = np.array(up.unpack_farray(N, func), dtype=dt)

    return np.reshape(farray, dim, order='F')


def write_farray(fa, pck, dtype="double"):

    if dtype == "float":
        func = pck.pack_float
    elif dtype == "double":
        func = pck.pack_double
    elif dtype == "int":
        func = pck.pack_int

    N      = np.prod(fa.shape)
    farray = np.reshape(fa, N, order='F')

    pck.pack_farray(N, farray, func)


def read_string(up, size=0):
    """Compensate for IDL string quirks"""

    up.unpack_int()

    if size == 0:
        return str(up.unpack_string().strip(), 'utf-8')
    else:
        up.unpack_int()
        return str(up.unpack_fstring(size), 'utf-8')


class spectrum:

    def __init__(self, inputs, geometry, atmos, filename='spectrum.out'):
        self.filename = filename
        self.read(inputs, geometry, atmos)

    def read(self, inputs, geometry, atmos):

        f  = open(self.filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        self.Nspect = up.unpack_int()
        self.waves = read_farray(self.Nspect, up, "double")

        if geometry.type == "ONE_D_PLANE" or\
           geometry.type == 'SPHERICAL_SYMMETRIC':
            dim = [geometry.Nrays, self.Nspect]

        elif geometry.type == 'TWO_D_PLANE':
            dim = [geometry.Nx, geometry.Nrays, self.Nspect]
        elif geometry.type == 'THREE_D_PLANE':
            dim = [geometry.Nx, geometry.Ny, geometry.Nrays, self.Nspect]

        self.I = read_farray(dim, up, "double")

        self.vacuum_to_air = up.unpack_int()
        self.air_limit     = float(up.unpack_double())

        if atmos.stokes or inputs.backgr_pol:
            self.Q = read_farray(dim, up, "double")
            self.U = read_farray(dim, up, "double")
            self.V = read_farray(dim, up, "double")

        up.done()


class rays:

    def __init__(self, inputs, geometry, spectrum, filename='spectrum_1.00'):
        self.filename = filename
        self.read(inputs, geometry, spectrum)

    def read(self, inputs, geometry, spectrum):

        f  = open(self.filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        if geometry.type == "ONE_D_PLANE":
            self.muz = up.unpack_double()

            dim1 = [spectrum.Nspect]
            dim2 = [geometry.Ndep]

        elif geometry.type == 'SPHERICAL_SYMMETRIC':
            self.muz = up.unpack_double()

            dim1 = [spectrum.Nspect]
            dim2 = [geometry.Nradius]

        elif geometry.type == 'TWO_D_PLANE':
            self.mux = up.unpack_double()
            self.muz = up.unpack_double()

            dim1 = [geometry.Nx, spectrum.Nspect]
            dim2 = [geometry.Nx, geometry.Nz]

        elif geometry.type == 'THREE_D_PLANE':
            self.mux = up.unpack_double()
            self.muy = up.unpack_double()

            dim1 = [geometry.Nx, geometry.Ny, spectrum.Nspect]
            dim2 = [geometry.Nx, geometry.Ny, geometry.Nz]


        self.I = read_farray(dim1, up, "double")

        self.Nopac = up.unpack_int()
        if self.Nopac > 0:
            self.opac = {}
            for n in range(self.Nopac):
                self.opac[n] = opac(dim2, up)

        try: spectrum.Q
        except AttributeError:
            up.done()
            return
        else:
            self.Q = read_farray(dim1, up, "double")
            self.U = read_farray(dim1, up, "double")
            self.V = read_farray(dim1, up, "double")

            up.done()


class opac:

    def __init__(self, dim, up):
        self.read(dim, up)

    def read(self, dim, up):
        self.nspect = up.unpack_int()
        self.chi    = read_farray(dim, up, "double")
        self.S      = read_farray(dim, up, "double")



class geometry:

    def __init__(self, filename='geometry.out'):
        self.filename = filename
        self.read()

    def read(self):

        geometry_types = ['ONE_D_PLANE', 'TWO_D_PLANE',
                          'SPHERICAL_SYMMETRIC', 'THREE_D_PLANE']

        f  = open(self.filename, 'rb')
        up = xdrlib.Unpacker(f.read())
        f.close()

        self.type  = geometry_types[up.unpack_int()]
        self.Nrays = up.unpack_int()

        if self.type == 'ONE_D_PLANE':

            self.Ndep = up.unpack_int()

            self.xmu    = read_farray(self.Nrays, up, "double")
            self.wmu    = read_farray(self.Nrays, up, "double")

            self.height = read_farray(self.Ndep, up, "double")
            self.cmass  = read_farray(self.Ndep, up, "double")
            self.tau500 = read_farray(self.Ndep, up, "double")
            self.vz     = read_farray(self.Ndep, up, "double")

        elif self.type == 'SPHERICAL_SYMMETRIC':

            self.Nradius = up.unpack_int()
            self.Ncore   = up.unpack_int()
            self.radius  = up.unpack_double()

            self.xmu    = read_farray(self.Nrays, up, "double")
            self.wmu    = read_farray(self.Nrays, up, "double")

            self.r      = read_farray(self.Nradius, up, "double")
            self.cmass  = read_farray(self.Nradius, up, "double")
            self.tau500 = read_farray(self.Nradius, up, "double")
            self.vr     = read_farray(self.Nradius, up, "double")

        elif self.type == 'TWO_D_PLANE':

            self.Nx  = up.unpack_int()
            self.Nz  = up.unpack_int()

            self.AngleSet = up.unpack_int()
            self.xmu = read_farray(self.Nrays, up, "double")
            self.ymu = read_farray(self.Nrays, up, "double")
            self.wmu = read_farray(self.Nrays, up, "double")

            self.x = read_farray(self.Nx, up, "double")
            self.z = read_farray(self.Nz, up, "double")

            dim = [self.Nx, self.Nz]
            self.vx = read_farray(dim, up, "double")
            self.vz = read_farray(dim, up, "double")

        elif self.type == 'THREE_D_PLANE':

            self.Nx  = up.unpack_int()
            self.Ny  = up.unpack_int()
            self.Nz  = up.unpack_int()

            self.AngleSet = up.unpack_int()
            self.xmu = read_farray(self.Nrays, up, "double")
            self.ymu = read_farray(self.Nrays, up, "double")
            self.wmu = read_farray(self.Nrays, up, "double")

            self.dx = up.unpack_double()
            self.dy = up.unpack_double()
            self.z  = read_farray(self.Nz, up, "double")

            dim = [self.Nx, self.Ny, self.Nz]
            self.vx = read_farray(dim, up, "double")
            self.vy = read_farray(dim, up, "double")
            self.vz = read_farray(dim, up, "double")

        up.done()



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



script_folder=os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(script_folder)

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path=fix_path(dir_path)


home=expanduser("~")
intern_folder_name='d_'+str(time.time()).split('.')[0][-5:]+'_'
f=open('initialization.input','r')
Input_file_int=f.readlines()
f.close()

Input_file=[]

for i in Input_file_int:
    if i != '\n':
        if i.replace(' ','')[0] != '#':
            Input_file.append(i)
del Input_file_int
ncontrol=0
if len(glob.glob('run_files/*.trol'))>0:
    if len(glob.glob('run_files/*.trol'))==1:
        control_file=glob.glob('run_files/*.trol')[0][10:]
        run_source=0
        ncontrol+=1
    else:
        print('More than one trol file detected. Please include only one file inside the run_files folder')
        sys.exit("Exiting computation")

if len(glob.glob('run_files/*.dtrol'))>0:
    if len(glob.glob('run_files/*.dtrol'))==1:
        control_file=glob.glob('run_files/*.dtrol')[0][10:]
        run_source=1
        ncontrol+=1

    else:
        print('More than one dtrol file detected. Please include only one file inside the run_files folder')
        sys.exit("Exiting computation")



if len(glob.glob('run_files/*.mtrol'))>0:
    if len(glob.glob('run_files/*.mtrol'))==1:
        control_file=glob.glob('run_files/*.mtrol')[0][10:]
        run_source=2
        ncontrol+=1
    else:
        print('More than one mtrol file detected. Please include only one file inside the run_files folder')
        sys.exit("Exiting computation")



if (len(glob.glob('run_files/keyword.input'))>0 and len(glob.glob('run_files/*.dtrol'))==0):
    if len(glob.glob('run_files/keyword.input'))==1:
        control_file=glob.glob('run_files/keyword.input')[0][10:]
        run_source=3
        ncontrol+=1
    else:
        print('More than one mtrol file detected. Please include only one file inside the run_files folder')
        sys.exit("Exiting computation")






if ncontrol==0:
    print('No control file detected')
    sys.exit("Exiting computation")

if ncontrol>1:
    print('More than one control file detected.Please include only the desired control file in the run_files folder')
    sys.exit("Exiting computation")














#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




if run_source==0 or run_source ==1:
    init_file_keys=['n_cores=','data=','coordinates=','disk_position=','atmospheric_initialization=','source_location=','output_type=','email_adress=','atmospheric_model_header=','debug=']
    init_file_pos=np.zeros(len(init_file_keys))

    for i in range(len(init_file_keys)):
        for j in range(len(Input_file)):
            if Input_file[j].find(init_file_keys[i]) >=0:
                init_file_pos[i]=j

    #Select the location of SIR version and run files

    if run_source==0:
        sir_location=script_folder+'/sir/'
    elif run_source==1:
        sir_location=script_folder+'/desire/run/'

    run_file_folder=script_folder+'/run_files/'


    #Identify the number of cores or syntesis mode and atmos file name
    ff=open(run_file_folder+control_file,'r')


    input_drtrol=ff.readlines()
    ff.close()
    n_cycles=(input_drtrol[0].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0]
    n_cycles2=int(float(n_cycles))
    n_cycles3=str(n_cycles2)
    init_atmos_file_name=(input_drtrol[7].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0]
    init_atmos_file_name_mo_mod=init_atmos_file_name.split('.')[0]

    init_atmos_file_name_2=(input_drtrol[8].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0]
    init_atmos_file_name_mo_mod_2=init_atmos_file_name_2.split('.')[0]



    wave_file_name=(input_drtrol[4].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0]
    profile_file_name=(input_drtrol[1].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0]
    if run_source==1:
        nlte_f=(input_drtrol[40].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0]
        nlte_f=float(nlte_f.replace(" ", ""))
    else:
        nlte_f=111

    ncores=int(Input_file[int(init_file_pos[0])].replace("\n", "").split('=')[1])
    ninde=Input_file[int(init_file_pos[1])].replace("\n", "").split('=')[1].split(',')
    nlines=len(ninde)



    if len(glob.glob('run_files/'+wave_file_name))==0:
        print('No grid files detected. Please include only one file inside the run_files folder')
        sys.exit("Exiting computation")

    ff=open(run_file_folder+wave_file_name,'r')
    input_grid=ff.readlines()
    ff.close()
    nindexa=[]
    inwave=[]
    fiwave=[]
    stepwave=[]
    num_lines_per=0
    for i in input_grid:
        if i[0].isnumeric():
            i_int=i.replace(' ','').replace('\n','').split(':')
            nindexa.append(int(i_int[0].split(',')[0]))
            inwave.append(float(i_int[-1].split(',')[0]))
            fiwave.append(float(i_int[-1].split(',')[2]))
            stepwave.append(float(i_int[-1].split(',')[1]))
            temp_lin_s=(float(i_int[-1].split(',')[2])-float(i_int[-1].split(',')[0]))/float(i_int[-1].split(',')[1])
            if round(temp_lin_s,4)-int(round(temp_lin_s,4)) !=0:
                sys.exit("Wave length range is not a multiple integer of the wave step in the line "+(i_int[0].split(',')[0]))
##HU        num_lines_per=num_lines_per+temp_lin_s+1
            num_lines_per=num_lines_per + round(temp_lin_s, 4) + 1
    num_lines_per=int(num_lines_per)



    #Select wavelength
    wave_leng={}
    for kk,k in enumerate(nindexa):
        wave_leng[k]=np.arange(inwave[kk],fiwave[kk]+0.00001,stepwave[kk])


    all_wave=[]
    all_ind=[]
    for iii in range(len(nindexa)):
            for iiii in np.arange(inwave[iii],fiwave[iii]+0.00000001,stepwave[iii]):
                all_wave.append(iiii)
                all_ind.append(nindexa[iii])










    #Read data
    data_path=Input_file[int(init_file_pos[1])].replace("\n", "").split('=')[1].split(',')

    if n_cycles2>0:
        if len(data_path) != len(nindexa):
            print('Diferent number of lines in the grid file and data cubes provided.')
            sys.exit("Exiting computation")


    dataa={}
    shapea={}
    if n_cycles2>0:
        try:
            for n,k in enumerate(nindexa):
                temp_dat=fits.open(data_path[n])[0].data
                shapea[k]=temp_dat.shape
                dataa[k]=temp_dat
        except Exception as e:
            print(e)
            print('Impossible to find one or more input data files')
            sys.exit("Exiting computation")


    if n_cycles2==0 or n_cycles2==-1:
        temp_dat=fits.open(data_path[0])[0].data
        shapea[nindexa[0]]=temp_dat.shape
        dataa[nindexa[0]]=temp_dat




    #Select x and y
    #coorde=list(map(int,Input_file[int(init_file_pos[9])].replace("\n", "").split('=')[1].split(',')))[0]
    coorde=Input_file[int(init_file_pos[2])].replace("\n", "").split('=')[1]


    if coorde[0] == '0':
        run_range_x_min=0
        run_range_y_min=0
        run_range_x_max=1
        run_range_y_max=1
    if coorde[0] == '1':
        run_range_x_min=0
        run_range_y_min=0
        run_range_x_max=shapea[nindexa[0]][-1]-1
        run_range_y_max=shapea[nindexa[0]][-2]-1
    if coorde[0] == '[':
        run_range_x_min=int(coorde.replace('[','').replace(']','').split(',')[0])
        run_range_y_min=int(coorde.replace('[','').replace(']','').split(',')[2])
        run_range_x_max=int(coorde.replace('[','').replace(']','').split(',')[1])
        run_range_y_max=int(coorde.replace('[','').replace(']','').split(',')[3])






    if n_cycles2>0:
        #Set the clustering
        clustering=Input_file[int(init_file_pos[4])].replace("\n", "").split('=')[1].split(',')[0]
        clustering=clustering.lower()
        if clustering =='yes':
            clust_ind_arr=fits.open(run_file_folder+'/atmos/profile_clustering.fits')[0]
            clust_ind_arr=clust_ind_arr.data
        if clustering =='nn':
            nn_init(dataa[0],run_range_x_max,run_range_x_min,run_range_y_max,run_range_y_min,ncores)
            ml_atmos_int=fits.open(run_file_folder+'/atmos/atmos.fits')[0]
            ml_atmos_int=ml_atmos_int.data
            del to_fit_atmos
            del atmos_full
            atmos_shape=ml_atmos_int.shape
            ml_atmos_base = multiprocessing.Array(ctypes.c_double, atmos_shape[0]*atmos_shape[1]*atmos_shape[2]*atmos_shape[3])
            ml_atmos = np.ctypeslib.as_array(ml_atmos_base.get_obj())
            ml_atmos = ml_atmos.reshape((atmos_shape[0],atmos_shape[1],atmos_shape[2],atmos_shape[3]))
            ml_atmos[:,:,:,:]=ml_atmos_int
            del ml_atmos_int
        if clustering =='atmos':
            ml_atmos_int=fits.open(run_file_folder+'/atmos/atmos.fits')[0]
            ml_atmos_int=ml_atmos_int.data
            atmos_shape=ml_atmos_int.shape
            ml_atmos_base = multiprocessing.Array(ctypes.c_double, atmos_shape[0]*atmos_shape[1]*atmos_shape[2]*atmos_shape[3])
            ml_atmos = np.ctypeslib.as_array(ml_atmos_base.get_obj())
            ml_atmos = ml_atmos.reshape((atmos_shape[0],atmos_shape[1],atmos_shape[2],atmos_shape[3]))
            ml_atmos[:,:,:,:]=ml_atmos_int
            del ml_atmos_int
            num_lines_mod=atmos_shape[1]
        if clustering.replace('.','',1).isnumeric():
            clustering_int=copy.deepcopy(clustering)
            clustering=int(float(clustering_int))
            clustering_chi=float(clustering_int)-clustering
            if clustering_chi==0.0: clustering_chi=1
            clustering=str(clustering)
        #Select saving the spectra option
        output_opt=int(Input_file[int(init_file_pos[6])].replace("\n", "").split('=')[1])
        if clustering =='single' or clustering =='ne':
            if os.path.isfile(run_file_folder+'atmos/'+init_atmos_file_name):
                num_lines_mod=np.loadtxt(run_file_folder+'atmos/'+init_atmos_file_name,skiprows=1).shape[0]
            else:
                sys.exit("No atmospheric file detected. Exiting computation")

        if clustering.isnumeric():
            for ati in range(int(clustering)):
                if os.path.isfile(run_file_folder+'atmos/atmcl'+str(ati+1)+'.mod'):
                    num_lines_mod=np.loadtxt(run_file_folder+'atmos/'+'atmcl1.mod',skiprows=1).shape[0]
                else:
                    sys.exit("At least one atmospheric file needed is not detected. Exiting computation")
    #Select the solar disk pixel position. coor file
    disk_pos=Input_file[int(init_file_pos[3])].replace("\n", "").split('=')[1].replace('[','').replace(']','').split(',')






    if n_cycles2 ==-1:
        para_num_phi=0
        for qqw in range(14,22):
            para_num_phi+=int((input_drtrol[qqw].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])
        if para_num_phi>1:
            sys.exit("Please select only one physical parameter. Exiting computation")
        if int((input_drtrol[14].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])>0:
            name_rf='temperature_rf.fits'
        if int((input_drtrol[15].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])>0:
            name_rf='electr_press_rf.fits'
        if int((input_drtrol[16].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])>0:
            name_rf='microturb_rf.fits'
        if int((input_drtrol[17].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])>0:
            name_rf='magnetic_field_rf.fits'
        if int((input_drtrol[18].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])>0:
            name_rf='velocity_rf.fits'
        if int((input_drtrol[19].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])>0:
            name_rf='gamma_rf.fits'
        if int((input_drtrol[20].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])>0:
           name_rf='phi_rf.fits'
        if int((input_drtrol[20].replace("\n", "").replace(" ","").split(':')[1]).split('!')[0])>0:
           name_rf='macroturbulence_rf.fits'

        disk_pos=Input_file[int(init_file_pos[3])].replace("\n", "").split('=')[1].replace('[','').replace(']','').split(',')






















    try:
        ver_int=float(disk_pos[0])
        disk_pos=list(map(float,disk_pos))
    except:
        temp_coor=fits.open(disk_pos[0])[0]
        disk_pos=temp_coor.data

    #Select atmosphere header in case of syntesis
    atmos_header=Input_file[int(init_file_pos[8])].replace("\n", "").split('=')[1].split(',')
    debug=int(Input_file[int(init_file_pos[9])].replace("\n", "").split('=')[1])















    #Read notification email
    email_adre=Input_file[int(init_file_pos[7])].replace("\n", "").split('=')[1].split(',')









    start= time.time()

    os.chdir(sir_location)

    #Creat results and erros folder
    if os.path.exists(dir_path+'/results/'):
        None
    else:
        make_dir(dir_path+'/results/')

    if os.path.exists(dir_path+'/results/errors/'):
        if glob.glob(dir_path+'/results/errors/*.txt'):
            make_dir(dir_path+'/results/errors/*')
    else:
        make_dir(dir_path+'/results/errors/')

    if os.path.exists(dir_path+'/results/run_folders/'):
        rm_dir('*'+dir_path+'/results/run_folders/')
    else:
        make_dir(dir_path+'/results/run_folders/')







    if coorde[0] != '2':
        list_run=[]
        rymax=run_range_y_max+1
        rxmax=run_range_x_max+1
        if coorde[0] == '1':
            rymax=run_range_y_max+1
            rxmax=run_range_x_max+1
        for i in range(run_range_x_min,rxmax):
            for l in range(run_range_y_min,rymax):
                if type(disk_pos) == np.ndarray:
                    if disk_pos[0,i,l]>=0:
                        list_run.append((i,l))
                else:
                    list_run.append((i,l))


    if coorde[0] == '2':
        coord_pix_temp=np.loadtxt(run_file_folder+'pix_coord.txt')
        run_range_x_max=int(coord_pix_temp[:,1].max())
        run_range_x_min=int(coord_pix_temp[:,1].min())
        run_range_y_max=int(coord_pix_temp[:,0].max())
        run_range_y_min=int(coord_pix_temp[:,0].min())
        list_run=[]
        for i in range(len(coord_pix_temp)):
            list_run.append((int(coord_pix_temp[i,1]),int(coord_pix_temp[i,0])))










    if len(list_run) ==0:
        print('No valid data!!! Please check your initialization file or data')
        sys.exit("Exiting computation")
    else:
        len_list_run=len(list_run)

    if debug==1 and len_list_run>1000:
        print('Please reduce the computation box size, up to maximum of 1000 points, in the degub mode.')
        sys.exit("Exiting computation")


    if n_cycles2==0:
        inv_res_array_per=c_ar_qd(4,num_lines_per,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)

    if n_cycles2>0:
        per_ori = c_ar_qd(4,num_lines_per,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)


        if str(clustering).isnumeric():
            atmcl_base = multiprocessing.Array(ctypes.c_float, (1+run_range_x_max-run_range_x_min)*(1+run_range_y_max-run_range_y_min))
            atmcl = np.ctypeslib.as_array(atmcl_base.get_obj())
            atmcl = atmcl.reshape((1+run_range_y_max-run_range_y_min,1+run_range_x_max-run_range_x_min))

        if output_opt==1 or output_opt==2:

            if nlte_f<10:
                fullp_chi2 =c_ar_td(3,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)

            if nlte_f>=10 or run_source==0:
                fullp_chi2 = c_ar_td(2,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)

            inv_res_array_per = c_ar_qd(4,num_lines_per,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)
            inv_res_array_mod= c_ar_qd(11,num_lines_mod,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)
            inv_res_macro_mod= c_ar_td(3,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)
            if len(init_atmos_file_name_2) >0:
                inv_res_array_mod_2= c_ar_qd(11,num_lines_mod,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)









        if output_opt==2:
            inv_error_mod= c_ar_qd(11,num_lines_mod,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)



        if output_opt==3:
            inv_res_array_per = c_ar_cd(n_cycles2,4,num_lines_per,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)
            inv_res_array_mod= c_ar_cd(n_cycles2,11,num_lines_mod,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)
            inv_res_macro_mod= c_ar_qd(n_cycles2,3,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)

            if len(init_atmos_file_name_2) >0:
                inv_res_array_mod_2= c_ar_cd(n_cycles2,11,num_lines_mod,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)

            inv_error_mod= c_ar_cd(n_cycles2,11,num_lines_mod,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)


            if nlte_f<10:
                fullp_chi2 = c_ar_qd(n_cycles2,3,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)

            if nlte_f>=10 or run_source==0:
                fullp_chi2 = c_ar_qd(n_cycles2,2,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)


    if n_cycles2==-1:
        rf = c_ar_cd(4,dataa[nindexa[0]].shape[1],num_lines_per,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)


    if len(list_run) < ncores:
        ncores=len(list_run)
    if clustering =='ne':
        if len(list_run)<10:
            sys.exit("Position range is too small for this inti")


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




if run_source==2:
    init_file_keys=['n_cores=','data=','coordinates=','run_files=','source_location=','email_adress=','atmospheric_model_header=','debug=']
    init_file_pos=np.zeros(len(init_file_keys))

    for i in range(len(init_file_keys)):
        for j in range(len(Input_file)):
            if Input_file[j].find(init_file_keys[i]) >=0:
                init_file_pos[i]=j


    #Select the location of SIR version and run files
    init_files=Input_file[int(init_file_pos[3])].replace("\n", "").split('=')[1].split(',')
    sir_location=Input_file[int(init_file_pos[4])].replace("\n", "").split('=')[1].split(',')

    if len(sir_location[0]) >1:
        sir_location=sir_location[0]
    else:
        sir_location=home+'/modelador/'

    run_file_folder=script_folder+'/run_files/'

    ncores=int(Input_file[int(init_file_pos[0])].replace("\n", "").split('=')[1])


    #Read data
    data_path=Input_file[int(init_file_pos[1])].replace("\n", "").split('=')[1].split(',')

    temp_dat=fits.open(data_path[0])[0]
    temp_dat=temp_dat.data
    shapea=temp_dat.shape
    dataa_int=copy.deepcopy(temp_dat)
    dataa_base= multiprocessing.Array(ctypes.c_float, dataa_int.shape[0]*dataa_int.shape[1]*dataa_int.shape[2]*dataa_int.shape[3])
    dataa= np.ctypeslib.as_array(dataa_base.get_obj())
    dataa= dataa.reshape(11,dataa_int.shape[1],dataa_int.shape[2],dataa_int.shape[3])
    dataa[:,:,:,:]=dataa_int
    del temp_dat
    del dataa_int

    #Select x and y
    #coorde=list(map(int,Input_file[int(init_file_pos[9])].replace("\n", "").split('=')[1].split(',')))[0]
    coorde=Input_file[int(init_file_pos[2])].replace("\n", "").split('=')[1]


    if coorde[0] == '0':
        run_range_x_min=0
        run_range_y_min=0
        run_range_x_max=1
        run_range_y_max=1
    if coorde[0] == '1':
        run_range_x_min=0
        run_range_y_min=0
        run_range_x_max=shapea[-1]-1
        run_range_y_max=shapea[-2]-1
    if coorde[0] == '[':
        run_range_x_min=int(coorde.replace('[','').replace(']','').split(',')[0])
        run_range_y_min=int(coorde.replace('[','').replace(']','').split(',')[2])
        run_range_x_max=int(coorde.replace('[','').replace(']','').split(',')[1])
        run_range_y_max=int(coorde.replace('[','').replace(']','').split(',')[3])



    atmos_header=Input_file[int(init_file_pos[6])].replace("\n", "").split('=')[1].split(',')
    debug=int(Input_file[int(init_file_pos[7])].replace("\n", "").split('=')[1])

    #Read notification email
    email_adre=Input_file[int(init_file_pos[5])].replace("\n", "").split('=')[1].split(',')




    start= time.time()

    os.chdir(sir_location)

    #Creat results and erros folder
    if os.path.exists(dir_path+'/results/'):
        None
    else:
        call('mkdir '+dir_path+'/results/', shell=True)

    if os.path.exists(dir_path+'/results/errors/'):
        if glob.glob(dir_path+'/results/errors/*.txt'):
            rm_file('*',dir_path+'/results/errors/*')
    else:
        call('mkdir '+dir_path+'/results/errors/', shell=True)













    list_run=[]
    for i in range(run_range_x_min,run_range_x_max+1):
        for l in range(run_range_y_min,run_range_y_max+1):
            list_run.append((i,l))









    if len(list_run) ==0:
        print('No valid data!!! Please check your initialization file or data')
        sys.exit("Exiting computation")
    else:
        len_list_run=len(list_run)

    if debug==1 and len_list_run>1000:
        print('Please reduce the computation box size, up to maximum of 1000 points, in the degub mode.')
        sys.exit("Exiting computation")


    os.chdir(sir_location)
    mdir=subprocess.Popen('mkdir initialization_folder', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = mdir.communicate()


    os.chdir('initialization_folder')
    call('cp '+run_file_folder+'*.* '+sir_location+'initialization_folder/', shell=True)
    call('cp '+run_file_folder+'* '+sir_location+'initialization_folder/', shell=True)
    np.savetxt('input.mod',dataa[:,:,0,0].transpose(),fmt='%1.5e', delimiter=' ', newline='\n', header=atmos_header[0],comments='')
    try:
        run_desire=subprocess.Popen('echo tau.mtrol | '+sir_location+'modelador*.x', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = run_desire.communicate()


        mod_file=open('output.mod','r')
        mod_file.readline()
        kl=0
        for tline in mod_file:
            kl=kl+1
        mod_file.close()
        num_lines_mod=kl
        os.chdir(sir_location)
        call('rm -r initialization_folder', shell=True)
    except:
        os.chdir(sir_location)
        call('cp -r initialization_folder '+dir_path+'/results/errors/initialization_folder', shell=True)
        os.chdir(dir_path)
        file = open(dir_path+'/results/errors/error.txt', 'w')
        file.write(out.decode("utf-8"))
        file.write('\n')
        file.write('----------------------------------------------------------')
        file.write('\n')
        file.write(err.decode("utf-8"))
        file.close()
        os.chdir(sir_location)
        call('rm -r initialization_folder', shell=True)
        sys.exit('Something went wrong!! check error.txt in error folder for more details')

    inv_res_array_mod= c_ar_qd(11,num_lines_mod,run_range_y_max-run_range_y_min,run_range_x_max-run_range_x_min)





    if len(list_run) < ncores:
        ncores=len(list_run)



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



if run_source==0 or run_source==1:
    #runs the inversion
    if n_cycles2>0 and (coorde[0] == '1' or coorde[0] == '[' or coorde[0] =='2'):


        if ncores>0:
            if __name__ == '__main__':
                p = multiprocessing.get_context("fork").Pool(ncores)
                create_folder(ncores)
                r=p.map(invertint, list_run,chunksize=1)
        else:
            print('no available cores')

    #runs the sint
    if n_cycles2==0 and (coorde[0] == '1' or coorde[0] == '['):


        if ncores>0:
            if __name__ == '__main__':
                p = multiprocessing.get_context("fork").Pool(ncores)
                create_folder_synt(ncores)
                p.map(sint, list_run,chunksize=1)
        else:
            print('no available cores')



    if n_cycles2==-1 and (coorde[0] == '1' or coorde[0] == '['):

        if ncores>0:
            if __name__ == '__main__':
                p = multiprocessing.get_context("fork").Pool(ncores)
                p.map(rffp, list_run,chunksize=1)
        else:
            print('no available cores')





    end= time.time()


    #Save arrays to fits files in the selected directory
    os.chdir(dir_path+'/results/')
    if coorde[0] != '2':
        if n_cycles2==0:
            save_fits_p(np.float32(inv_res_array_per),all_wave,all_ind,'synt_prof.fits')

        if n_cycles2>=1:
            save_fits(np.float32(inv_res_array_mod),'inv_res_mod.fits')
            if len(init_atmos_file_name_2) >0:
                save_fits(np.float32(inv_res_array_mod_2),'inv_res_sec_mods.fits')
            save_fits_p(np.float32(inv_res_array_per),all_wave,all_ind,'inv_res_prof.fits')
            save_fits(np.float32(inv_res_macro_mod),'inv_res_macro_mod.fits')

            save_fits_p(np.float32(per_ori),all_wave,all_ind,'observed_prof.fits')
            save_fits(np.float32(fullp_chi2),'chi2.fits')
            if output_opt==2 or output_opt==3:save_fits(np.float32(inv_error_mod),'erro_mod_atmos.fits')
    if n_cycles2>=1:
        if str(clustering).isnumeric():
            save_fits(np.float32(atmcl),'atmcl.fits')

    if coorde[0] == '2' and n_cycles2>=1:
        if os.path.exists('inv_res_mod.fits'):
            atmos_temp_dat=fits.open('inv_res_mod.fits')[0]
            atmos_temp_dat=atmos_temp_dat.data
            fitted_temp_dat=fits.open('inv_res_prof.fits')[0]
            fitted_temp_dat=fitted_temp_dat.data
            ori_per_temp_dat=fits.open('observed_prof.fits')[0]
            ori_per_temp_dat=ori_per_temp_dat.data
            fullp_chi2_temp_dat=fits.open('chi2.fits')[0]
            fullp_chi2_temp_dat=fullp_chi2_temp_dat.data
            for i in range(len(coord_pix_temp)):
                atmos_temp_dat[:,:,int(coord_pix_temp[i,1])-run_range_x_min+abs(inv_res_array_mod.shape[2]-atmos_temp_dat.shape[2]),int(coord_pix_temp[i,0])-run_range_y_min+abs(inv_res_array_mod.shape[3]-atmos_temp_dat.shape[3])]=inv_res_array_mod[:,:,int(coord_pix_temp[i,1])-run_range_x_min,int(coord_pix_temp[i,0])-run_range_y_min]
                fitted_temp_dat[:,:,int(coord_pix_temp[i,1])-run_range_x_min+abs(inv_res_array_mod.shape[2]-atmos_temp_dat.shape[2]),int(coord_pix_temp[i,0])-run_range_y_min+abs(inv_res_array_mod.shape[3]-atmos_temp_dat.shape[3])]=inv_res_array_per[:,:,int(coord_pix_temp[i,1])-run_range_x_min,int(coord_pix_temp[i,0])-run_range_y_min]
                fullp_chi2_temp_dat[:,int(coord_pix_temp[i,1])-run_range_x_min+abs(inv_res_array_mod.shape[2]-atmos_temp_dat.shape[2]),int(coord_pix_temp[i,0])-run_range_y_min+abs(inv_res_array_mod.shape[3]-atmos_temp_dat.shape[3])]=fullp_chi2[:,int(coord_pix_temp[i,1])-run_range_x_min,int(coord_pix_temp[i,0])-run_range_y_min]
                ori_per_temp_dat[:,:,int(coord_pix_temp[i,1])-run_range_x_min+abs(inv_res_array_mod.shape[2]-atmos_temp_dat.shape[2]),int(coord_pix_temp[i,0])-run_range_y_min+abs(inv_res_array_mod.shape[3]-atmos_temp_dat.shape[3])]=per_ori[:,:,int(coord_pix_temp[i,1])-run_range_x_min,int(coord_pix_temp[i,0])-run_range_y_min]
            plot_maps(atmos_temp_dat,fitted_temp_dat,ori_per_temp_dat,fullp_chi2_temp_dat,output_opt)
            save_fits(np.float32(atmos_temp_dat),'inv_res_mod.fits')
            save_fits_p(np.float32(fitted_temp_dat),all_wave,all_ind,'inv_res_prof.fits')
            save_fits(np.float32(fullp_chi2_temp_dat),'chi2.fits')
            #save_fits(np.float32(inv_res_macro_mod),'inv_res_macro_mod.fits')
        else:
            if len(init_atmos_file_name_2) >0:
                save_fits(np.float32(inv_res_array_mod_2),'inv_res_sec_mod.fits')
            save_fits(np.float32(inv_res_array_mod),'inv_res_mod.fits')
            save_fits_p(np.float32(inv_res_array_per),all_wave,all_ind,'inv_res_prof.fits')
            save_fits_p(np.float32(per_ori),all_wave,all_ind,'observed_prof.fits')
            save_fits(np.float32(fullp_chi2),'chi2.fits')
            plot_maps(inv_res_array_mod,inv_res_array_per,per_ori,fullp_chi2,output_opt)

    if n_cycles2==-1:
            hdu1 = fits.PrimaryHDU(np.float32(rf))
            hdu2 = fits.ImageHDU(all_wave)
            hdu3 = fits.ImageHDU(dataa[nindexa[0]][0,:,0,0])
            hdu4 = fits.ImageHDU(all_ind)
            new_hdul = fits.HDUList([hdu1, hdu2,hdu3,hdu3])
            new_hdul.writeto(name_rf,overwrite=True)


    try:
        if n_cycles2>=1 and coorde[0] != '2':
            plot_maps(inv_res_array_mod,inv_res_array_per,per_ori,fullp_chi2,output_opt)
    except Exception as e:
        print('Not possible to plot data.',e)


    os.chdir(dir_path+'/results/errors/')
    list_error=glob.glob('pixel*.txt')
    list_error.sort()
    os.chdir(dir_path+'/results/')
    if len(list_error)>0:
        file_no_cov = open('no_converged_pixels.txt','w')
        for i in range(len(list_error)):
            qq=list_error[i].split('.')[0].split('_')
            file_no_cov.write(str(qq[1])+'  '+str(qq[2])+'\n')
        file_no_cov.close()




if run_source==2:
    if len(list_run) < ncores:
        ncores=len(list_run)
    create_folder_model(ncores)

    if ncores>0:
        if __name__ == '__main__':
            p = multiprocessing.get_context("fork").Pool(ncores)
            p.map(modelador, list_run,chunksize=1)
    else:
        print('no available cores')

    end= time.time()



    os.chdir(dir_path+'/results/')

    save_fits(np.float32(inv_res_array_mod),'new_tau_atmos.fits')





    os.chdir(dir_path+'/results/errors/')
    list_error=glob.glob('pixel*.txt')
    list_error.sort()
    os.chdir(dir_path+'/results/')
    if len(list_error)>0:
        file_no_cov = open('no_converged_pixels.txt','w')
        for i in range(len(list_error)):
            qq=list_error[i].split('.')[0].split('_')
            file_no_cov.write(str(qq[1])+'  '+str(qq[2])+'\n')
        file_no_cov.close()



    os.chdir(sir_location)
    shp = subprocess.Popen('rm -r '+intern_folder_name+'*', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outs, errs = shp.communicate()





























if run_source<=2:
    os.chdir(sir_location)
    shp = subprocess.Popen('rm -rf '+intern_folder_name+'*', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outs, errs = shp.communicate()
    '''
    if run_source==0:
        if len(email_adre[0]) >1:
            me="pythongaf@gmail.com"
            mes=email_adre
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(me, "caiih3968")

            msg = "\r\n".join([
          "From: SIR computation",
          "To: user",
          "Subject: Computation done",
          "",
          "Computation done in "+str(datetime.timedelta(seconds=int(end-start)))
          ])
            server.sendmail(me, mes, msg)
            server.quit()


    if run_source==1:
        if len(email_adre[0]) >1:
            me="pythongaf@gmail.com"
            mes=email_adre
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(me, "caiih3968")

            msg = "\r\n".join([
          "From: DeSIRe computation",
          "To: user",
          "Subject: Computation done",
          "",
          "Computation done in "+str(datetime.timedelta(seconds=int(end-start)))
          ])
            server.sendmail(me, mes, msg)
            server.quit()




    if run_source==2:
        if len(email_adre[0]) >1:
            me="pythongaf@gmail.com"
            mes=email_adre
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(me, "caiih3968")

            msg = "\r\n".join([
          "From: Modelador computation",
          "To: user",
          "Subject: Computation done",
          "",
          "Computation done in "+str(datetime.timedelta(seconds=int(end-start)))
          ])
            server.sendmail(me, mes, msg)
            server.quit()


    '''





#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------







if run_source==3:
    import xdrlib
    dir_path = os.path.dirname(os.path.realpath(__file__))

    error=0


    if os.path.exists(dir_path+'/results/'):
        None
    else:
        call('mkdir '+dir_path+'/results/', shell=True)

    if os.path.exists(dir_path+'/results/errors/'):
        if glob.glob(dir_path+'/results/errors/*.txt'):
            call('rm -rf '+dir_path+'/results/errors/*', shell=True)
    else:
        call('mkdir '+dir_path+'/results/errors/', shell=True)


    home=expanduser("~")
    file_in=open('initialization.input','r')
    Input_file_temp=file_in.readlines()
    file_in.close()
    Input_file=[]

    for i in range(len(Input_file_temp)):
        if Input_file_temp[i].find('#')<0:
             Input_file.append(Input_file_temp[i])


    init_file_keys=['n_cores=','data=','coordinates=','source_location=','email_adress=','debug=','depth_s_t=','output_type=']
    init_file_pos=np.zeros(len(init_file_keys))

    for i in range(len(init_file_keys)):
        for j in range(len(Input_file)):
            if Input_file[j].find(init_file_keys[i]) >=0:
                init_file_pos[i]=j




    ncores=int(Input_file[int(init_file_pos[0])].replace("\n", "").split('=')[1])
    data_path=Input_file[int(init_file_pos[1])].replace("\n", "").split('=')[1].split(',')
    coorde=Input_file[int(init_file_pos[2])].replace("\n", "").split('=')[1]

    temp_dat=fits.open(data_path[0])[0]
    atmos_dat=temp_dat.data
    shapea=temp_dat.shape

    if coorde[0] == '0':
        run_range_x_min=0
        run_range_y_min=0
        run_range_x_max=1
        run_range_y_max=1
    if coorde[0] == '1':
        run_range_x_min=0
        run_range_y_min=0
        run_range_x_max=shapea[-1]-1
        run_range_y_max=shapea[-2]-1
    if coorde[0] == '[':
        run_range_x_min=int(coorde.replace('[','').replace(']','').split(',')[0])
        run_range_y_min=int(coorde.replace('[','').replace(']','').split(',')[2])
        run_range_x_max=int(coorde.replace('[','').replace(']','').split(',')[1])
        run_range_y_max=int(coorde.replace('[','').replace(']','').split(',')[3])

    rh_location=Input_file[int(init_file_pos[3])].replace("\n", "").split('=')[1].split(',')
    email_adre=Input_file[int(init_file_pos[4])].replace("\n", "").split('=')[1].split(',')
    debug=int(Input_file[int(init_file_pos[5])].replace("\n", "").split('=')[1])
    depth_s_t=Input_file[int(init_file_pos[6])].replace("\n", "").split('=')[1].split(',')[0].lower()
    output_t=int(Input_file[int(init_file_pos[7])].replace("\n", "").split('=')[1].split(',')[0])




    if len(rh_location[0]) >1:
        rh_location=rh_location[0]
    else:
        rh_location=home+'/rh/rhf1d/'


    run_file_folder=script_folder+'/run_files/'




    change_keyword(2)

    file_ke=open(run_file_folder+'keyword.input','r')
    Input_file_temp=file_ke.readlines()
    file_ke.close()
    Input_file_k=[]

    for i in range(len(Input_file_temp)):
        if Input_file_temp[i].find('#')<0:
             Input_file_k.append(Input_file_temp[i].replace("\n",'').replace(" ",''))


    for j in range(len(Input_file_k)):
        if Input_file_k[j].find('STOKES_MODE') >=0:
            pol_int=Input_file_k[j].split('=')[1].lower()

    if pol_int=='no_stokes':
        n_p_pol=2
    else:
        n_p_pol=5









    if run_range_x_max == -1:
        run_range_x_max=shapea[2]

    if run_range_y_max == -1:
        run_range_y_max=shapea[3]
    atmos_points=shapea[0]



    if (shapea[1]!=14):
        if (shapea[1]!=8):
            sys.exit("Check input data. Invalid number of physical parameters or height stratification definition on initialization.input file")




    start= time.time()

    os.chdir(rh_location)




    #Clean folder from previous incomplete runs
    if len(glob.glob('prh_*')) >1:
        call('rm -r prh_*', shell = True)
    else:
        None

    if len(glob.glob('initiallization_folder')) >0:
        rm_dir(rh_location+'initiallization_folder')
    else:
        None

    list_run=[]
    for i in range(run_range_x_min,run_range_x_max+1):
        for l in range(run_range_y_min,run_range_y_max+1):
            list_run.append((i,l))


    call('mkdir initiallization_folder', shell=True)
    os.chdir('initiallization_folder')
    call('cp '+run_file_folder+'*.* '+rh_location+'initiallization_folder/', shell=True)
    if os.path.exists(run_file_folder+'input_files/'):
        call('cp '+run_file_folder+'input_files/* '+rh_location+'initiallization_folder/', shell=True)
    save_atmos(atmos_dat,atmos_points,0,0,depth_s_t)


    if len(list_run) ==0:
        print('No valid data!!! Please check your initialization file or data')
        sys.exit("Exiting computation")
    else:
        len_list_run=len(list_run)

    if debug==1 and len_list_run>1000:
        print('Please reduce the computation box size, up to maximum of 1000 points, in the degub mode.')
        sys.exit("Exiting computation")
    if debug==1:
        if os.path.exists(dir_path+'/results/run_folders/'):
            rm_dir('*'+dir_path+'/results/run_folders/')
        else:
            make_dir(dir_path+'/results/run_folders/')


    try:
        run_rh=subprocess.Popen('../rhf1d', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = run_rh.communicate(input=None)
        run_solveray=subprocess.Popen('../solveray', shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out1, err1 = run_solveray.communicate(input=None)
        if os.path.exists('spectrum.out'):
            rhd1 = rhout(rh_location+'initiallization_folder')

            wave_leng_array=rhd1.spectrum.waves
            wave_leng_number=len(wave_leng_array)
            os.chdir(rh_location)
            rh_spect_base = multiprocessing.Array(ctypes.c_double, (1+run_range_x_max-run_range_x_min)*(1+run_range_y_max-run_range_y_min)*wave_leng_number*(n_p_pol-1))
            rh_spect = np.ctypeslib.as_array(rh_spect_base.get_obj())
            rh_spect = rh_spect.reshape((n_p_pol-1,wave_leng_number,1+run_range_y_max-run_range_y_min,1+run_range_x_max-run_range_x_min))
            if output_t==2:
                geometri_shape=rhd1.geometry.tau500.shape[0]
                rh_geo_base = multiprocessing.Array(ctypes.c_double, (1+run_range_x_max-run_range_x_min)*(1+run_range_y_max-run_range_y_min)*geometri_shape*(3))
                rh_geo = np.ctypeslib.as_array(rh_geo_base.get_obj())
                rh_geo = rh_geo.reshape((3,geometri_shape,1+run_range_y_max-run_range_y_min,1+run_range_x_max-run_range_x_min))

        else:
            os.chdir(rh_location)
            call('cp -r '+rh_location+'initiallization_folder '+dir_path+'/results/errors/initiallization_folder/', shell=True)
            os.chdir(dir_path)
            file = open('results/errors/error.txt', 'w')
            file.write(out.decode("utf-8"))
            file.write(err.decode("utf-8"))
            file.write(out1.decode("utf-8"))
            file.write(err1.decode("utf-8"))
            file.close()
            sys.exit('ERROR!! check error.txt for more details')
    except:
        os.chdir(rh_location)
        call('cp -r initiallization_folder/ '+dir_path+'/results/errors/initiallization_folder/', shell=True)
        os.chdir(dir_path)
        file = open('results/errors/error.txt', 'w')
        file.write(out.decode("utf-8"))
        file.write(err.decode("utf-8"))
        file.write(out1.decode("utf-8"))
        file.write(err1.decode("utf-8"))
        file.close()
        sys.exit('ERROR!! check error.txt for more details')











    #Create the global arrays where the data is stored
    os.chdir(rh_location)


    #runs the inversion
    if ncores>0:
        p = multiprocessing.get_context("fork").Pool(ncores)
    else:
        p = multiprocessing.get_context("fork").Pool()
    p.map(frh, list_run,chunksize=1)


    end= time.time()


    #Save arrays to fits flies in the selected directory
    os.chdir(dir_path+'/results')
    #save_fits(np.float32(rh_spect),'rh_spect.fits')

    hdu1 = fits.PrimaryHDU(np.float32(rh_spect))
    hdu2 = fits.ImageHDU(wave_leng_array)
    new_hdul = fits.HDUList([hdu1, hdu2])
    new_hdul.writeto("rh_spect.fits",overwrite=True)
    if output_t==2:
        hdu1 = fits.PrimaryHDU(np.float32(rh_geo))
        new_hdul = fits.HDUList(hdu1)
        new_hdul.writeto("rh_geometry.fits",overwrite=True)

    rm_dir(rh_location+'initiallization_folder')
