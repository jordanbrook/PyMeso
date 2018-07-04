"""
LLSD azimuthal shear calculation
Miller, M. L., Lakshmanan, V., and Smith, T. M. (2013). An Automated Method for Depicting Mesocy-
clone Paths and Intensities. Weather and Forecasting, 28(3): 570-585.
Jordan Brook - 3 July 2018
"""

import pyart
import numpy as np
import scipy

def dealiase(radar,vel_name):
    """
    Dealiase and replace the doppler velocity using pyart's region based method
    Parameters:
    ===========
    radar: struct
        pyart radar object
    vel_name: string
        name of doppler velocity field
    Returns:
    ========
    none
    """
    gatefilter = pyart.correct.GateFilter(radar)
    corr_vel = pyart.correct.dealias_region_based(
        radar, vel_field=vel_name, keep_original=False, gatefilter = gatefilter)
    radar.add_field(vel_name, corr_vel, True)

def smooth_data(radar,data_name):
    """
    Smooth and replace input data using a median filter technique built into scipy
    Parameters:
    ===========
    radar: struct
        pyart radar object
    data_name: string
        name of data field in radar object to be smoothed
    Returns:
    ========
    none
    """
    data = radar.fields[data_name]['data']
    smooth_data=scipy.signal.medfilt(data)
    radar.add_field_like(data_name, 
                     data_name, 
                     smooth_data, replace_existing = True)
    
def ref_mask(ref,shear,threshold,dilution):
    """
    Mask shear values based on reflectivity and dilate using scipy's dilation tools 
    Parameters:
    ===========
    ref: array
        reflectivity array in radar coordinates
    shear: array
        azimuthal shear array in radar coordinates
    threshold: float
        reflecitvity threshold for masking
    dilution: int
        number of dilation pixels, refer to scipy.ndimage.binary_dilation() doc.
    Returns:
    ========
    masked azimuthal shear field
    """
    mask = np.zeros(ref.shape)
    mask[ref>threshold]=1
    mask=scipy.ndimage.binary_dilation(mask,iterations=dilution).astype(mask.dtype)
    return mask*shear



def main(radar, ref_name, vel_name,sweep):
    """
    Hail Differential Reflectity Retrieval
    Required DBZH and ZDR fields
    Parameters:
    ===========
    radar: struct
        pyart radar object
    ref_name: string
        name of reflecitivty field
    vel_name: string
        name of doppler velocity field
    Returns:
    ========
    hdr:
        azimuthal shear calculated via the linear least squares derivitives method
    """
    
    #define the indices for the required sweep
    sweepidx = radar.get_start_end(sweep)
    
    #data quality controls 
    #dealiase(radar,vel_name)
    smooth_data(radar,ref_name)
    smooth_data(radar,vel_name)
    
    #extract data
    r=radar.range['data']
    theta=radar.azimuth['data'][sweepidx[0]:sweepidx[1]+1]
    theta=theta*np.pi/180
    refl = radar.fields[ref_name]['data'][sweepidx[0]:sweepidx[1]+1]
    vrad = radar.fields[vel_name]['data'][sweepidx[0]:sweepidx[1]+1]
    r,theta = np.meshgrid(r,theta)
    
    #set the constants definining the LLSD grid in the azimuthal and radial directions
    azi_saxis = 2000 #km
    rng_saxis = 1  #idx away from i
    
    #convert from cylindinrical to cartesian coords
    x  = r* np.cos(theta)
    y  = r* np.sin(theta)

    sz = vrad.shape
    azi_shear = np.zeros(sz)
    
    #begin looping over grid
    for i in range(0,sz[0]):
        for j in range(0+rng_saxis,sz[1]-rng_saxis):
            #defining the amount of index offsets for azimuthal direction
            arc_len_idx_offset = int(azi_saxis//((2*r[i,j]*np.pi)/360)) #arc length as a fraction or circ
            #limit the offset to 100 
            if arc_len_idx_offset>100:
                arc_len_idx_offset=100
            #define the indices for the LLSd grid and deal with wrapping
            lower_arc_idx      = i - arc_len_idx_offset              
            upper_arc_idx      = i + arc_len_idx_offset
            if lower_arc_idx < 0:
                lower_arc_idx = lower_arc_idx + sz[0]
            if upper_arc_idx > sz[0]-1:
                upper_arc_idx = upper_arc_idx - sz[0]                 
            if upper_arc_idx < lower_arc_idx:
                ii_range = np.concatenate((np.arange(lower_arc_idx,sz[0],1),np.arange(0,upper_arc_idx+1,1)),axis=0)
            else:
                ii_range = range(lower_arc_idx,upper_arc_idx+1)
                
            #perform calculations according to Miller et al., (2013)
            topsum=0
            botsum=0
            dsum=0
            for ii in ii_range:         
                for jj in range(j-rng_saxis+1,j+rng_saxis):
                    dtheta=(theta[ii,jj]-theta[i,j])
                    #ensure the angle difference doesnt wrap onto another tilt
                    if (abs(dtheta)>np.pi) and (dtheta>0):
                        dtheta=((theta[ii,jj]-2*np.pi)-theta[i,j])
                    elif (abs(dtheta)>np.pi) and (dtheta<0):
                        dtheta=(theta[ii,jj])-(theta[i,j]-2*np.pi)
                    topsum = topsum + (r[ii,jj]*dtheta)*vrad[ii,jj]
                    botsum = botsum + (r[ii,jj]*dtheta)**2
               
            azi_shear[i,j]=topsum/botsum
            
            #exclude areas where there is only one point in each grid
            if botsum==0:
                azi_shear[i,j]=np.nan
                
    #mask according to reflectivity 
    azi_shear=ref_mask(refl,azi_shear,40,4)
    
    #define meta data
    azi_shear_meta = {'data': azi_shear, 'long_name': 'LLSD Azimuthal Shear', 
                      'standard_name': 'Azimuthal Shear', 'units': 'second$^{-1}$'}
    #return shear data 
    return azi_shear_meta

