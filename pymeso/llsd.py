"""
LLSD azimuthal shear calculation
Miller, M. L., Lakshmanan, V., and Smith, T. M. (2013). An Automated Method for Depicting Mesocy-
clone Paths and Intensities. Weather and Forecasting, 28(3): 570-585.
Jordan Brook - 3 July 2018
"""

import pyart
from numba import int64, float64, jit
import numpy as np
import scipy

def smooth_data(radar, data_name):
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
    data           = radar.fields[data_name]['data']
    smooth_data    = scipy.ndimage.filters.median_filter(data, 3)
    smooth_data_ma = np.ma.masked_where(np.ma.getmask(data), smooth_data)
    radar.add_field_like(data_name, 
                     data_name, 
                     smooth_data_ma, replace_existing = True)
    
def ref_mask(ref,threshold,dilution):
    """
    Mask shear values based on reflectivity and dilate using scipy's dilation tools 
    Parameters:
    ===========
    ref: array
        reflectivity array in radar coordinates
    threshold: float
        reflecitvity threshold for masking
    dilution: int
        number of dilation pixels, refer to scipy.ndimage.binary_dilation() doc.
    Returns:
    ========
    mask
    
    """
    mask = np.zeros(ref.shape)
    mask[ref > threshold] = 1
    mask = scipy.ndimage.binary_dilation(mask, iterations = dilution).astype(bool)
    return np.invert(mask)

def main(radar, ref_name, vel_name):
    """
    Main processing function for LLSD, applies smoothing and masks before calling llsd compute
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
    FILLVALUE = -9999
    SCALING = 1000
    
    #define the indices for the required sweep
    sweep_startidx = int64(radar.sweep_start_ray_index['data'][:])
    sweep_endidx = int64(radar.sweep_end_ray_index['data'][:])
    
    #data quality controls on entire volume
    smooth_data(radar, vel_name)
    
    #extract data
    r        = radar.range['data']
    theta    = radar.azimuth['data']
    theta    = theta*np.pi/180
    refl_ma  = radar.fields[ref_name]['data']
    vrad_ma  = radar.fields[vel_name]['data']
    vrad     = np.ma.filled(vrad_ma, fill_value=0)
    mask     = np.ma.getmask(vrad_ma)
    r, theta = np.meshgrid(r, theta)
    
    #call llsd compute function
    azi_shear = lssd_compute(r, theta, vrad, mask, sweep_startidx, sweep_endidx)
    #scale
    azi_shear = azi_shear*SCALING
    
    #generate mask according to reflectivity 
    refl_mask = ref_mask(refl_ma, 40, 8)
    #combine with vrad mask
    azi_mask  = np.logical_or(refl_mask, mask)
    # apply combined mask to azi_shear
    azi_shear = np.ma.masked_where(azi_mask, azi_shear).astype(np.float32)
    
    #define meta data
    azi_shear_meta = {'data': azi_shear,
                      'long_name': 'LLSD Azimuthal Shear', 
                      '_FillValue': FILLVALUE,
                      '_Least_significant_digit': 2,
                      'comment': f'Scaled by x{SCALING}. LLSD azimuthal shear calculation from Miller, M. L., Lakshmanan, V., and Smith, T. M. (2013). An Automated Method for Depicting Mesocyclone Paths and Intensities. Weather and Forecasting, 28(3): 570-585. Effective range of this technique is limited by the window size.',
                      'units': 'Hz'}
    #return shear data 
    return azi_shear_meta

#compile using jit
@jit(nopython=True)
def lssd_compute(r, theta, vrad, mask, sweep_startidx, sweep_endidx):
    """
    Compute core for llsd, uses numpy only functions and numba jit.
    Parameters:
    ===========
    r: array
        volume radar range array (2D) (m)
    theta: array
        volume radar azimuth angle array (2D) (radians)
    vrad: array
        volume radial velocity array (2D) (m/s)
    mask: logical array
        volume mask for valid radial velocities (2D)
    sweep_startidx: numba int64 array
        index of starting rays for tilts
    sweep_endidx: numba int64 array
        index of ending rays for tilts
                
    Returns:
    ========
    azi_shear:
        azimuthal shear calculated via the linear least squares derivitives method
    """
    
    #set the constants definining the LLSD grid in the azimuthal and radial directions
    azi_saxis = 2000 #m              #notes: total azimuthal size = 2*azi_saxis
    rng_saxis = 1  #idx away from i  #notes: total range size = 2*rng_saxis
    
    #init az_shear for the volume
    azi_shear = np.zeros(vrad.shape)
    
    #begin looping over grid
    for k in np.arange(len(sweep_startidx)):
        #subset volume into tilt
        r_tilt     = r[sweep_startidx[k]:sweep_endidx[k]+1]
        theta_tilt = theta[sweep_startidx[k]:sweep_endidx[k]+1]
        vrad_tilt  = vrad[sweep_startidx[k]:sweep_endidx[k]+1]
        mask_tilt  = mask[sweep_startidx[k]:sweep_endidx[k]+1]
        #convert from cylindinrical to cartesian coords
        x  = r_tilt * np.cos(theta_tilt)
        y  = r_tilt * np.sin(theta_tilt)
        #get size and init az_shear_tilt
        sz = vrad_tilt.shape
        azi_shear_tilt = np.zeros(sz)
        
        
        for i in np.arange(0, sz[0]):
            for j in np.arange(0 + rng_saxis, sz[1] - rng_saxis):
                #skip if j is invalid
                if mask_tilt[i, j]:
                    continue
                #defining the amount of index offsets for azimuthal direction
                arc_len_idx_offset = int64([(azi_saxis//((2*r_tilt[i, j]*np.pi)/360))])[0] #arc length as a fraction or circ
                #limit the offset to 100 
                if arc_len_idx_offset > 100:
                    arc_len_idx_offset = 100
                #define the indices for the LLSd grid and deal with wrapping
                lower_arc_idx = i - arc_len_idx_offset              
                upper_arc_idx = i + arc_len_idx_offset
                if lower_arc_idx < 0:
                    lower_arc_idx = lower_arc_idx + sz[0]
                if upper_arc_idx > sz[0]-1:
                    upper_arc_idx = upper_arc_idx - sz[0]                 
                if upper_arc_idx < lower_arc_idx:
                    ii_range = np.concatenate((np.arange(lower_arc_idx, sz[0], 1), np.arange(0, upper_arc_idx+1 ,1)), axis=0)
                else:
                    ii_range = np.arange(lower_arc_idx, upper_arc_idx+1)
                #define jj range
                jj_range = np.arange(j-rng_saxis+1, j+rng_saxis)
                #perform calculations according to Miller et al., (2013)
                topsum = 0
                botsum = 0
                masked = False
                for ii in ii_range:         
                    for jj in jj_range:
                        dtheta = (theta_tilt[ii, jj] - theta_tilt[i, j])
                        #ensure the angle difference doesnt wrap onto another tilt
                        if (abs(dtheta) > np.pi) and (dtheta > 0):
                            dtheta = ((theta_tilt[ii, jj]-2*np.pi) - theta_tilt[i, j])
                        elif (abs(dtheta) > np.pi) and (dtheta < 0):
                            dtheta=(theta_tilt[ii, jj]) - (theta_tilt[i, j]-2*np.pi)
                        topsum = topsum + (r_tilt[ii, jj]*dtheta) * vrad_tilt[ii, jj]
                        botsum = botsum + (r_tilt[ii, jj]*dtheta)**2
                        if mask_tilt[ii, jj]:
                            masked = True
                
                if masked:
                    #exclude regions which contain any masked pixels
                    pass
                elif botsum == 0:
                    #exclude areas where there is only one point in each grid
                    pass
                else:
                    azi_shear_tilt[i, j] = topsum/botsum

        #insert az shear tilt into volume array
        azi_shear[sweep_startidx[k]:sweep_endidx[k]+1] = azi_shear_tilt

    return azi_shear