# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:45:16 2023

@author: Luisa Lucchese, based on r.sunmask, tool from GRASS GIS
This was translated from C to Python. I kept the original citations that were
included in the original C files. 

Contributors of the original code:
    Janne Soimasuo, Finland 1994 (original contributor)
    update to FP by Huidae Cho <grass4u gmail.com> 2001
    added solpos algorithm feature by Markus Neteler 2001
    Brad Douglas <rez touchofmadness.com>,
    Glynn Clements <glynn gclements.plus.com>,
    Hamish Bowman <hamish_b yahoo.com>,
    Paul Kelly <paul-grass stjohnspoint.co.uk>

inputs:
    - filename: filename and path of the DEM
    - pointlon: longitude of the in situ point
    - pointlat: latitude of the in situ point
    - yearstr: string of the year
    - daystr: string of the day in julian format
    - hourstr: string of the hour
    - minstr: string of the minute
    - secstr: string of the seconds
    - dst_filename: filename in which I want to save my shadow output
    
outputs:
    - output: array of the sunmask
"""


def function_sunmask(filename,pointlon,pointlat,yearstr,daystr,hourstr,minstr,secstr,dst_filename):

    import math
    import numpy as np
    import osr
    from osgeo import gdal
    import copy
    
    #year=float(2000)
    #daynum = float(172)#float(250)
    degrad = 57.295779513# /* converts from radians to degrees */
    raddeg = 0.0174532925# /* converts from degrees to radians */
    interval =0.0 #default value
    year=float(yearstr)
    daynum=float(daystr)
    hour=float(hourstr)
    minute=float(minstr)
    second=float(secstr)
    timezone=0.0
    
    #aspect  # /* Azimuth of panel surface (direction it faces) N=0, E=90, S=180, W=270 */
    #tilt  # /* Degrees tilt from horizontal of panel */   
    #solcon #/* Solar constant, 1367 W/sq m */
    
    press = 1013.0    # /* Surface pressure, millibars */
    temp = 15.0       # /* Ambient dry-bulb temperature, degrees C */
    
    sd=-999 #initialization 
    
    # things that would be in the GRASS region definition
    #ns_res
    #ew_res
    
    
    # I will load a raster and test how it performs running this code here
    
    #filename="D:/Luisa/start_insitu_data_wrongdata/DEM_MERIT_DEM/reprojected_for_test.tif"
    
    step1 = gdal.Open(filename, gdal.GA_ReadOnly)
    #get location and size of the file
    GT_input = step1.GetGeoTransform()
    minx = GT_input[0] #lon
    maxy = GT_input[3] #lat
    maxx = minx + GT_input[1] * step1.RasterXSize #step1.RasterXSize is the same as size2
    miny = maxy + GT_input[5] * step1.RasterYSize
    #projection=step1.GetProjection()
    
    step2 = step1.GetRasterBand(1)
    img_as_array = step2.ReadAsArray()
    size1,size2=img_as_array.shape
    
    output=np.ones(shape=(size1,size2)) #the shadow
    
    ns_res=abs(GT_input[1])
    ew_res=abs(GT_input[5])
    
    longitude=pointlon#-59.8#maxx #longitude=107.6
    latitude=pointlat#-2.9#maxy #latitude=12.26
    
    windowrows=size1
    windowcols=size2

    
    #day angle
    dayang = 360.0 * (daynum - 1) / 365.0;
    
    #    /* Earth radius vector * solar constant = solar energy */
    #    /*  Spencer, J. W.  1971.  Fourier series representation of the
    #       position of the sun.  Search 2 (5), page 172 */
    
    sineday = math.sin(raddeg * dayang); #previously sd but caused conflict
    cd = math.cos(raddeg * dayang);
    d2 = 2.0 * dayang;
    c2 = math.cos(raddeg * d2);
    s2 = math.sin(raddeg * d2);
    
    erv = 1.000110 + 0.034221 * cd + 0.001280 * sineday; #sineday previously sd but caused conflict
    erv = erv + 0.000719 * c2 + 0.000077 * s2;
    
    #    /* Universal Coordinated (Greenwich standard) time */
    #    /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #       approximate solar position (1950-2050).  Solar Energy 40 (3),
    #       pp. 227-235. */
    utime = hour * 3600.0 + minute * 60.0 + second -interval / 2.0;
    utime = utime / 3600.0 - timezone;
    
        # /* Julian Day minus 2,400,000 days (to eliminate roundoff errors) */
        # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
        #    approximate solar position (1950-2050).  Solar Energy 40 (3),
        #    pp. 227-235. */
    
        # /* No adjustment for century non-leap years since this function is
        #    bounded by 1950 - 2050 */
    delta = year - 1949;
    leap = int(delta / 4.0);
    julday = 32916.5 + delta * 365.0 + leap + daynum +utime / 24.0;
    
        # /* Time used in the calculation of ecliptic coordinates */
        # /* Noon 1 JAN 2000 = 2,400,000 + 51,545 days Julian Date */
        # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
        #    approximate solar position (1950-2050).  Solar Energy 40 (3),
        #    pp. 227-235. */
    ectime = julday - 51545.0;
    
    # /* Mean longitude */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    mnlong = 280.460 + 0.9856474 * ectime;
    
    # /* (dump the multiples of 360, so the answer is between 0 and 360) */
    mnlong -= 360.0 * int(mnlong / 360.0)
    if (mnlong < 0.0):
        mnlong =mnlong+ 360.0
    
    # /* Mean anomaly */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    mnanom = 357.528 + 0.9856003 * ectime
    
    # /* (dump the multiples of 360, so the answer is between 0 and 360) */
    mnanom -= 360.0 * int(mnanom / 360.0)
    if (mnanom < 0.0):
        mnanom += 360.0;
    
    # /* Ecliptic longitude */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    eclong = mnlong + 1.915 * math.sin(mnanom * raddeg) + 0.020 * math.sin(2.0 * mnanom * raddeg);
    
    # /* (dump the multiples of 360, so the answer is between 0 and 360) */
    eclong -= 360.0 * int(eclong / 360.0);
    if (eclong < 0.0):
        eclong += 360.0;
    
    # /* Obliquity of the ecliptic */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    
    # /* 02 Feb 2001 SMW corrected sign in the following line */
    # /*  pdat->ecobli = 23.439 + 4.0e-07 * pdat->ectime;     */
    ecobli = 23.439 - 4.0e-07 * ectime;
    
    # /* Declination */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    declin = degrad * math.asin(math.sin(ecobli * raddeg) * math.sin(eclong * raddeg))
    
    # /* Right ascension */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    top = math.cos(raddeg * ecobli) * math.sin(raddeg * eclong);
    bottom = math.cos(raddeg * eclong);
    
    rascen = degrad * math.atan2(top, bottom);
    
    # /* (make it a positive angle) */
    if (rascen < 0.0):
        rascen += 360.0;
    
    # /* Greenwich mean sidereal time */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    gmst = 6.697375 + 0.0657098242 * ectime + utime;
    
    # /* (dump the multiples of 24, so the answer is between 0 and 24) */
    gmst -= 24.0 * int(gmst / 24.0);
    if (gmst < 0.0):
        gmst += 24.0;
    
    # /* Local mean sidereal time */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    lmst = gmst * 15.0 + longitude;
    
    # /* (dump the multiples of 360, so the answer is between 0 and 360) */
    lmst -= 360.0 * int(lmst / 360.0);
    if (lmst < 0.):
        lmst += 360.0;
    
    # /* Hour angle */
    # /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
    #    approximate solar position (1950-2050).  Solar Energy 40 (3),
    #    pp. 227-235. */
    hrang = lmst - rascen;
    
        # /* (force it between -180 and 180 degrees) */
    if (hrang < -180.0):
        hrang += 360.0
    else:
        if (hrang > 180.0):
            hrang -= 360.0
    
    
    #localtrig(pdat, tdat);
    #print(sd)
    if (sd < -900.0): #{ /* sd was initialized -999 as flag */
        sd = 1.0      #/* reflag as having completed calculations */
        cd = math.cos(raddeg * declin)
        ch = math.cos(raddeg * hrang)
        cl = math.cos(raddeg * latitude)
        sd = math.sin(raddeg * declin)
        sl = math.sin(raddeg * latitude)
        
    #end of localtrig
    
    cz = sd * sl + cd * cl * ch;
    
    # /* (watch out for the roundoff errors) */
    if (abs(cz) > 1.0):
        if (cz >= 0.0):
            cz = 1.0
        else:
            cz = -1.0
    
    zenetr = math.acos(cz) * degrad;
    
    # /* (limit the degrees below the horizon to 9 [+90 -> 99]) */
    if (zenetr > 99.0):
        zenetr = 99.0
    
    elevetr = 90.0 - zenetr;
    
    '''
        if (pdat->function & L_ZENETR) /* etr at non-refracted zenith angle */
            zen_no_ref(pdat, tdat);
    
        if (pdat->function & L_SSHA) /* Sunset hour calculation */
            ssha(pdat, tdat);
            '''
    
    #localtrig(pdat, tdat);
    #print(sd)
    if (sd < -900.0): #{ /* sd was initialized -999 as flag */
        sd = 1.0      #/* reflag as having completed calculations */
        cd = math.cos(raddeg * declin)
        ch = math.cos(raddeg * hrang)
        cl = math.cos(raddeg * latitude)
        sd = math.sin(raddeg * declin)
        sl = math.sin(raddeg * latitude)
        
    #end of localtrig

    
    
    '''
    
        if (pdat->function & L_SBCF) /* Shadowband correction factor */
            sbcf(pdat, tdat);
            '''
        
            
            
    '''
    
        if (pdat->function & L_TST) /* true solar time */
            tst(pdat);
            '''
            
            
     # /*============================================================================
     #  *    Local Void function tst
     #  *
     #  *    TST -> True Solar Time = local standard time + TSTfix, time
     #  *      in minutes from midnight.
     #  *        Iqbal, M.  1983.  An Introduction to Solar Radiation.
     #  *            Academic Press, NY., page 13
     #  *----------------------------------------------------------------------------*/
     # static void tst(struct posdata *pdat)
     # {
    tst = (180.0 + hrang) * 4.0;
    tstfix =tst - hour * 60.0 - minute - second / 60.0 + interval / 120.0 #/* add back half of the interval */
    
    #/* bound tstfix to this day */
    while (tstfix > 720.0):
        tstfix = tstfix- 1440.0
    while (tstfix < -720.0):
        tstfix = tstfix + 1440.0
    
  
            
    '''
    
        if (pdat->function & L_SRSS) /* sunrise/sunset calculations */
            srss(pdat);
            '''
            
    # /*============================================================================
    #  *    Local Void function srss
    #  *
    #  *    Sunrise and sunset times (minutes from midnight)
    #  *----------------------------------------------------------------------------*/
    # static void srss(struct posdata *pdat)
    # {


     
            
            
    '''
    
        if (pdat->function & L_SOLAZM) /* solar azimuth calculations */
            sazm(pdat, tdat);
            '''
            
    # /*============================================================================
    #  *    Local Void function sazm
    #  *
    #  *    Solar azimuth angle
    #  *       Iqbal, M.  1983.  An Introduction to Solar Radiation.
    #  *            Academic Press, NY., page 15
    #  *----------------------------------------------------------------------------*/
    # static void sazm(struct posdata *pdat, struct trigdata *tdat)
    # {
    #     float ca;   /* cosine of the solar azimuth angle */
    #     float ce;   /* cosine of the solar elevation */
    #     float cecl; /* ( ce * cl ) */
    #     float se;   /* sine of the solar elevation */
    
        #localtrig(pdat, tdat);
    ce = math.cos(raddeg * elevetr);
    se = math.sin(raddeg * elevetr);
    
    azim = 180.0
    cecl = ce * cl
    if (abs(cecl) >= 0.001):
        ca = (se * sl - sd) / cecl
        if (ca > 1.0):
            ca = 1.0
        else:
            if (ca < -1.0):
                ca = -1.0
    
        azim = 180.0 - math.acos(ca) * degrad;
        if (hrang > 0):
            azim = 360.0 - azim
              
        
    '''
    
        if (pdat->function & L_REFRAC) /* atmospheric refraction calculations */
            refrac(pdat);
            '''
            
    # /*============================================================================
    #  *    Local Int function refrac
    #  *
    #  *    Refraction correction, degrees
    #  *        Zimmerman, John C.  1981.  Sun-pointing programs and their
    #  *            accuracy.
    #  *            SAND81-0761, Experimental Systems Operation Division 4721,
    #  *            Sandia National Laboratories, Albuquerque, NM.
    #  *----------------------------------------------------------------------------*/
    # static void refrac(struct posdata *pdat)
    # {
    #     float prestemp; /* temporary pressure/temperature correction */
    #     float refcor;   /* temporary refraction correction */
    #     float tanelev;  /* tangent of the solar elevation angle */
    
    
    
    #     /* If the sun is near zenith, the algorithm bombs; refraction near 0 */
    if (elevetr > 85.0):
        refcor = 0.0
    
        # /* Otherwise, we have refraction */
    else:
        tanelev = math.tan(raddeg * elevetr)
        if (elevetr >= 5.0):
            refcor = (58.1 / tanelev - 0.07 / (tanelev**3) + 0.000086 / (tanelev**5))/3600
        else:
            if (elevetr >= -0.575):
                # refcor = 1735.0 +
                #          pdat->elevetr *
                #              (-518.2 +
                #               pdat->elevetr *
                #                   (103.4 + pdat->elevetr *
                #                                (-12.79 + pdat->elevetr * 0.711)));
                # couldnt find the source,
                # replaced it by the NOAA equation
                # https://gml.noaa.gov/grad/solcalc/calcdetails.html
                refcor = (1/3600)*(1735-518.2*elevetr+103.4*elevetr**2-12.79*elevetr**3+0.711*elevetr**4)  #1735.0 + elevetr *(-518.2 + elevetr *( 103.4 + elevetr *(-12.79 + elevetr * 0.711)))
            else:
                refcor = -20.774 / tanelev
                prestemp = (press * 283.0) / (1013.0 * (273.0 + temp))
                refcor = refcor*( prestemp / 3600.0)
    
    #something is going wrong on the refcor (coeficient of refraction calculation)
    #so I will just replace whatever is going on
    
    #tanelev = math.tan(raddeg * elevetr)
    #R=1.02/(math.tan(raddeg * elevetr + raddeg *(10.3/(elevetr+5.11)))) #Saemundsson's formula
    #prestemp = (press * 283.0) / (1010.0 * (273.0 + temp))
    #refcor=(R*prestemp)
    
    
    #print(refcor)
    # refcor = 58.1 / tanelev - 0.07 / (tanelev**3) + 0.000086 / (tanelev**5)
    
    # refcor = -20.774 / tanelev
    # prestemp = (press * 283.0) / (1013.0 * (273.0 + temp));
    # refcor = refcor*( prestemp / 3600.0)
    
    
    
    # /* Refracted solar elevation angle */
    elevref = elevetr + refcor
    
    #/* (limit the degrees below the horizon to 9) */
    if (elevref < -9.0):
        elevref = -9.0
    
    #/* Refracted solar zenith angle */


         
            
    '''
        if (pdat->function & L_AMASS) /* airmass calculations */
            amass(pdat);
            '''
    
    # /*============================================================================
    #  *    Local Void function  amass
    #  *
    #  *    Airmass
    #  *       Kasten, F. and Young, A.  1989.  Revised optical air mass
    #  *            tables and approximation formula.  Applied Optics 28 (22),
    #  *            pp. 4735-4738
    #  *----------------------------------------------------------------------------*/
    # static void amass(struct posdata *pdat)
    # {

    
    
    '''
        if (pdat->function & L_PRIME) /* kt-prime/unprime calculations */
            prime(pdat);
            '''
    
    # /*============================================================================
    #  *    Local Void function prime
    #  *
    #  *    Prime and Unprime
    #  *    Prime  converts Kt to normalized Kt', etc.
    #  *       Unprime deconverts Kt' to Kt, etc.
    #  *            Perez, R., P. Ineichen, Seals, R., & Zelenka, A.  1990.  Making
    #  *            full use of the clearness index for parameterizing hourly
    #  *            insolation conditions. Solar Energy 45 (2), pp. 111-114
    #  *----------------------------------------------------------------------------*/
    # static void prime(struct posdata *pdat)
    # {


    
    
    
    '''
        if (pdat->function & L_ETR) /* ETR and ETRN (refracted) */
            etr(pdat);
            '''
            
            
     # /*============================================================================
     #  *    Local Void function etr
     #  *
     #  *    Extraterrestrial (top-of-atmosphere) solar irradiance
     #  *----------------------------------------------------------------------------*/
     # static void etr(struct posdata *pdat)
     # {
    
            
            
            
            
    '''
        if (pdat->function & L_TILT) /* tilt calculations */
            tilt(pdat);
            '''
    # /*============================================================================
    #  *    Local Void function tilt
    #  *
    #  *    ETR on a tilted surface
    #  *----------------------------------------------------------------------------*/
    # static void tilt(struct posdata *pdat)
    # {
        # float ca; /* cosine of the solar azimuth angle */
        # float cp; /* cosine of the panel aspect */
        # float ct; /* cosine of the panel tilt */
        # float sa; /* sine of the solar azimuth angle */
        # float sp; /* sine of the panel aspect */
        # float st; /* sine of the panel tilt */
        # float sz; /* sine of the refraction corrected solar zenith angle */
        # /* Cosine of the angle between the sun and a tipped flat surface,
        #    useful for calculating solar energy on tilted surfaces */
        
        
     
    
    
    ca = math.cos(raddeg * azim)


    

        
    #retval = S_solpos(pdat);
          
            
            
    '''
            start of the remaining of the function - calculating shadows
            
    '''
    
    # if solpos --> if calculating the solar position
    dalti = elevref #confirmed that these work
    dazi = azim #confirmed that these work
    
    
    #just as a test
    #dalti=30
    #dalti=elevetr
    #dazi=0
    
    
    azi = 2 * math.pi * dazi / 360
    alti = 2 * math.pi * dalti / 360
    nstep = abs(math.cos(azi) * ns_res)#window.ns_res #ns - north south resolution
    estep = abs(math.sin(azi) * ew_res)#window.ew_res # ew - east west resolution 
    row1 = -1
    
    if (GT_input[5]>0): #same for y, necessary to keep the order of the direction
        north=miny
    else: 
        north=maxy
    
    #get the maximum value of the raster
    dmax = np.max(img_as_array)
    
    #G_message(_("Calculating shadows from DEM..."));
    while (row1 < (windowrows-1)):
        #G_percent(row1, windowrows, 2) #percent of work complete, not useful here
        col1 = -1
        drow = -1
        row1 = row1+ 1
        # Rast_get_row(elev_fd, elevbuf.v, row1, data_type) #read the data and buffer  - commented because we will already have read it
        
    
        
        #print('updated north and north1, row, col '+str(row1)+ ' , '+str(col1))
       
        while (col1 < (windowcols-1) ):
            col1 =col1+ 1
            dvalue= img_as_array[row1,col1]
            #dvalue = raster_value(elevbuf, data_type, col1); #unnecessary because we will already be acessing the whole raster
            #/*              outbuf.c[col1]=1; */
            # Rast_set_null_value(&outbuf.c[col1], 1, CELL_TYPE); #buffers a null value - unnecessary
            
            if (GT_input[1]>0): #necessary to keep the order of the direction
                coordlocal_x=minx + (col1+0.5) * GT_input[1]
            else:
                coordlocal_x=maxx + (col1+0.5) * GT_input[1]    
    
            
            east=copy.deepcopy(coordlocal_x)
            east1 = east
            #north=copy.deepcopy(coordlocal_y)
            #print(str(east)+' , '+str(north))
            #east = Rast_col_to_easting(col1 + 0.5, &window);
            #north = Rast_row_to_northing(row1 + 0.5, &window);
            #get the easting and northing specific to each point here
            
            
            
            OK = 1 #not okay yet
            #if (dvalue == 0.0): #if the value in the raster point is zero -- this was removed because we always use the NO option on QGIS
            #    OK = 0
            while (OK == 1) and (drow<(windowrows-1)):
                if (GT_input[1]>0): #necessary to keep the order of the direction
                    east=east+estep
                else:
                    east=east-estep
                #east = east+ estep
    
    
    
                if (GT_input[5]>0): #same for y, necessary to keep the order of the direction
                    coordlocal_y=miny + (row1+0.5) * GT_input[5]
                    #north=miny
                else:
                    coordlocal_y=maxy + (row1+0.5) * GT_input[5] 
                    #north=maxy
                    #north1 = north
       
                north=copy.deepcopy(coordlocal_y)
                north1=north
                if (GT_input[5]>0): #same for y, necessary to keep the order of the direction
                    north=north+nstep
                else:
                    north=north-nstep
    
                if (north>maxy) or (north<miny) or (east>maxx) or (east<minx):
                    OK=0
                else:
                    
                    #north=north+nstep
                    maxh = math.tan(alti) * math.sqrt((north1 - north) * (north1 - north) + (east1 - east) * (east1 - east))
                    if ((maxh) > (dmax - dvalue)): #checks maximum value
                        OK = 0 #okay
                    else:
                        #come back and transform given easting to column number
                        if (GT_input[1]>0): #necessary to keep the order of the direction
                            dcol=round((east-minx)/(GT_input[1])-0.5)
                        else:
                            dcol=round((east-maxx)/(GT_input[1])-0.5)
                            #
                        #dcol = Rast_easting_to_col(east, &window)
    # =============================================================================
                        if (GT_input[5]>0): #same for y, necessary to keep the order of the direction
    #                         #coordlocal_y=miny + (row1+0.5) * GT_input[5]
                             drow_test=round((north-miny)/(GT_input[5])-0.5)
                        else:
                             drow_test=round((north-maxy)/(GT_input[5])-0.5)
    # =============================================================================
                            #coordlocal_y=maxy + (row1+0.5) * GT_input[5] 
                        #row_use=row1 #save before it changes
                        if (drow != drow_test):
                             drow=copy.deepcopy(drow_test)
                             #row_use=drow
                             
                        #if (drow != Rast_northing_to_row(north, &window)):
                        #   drow = Rast_northing_to_row(north, &window)
                        #   Rast_get_row(elev_fd, tmpbuf.v, (int)drow,
                        #                 data_type)
                        #print(drow)
                        #print(OK)
                        dvalue2 = img_as_array[drow,dcol]#raster_value(tmpbuf, data_type, (int)dcol);
                        if ((dvalue2 - dvalue) > (maxh)):
                            OK = 0 #okay
                            output[row1,col1]=np.nan
                            #print(OK)
                            #outbuf.c[col1] = 1 #this place has shadows, that is what it means
                            
                            
    

                
    driver = gdal.GetDriverByName( 'GTiff' )
    dst_ds=driver.Create(dst_filename,size2,size1,1,gdal.GDT_Float64)
    dst_ds.SetGeoTransform(GT_input)
    srs = osr.SpatialReference()
    #srs.SetWellKnownGeogCS( 'WGS84' )
    dst_ds.SetProjection( srs.ExportToWkt() )
    export=dst_ds.GetRasterBand(1).WriteArray( output )
    export = None
    

    return output    
        
        