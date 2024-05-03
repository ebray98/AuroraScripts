import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import statistics as st
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

plt.ion()
from omfit_classes import omfit_eqdsk
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import sys
import os
import time
import aug_sfutils as sf 

plt.close('all')

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

nzSumToCompare = [] 
DaveToCompare = []
VaveToCompare = []

# Import bolometric experimental results 

try:
    shot = int(sys.argv[1])
except:
    shot =  41279 #39233 #30 #29 #40172 #40043 #39649
print(shot)
sigs = {} 

spred = sf.SFREAD(shot, "SCL", ed=5)
blb = sf.SFREAD(shot,"BLB", ed=1)
 
timeFVC = blb.gettimebase("powFVC")
powerFVC = blb.getobject("powFVC")
radFVC = np.linspace(1,32,32) # ho 32 los del FVC   

TotPowFVC = np.nansum(powerFVC, axis=1) # faccio la somma di tutte le los available nel tempo  

# from 6.2 to 6.4 s - 2.5 to 2.7 s
lowIndex6 = np.argmax(timeFVC >= 6.2)
upIndex6 = np.argmax(timeFVC >= 6.4)
lowIndex2 = np.argmax(timeFVC >= 2.5)
upIndex2 = np.argmax(timeFVC >= 2.7)

# Per valutare il valore di P del Sn faccio la differenza tra il valore a 6s e quello a 2 e poi moltiplico per 1.15  
# Lo faccio per ogni line of sight, poi semmai faccio anche quello della somma 
powSn = np.zeros(powerFVC[lowIndex6:upIndex6].shape)
timeSn = np.linspace(0,0.2,len(powSn))
for los in range(1,len(radFVC)):
    powSn[:,los]  = (powerFVC[lowIndex6:upIndex6,los] - powerFVC[lowIndex2:upIndex2-1,los])*1.15

# Average of powSn nel tempo 
powSnMean = np.average(powSn, axis=0)
# Lo plotto ad ogni los -- poi trovo un modo per riportarlo sul raggio 
'''
plt.figure(1)
plt.scatter(radFVC, powSnMean)  
plt.title("FVC signals")
plt.xlabel('lines of sight')
plt.ylabel('W/m^2')
plt.grid()
plt.legend()
'''

timeFHC = blb.gettimebase("powFHC")
powerFHC = blb.getobject("powFHC")
radFHC = np.linspace(1,48,48) # ho 48 los dell' FHC   

TotPowFHC = np.nansum(powerFHC, axis=1) # faccio la somma di tutte le los available nel tempo  

# from 6.2 to 6.4 s - 2.5 to 2.7 s
lowIndex6H = np.argmax(timeFHC >= 6.2)
upIndex6H = np.argmax(timeFHC >= 6.4)
lowIndex2H = np.argmax(timeFHC >= 2.1) #2.5)
upIndex2H = np.argmax(timeFHC >= 2.3) #2.7)

# Per valutare il valore di P del Sn faccio la differenza tra il valore a 6s e quello a 2 e poi moltiplico per 1.15  
# Lo faccio per ogni line of sight, poi semmai faccio anche quello della somma 
powSnH = np.zeros(powerFHC[lowIndex6H:upIndex6H].shape)
timeSnH = np.linspace(0,0.2,len(powSnH))
for los in range(1,len(radFHC)):
    powSnH[:,los]  = (powerFHC[lowIndex6H:upIndex6H,los] - powerFHC[lowIndex2H:upIndex2H-1,los])*1.15

# Average of powSn nel tempo 
powSnMeanH = np.average(powSnH, axis=0)
# Lo plotto ad ogni los -- poi trovo un modo per riportarlo sul raggio 
'''
plt.figure(2)
plt.scatter(radFHC, powSnMeanH)  
plt.title("FHC signals")
plt.xlabel('lines of sight')
plt.ylabel('W/m^2')
plt.grid()
plt.legend()
'''

TotPow = TotPowFVC + TotPowFHC

#time = spred.gettimebase("B2_518")
Time = spred.gettimebase("Sn39_138")

FVCToPlot = np.interp(Time, timeFVC, TotPowFVC*0.286)
FHCToPlot = np.interp(Time, timeFHC, TotPowFHC*0.286)
PowerToPlot = np.interp(Time, timeFVC, TotPow*0.286)

idx6 = np.abs(Time-6.3).argmin()
idx2 = np.abs(Time-2.6).argmin()
MWave = (PowerToPlot[idx6] - PowerToPlot[idx2])*1e-6
print(f"Power 6.3-2.6s [MW] ", MWave)  
'''
plt.figure(3)
plt.title("Power from BLB")
plt.ylabel("Power [$W$]")
plt.xlabel("time[s]")
plt.grid()
plt.plot(Time, FVCToPlot, label = "FVC")
plt.plot(Time, FHCToPlot, label = "FHC")
plt.plot(Time, PowerToPlot, label = "Tot")
plt.legend()
'''


# Spectral results  

for sig_name in spred.getlist():
    if sig_name != "TIME" and "EMPTY" not in sig_name:
        sigs[sig_name] = spred.getobject(sig_name)


# plot all available line fits
nplots = len(sigs.keys())
ncol = int(np.sqrt(nplots))
nrow = int(np.ceil(float(nplots) / ncol))

'''
# plot all lines in a single plot
fig, axs = plt.subplots(nrow, ncol, sharex=True, figsize=(15, 8))
axx = axs.flatten()

ls_cycle = aurora.get_ls_cycle()
for ax, sig_name in zip(axx[: len(sigs.keys())], sigs.keys()):
    ax.plot(Time, sigs[sig_name], next(ls_cycle), label=sig_name)


for _ax in axs[-1, :]:
    _ax.set_xlabel("time [s]")
for _ax in axs[:, 0]:
    _ax.set_ylabel("Amp [A.U.]")

for ax in axs.flatten():
    ax.grid("on")
    ax.set_ylim([0.0, None])
    if ax.lines: # check if anything was plotted
        ax.legend(loc='best').set_draggable(True)
    else:
        ax.axis('off')


plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.95, left = 0.05, hspace = 0)
ax.set_xlim([0,10])
'''
# Ti e Vtor from experimental results #41278   
shot2 = 41278
time_shot2 = 6.3 

cuz = sf.SFREAD(shot2, "cuz")
time_cuz = cuz.gettimebase("Ti") #(250) 
it_cuz = np.argmin(np.abs(time_cuz-time_shot2))
rhop_cuz = cuz.getareabase("Ti") #(30,250)
Ticuz = cuz.getobject("Ti") #(250,30) 

#### AURORA RUN #####  

K_ave = []
H_ave = [] 
HoverK = []  
TSC = [] 

# I do it only for the case of rotation  
for m in range(0,3,2): 
    print(m)
    for k in [1,2,5]: 
        # user choices for testing:
        rotation_model = m  # recommended: 0 for light impurities, 2 for heavier than Ar
        '''
        a = 0.3e4
        b = 3.7e4
        c = 0.5
        def turbo(x):
            D_an = a+b*(1-np.exp(-c*x))
            return D_an 
        '''

        D_an = 1e4*k  # anomalous D, cm^2/s
        V_an = -1e2*k  # anomalous V, cm/s

        # function for the Z_Sn profile 

        def ZTIN(Temp): 
            
            T = np.arange(1, 10001)
            Z = np.log10(T.astype(float))
            Y = []


            for temperature in T:
                    if temperature < 100:
                        Y.append(0.)
                    elif 100 <= temperature < 300:
                        Y.append(-1.74287118071950e-01 * Z[temperature - 1]**3 +
                                1.54636650598728e+00 * Z[temperature - 1]**2 -
                                3.99141529108946e+00 * Z[temperature - 1]**1 +
                                4.33147417838939e+00 * Z[temperature - 1]**0)
                    elif 300 <= temperature < 1000:
                        Y.append(1.00244293446694e+00 * Z[temperature - 1]**3 -
                                7.97732921344918e+00 * Z[temperature - 1]**2 +
                                2.13382972994841e+01 * Z[temperature - 1]**1 -
                                1.78615834244534e+01 * Z[temperature - 1]**0)
                    elif 1000 <= temperature < 2000:
                        Y.append(3.42895052030529e-01 * Z[temperature - 1]**3 -
                                3.06822566369654e+00 * Z[temperature - 1]**2 +
                                9.53786318906057e+00 * Z[temperature - 1]**1 -
                                8.83692882480517e+00 * Z[temperature - 1]**0)
                    elif 2000 <= temperature < 5000:
                        Y.append(4.81585016923541e-01 * Z[temperature - 1]**3 -
                                5.25915388459379e+00 * Z[temperature - 1]**2 +
                                1.92606216460337e+01 * Z[temperature - 1]**1 -
                                2.20499427661916e+01 * Z[temperature - 1]**0)
                    elif 5000 <= temperature < 10000:
                        Y.append(-2.08321206186342e+00 * Z[temperature - 1]**3 +
                                2.39727274395118e+01 * Z[temperature - 1]**2 -
                                9.17468909033947e+01 * Z[temperature - 1]**1 +
                                1.18408176981176e+02 * Z[temperature - 1]**0)
                    else:
                        Y.append(9.91829921918504e-02 * Z[temperature - 1]**3 -
                                1.32853805480940e+00 * Z[temperature - 1]**2 +
                                5.94848074638099e+00 * Z[temperature - 1]**1 -
                                7.22498252575176e+00 * Z[temperature - 1]**0)

            ZITIN = 10**np.array(Y)

            ZTin_Interp = np.interp(Temp, T, ZITIN)
            return ZTin_Interp

        # ------------------

        shot = 41279
        time_shot = 6.3
        namelist = aurora.default_nml.load_default_namelist()

        geqdsk = OMFITgeqdsk("").from_aug_sfutils(shot= shot, time= time_shot, eq_shotfile="EQI")

        kp = namelist["kin_profs"] 

        ida = sf.SFREAD(shot, "ida")
        list_objects = ida.getlist()
        time_ida = ida.gettimebase("Te")# time base of the temperature profile 
        it_ida = np.argmin(np.abs(time_ida-time_shot))# returns the index of the time "time" in the array extracted from the aug data
        rhop_ida = ida.getareabase("Te")# Reads the areabase  
        Te_eV = ida.getobject("Te")
        ne_m3 = ida.getobject("ne")


        Tempo = np.linspace(0,7.298,1000)
        TeInterpolata = np.interp(Tempo, time_ida, Te_eV[0,:])
        #plt.figure()
        #plt.plot(time_ida, Te_eV[0,:])
        #plt.plot(Tempo, TeInterpolata)

        # assign the extract data to the namelist   
        rhop_kp = kp["Te"]["rhop"] = kp["ne"]["rhop"] = rhop_ida[:, it_ida] 
        kp["Te"]["vals"] = Te_eV[:,it_ida] # eV
        kp["ne"]["vals"] = ne_m3[:,it_ida] * 1e-6 # from m^-3 --> to cm^-3

        #lowIndex6H = np.argmax(Tempo >= 6.4)
        #upIndex6H = np.argmax(Tempo >= 6.2)
        idx2Inter = np.abs(Tempo-6.2).argmin()
        idx6Inter = np.abs(Tempo-6.4).argmin()
        Te_ave = (TeInterpolata[idx2Inter] + TeInterpolata[idx6Inter])/2

        # set impurity species and sources rate
        imp = namelist["imp"] = "Sn"  
        namelist["source_type"] = "const"
        namelist["source_rate"] = 0.8e19 #2e14 #particles/s

        oldSource = namelist["source_rate"]
        f = 0
        Ptarget = MWave*1e6
        err = 100
        tol = 1e-2
        PradTot = 1.7e6 

        while err > tol: 

            print(f'Source: ', oldSource)
            namelist["source_rate"] = oldSource*(1+f)

            # Now get aurora setup
            asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

            times_DV = np.array([0])
            nz_init = np.zeros((asim.rvol_grid.size, asim.Z_imp + 1))

            # initialize transport coefficients
            D_z = np.zeros((asim.rvol_grid.size, times_DV.size, asim.Z_imp + 1))  # space, time, nZ
            V_z = np.zeros(D_z.shape)
            K_z = np.zeros(D_z.shape)
            H_z = np.zeros(D_z.shape)
            K_zPS = np.zeros(D_z.shape)
            H_zPS = np.zeros(D_z.shape)
            K_zBP = np.zeros(D_z.shape)
            H_zBP = np.zeros(D_z.shape)

            # set time-independent anomalous transport coefficients
            Dz_an = np.zeros(D_z.shape)  # space, time, nZ
            Vz_an = np.zeros(D_z.shape)

            # set anomalous transport coefficients
            Dz_an[:20,0,:] = 0.5e4
            Dz_an[20:,0,:] = D_an
            Vz_an[:] = V_an

            # -------------------
            # prepare FACIT input
            rr = asim.rvol_grid / 100  # in m
            idxsep = np.argmin(np.abs(1.0 - asim.rhop_grid))  # index of radial position of separatrix 
            amin = rr[idxsep]  # minor radius in m
            roa = rr[: idxsep + 1] / amin  # normalized radial coordinate 

            B0 = np.abs(geqdsk["BCENTR"])  # magnetic field on axis
            R0 = geqdsk["fluxSurfaces"]["R0"]  # major radius

            qmag = np.interp(roa, geqdsk["RHOVN"], -geqdsk["QPSI"])[: idxsep + 1]  # safety factor
            rhop = asim.rhop_grid[: idxsep + 1]

            # profiles
            Ni = (np.interp(roa, rhop_kp, kp["ne"]["vals"]) * 1e6)  # in m**3 instead of cm**3 in FACIT 
            TeovTi =  2.0 # kp["Te"]["vals"]/kp["Ti"]["vals"] # electron to ion temperature ratio
            Ti = np.interp(roa, rhop_kp, kp["Te"]["vals"]) / TeovTi
            Te = np.interp(roa, rhop_kp, kp["Te"]["vals"])

            vtor = cuz.getobject("vrot")
            rhop_cuzInv = rhop_cuz[::-1] #(30,250) 
            TicuzInv = Ticuz[it_cuz,:][::-1] #(30) 
            vtorInv = vtor[it_cuz,:][::-1] #(30,) 
            indiceMax = np.argmax(TicuzInv)
            RhoFromLOS = np.linspace(0,1,30-indiceMax) #(22) 
            vtorInterp = np.interp(roa, RhoFromLOS[:-3], vtorInv[indiceMax:-3])
            Mi = abs(vtorInterp*1e-3)/(np.sqrt(2*Ti*1e3/asim.A_imp))
            '''
            plt.figure(1)
            plt.title("vTor")
            plt.plot(rhop_cuzInv[:,it_cuz], vtorInv)

            plt.figure(6)
            plt.title("vtorRho")
            plt.plot(roa,vtorInterp)
            plt.plot(RhoFromLOS, vtorInv[indiceMax:])

            plt.figure(2)
            plt.title("Ti")
            plt.plot(rhop_cuzInv[:,it_cuz], TicuzInv)
            '''
            gradNi = np.gradient(Ni, roa*amin)
            gradTi = np.gradient(Ti, roa*amin)

            gradTi[-1] = gradTi[-2]
            gradNi[-1] = gradNi[-2]

            grad_log_Ni = gradNi/Ni
            grad_log_Ti = gradTi/Ti
            OneOverEta = grad_log_Ni/grad_log_Ti

            Zeff = 1.5 * np.ones(roa.size)  # typical AUG value 

            ZIMP = ZTIN(kp["Te"]["vals"])
            ZIMP_INTERP = np.interp(roa, rhop_kp, ZIMP)

            #Actual charge states that are present in this simulation 
            ChargeStates = list(set(ZIMP_INTERP.astype(int)))
            Nstates = len(ChargeStates)

            # I tried to derive Machi starting from the profile of Mz_star in figure 2 that I reproduced in a .csv file 

            import csv 
            file = open('plot_data.csv')
            csvreader = csv.reader(file)
            header = [] 
            header = next(csvreader)
            rows = []

            for row in csvreader: 
                rows.append(row)

            file.close()

            gridCSV = [] 
            Mz_starRough = [] 
            for sottolista in rows: 
                primoValore = float(sottolista[0])
                secondoValore = float(sottolista[1])
                gridCSV.append(primoValore)
                Mz_starRough.append(secondoValore)

            Mz_star = np.interp(roa, gridCSV, Mz_starRough)

            # uncomment to begin simulation from a pre-existing profile
            c_imp = 1e-4 # trace concentration
            for k in range(nz_init.shape[1]):
                nz_init[:idxsep+1,k] = c_imp*Ni*1e-6 # in 1/cm**3


            if rotation_model == 0:

                Machi = np.zeros(
                    roa.size
                )  # no rotation (not that it matters with rotation_model=0)
                RV = None
                ZV = None

            elif rotation_model == 2:

                Machi = np.sqrt(Mz_star**2/(asim.A_imp/asim.main_ion_A - ZIMP_INTERP/asim.main_ion_Z * Zeff/(Zeff+TeovTi)))
                #Machi = abs(vtorInterp*1e-3)/(np.sqrt(2*Ti*1e3/asim.A_imp))

                nth = 51
                theta = np.linspace(0, 2 * np.pi, nth)

                RV, ZV = aurora.rhoTheta2RZ(geqdsk, rhop, theta, coord_in="rhop", n_line=201)
                RV, ZV = RV.T, ZV.T

            else:
                raise ValueError("Other options of rotation_model are not enabled in this example!")

            # ----------
            # call FACIT


            starttime = time.time()
            for j, tj in enumerate(times_DV):

                #for i, zi in enumerate(ChargeStates):
                for i, zi in enumerate(range(asim.Z_imp + 1)):

                    if zi != 0:
                        Nz = nz_init[: idxsep + 1, zi] * 1e6  # in 1/m**3
                        gradNz = np.gradient(Nz, roa * amin)

                        fct = aurora.FACIT(
                            roa,
                            zi,
                            asim.A_imp,
                            asim.main_ion_Z,
                            asim.main_ion_A,
                            Ti,
                            Ni,
                            Nz,
                            Machi,
                            Zeff,
                            gradTi,
                            gradNi,
                            gradNz,
                            amin / R0,
                            B0,
                            R0,
                            qmag,
                            rotation_model=rotation_model,
                            Te_Ti=TeovTi,
                            RV=RV,
                            ZV=ZV,
                        )

                        D_z[: idxsep + 1, j, zi] = fct.Dz * 100**2  # convert to cm**2/s
                        V_z[: idxsep + 1, j, zi] = fct.Vconv * 100  # convert to cm/s
                        K_z[: idxsep + 1, j, zi] = fct.Kz * 100**2  # convert to cm**2/s
                        H_z[: idxsep + 1, j, zi] = fct.Hz * 100**2  # convert to cm**2/s
                        K_zPS[: idxsep + 1, j, zi] = fct.Kz_PS * 100**2  # convert to cm**2/s
                        H_zPS[: idxsep + 1, j, zi] = fct.Hz_PS * 100**2
                        K_zBP[: idxsep + 1, j, zi] = fct.Kz_BP * 100**2  # convert to cm**2/s
                        H_zBP[: idxsep + 1, j, zi] = fct.Hz_BP * 100**2


            time_exec = time.time() - starttime
            print("FACIT exec time [s]: ", time_exec)

            # add anomalous transport
            D_z += Dz_an
            V_z += Vz_an


            # correction diffusion coefficients  
            target_value = 0.05
            diff = np.abs(asim.rhop_grid - target_value)
            indCorrection = np.argmin(diff)
            D_z[:indCorrection]  = D_z[indCorrection] 
            V_z[:indCorrection]  = V_z[indCorrection] 

            # run Aurora forward model and plot results
            out = asim.run_aurora(D_z, V_z, times_DV=times_DV, nz_init=None, plot=False)

            # extract densities and particle numbers in each simulation reservoir
            nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
    
            ## Definisci n0 con shape parabolica (guarda TestFACIT_Tin.py) se ottieni adas cx ##    

    
            # I try to compute the radiated power  !!! POI AGGIUNGI CX E NEUTRALS !!! 
            asim.rad = aurora.compute_rad(imp, nz.transpose(2,1,0), asim.ne, asim.Te, Ti=Ti, 
            prad_flag=True, spectral_brem_flag=False)
            PradTot = aurora.grids_utils.vol_int(asim.rad['tot'].transpose(1,0)[:,-1],asim.rvol_grid,asim.pro_grid, asim.Raxis_cm)
            print(f'PradTot: %d [MW] ', PradTot*1e-6)
            # I computed the average values of the transport coefficients K and H to reproduce the ones in the paper

            oldSource = namelist["source_rate"]

            weigth = np.arange(76,1,-1)
            K_ave.append(np.average(K_z, axis=2, weights=None))
            H_ave.append(np.average(H_z, axis=2, weights=None))
            #HoverK = H_ave/K_ave
            K_zPS_ave = np. average(K_zPS, axis=2, weights = None)
            H_zPS_ave = np. average(H_zPS, axis=2, weights = None)
            HoverK_PS = H_zPS_ave/K_zPS_ave
            K_zBP_ave = np. average(K_zBP, axis=2, weights = None)
            H_zBP_ave = np. average(H_zBP, axis=2, weights = None)
            HoverK_BP = H_zBP_ave/K_zBP_ave

            f = (Ptarget-PradTot)/Ptarget
            
            print(f'f:', f)
            err = abs(f)

            
            #HoverK = [H / K for H, K in zip(H_ave, K_ave)]

            print(f'PradTot: %d [MW] ', PradTot*1e-6)
            print(f'err:', err)
            print(f'Source: ', namelist["source_rate"])


            # Weighting of D and V to plot with respect to nz and nztot
            sum_curve = np.sum(nz*1e6, axis=1) 
        # end of while per matchare prad   



        fig, axs = plt.subplots(2,2, figsize=(10,8))
        if m == 0:
            fig.suptitle("No rotation")
            condition = 'no_rotation'
        elif m == 2: 
            fig.suptitle("Rotation")
            condition = 'rotation'

        Dneo_weight = []  
        
        for i, zi in enumerate(range(asim.Z_imp + 1)):

            Dneo = (D_z - Dz_an)[:, -1, zi] / 100**2
            Dneo_weight.append(Dneo*nz[:,zi,-1]*1e6) # /sum_curve[:,-1])


            '''
            #if i %5 == 0 and i!=0: 
            curveD, = axs[0,0].plot(asim.rhop_grid, (D_z - Dz_an)[:, -1, zi] / 100**2)
            colorD = curveD.get_color()

            max_value_index_D = np.argmax((D_z - Dz_an)[:, -1, zi])
            max_value = (D_z - Dz_an)[max_value_index_D, -1, zi] /100**2
            axs[0,0].text(asim.rhop_grid[max_value_index_D], max_value, f'{zi}', color=colorD,fontsize=13)
            '''
            '''
            plt.figure(20)
            plt.plot(asim.rhop_grid, Dneo_weight)
            '''

        DaveNeo = np.sum(Dneo_weight, axis = 0)/sum_curve[:,-1]
        #axs[0,0].plot(asim.rhop_grid, D_z[:,0,-1]/100**2, label = "tot") 
        axs[0,0].plot(asim.rhop_grid, DaveNeo , label="neo") 
        axs[0,0].plot(asim.rhop_grid, Dz_an[:,0,-1] /100**2, label="turb") 
        axs[0,0].set_ylabel(r"D [m$^2$/s]")
        axs[0,0].set_title("D")
        axs[0,0].grid(True)
        axs[0,0].legend()
        '''
        bbox = axs[0,1].get_position()
        left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        zoom_factor = 0.5
        ax_zoom = fig.add_axes([left+width*(1-0.8), bottom+height*(1-0.8),
                                    width*zoom_factor, height*zoom_factor])
        '''

        Vneo_weight = [] 
        for i, zi in enumerate(range(asim.Z_imp + 1)):
            #if i %5 == 0 and i!=0: 
            #axs[0,1].plot(asim.rhop_grid, (V_z - Vz_an)[:, -1, zi] / 100)
            

            Vneo = (V_z - Vz_an)[:, -1, zi] / 100
            Vneo_weight.append(Vneo*nz[:,zi,-1]*1e6)
            
            #axs[0,1].plot(asim.rhop_grid, Vneo_weight)

            Vplot = (V_z - Vz_an)[:, -1, zi] / 100
            # zoom per V   
            '''
            x_zoom_min, x_zoom_max = asim.rhop_grid[-150], asim.rhop_grid[-20]
            y_zoom =  Vneo_weight[(asim.rhop_grid>x_zoom_min) & (asim.rhop_grid<x_zoom_max)]
            x_zoom =  asim.rhop_grid[(asim.rhop_grid>x_zoom_min) & (asim.rhop_grid<x_zoom_max)] 

            ax_zoom.plot(x_zoom,y_zoom)
            '''
        VaveNeo = np.sum(Vneo_weight, axis = 0)/sum_curve[:,-1]
        '''
        buffer = 0.1  # 10% buffer
        y_min = min(y_zoom) - (max(y_zoom) - min(y_zoom)) * buffer
        y_max = max(y_zoom) + (max(y_zoom) - min(y_zoom)) * buffer
        ax_zoom.set_xlim(x_zoom_min, x_zoom_max)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_title('V zoom-edge')
        
        #ax_zoom.set_xlim(x_zoom_min, x_zoom_max) 
        #ax_zoom.set_ylim(min(y_zoom), max(y_zoom))
        ax_zoom.grid(True)
        '''
        #axs[0,1].plot(asim.rhop_grid, V_z[:,0,-1] /100, label = "tot") 
        axs[0,1].plot(asim.rhop_grid, VaveNeo, label="neo") 
        axs[0,1].plot(asim.rhop_grid, Vz_an[:,0,-1] /100, label="turb") 
        axs[0,1].set_ylabel(r"v [m/s]")
        axs[0,1].set_title("V")
        axs[0,1].grid(True)
        axs[0,1].legend() 


        #for i, zi in enumerate(ChargeStates): 
        for i, zi in enumerate(range(asim.Z_imp + 1)):
            if zi == 20 or zi == 21 or zi == 22 or zi == 23 or zi == 38 or zi == 39 or zi == 37:
                curve, = axs[1,0].plot(asim.rhop_grid, nz[:,zi,-1]*1e6, linewidth=2.5) #/np.max(sum_curve))
                color = curve.get_color()

                max_value_index = np.argmax(nz[:,zi,-1]*1e6) #/ np.max(sum_curve))
                axs[1,0].text(asim.rhop_grid[max_value_index], nz[max_value_index, zi, -1]*1e6, f'{zi}', color=color,fontsize=13) #/ np.max(sum_curve)

        axs[1,0].plot(asim.rhop_grid, sum_curve[:,-1], 'k', label = "sum", linewidth=2.5) #/np.max(sum_curve) , 'k', label = "sum") 
        axs[1,0].set_xlabel(r"$\rho_p$")
        axs[1,0].set_ylabel(r"$N_z$ $[part/m^3]$")
        axs[1,0].set_title("Impurity distribution")
        axs[1,0].grid(True)
        axs[1,0].legend()

        nzSumToCompare.append(sum_curve[:,-1])
        DaveToCompare.append(DaveNeo)
        VaveToCompare.append(VaveNeo)


        ax1 = axs[1,1]
        ax2 = ax1.twinx()
        ax1.plot(roa, Te, color='r', label = "Te", linewidth=2.5)
        ax1.plot(roa, Ti, color='b', label = "Ti", linewidth=2.5)
        ax2.plot(roa, Ni, color='g', label = "ne", linewidth=2.5)
        ax1.set_title("Background plasma")
        ax1.set_ylabel("T [keV]")
        ax2.set_ylabel("ne $[m^{-3}$]", color = 'g')
        ax1.set_xlabel(r"$\rho_p$")
        ax1.grid(True)
        ax1.legend(loc = (0.03,0.6))
        ax2.legend(loc = 'center right')
        plt.tight_layout()



        Npart = aurora.grids_utils.vol_int(nz.transpose(2,1,0)[-1,:,:], asim.rvol_grid, asim.pro_grid, asim.Raxis_cm)
        print(np.sum(Npart))

        TSC.append(H_ave[-1] / K_ave[-1])


nzSumToCompare = np.array(nzSumToCompare)
plt.figure() 
D_values = [1,2,5] 
V_values = [0.01, 0.02, 0.05] 
for p in[0,1,2]: 
    plt.plot(asim.rhop_grid, nzSumToCompare[p,:], label=f'$D_t$={D_values[p]} $m^2/s$',linewidth=2.5)
plt.title("$N_z$ with no rotation")
plt.xlabel(r"$\rho_p$")
plt.ylabel(r"$N_z$ $[part/m^3]$")
plt.grid(True)
plt.legend()

DaveToCompare = np.array(DaveToCompare)
plt.figure() 
for p in[0,1,2]: 
    plt.plot(asim.rhop_grid, DaveToCompare[p,:], label=f'$D_t$={D_values[p]} $m^2/s$')
plt.title(r"$D neo$ with no rotation")
plt.xlabel(r"$\rho_p$")
plt.ylabel(r"D [m$^2$/s]")
plt.grid(True)
plt.legend()


line_style = ['-', '--', '-.']
VaveToCompare = np.array(VaveToCompare)
plt.figure() 
for p, style in zip([0,1,2], line_style):  
    plt.plot(asim.rhop_grid, VaveToCompare[p,:], label=f'$V_t$={V_values[p]} $m^2/s$', linestyle=style)
plt.title("V neo with no rotation")
plt.xlabel(r"$\rho_p$")
plt.ylabel(r"v [m/s]")
plt.grid(True)
plt.legend()


plt.figure() 
D_values = [1,2,5,1,2,5] 
V_values = [0.01, 0.02, 0.05,0.01, 0.02, 0.05] 
for p in[3,4,5]: 
    plt.plot(asim.rhop_grid, nzSumToCompare[p,:], label=f'$D_t$={D_values[p]} $m^2/s$',linewidth=2.5)
plt.title("$N_z$ with rotation")
plt.xlabel(r"$\rho_p$")
plt.ylabel(r"$N_z$ $[part/m^3]$")
plt.grid(True)
plt.legend()

plt.figure() 
for p in [3,4,5]: 
    plt.plot(asim.rhop_grid, DaveToCompare[p,:], label=f'$D_t$={D_values[p]} $m^2/s$')
plt.title(r"$D neo$ with rotation")
plt.xlabel(r"$\rho_p$")
plt.ylabel(r"D [m$^2$/s]")
plt.grid(True)
plt.legend()


plt.figure() 
for p, style in zip([3,4,5], line_style): 
    plt.plot(asim.rhop_grid, VaveToCompare[p,:], label=f'$V_t$={V_values[p]} $m/s$',linestyle = style)
plt.title("V neo with  rotation")
plt.xlabel(r"$\rho_p$")
plt.ylabel(r"v [m/s]")
plt.grid(True)
plt.legend()

'''
plt.figure(3)
plt.plot(Time[idx6], PradTot + PowerToPlot[idx2], 'ro', label='model')
plt.savefig('TotPrad.png')
plt.legend()
'''

neInterp = np.interp(asim.rhop_grid, rhop_kp, kp["ne"]["vals"])

plt.figure()
plt.plot(asim.rhop_grid, sum_curve[:,-1]/(neInterp*1e6), label="cSn",linewidth=2.5)
plt.plot(asim.rhop_grid, 3e-4*np.ones(len(asim.rhop_grid)), label="limit [Sn]", linewidth=2.5)
plt.title("Sn concentration")
plt.grid(True)
plt.legend()
plt.xlabel(r"$\rho_p$")
plt.ylabel("[-]")


'''
plt.figure()
for i, zi in enumerate(range(asim.Z_imp + 1)):
    if zi == 20 or zi == 21 or zi == 22 or zi == 23 or zi == 38 or zi == 39 or zi == 37 or zi == 40:
        curve, = plt.plot(asim.rhop_grid, nz[:,zi,-1]*1e6) #/np.max(sum_curve))
        color = curve.get_color()

        max_value_index = np.argmax(nz[:,zi,-1]*1e6) #/ np.max(sum_curve))
        plt.text(asim.rhop_grid[max_value_index], nz[max_value_index, zi, -1]*1e6, f'{zi}', color=color,fontsize=13) #/ np.max(sum_curve)

plt.plot(asim.rhop_grid, sum_curve[:,-1], 'k', label = "sum") #/np.max(sum_curve) , 'k', label = "sum") 
plt.xlabel(r"$\rho_p$")
plt.ylabel(r"$N_z$ $[part/m^3]$")
plt.title("Impurity distribution")
plt.grid(True)
plt.legend()
'''

# Plots TSC 
'''
plt.figure()
plt.title("TSC")
plt.xlabel(r"$\rho_p$")
plt.ylabel("$H/K$")

for i in range(2,5,2):
    if i == 2:
        label = "noRotation"
        color = 'darkorange'
    else:
        label = "Rotation" 
        color = 'brown'  
    plt.plot(asim.rhop_grid, TSC[i][:,0], color=color, label = label)

plt.grid(True)
plt.legend()
'''

'''

RHOinv = RhoFromLOS[::-1]
vtorInterp = np.interp(roa, RHOinv, vtorInv)
vtorSmooth = interp1d(roa, vtorInterp, kind='cubic')
#xnew = np.linspace(0,, num=320, endpoint=True)
ynew = vtorSmooth(roa)

plt.figure(50)
plt.plot(roa,ynew) 

Mi = abs(vtorInterp*1e-3)/(np.sqrt(2*Ti*1e3/asim.A_imp))
''' 

'''
# PLOTS  
# Fractional abundance of Sn depending on temperature 

atom_data = aurora.atomic.get_atom_data(imp, ["scd", "acd"])
_Te, fz = aurora.atomic.get_frac_abundances(atom_data, kp["ne"]["vals"], kp["Te"]["vals"], asim.rhop_grid, plot=False)

fz_interp = np.zeros((320,51))
plt.figure()
plt.title("Fractional abundance")
for i, zi in enumerate(ChargeStates): 
    if zi == 20 or zi == 21 or zi == 38 or zi == 39:
        curve, = plt.plot(_Te, fz[:,zi]) 
        color = curve.get_color()

        max_value_index = np.argmax(fz[:,zi]) 
        plt.text(_Te[max_value_index], fz[max_value_index, zi], f'{zi}', color=color,fontsize=13) 

plt.xlabel("Te")
plt.ylabel("[-]")
plt.grid(True)
plt.legend

# Plot of the cooling factors 

coolFactors = aurora.radiation.get_cooling_factors(imp, kp["ne"]["vals"], kp["Te"]["vals"], ion_resolved=True) 
plt.figure()
for i, zi in enumerate(ChargeStates): 
    if zi == 20 or zi == 21 or zi == 38 or zi == 39 or zi == 36 or zi == 37:
        curve, = plt.plot(_Te, coolFactors[0][:,zi] ) 
        color = curve.get_color()

        max_value_index = np.argmax(coolFactors[0][:,zi]) 
        plt.text(_Te[max_value_index], coolFactors[0][:,zi][max_value_index] , f'{zi}', color=color,fontsize=13) 

plt.title('Line radiation')
plt.grid()
plt.legend()
plt.xlabel('Te')
plt.ylabel('Lz $[W/m^3]$')

plt.figure()
for i, zi in enumerate(ChargeStates): 
    if zi == 20 or zi == 21 or zi == 38 or zi == 39 or zi == 36 or zi == 37:
        curve, = plt.plot(_Te, coolFactors[1][:,zi] ) 
        color = curve.get_color()

        max_value_index = np.argmax(coolFactors[1][:,zi]) 
        plt.text(_Te[max_value_index], coolFactors[1][:,zi][max_value_index] , f'{zi}', color=color,fontsize=13) 

plt.title('Continuum radiation')
plt.grid()
plt.legend()
plt.xlabel('Te')
plt.ylabel('Brm $[W/m^3]$')

# !!! PLOTTING SOME RADIATION RESULTS !!! #  

# Plotting the total Prad 
TotCont = np.sum(asim.rad['cont_rad'],axis=1)
TotLine = np.sum(asim.rad['line_rad'],axis=1)

plt.figure()
plt.title('Total Prad Sn')
plt.plot(asim.rvol_grid, asim.rad['tot'].transpose(1,0)[:,-1]*1e6, color = 'k', label='Tot') 
plt.plot(asim.rvol_grid, TotCont.transpose(1,0)[:,-1]*1e6,color = 'b', label='Cont') 
plt.plot(asim.rvol_grid, TotLine.transpose(1,0)[:,-1]*1e6,color = 'r', label='LineRad') 
plt.xlabel("[$W/m^3$]")
plt.grid()
plt.legend()

# -------------------------------------------------------------------------------------- #

## Plot dei vari coefficienti K,H - se poi ti servono guarda TestFACIT_Tin.py ##   
#''' 

