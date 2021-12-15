#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:04:09 2021

@author: jools

dependancies:
F:\My Files\DOCUMENTS\_________UNI\PROJECT\Code\ver_3-0_20211025_threading\daisyworld_ext_ver_3_0_threading.py
F:\My Files\DOCUMENTS\_________UNI\PROJECT\Code\ver_3-0_20211025_threading\daisyworld_main_ver_2_0.py
F:\My Files\DOCUMENTS\_________UNI\PROJECT\Code\ver_3-0_20211025_threading\run_store.txt
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import telegram_send
import os
import random as rd
########################################################################################################
test = False
#test = True
#########################################################################################################

def callFunc(dir_path, fluxes, t_max, size, Td_ideal_black_C, Td_ideal_white_C, drate, min_brate, neighbourhood_reach):
    mean_temp = np.zeros_like(fluxes)
    black = np.zeros_like(fluxes)
    white = np.zeros_like(fluxes)

    

        # Temperatures
    KELVIN_OFFSET = 273.15
    Td_min = 5 + KELVIN_OFFSET
    Td_max = 40 + KELVIN_OFFSET
    Td_ideal_black = Td_ideal_black_C + KELVIN_OFFSET #22.5
    Td_ideal_white = Td_ideal_white_C + KELVIN_OFFSET #22.5

        # Albedo
    alb_white = 0.75
    area_white = 0.01 #0.01
    alb_black = 0.25
    area_black = 0.01 #0.01
    alb_barren = 0.5
    insul = 20

        # Flux terms

    So = 1000
    sigma = 5.67032e-8

    with open("run_store.txt", 'r+') as rn:
        run = int(rn.readlines()[0])



        #print('parameters initialised')
        #%%
        # If run from command line, do the whole thing
    for flux_i, flux in enumerate(fluxes):

        """Run the daisyworld model"""
        #print('commencing main')
        # Initialize arrays
        time = np.arange(t_max)
        alb_p = np.zeros_like(time, dtype = np.float64) #planet albido
        Tp = np.zeros_like(time, dtype = np.float64) #planet temp
        space = np.zeros((t_max,size,size))
        #alb_space = np.full(shape=(t_max,size,size),fill_value=alb_barren) #albido spacetime of daisyworld
        #tmp_space = np.zeros((t_max,size,size)) #temperature spacetime

        neighbourhood_size = neighbourhood_reach*2 + 1
        neighbourhood_mask = np.zeros((neighbourhood_size, neighbourhood_size))
        for x_ in range(neighbourhood_size):
            for y_ in range(neighbourhood_size):
                neighbourhood_mask[x_,y_] = np.exp(-0.1*(x_-neighbourhood_reach)**2-0.1*(y_-neighbourhood_reach)**2)
        neighbourhood_mask[neighbourhood_reach,neighbourhood_reach] = 0
        neighbourhood_mask = neighbourhood_mask/(np.sum(neighbourhood_mask))
        #print('mask initialised')

        #initialise white
        i=0
        while i<area_white*(size**2):
            x,y = rd.randrange(0,size),rd.randrange(0,size)
            if space[0,x,y] == 0:
                space[0,x,y] = 1
                i+=1

        #initialise black
        i=0
        while i<area_black*(size**2):
            x,y = rd.randrange(0,size),rd.randrange(0,size)
            if space[0,x,y] == 0:
               space[0,x,y] = -1
               i+=1
        #print('daisies initialised')
       # a_black, a_white, a_tot = np.zeros_like(time), np.zeros_like(time), np.zeros_like(time)
       # a_black[0] = area_black
       # a_white[0] = area_white
       # a_tot[0] = area_black+area_white
        a_black = np.array([np.count_nonzero(i == -1) / size**2 for i in space])
        a_white = np.array([np.count_nonzero(i == 1) / size**2 for i in space])
        a_tot = np.array([np.count_nonzero(i != 0) / size**2 for i in space])

        birth_black,birth_white = 0,0

        alb_p[0] = (a_black[0] * alb_black + a_white[0] * alb_white + (1 - a_tot[0]) * alb_barren)
        Tp[0] = np.power(flux*So*(1-alb_p[0])/sigma, 0.25)
        #print('percentages intialised\n~begining simulation...')

        # Loop over all time
        for j in range(len(time)-1):
            space[j+1] = space[j]
            pad_ =  np.pad(space[j], neighbourhood_reach, mode='wrap')
            # Local temperatures
            for x_ in range(size):
                for y_ in range(size):
                    if space[j,x_,y_] != 0 and np.random.choice((True, False), 1, p=[drate,1-drate]):
                        space[j+1,x_,y_] = 0
                    elif space[j,x_,y_] == 0:
                        neighbours = pad_[x_:x_+2*neighbourhood_reach+1,y_:y_+2*neighbourhood_reach+1]
                        prob = (neighbours * neighbourhood_mask).sum()
                        if abs(prob) < min_brate:
                            prob = np.random.choice((-min_brate, min_brate))
                        #print(f"prob={prob}    birth_black={birth_black}    birth_white={birth_white}")
                        if prob<0 and np.random.choice((True, False), 1, p=[abs(prob*birth_black), 1-abs(prob*birth_black)]):
                            space[j+1,x_,y_] = -1
                        elif prob > 0 and np.random.choice((True, False), 1, p=[abs(prob*birth_white), 1-abs(prob*birth_white)]):
                            space[j+1,x_,y_] = 1




            a_black[j+1] = np.count_nonzero(space[j+1] == -1) / size**2
            a_white[j+1] = np.count_nonzero(space[j+1] == 1) / size**2
            a_tot[j+1] = np.count_nonzero(space[j+1] != 0) / size**2
            #planetary albido
            alb_p[j+1] = (a_black[j+1] * alb_black + a_white[j+1] * alb_white + (1 - a_tot[j+1]) * alb_barren)
            Tp[j+1] = np.power(flux*So*(1-alb_p[j+1])/sigma, 0.25)

            Td_black = insul*(alb_p[j+1]-alb_black) + Tp[j+1]
            Td_white = insul*(alb_p[j+1]-alb_white) + Tp[j+1]
            #print(Td_black, Td_white)

            if Td_black >= Td_min and Td_black <= Td_max:
                birth_black = 1 - 0.003265*(Td_ideal_black-Td_black)**2
            else:
                birth_black = 0.0

            if Td_white >= Td_min and Td_white <= Td_max:
                birth_white = 1 - 0.003265*(Td_ideal_white-Td_white)**2
            else:
                birth_white = 0.0

        #print(birth_white[j], birth_black[j])

        mean_temp[flux_i], black[flux_i], white[flux_i] =  np.mean(Tp[int(-(t_max/20)):] - KELVIN_OFFSET), np.mean(a_black[int(-(t_max/20)):]*100), np.mean(a_white[int(-(t_max/20)):]*100)

        #print('simulation completed\n~graphing up...')
        #%%

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(time, a_black*100, color='k', label="area black daisies")
        ax[0].plot(time, a_white*100, color='red', label="area white daisies")

        #ax[0].plot(time, a_tot*100, "b.", label="total amount of daisies")
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('%')
        ax[0].legend()

        ax[1].plot(time, Tp-KELVIN_OFFSET, color='r', label="Planetary Temperature")


        ax[1].set_xlabel('time')
        ax[1].set_ylabel('°C')

        if Td_ideal_black == Td_ideal_white:
            ax[1].axhline(y=(Td_ideal_black - KELVIN_OFFSET), color='g', linestyle='dotted',label="Ideal Temperature for Daisies")
        else:
            ax[1].axhline(y=(Td_ideal_black - KELVIN_OFFSET), color='g', linestyle='dotted',label="Ideal Temperature for Black Daisies")
            ax[1].axhline(y=(Td_ideal_white - KELVIN_OFFSET), color='b', linestyle='dotted',label="Ideal Temperature for White Daisies")

        ax[1].legend()



        #save things
        dir = dir_path+"/timeplots/timeplots_run{}_size{}_time{}".format(run,size,t_max)

        if not os.path.exists(dir):
            os.makedirs(dir)
        lnlent=14
        lnlen=28
        plt.savefig(dir_path+'/timeplots/timeplots_run{}_size{}_time{}/flux_{:.3f}.png'.format(run,size,t_max,flux))
        telegram_send.send(messages = ["""{:.3f}, {}\n{}\n{}{}""".format(flux,neighbourhood_reach,(int(lnlent*mean_temp[flux_i]/60)*"\U0001F7E5") if mean_temp[flux_i]>0 else (int(lnlent*(-mean_temp[flux_i])/60)*"\U0001F7E6"),int(lnlen*black[flux_i]/100)*"\u2B1C",int(lnlen*white[flux_i]/100)*"\u2B1B")])

        #with open(dir_path+'/timeplots_run{}_size{}_time{}/flux_{:.3f}.png'.format(run,size,t_max,flux), "rb") as f:
        #    telegram_send.send(images=[f])


        out = [flux,time,a_black,a_white,Tp]
        np.save(dir_path+'/timeplots/timeplots_run{}_size{}_time{}/flux_{:.3f}.npy'.format(run,size,t_max,flux),out)
        #print('graphing complete...')

    return mean_temp, black, white






"""
def callFunc(dir_path ,run, fluxes, t_max, size, Td_ideal_black_C, Td_ideal_white_C, drate, min_brate, neighbourhood_reach):
    mean_temp = np.zeros_like(fluxes)
    black = np.zeros_like(fluxes)
    white = np.zeros_like(fluxes)

    for i, flux in enumerate(fluxes):
        #print(flux)
        mean_temp[i], black[i], white[i] = dw(dir_path, run, flux, t_max, size, Td_ideal_black_C, Td_ideal_white_C, drate, min_brate, neighbourhood_reach)
        telegram_send.send(messages=['flux {}'.format(flux)])
    return mean_temp, black, white
"""

    
def main(Td_ideal_black_C = 22.5,Td_ideal_white_C = 22.5):
    telegram_send.send(messages=['initialising'])
    #Td_ideal_black_C=22.5
    #Td_ideal_white_C=22.5
    drate = 0.3
    min_brate = 0.01
    neighbourhood_reach = 3


    if test:
        size=3
        t_max = 10
        Sflux_step = 0.1

    else:
        size=30
        t_max = 1000
        Sflux_step = 0.02



    dir_path = os.path.dirname(os.path.realpath(__file__))

    telegram_send.send(messages=['initialising in {}'.format(str(dir_path))])


    with open("run_store.txt", 'r+') as rn:
        run = int(rn.readlines()[0])
        rn.seek(0)
        rn.write(str(run+1))
        rn.truncate()


    Sflux_min = 0.5
    Sflux_max = 1.4
    target_fluxes = np.arange(Sflux_min, Sflux_max, Sflux_step)
    
    telegram_send.send(messages=[f"""
################
          run   :   {run}
################
  T max_________{t_max}
  T ideal black___{Td_ideal_black_C}
  T ideal white___{Td_ideal_white_C}
  drate____________{drate}
  min birth rate___{min_brate}
  neighbourhood____{neighbourhood_reach}
  size_____________{size}
################
  Sflux min________{Sflux_min}
  Sflux max_______{Sflux_max}
  Sflux step______{Sflux_step}
################
"""])


    fluxes_cr_1 = target_fluxes[:int(len(target_fluxes)/4)]
    fluxes_cr_2 = target_fluxes[int(len(target_fluxes)/4):int(len(target_fluxes)/4)*2]
    fluxes_cr_3 = target_fluxes[int(len(target_fluxes)/4)*2:int(len(target_fluxes)/4)*3]
    fluxes_cr_4 = target_fluxes[int(len(target_fluxes)/4)*3:]
    
    fluxes = [fluxes_cr_1,fluxes_cr_2,fluxes_cr_3,fluxes_cr_4]

    
    
    
    with mp.Pool(processes=4) as pool:
        global results
        results = [pool.apply(callFunc, args=(dir_path, j, t_max, size, Td_ideal_black_C, Td_ideal_white_C, drate, min_brate, neighbourhood_reach)) for j in fluxes]
   
    
    

    
   
    mean_temp = np.array([item for sublist in [t[0] for t in results] for item in sublist])
    black = np.array([item for sublist in [t[1] for t in results] for item in sublist])
    white = np.array([item for sublist in [t[2] for t in results] for item in sublist])

    telegram_send.send(messages=['plotting'])     
        
    fig, ax = plt.subplots(2, 1,figsize=(8,10))
    ax[0].plot(target_fluxes, black, color='k', label="black daisies")
    ax[0].plot(target_fluxes, white, color='red', label="white daisies")
    ax[0].plot(target_fluxes, black+white,"k--", label="total amount of daisies for incr. L")
    ax[0].set_xlabel('solar luminosity')
    ax[0].set_ylabel('area (%)')
    ax[0].legend()
    
    ax[1].plot(target_fluxes, mean_temp, color='r', label="T with Daisies")
    ax[1].set_xlabel('solar luminosity')
    ax[1].set_ylabel('global temperature (C)')
    if Td_ideal_black_C == Td_ideal_white_C:
      ax[1].axhline(y=(Td_ideal_black_C), color='g', linestyle='dotted',label="Ideal Temperature for Daisies")
    else:
      ax[1].axhline(y=(Td_ideal_black_C), color='g', linestyle='dotted',label="Ideal Temperature for Black Daisies")
      ax[1].axhline(y=(Td_ideal_white_C), color='b', linestyle='dotted',label="Ideal Temperature for White Daisies")
    ax[1].legend()
    
    
    plt.savefig('runfigs/run{}_{}time_{}size_{}step.png'.format(run,t_max,size,str(Sflux_step).replace('.','-')))
    

    out = [t_max,Td_ideal_black_C,Td_ideal_white_C,drate,min_brate,neighbourhood_reach,size,Sflux_min,Sflux_max,Sflux_step,target_fluxes,mean_temp,black,white]
    np.save('runsaves/run{}_{}time_{}size_{}step.npy'.format(run,t_max,size,str(Sflux_step).replace('.','-')),out)


    telegram_send.send(messages=['successful completion'])

    with open("runfigs/run{}_{}time_{}size_{}step.png".format(run,t_max,size,str(Sflux_step).replace('.','-')), "rb") as f:
        telegram_send.send(images=[f])
    
    


if __name__ == '__main__':
    for i in range(3):
        main(Td_ideal_white_C = 15)
        main(Td_ideal_white_C = 30) 
        main(Td_ideal_black_C = 15)
        main(Td_ideal_black_C = 30)
    
    
