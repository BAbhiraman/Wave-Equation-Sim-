# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:10:24 2020

@author: bhask
"""
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

plt.style.use('seaborn-pastel')
plt.rc('xtick', direction='in'); plt.rc('ytick', direction='in')


def calc_coeffs(f0, g0, name = None):
    m = len(f0)
    alphas = np.zeros(m)
    betas = np.zeros(m)
    for n in range(m):
        alphas[n] = np.sum(2*f0*np.sin((n+1)*pi*x)*(1/m))
        betas[n] = np.sum((2/((n+1)*pi))*g0*np.sin((n+1)*pi*x)*(1/m))
        
    fig,ax =  plt.subplots(2,2, constrained_layout = True)
    ax[0,0].plot(f0, color = 'C0');     ax[0,0].title.set_text("String's Initial Displacement")
    ax[0,1].plot(alphas, color = 'C1'); ax[0,1].title.set_text(r'$\alpha$')
    ax[1,0].plot(g0, color = 'C2');     ax[1,0].title.set_text("String's Initial Velocity")
    ax[1,1].plot(betas, color = 'C3');  ax[1,1].title.set_text(r'$\beta$')
    ax[0,0].set_xlabel('Position');     ax[1,0].set_xlabel('Position'); 
    ax[0,1].set_xlabel('Frequency');        ax[1,1].set_xlabel('Frequency'); 
    if name is not None:
        plt.savefig('outputs/{} graphs.png'.format(name), dpi = 300)      
    return alphas, betas

# CRUNCH THE TIME NUMBERS
def time_evolve(alphas, betas, fidel = 1):
    fidel = min(fidel, 1)
    m = len(alphas)
    dt = 1/60
    n_t = 10000
    ts = np.arange(start = 0, stop = n_t*dt, step = dt)
    u = np.zeros((m, n_t)) #xs, ts
    for it in range(n_t):
        if (10*it) % n_t == 0:
            print("{} % done with u(x,t)".format(100*it/n_t))
        for n in range(int(fidel*m)): #m
            u[:, it] += (alphas[n]*np.cos((n+1)*pi*ts[it]) + betas[n]*np.sin((n+1)*pi*ts[it]))*np.sin((n+1)*pi*x)
    u /= 1.1*np.max(u)
    return u, ts


def animate(u, name = None):
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
    ax.axis('off')
    line, = ax.plot([], [], lw=3)
    
    def init():
        line.set_data([], [])
        return line,
    def frame_func(i):
        y = u[:, i]
        line.set_data(x, y)
        #if i % 100 == 0:
        #    print(i)
        return line,
    anim = FuncAnimation(fig, frame_func, init_func=init,
             frames=300, interval=60, blit=True, repeat = True, repeat_delay = 1000)
    if name is not None:
        anim.save('outputs/{} vid.mp4'.format(name), dpi = 300)

def sound(u, ts, n_samps = 1, name = None):
    fs = 10000
    env = np.exp(-0.03*ts) #exponentially decaying envelope
    u_env = u*env 
    samp_inds = np.round((np.linspace(0, m, n_samps + 2)))
    audio= u_env[samp_inds[1:-1].astype(int), :]
    audio = np.sum(audio, axis = 0)
    #plt.figure()
    #plt.plot(ts, audio/max(audio))
    
    sd.play(audio/max(audio), fs)
    #audio_out = np.asarray(audio/max(audio), dtype=np.int16)
    if name is not None:
        wav_write('outputs/{} audio.wav'.format(name), fs, audio/max(audio))
    
    
#%%
plt.close('all')
m = 100
ns = np.arange(1, m+1, 1)
x = np.arange(0,1, 1/m)

pos_weird =  np.hstack((5*np.arange(0,1/4, 1/m), np.ones(int(m/4)), np.sin(3*pi*x[75:]), np.sin(7*pi*x[75:])))# ugly thing
pos_harm = np.sin(pi*x) + np.sin(2*pi*x) + np.sin(3*pi*x) + np.sin(4*pi*x) + np.sin(5*pi*x) + np.sin(6*pi*x) + np.sin(8*pi*x)# 
pos_0 = np.zeros(m)
pos_tri = np.hstack((np.arange(0,1/4, 1/m), np.arange(1/4,0, -1/m), -np.arange(0,1/4, 1/m), -np.arange(1/4,0, -1/m)))#
pos_sq = np.hstack((np.zeros(int(m/5)), np.ones(int(m/5)),-1*np.ones(int(m/5)),1*np.ones(int(m/5)),np.zeros(int(m/5))))#


vel_v = -10*np.hstack((np.arange(0,1/2, 1/m), np.arange(1/2,0, -1/m))) #np.zeros(m)
vel_0 = np.zeros(m) #
vel_spikes = np.zeros(m); vel_spikes[int(m/3)] = 1; vel_spikes[int(2*m/3)] = -1
vel_sq = np.hstack((np.zeros(int(2*m/5)), -1*np.ones(int(m/5)),np.zeros(int(2*m/5))))

pairs = [(pos_harm, vel_0), (pos_sq, vel_0), (pos_0, vel_v), (pos_weird, vel_0), (pos_tri, vel_0), (pos_0, vel_sq)]
names = ['harm_0', 'sq_0', '0_v', 'weird_0', 'tri_0', '0_sq']


for i, pair in enumerate(pairs):
    plt.close('all')
    f0, g0 = pair
    alphas, betas = calc_coeffs(f0, g0, name = names[i])
    u, ts = time_evolve(alphas, betas, fidel = 1)
    animate(u, names[i])
    for n_samp in range(2):
        sound(u, ts, n_samps = n_samp+1, name = '{}_{}samps'.format(names[i], n_samp+1))

