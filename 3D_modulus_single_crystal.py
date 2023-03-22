#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:45:27 2022

@author: jiafanhao

This code is based on 
https://github.com/coudertlab/elate

Here, I just delete all of web part. 

"""

import numpy as np
import json
from scipy import optimize
import matplotlib.pyplot as plt


def dirVec1(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def dirVec2(theta, phi, chi):
    return np.array([np.cos(theta)*np.cos(phi)*np.cos(chi) - np.sin(phi)*np.sin(chi),
             np.cos(theta)*np.sin(phi)*np.cos(chi) + np.cos(phi)*np.sin(chi),
             - np.sin(theta)*np.cos(chi)])

# Functions to minimize/maximize
def minimize(func, dim):
    if dim == 2:
        r = ((0, np.pi), (0, 2*np.pi))
        n = 20
    elif dim == 3:
        r = ((0, np.pi), (0, 2*np.pi), (0, 2*np.pi))
        n = 20
    # TODO -- try basin hopping or annealing
    return optimize.brute(func, r, Ns = n, full_output = True, finish = optimize.fmin)[0:2]
def maximize(func, dim):
    res = minimize(lambda x: -func(x), dim)
    return (res[0], -res[1])


class Elastic:
    """An elastic tensor, along with methods to access it"""
    def __init__(self, s):
        """Initialize the elastic tensor from a string"""
        
        if s is None:
            raise ValueError("no matrix was provided")
        # Argument can be a 6-line string, a list of list, or a string representation of the list of list
        try:
            if type(json.loads(s)) == list: s = json.loads(s)
        except:
            pass
        if type(s) == str:
            # Remove braces and pipes
            s = s.replace("|", " ").replace("(", " ").replace(")", " ")
            # Remove empty lines
            lines = [line for line in s.split('\n') if line.strip()]
            if len(lines) != 6:
                raise ValueError("should have six rows")
            # Convert to float
            try:
                mat = [list(map(float, line.split())) for line in lines]
            except:
                raise ValueError("not all entries are numbers")
        elif type(s) == list:
            # If we already have a list, simply use it
            mat = s
        elif type(s) == np.ndarray:
            mat = s
        else:
            print (s)
            raise ValueError("invalid argument as matrix")
        # Make it into a square matrix
        mat = np.array(mat)
        if mat.shape != (6,6):
            # Is it upper triangular?
            if list(map(len, mat)) == [6,5,4,3,2,1]:
                mat = [ [0]*i + mat[i] for i in range(6) ]
                mat = np.array(mat)
        # Is it lower triangular?phi
        if list(map(len, mat)) == [1,2,3,4,5,6]:
            mat = [ mat[i] + [0]*(5-i) for i in range(6) ]
            mat = np.array(mat)
        if mat.shape != (6,6):
            raise ValueError("should be a square matrix")
        # Check that is is symmetric, or make it symmetric
        if np.linalg.norm(np.tril(mat, -1)) == 0:
            mat = mat + np.triu(mat, 1).transpose()
        if np.linalg.norm(np.triu(mat, 1)) == 0:
            mat = mat + np.tril(mat, -1).transpose()
        if np.linalg.norm(mat - mat.transpose()) > 1e-3:
            raise ValueError("should be symmetric, or triangular")
        elif np.linalg.norm(mat - mat.transpose()) > 0:
            mat = 0.5 * (mat + mat.transpose())
        # Store it
        self.CVoigt = mat
        # Put it in a more useful representation
        try:
            self.SVoigt = np.linalg.inv(self.CVoigt)
        except:
            raise ValueError("matrix is singular")
        VoigtMat = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
        def SVoigtCoeff(p,q): return 1. / ((1+p//3)*(1+q//3))
        self.Smat = [[[[ SVoigtCoeff(VoigtMat[i][j], VoigtMat[k][l]) * self.SVoigt[VoigtMat[i][j]][VoigtMat[k][l]]
                         for i in range(3) ] for j in range(3) ] for k in range(3) ] for l in range(3) ]
        return
    
    def Young(self, theta, phi):
        a = dirVec1(theta, phi)
        r = sum([a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/r

    def LC(self, theta, phi):
        a = dirVec1(theta, phi)
        r = sum([ a[i]*a[j] * self.Smat[i][j][k][k]
                  for i in range(3) for j in range(3) for k in range(3) ])
        return 1000 * r

    def bulk_modulus(self, theta, phi):
        a = dirVec1(theta, phi)
        r = sum([ a[i]*a[j] * self.Smat[i][j][k][k]
                  for i in range(3) for j in range(3) for k in range(3) ])
        return 1/r

    def shear(self, theta, phi, chi):
        a = dirVec1(theta, phi)
        b = dirVec2(theta, phi, chi)
        r = sum([ a[i]*b[j]*a[k]*b[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/(4*r)

    def Poisson(self, theta, phi, chi):
        a = dirVec1(theta, phi)
        b = dirVec2(theta, phi, chi)
        r1 = sum([ a[i]*a[j]*b[k]*b[l] * self.Smat[i][j][k][l]
                   for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        r2 = sum([ a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                   for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return -r1/r2      
    def averages(self):
        A = (self.CVoigt[0][0] + self.CVoigt[1][1] + self.CVoigt[2][2]) / 3
        B = (self.CVoigt[1][2] + self.CVoigt[0][2] + self.CVoigt[0][1]) / 3
        C = (self.CVoigt[3][3] + self.CVoigt[4][4] + self.CVoigt[5][5]) / 3
        a = (self.SVoigt[0][0] + self.SVoigt[1][1] + self.SVoigt[2][2]) / 3
        b = (self.SVoigt[1][2] + self.SVoigt[0][2] + self.SVoigt[0][1]) / 3
        c = (self.SVoigt[3][3] + self.SVoigt[4][4] + self.SVoigt[5][5]) / 3

        KV = (A + 2*B) / 3
        GV = (A - B + 3*C) / 5

        KR = 1 / (3*a + 6*b)
        GR = 5 / (4*a - 4*b + 3*c)

        KH = (KV + KR) / 2
        GH = (GV + GR) / 2

        return [ [KV, 1/(1/(3*GV) + 1/(9*KV)), GV, (1 - 3*GV/(3*KV+GV))/2],
                 [KR, 1/(1/(3*GR) + 1/(9*KR)), GR, (1 - 3*GR/(3*KR+GR))/2],
                 [KH, 1/(1/(3*GH) + 1/(9*KH)), GH, (1 - 3*GH/(3*KH+GH))/2] ]

    def shear2D(self, theta, phi):
        ftol = 0.001
        xtol = 0.01
        def func1(z): return self.shear([theta, phi, z])
        r1 = optimize.minimize(func1, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.shear([theta, phi, z])
        r2 = optimize.minimize(func2, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))

    def shear3D(self, theta, phi, guess1 = np.pi/2.0, guess2 = np.pi/2.0):
        tol = 0.005
        def func1(z): return self.shear([theta, phi, z])
        r1 = optimize.minimize(func1, guess1, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.shear([theta, phi, z])
        r2 = optimize.minimize(func2, guess2, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))
    
    def shear3D_new(self, theta, phi):
        npoints=360
        chi   = np.linspace(0, 2*np.pi, npoints)
        
        sh_max=[]; sh_min=[]
        for i in range (len(theta)):
            s_max=[]; s_min=[]
            for j in range (len(theta[0])):
                x=[]; y=[]
                for k in range (npoints):
                    x.append(theta[i][j])
                    y.append(phi[i][j])
                sh=self.shear(x, y, chi)
                s_max.append(max(sh))
                s_min.append(min(sh))
            sh_max.append(s_max)
            sh_min.append(s_min)
        return np.array(sh_max), np.array(sh_min)

    def poisson3D_new(self, theta, phi):
        npoints=360
        chi   = np.linspace(0, 2*np.pi, npoints)
        sh_max=[]; sh_min=[]
        for i in range (len(theta)):
            s_max=[]; s_min=[]
            for j in range (len(theta[0])):
                x=[]; y=[]
                for k in range (npoints):
                    x.append(theta[i][j])
                    y.append(phi[i][j])
                sh=self.Poisson(x, y, chi)
                s_max.append(max(sh))
                s_min.append(min(sh))
            sh_max.append(s_max)
            sh_min.append(s_min)
        return np.array(sh_max), np.array(sh_min)

    def Poisson2D(self, theta, phi):
        ftol = 0.001
        xtol = 0.01
        def func1(z): return self.Poisson([theta, phi, z])
        r1 = optimize.minimize(func1, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.Poisson([theta, phi, z])
        r2 = optimize.minimize(func2, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        return (min(0,float(r1.fun)), max(0,float(r1.fun)), -float(r2.fun))

    def poisson3D(self, theta, phi, guess1 = np.pi/2.0, guess2 = np.pi/2.0):
        tol = 0.005
        def func1(z): return self.Poisson([theta, phi, z])
        r1 = optimize.minimize(func1, guess1, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.Poisson([theta, phi, z])
        r2 = optimize.minimize(func2, guess2, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        return (min(0,float(r1.fun)), max(0,float(r1.fun)), -float(r2.fun), float(r1.x), float(r2.x))



font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}


def makePolarPlot(func, theta, phi, npoints=100,pic_name='shear.png'):
    chi=np.linspace(0, 2*np.pi, npoints)
    xx=[]; yy=[]
    for i in range(npoints):
        xx.append(theta)
        yy.append(phi)
    p= func(xx, yy, chi)
    x = p * np.cos(chi)
    y = p * np.sin(chi)
    plt.plot(x, y, lw=2, color='green' )
    plt.rc('font', **font)
    plt.tick_params(axis='both',direction='in')
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()

def ela_normal_lattice(func, theta=[np.pi/2, np.pi/2, 0], phi=[0, np.pi/2, 0]):
    d={'0': 'a', '1': 'b', '2': 'c'}
    p= func(theta, phi)
    for i in range(3):
        print (d[str(i)], '  X  ')
        print ('    ', round(p[i],1))
        
    
def makePolarPlot_all_S(func, theta=[np.pi/2, np.pi/2, 0], phi=[0, np.pi/2, 0], npoints=100,pic_name='shear.png'):
    d={'0': 'a', '1': 'b', '2': 'c'}
    chi=np.linspace(0, 2*np.pi, npoints)
    xxx=[]; yyy=[]
    for i in range(3):
        xx=[]; yy=[]; 
        for j in range(npoints):
            xx.append(theta[i])
            yy.append(phi[i])
        p= func(xx, yy, chi)
        print (d[str(i)], '--> max  ', '  min  ', '  A  ')
        print ('    ', round(max(p),1), '  ', round(min(p),1), ' ', round(max(p)/min(p), 3))
        x = p * np.cos(chi)
        y = p * np.sin(chi)
        xxx.append(x)
        yyy.append(y)

    p1, =plt.plot(xxx[0], yyy[0], lw=2, color='green' )
    p2, =plt.plot(xxx[1], yyy[1], lw=2, color='red' )
    p3, =plt.plot(xxx[2], yyy[2], lw=2, color='black' )
    plt.legend([p1,p2,p3,], ['{100}', '{010}', '{001}', ], loc='upper right', fontsize=14, frameon=False)
    
    plt.xlim(-205, 205)
    plt.ylim(-205, 205)
    plt.rc('font', **font)
    plt.tick_params(axis='both',direction='in')
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()

def makePolarPlot_all_P(func, theta=[np.pi/2, np.pi/2, 0], phi=[0, np.pi/2, 0], npoints=100,pic_name='shear.png'):
    d={'0': 'a', '1': 'b', '2': 'c'}
    chi=np.linspace(0, 2*np.pi, npoints)
    xxx=[]; yyy=[]
    for i in range(3):
        xx=[]; yy=[]; 
        for j in range(npoints):
            xx.append(theta[i])
            yy.append(phi[i])
        p= func(xx, yy, chi)
        print (d[str(i)], '--> max  ', '  min  ', '  A  ')
        print ('    ', round(max(p),3), '  ', round(min(p),3), ' ', round(max(p)/min(p), 3))
        x = p * np.cos(chi)
        y = p * np.sin(chi)
        xxx.append(x)
        yyy.append(y)

    p1, =plt.plot(xxx[0], yyy[0], lw=2, color='green' )
    p2, =plt.plot(xxx[1], yyy[1], lw=2, color='red' )
    p3, =plt.plot(xxx[2], yyy[2], lw=2, color='black' )
    plt.legend([p1,p2,p3,], ['{100}', '{010}', '{001}', ], loc='upper right', fontsize=14, frameon=False)
    
    #plt.xlim(-205, 205)
    #plt.ylim(-205, 205)
    plt.rc('font', **font)
    plt.tick_params(axis='both',direction='in')
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()


def make3DPlot_Y(func, npoints=100, cmp='plasma', pic_name='pic.png'):
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    theta, phi = np.meshgrid(theta,phi)
    
    p= func(theta, phi)
    p_max_ind=np.unravel_index(np.argmax(p, axis=None), p.shape)
    p_min_ind=np.unravel_index(np.argmin(p, axis=None), p.shape)
    
    p_max=round(p[p_max_ind],3)
    p_min=round(p[p_min_ind],3)
    print (" ")
    print (pic_name, p_max_ind, p_min_ind)
    print ("max:", p_max, dirVec1(theta[p_max_ind], phi[p_max_ind]))
    print ("min:", p_min, dirVec1(theta[p_min_ind], phi[p_min_ind]))
    print ("Anisotropy", round(p_max/p_min, 3))

    #elas.Young_2(x, y)
    x = np.sin(theta)*np.cos(phi)*p
    y = np.sin(theta)*np.sin(phi)*p
    z =  np.cos(theta)*p
    
    ax = plt.subplot(111, projection='3d')
    #ax.set_xlabel('$\it^{x}$')
    #ax.set_ylabel('$\it^{y}$')
    #ax.set_zlabel('$\it^{z}$', rotation=90, ha='right')
    ax.set_xticks([-300,0,300])
    ax.set_yticks([-300,0,300])
    ax.set_zticks([-300,0,300])
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_zlim(-400, 300)    
    ax.set_xticklabels([ '-300', '0', '300'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels([ '-300', '0', '300'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_zticklabels([ '-300', '0', '300'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    
    
    #Sc=ax.plot_surface(x, y, z, cmap=newcmp , vmin=-p_max, vmax=p_max)
    Sc=ax.scatter(x, y, z, c=p, cmap=cmp, vmin=245, vmax=345)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([250, 280, 310, 340])
    cbar.ax.tick_params(labelsize=8)
    
    ax.view_init(elev=20, azim=300)
    plt.rc('font', **font)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()


def make3DPlot_B(func, npoints=100, cmp='plasma', pic_name='pic.png'):
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    theta, phi = np.meshgrid(theta,phi)
    
    p= func(theta, phi)
    p_max_ind=np.unravel_index(np.argmax(p, axis=None), p.shape)
    p_min_ind=np.unravel_index(np.argmin(p, axis=None), p.shape)
    
    p_max=round(p[p_max_ind],3)
    p_min=round(p[p_min_ind],3)
    print (" ")
    print (pic_name, p_max_ind, p_min_ind)
    print ("max:", p_max, dirVec1(theta[p_max_ind], phi[p_max_ind]))
    print ("min:", p_min, dirVec1(theta[p_min_ind], phi[p_min_ind]))
    print ("Anisotropy", round(p_max/p_min, 3))

    x = np.sin(theta)*np.cos(phi)*p
    y = np.sin(theta)*np.sin(phi)*p
    z = np.cos(theta)*p
    
    ax = plt.subplot(111, projection='3d')
    #ax.set_xlabel('$\it^{x}$')
    #ax.set_ylabel('$\it^{y}$')
    #ax.set_zlabel('$\it^{z}$', rotation=90, ha='right')
    ax.set_xticks([-600,0,600])
    ax.set_yticks([-600,0,600])
    ax.set_zticks([-600,0,600])
    ax.set_xlim(-900, 900)
    ax.set_ylim(-900, 900)
    ax.set_zlim(-900, 600)    
    ax.set_xticklabels(['-600', '0', '600'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels(['-600', '0', '600'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_zticklabels(['-600', '0', '600'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})

    #Sc=ax.plot_surface(x, y, z, cmap=newcmp , vmin=-p_max, vmax=p_max)
    Sc=ax.scatter(x, y, z, c=p, cmap=cmp, vmin=430, vmax=770)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([450, 550, 650, 750])
    cbar.ax.tick_params(labelsize=8)
    
    ax.view_init(elev=20, azim=300)
    plt.rc('font', **font)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()
                

def make3DPlot2_G(func, npoints=100, cmp='plasma', pic_name='pic.png'):
    
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    theta, phi = np.meshgrid(theta, phi)
    p_max, p_min  = func(theta, phi)
    
    #elas.Young_2(x, y)    
    x = np.sin(theta)*np.cos(phi)*p_min
    y = np.sin(theta)*np.sin(phi)*p_min
    z = np.cos(theta)*p_min
    
    x2= np.sin(theta)*np.cos(phi)*p_max
    y2= np.sin(theta)*np.sin(phi)*p_max
    z2= np.cos(theta)*p_max
    
    p_max_ind=np.unravel_index(np.argmax(p_max, axis=None), p_max.shape)
    p_min_ind=np.unravel_index(np.argmin(p_min, axis=None), p_min.shape)
    
    p_max_3D=round(p_max[p_max_ind],3)
    p_min_3D=round(p_min[p_min_ind],3)
    print (" ")
    print (pic_name, p_max_ind, p_min_ind)
    print ("max:", p_max_3D, dirVec1(theta[p_max_ind], phi[p_max_ind]))
    print ("min:", p_min_3D, dirVec1(theta[p_min_ind], phi[p_min_ind]))
    print ("Anisotropy", round(p_max_3D/p_min_3D, 3))
    #print (np.max(p_max), np.min(p_min))

    ax = plt.subplot(111, projection='3d')
    #ax.set_xlabel('$\it^{x}$')
    #ax.set_ylabel('$\it^{y}$')
    #ax.set_zlabel('$\it^{z}$', rotation=90, ha='right')
    ax.set_xticks([-120,0,120])
    ax.set_yticks([-120,0,120])
    ax.set_zticks([-120,0,120])
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(-150, 120)    
    ax.set_xticklabels(['-120', '0', '120'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels(['-120', '0', '120'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_zticklabels(['-120', '0', '120'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})

    #Sc=ax.scatter(x, y, z, c=p_min, cmap=cmp)
    Sc=ax.scatter(x, y, z, c=p_min, cmap=cmp, vmin=95, vmax=145)
    Sc2=ax.plot_surface(x2, y2, z2, alpha=0.2)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    
    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([100, 110, 120, 130, 140])
    cbar.ax.tick_params(labelsize=8)
    
    ax.view_init(elev=20, azim=300)
    plt.rc('font', **font)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()


def make3DPlot2_P(func, npoints=100, cmp='plasma', pic_name='pic.png'):
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    theta, phi = np.meshgrid(theta, phi)
    p_max, p_min  = func(theta, phi)
    
    #elas.Young_2(x, y)    
    x = np.sin(theta)*np.cos(phi)*p_min
    y = np.sin(theta)*np.sin(phi)*p_min
    z = np.cos(theta)*p_min
    
    x2= np.sin(theta)*np.cos(phi)*p_max
    y2= np.sin(theta)*np.sin(phi)*p_max
    z2= np.cos(theta)*p_max
    
    p_max_ind=np.unravel_index(np.argmax(p_max, axis=None), p_max.shape)
    p_min_ind=np.unravel_index(np.argmin(p_min, axis=None), p_min.shape)
    
    p_max_3D=round(p_max[p_max_ind],3)
    p_min_3D=round(p_min[p_min_ind],3)
    print (" ")
    print (pic_name, p_max_ind, p_min_ind)
    print ("max:", p_max_3D, dirVec1(theta[p_max_ind], phi[p_max_ind]))
    print ("min:", p_min_3D, dirVec1(theta[p_min_ind], phi[p_min_ind]))
    print ("Anisotropy", round(p_max_3D/p_min_3D, 3))
    #print (np.max(p_max), np.min(p_min))
    
    #Sc=ax.scatter(x, y, z, c=p_min, cmap=cmp)

    ax = plt.subplot(111, projection='3d')
    #ax.set_xlabel('$\it^{x}$')
    #ax.set_ylabel('$\it^{y}$')
    #ax.set_zlabel('$\it^{z}$', rotation=90, ha='right')
    ax.set_xticks([-0.24,0,0.24])
    ax.set_yticks([-0.24,0,0.24])
    ax.set_zticks([-0.24,0,0.24])
    ax.set_xlim(-0.32,0.32)
    ax.set_ylim(-0.32,0.32)
    ax.set_zlim(-0.32,0.24)    
    ax.set_xticklabels(['-0.24', '0', '0.24'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels(['-0.24', '0', '0.24'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_zticklabels(['-0.24', '0', '0.24'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})

    Sc=ax.scatter(x, y, z, c=p_min, cmap=cmp, vmin=0.08, vmax=0.27)
    Sc2=ax.plot_surface(x2, y2, z2, alpha=0.2)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    
    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([0.10,0.15,0.20,0.25])
    cbar.ax.tick_params(labelsize=8)
    
    ax.view_init(elev=20, azim=300)
    plt.rc('font', **font)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()


def make4DPlot(func, npoints=30, cmp='plasma', pic_name='pic.png'):
    plt.rc('font',family='Times New Roman',size=16)
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    chi   = np.linspace(0, 2*np.pi, 2*npoints)
    theta, phi, chi = np.meshgrid(theta, phi, chi)
    
    p= func(theta, phi, chi)
    p_max_ind=np.unravel_index(np.argmax(p, axis=None), p.shape)
    p_min_ind=np.unravel_index(np.argmin(p, axis=None), p.shape)
    p_max=round(p[p_max_ind],3)
    p_min=round(p[p_min_ind],3)
    print (" ")
    print (pic_name, p_max_ind, p_min_ind)
    print ("max:", p_max, dirVec2(theta[p_max_ind], phi[p_max_ind], chi[p_max_ind]))
    print ("min:", p_min, dirVec2(theta[p_min_ind], phi[p_min_ind], chi[p_max_ind]))
    print ("Anisotropy", round(p_max/p_min, 3))

    #elas.Young_2(x, y)    
    x = np.cos(theta)*np.cos(phi)*np.cos(chi) - np.sin(phi)*np.sin(chi)
    y = np.cos(theta)*np.sin(phi)*np.cos(chi) + np.cos(phi)*np.sin(chi)
    z = - np.sin(theta)*np.cos(chi)
    
    #Sc=ax.plot_surface(x, y, z, cmap=newcmp , vmin=-p_max, vmax=p_max)
    Sc=ax.scatter(x, y, z, c=p, cmap=cmp)
    #Sc=ax.scatter(theta, phi, chi, c=p, cmap=cmp)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'green'})
    plt.colorbar(Sc, orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    ax.view_init(elev=20, azim=300)
    plt.rc('font', **font)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()

##########################################main#############################################################

if __name__ == "__main__":
    Gd=np.array([ [387.7,  125.1,  148.5,  0.0,  0.0, 0.0],
                  [125.1,  297.1,  125.7,  0.0,  0.0, 0.0],
                  [148.5,  125.7,  349.4,  0.0,  0.0, 0.0],
                  [0.0,  0.0,  0.0,  158.6,  0.0, 0.0],
                  [0.0,  0.0,  0.0,  0.0,  139.7, 0.0],
                  [0.0,  0.0,  0.0,  0.0,  0.0, 104.7]])
                              
    mat=Gd
    print ('Gd')              
    elas = Elastic(mat)
    
    for i in range(6):
        print(("   " + 6 * "%7.5g  ") % tuple(elas.CVoigt[i]))
    
    make3DPlot_Y(lambda x, y: elas.Young(x, y), npoints=100, cmp='viridis', pic_name="Youngs_modulus.png")
    #make3DPlot_B(lambda x, y: elas.bulk_modulus(x, y), npoints=100, cmp='viridis', pic_name="Bulk_modulus.png")
    #make3DPlot2_G(lambda x, y: elas.shear3D_new(x, y),  npoints=100, cmp='viridis', pic_name="Shear_modulus_3D.png")
    #make3DPlot2_P(lambda x, y: elas.poisson3D_new(x, y),  npoints=100, cmp='viridis', pic_name="Poisson_ratio_3D.png")
    
    print ('Young')
    ela_normal_lattice(elas.Young)
    print ('Bulk_modulus')
    #ela_normal_lattice(elas.bulk_modulus)
    print ('Shear')
    #makePolarPlot_all_S(elas.shear, npoints=100, pic_name="Shear_modulus_a.png")
    print ('Poisson')
    #makePolarPlot_all_P(elas.Poisson, npoints=100, pic_name="Poisson_ratio_a.png")
    
    #print (elas.shear3D_new(np.pi/2, 0.0))
    #make4DPlot(lambda x, y, z: elas.shear(x, y, z), npoints=60, cmp='viridis', pic_name="Shaer_modulus.png")
    #make4DPlot(lambda x, y, z: elas.Poisson(x, y, z), npoints=60, cmp='viridis', pic_name="Poisson_ratio.png")




