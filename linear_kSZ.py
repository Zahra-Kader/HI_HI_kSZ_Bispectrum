'This script is used to compute the Ostriker Vishniac signal'

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
from useful_functions import *
from cosmolopy_codes import constants, density


n_points=60
z_r=10

ell=np.geomspace(200.,1.e4,n_points)

def C_l_mu_integral(ell,z_min):
    Kp=np.geomspace(1.e-6,10.,n_points)
    Mu=np.linspace(-0.9999,0.9999,n_points)
    Z=np.geomspace(z_min,z_r,n_points)
    mu,kp,z=np.meshgrid(Mu,Kp,Z)
    H_z=H(z)
    chi_z=chi(z)
    tau_z=tau_inst(z)
    f_z=f(z)
    D_z=D_1(z)
    T_rad=2.725 #In Kelvin
    x=ionized_elec(0.24,0)
    sigma_T=constants.sigma_T_Mpc
    rho_c=density.cosmo_densities(**cosmo)[0]*constants.M_sun_g #in units of g/Mpc^3
    rho_g0=cosmo['omega_b_0']*rho_c
    mu_e=1.14
    m_p=constants.m_p_g #in grams
    const=1.e12*T_rad**2*x**2/constants.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2/8/np.pi**2
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
    theta_kp_arr=np.array([])
    theta_K_arr=np.array([])
    C_l=np.array([])
    for i in ell:
        k=i/chi_z
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
        theta_kp=np.sqrt(1-mu**2)
        theta_K=-np.sqrt(1-mu**2)*kp/K
        theta_kp_arr=np.append(theta_kp_arr,theta_kp)
        theta_K_arr=np.append(theta_K_arr,theta_K)
        I=theta_kp**2/kp**2+theta_K*theta_kp/K/kp
        #I=k*(k-2*kp*mu)*(1-mu**2)/K**2
        integral=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(kp**2*I*Mps_interpf(kp)*Mps_interpf(K)*(1+z)**2*np.exp(-2*tau_z)*f_z**2*D_z**4/chi_z**2*H_z,Mu,axis=0),Kp,axis=0),Z,axis=0)
        C_l=np.append(C_l,integral)
    return C_l, theta_kp_arr, theta_K_arr


#np.save('C_l_ksz_z_0_10_100pts',(ell,C_l_mu_integral(ell,redshift_max)[0]))

plt.plot(ell,ell*(ell+1)*(C_l_mu_integral(ell,1.e-4)[0])/2/np.pi)
plt.xlabel('l')
plt.show()
