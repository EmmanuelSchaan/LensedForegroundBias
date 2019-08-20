import imp

import universe
reload(universe)
from universe import *

import mass_function
reload(mass_function)
from mass_function import *

import profile
reload(profile)
from profile import *

import halo_fit
reload(halo_fit)
from halo_fit import *

import i_halo_model
reload(i_halo_model)
from i_halo_model import *

import pn_3d
reload(pn_3d)
from pn_3d import *

import weight
reload(weight)
from weight import *

import pn_2d
reload(pn_2d)
from pn_2d import *

import cmb
reload(cmb)
from cmb import *

import cmb_lensing_rec
imp.reload(cmb_lensing_rec)
from cmb_lensing_rec import *

import flat_map
reload(flat_map)
from flat_map import *


##################################################################################
##################################################################################
# Colors for plots

cCmb = 'k'
cKszLate = 'r'
cKszReio = 'orange'
cTsz = 'b'
cCib = 'g'
cCibTsz = 'm'
cRadiops = 'y'
#
cTszxCib = 'c'

pathFig = "./figures/kcmb_lensed_foregrounds/"
pathOut = "./output/kcmb_lensed_foregrounds/"


##################################################################################
##################################################################################
# Lensing convergence power spectrum

u = UnivPlanck15()
profNFW = ProfNFW(u)
massFunc = MassFuncTinker(u, save=0)
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


##################################################################################
# kSZ from reionization: put a single source at z=8

w_kszreiolens = WeightLensSingle(u, z_source=8., name="kszreiolens")
p2d_kszreiolens = P2dAuto(u, halofit, w_kszreiolens, doT=False, nProc=3, save=False)

##################################################################################
# radio point sources

# Eq 26 in de Zotti+10, giving the redshift distribution of the radio sources
# in the sample from Brookes+08, selected from NVSS data at 1.4GHz.
fdndz_radiops = lambda z: 1.29 + 32.37*z - 32.89*z**2 + 11.13*z**3 - 1.25*z**4

w_radiopslens = WeightLensCustom(u, fdndz_radiops, zMin=1.e-4, zMax=3.5, name="radiopslens")
p2d_radiopslens = P2dAuto(u, halofit, w_radiopslens, doT=False, nProc=3, save=False)


##################################################################################
# late time kSZ lensing:

# Use dC_(ell=3000)/dz from Shaw+12, fig6, simulation L60CSFz2
# as the source distribution
path = "./input/ksz_dcldz_Shaw+12_fig6/Shaw+12_fig6_L60CSFz2.txt"
data = np.genfromtxt(path)
fdndz_ksz = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)

w_kszlens = WeightLensCustom(u, fdndz_ksz, zMin=1.e-4, zMax=10., name="kszlens")
p2d_kszlens = P2dAuto(u, halofit, w_kszlens, doT=False, nProc=3, save=False)

##################################################################################
# CIB lensing

massFunc = MassFuncTinker(u, save=False)
iHaloModel = IHaloModel(u, massFunc)

# 353GHz
profHODPenin12_353 = ProfHODPenin12(u, massFunc, nu=353)
p3d_galPenin12_353 = P3dAuto(u, iHaloModel, profHODPenin12_353, fPnoise=profHODPenin12_353.fPshotNoise, fTnoise=profHODPenin12_353.fTshotNoise, name="galpenin353", doT=True, save=False)
w_cib353 = WeightCIBPenin12(u, nu=353.e9, fluxCut=315.e-3, name='cibpenin12')
p2d_cib353 = P2dAuto(u, p3d_galPenin12_353, w_cib353, fPnoise=w_cib353.fPshotNoise, fTnoise=w_cib353.fTshotNoise, doT=True, save=False, nProc=3)

# Use dC_(ell=3000)/dz as the source distribution
fdndz_cib = lambda z: p2d_cib353.fdPdz(3.e3, z) + p2d_cib353.fdPnoisedz(3.e3, z)

w_ciblens = WeightLensCustom(u, fdndz_cib, zMin=1.e-4, zMax=10., name="ciblens")
p2d_ciblens = P2dAuto(u, halofit, w_ciblens, doT=False, nProc=3, save=False)

##################################################################################
# tSZ lensing

profY = ProfY(u)
p3d_y = P3dAuto(u, iHaloModel, profY, doT=True, save=False)
w_y = WeightY(u)
p2d_y = P2dAuto(u, p3d_y, w_y, doT=True, nProc=3, save=False)

# Use dC_(ell=3000)/dz as the source distribution
fdndz_tsz = lambda z: p2d_y.fdPdz(3.e3, z)

w_tszlens = WeightLensCustom(u, fdndz_tsz, zMin=1.e-4, zMax=4., name="tszlens")
p2d_tszlens = P2dAuto(u, halofit, w_tszlens, doT=False, nProc=3, save=False)


##################################################################################
# Compare the various redshift distributions

Z = np.linspace(0.1, 10., 501)
A = 1./(1.+Z)

# Source distributions
fig=plt.figure(0)
ax=fig.add_subplot(111)
#
# CIB
dCIBdz = np.array(map(w_ciblens.fdpdz, Z))
ax.plot(Z, dCIBdz, c=cCib, label=r'CIB')
#
# kSZ late
dKSZLatedz = np.array(map(w_kszlens.fdpdz, Z))
ax.plot(Z, dKSZLatedz, c=cKszLate, label=r'kSZ late')
#
# kSZ reio
ax.axvline(8., ymin=0., c=cKszReio, label=r'kSZ reio')
#
# radio PS
dRadiopsdz = np.array(map(w_radiopslens.fdpdz, Z))
ax.plot(Z, dRadiopsdz, c=cRadiops, label=r'Radio PS')
#
# tSZ
dTSZdz = np.array(map(w_tszlens.fdpdz, Z))
ax.plot(Z, dTSZdz, c=cTsz, label=r'tSZ')
#
ax.legend(loc='upper center')
ax.set_ylim((0., 2.))
ax.set_xlim((0., 10.))
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'Redshift distribution')
#
fig.savefig(pathFig+"foregrounds_z_distributions.pdf", bbox_inches='tight')
fig.clf()
#plt.show()


# Lensing kernels, to be integrated wrt chi
Wcmb = np.array(map(w_cmblens.f, A))
Wcib = np.array(map(w_ciblens.f, A))
Wksz = np.array(map(w_kszlens.f, A))
Wkszreio = np.array(map(w_kszreiolens.f, A))
Wradiops = np.array(map(w_radiopslens.f, A))
Wtsz = np.array(map(w_tszlens.f, A))
# inverse hubble length: H/c in (h Mpc^-1)
H_A = u.Hubble(A) / 3.e5

# Lensing kernels
fig=plt.figure(1)
ax=fig.add_subplot(111)
#
ax.plot(Z, Wcmb/H_A, c=cCmb, label=r'CMB')
ax.plot(Z, Wcib/H_A, c=cCib, label=r'CIB')
ax.plot(Z, Wksz/H_A, c=cKszLate, label=r'kSZ late')
ax.plot(Z, Wkszreio/H_A, c=cKszReio, label=r'kSZ reio')
ax.plot(Z, Wradiops/H_A, c=cRadiops, label=r'Radio PS')
ax.plot(Z, Wtsz/H_A, c=cTsz, label=r'tSZ')
#
ax.legend(loc=1)
ax.set_ylim((0., 0.36))
ax.set_xlim((0., 10.))
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$W^\kappa(z)$')
#
fig.savefig(pathFig+"lensing_kernels.pdf", bbox_inches='tight')
fig.clf()
#plt.show()



##################################################################################
# Lensing bias from lensed foregrounds:
# reduction factor from the fact that kappa_f \neq kappa_cmb

# cross-correlations
p2d_ciblenscmblens = P2dCross(u, halofit, w_ciblens, w_cmblens, doT=False, nProc=3, save=False)
p2d_kszlenscmblens = P2dCross(u, halofit, w_kszlens, w_cmblens, doT=False, nProc=3, save=False)
p2d_kszreiolenscmblens = P2dCross(u, halofit, w_kszreiolens, w_cmblens, doT=False, nProc=3, save=False)
p2d_tszlenscmblens = P2dCross(u, halofit, w_tszlens, w_cmblens, doT=False, nProc=3, save=False)
p2d_radiopslenscmblens = P2dCross(u, halofit, w_radiopslens, w_cmblens, doT=False, nProc=3, save=False)

# ratios of cross to auto, relevant for bias to CMB lensing auto
L = np.genfromtxt("./input/Lc.txt")
rKszReio = p2d_kszreiolenscmblens.fPinterp(L) / p2d_cmblens.fPinterp(L)
rKszLate = p2d_kszlenscmblens.fPinterp(L) / p2d_cmblens.fPinterp(L)
rCib = p2d_ciblenscmblens.fPinterp(L) / p2d_cmblens.fPinterp(L)
rTsz = p2d_tszlenscmblens.fPinterp(L) / p2d_cmblens.fPinterp(L)
rRadiops = p2d_radiopslenscmblens.fPinterp(L) / p2d_cmblens.fPinterp(L)


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.semilogx(L, rKszReio, c=cKszReio, label=r'reio kSZ')
ax.semilogx(L, rKszLate, c=cKszLate, label=r'late kSZ')
ax.semilogx(L, rCib, c=cCib, label=r'CIB')
ax.semilogx(L, rTsz, c=cTsz, label=r'tSZ')
ax.semilogx(L, rRadiops, c=cRadiops, label=r'Radio PS')
#
ax.legend(loc=1)
ax.set_ylim((0., 1.))
ax.set_xlabel(r'L')
ax.set_ylabel(r'$\langle \kappa_\text{f}\; \kappa_\text{CMB} \rangle / \langle \kappa_\text{CMB} \; \kappa_\text{CMB} \rangle$')
ax.set_ylabel(r'$C^{\kappa_\text{f}\; \kappa_\text{CMB}}_L / C^{\kappa_\text{CMB} \; \kappa_\text{CMB}}_L$')
#
fig.savefig(pathFig+"kfk_over_kk.pdf", bbox_inches='tight')
fig.clf()
#plt.show()


##################################################################################
# Cross with galaxy tracer

w_lsstgold = WeightTracerLSSTGold(u)
p2d_lsstgold = P2dAuto(u, halofit, w_lsstgold, fPnoise=lambda l:1./w_lsstgold.ngal, nProc=3, save=False)
p2d_cmblenslsstgold = P2dCross(u, halofit, w_lsstgold, w_cmblens, nProc=3, save=False)

# cross-correlations
p2d_ciblenslsstgold = P2dCross(u, halofit, w_lsstgold, w_ciblens, nProc=3, save=False)
p2d_kszlenslsstgold = P2dCross(u, halofit, w_lsstgold, w_kszlens, nProc=3, save=False)
p2d_kszreiolenslsstgold = P2dCross(u, halofit, w_lsstgold, w_kszreiolens, nProc=3, save=False)
p2d_tszlenslsstgold = P2dCross(u, halofit, w_lsstgold, w_tszlens, nProc=3, save=False)
p2d_radiopslenslsstgold = P2dCross(u, halofit, w_lsstgold, w_radiopslens, nProc=3, save=False)

# ratios of cross to auto, relevant for bias to CMB lensing cross
L = np.genfromtxt("./input/Lc.txt")
rCibCross = p2d_ciblenslsstgold.fPinterp(L) / p2d_cmblenslsstgold.fPinterp(L)
rKszLateCross = p2d_kszlenslsstgold.fPinterp(L) / p2d_cmblenslsstgold.fPinterp(L)
rKszReioCross = p2d_kszreiolenslsstgold.fPinterp(L) / p2d_cmblenslsstgold.fPinterp(L)
rTszCross = p2d_tszlenslsstgold.fPinterp(L) / p2d_cmblenslsstgold.fPinterp(L)
rRadiopsCross = p2d_radiopslenslsstgold.fPinterp(L) / p2d_cmblenslsstgold.fPinterp(L)


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.semilogx(L, rKszReioCross, c=cKszReio, label=r'reio kSZ')
ax.semilogx(L, rKszLateCross, c=cKszLate, label=r'late kSZ')
ax.semilogx(L, rCibCross, c=cCib, label=r'CIB')
ax.semilogx(L, rTszCross, c=cTsz, label=r'tSZ')
ax.semilogx(L, rRadiopsCross, c=cRadiops, label=r'Radio PS')
#
ax.legend(loc=1)
ax.set_ylim((0., 1.))
ax.set_xlabel(r'L')
ax.set_ylabel(r'$C_L^{g \kappa_\text{f}} / C_L^{g \kappa_\text{CMB}}$')
#
fig.savefig(pathFig+"kfg_over_kcmbg.pdf", bbox_inches='tight')
fig.clf()
#plt.show()



##################################################################################
##################################################################################
# CMB specifications

# Adjust the lMin and lMax to the assumptions of the analysis
lMin = 30.
lMaxT = 3.5e3 #3.e3   # 4.e3
lMaxP = 5.e3

# CMB specs
cmb = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMaxT, lMaxP=lMaxP, atm=False)

# weights for lensing
forCtotal = lambda l: cmb.flensedTT(l) + cmb.fkSZ(l) + cmb.fCIB(l) + cmb.ftSZ(l) + cmb.ftSZ_CIB(l) + cmb.fradioPoisson(l) + cmb.fdetectorNoise(l)
# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin), np.log10(lMaxT), 1001, 10.)
F = np.array(map(forCtotal, L))
cmb.ftotalTT = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

# splitting the late time and reionization kSZ,
# from Calabrese Hlozek +14
cmb.fkSZLate = lambda l: 0.45 * cmb.fkSZ(l)
cmb.fkSZReio = lambda l: 0.55 * cmb.fkSZ(l)


##################################################################################
# Foreground power spectra

# Use high lMaxT just for plotting
cmbPlot = StageIVCMB(beam=1.4, noise=7., lMin=lMin, lMaxT=1.e4, lMaxP=lMaxP, atm=False)
# splitting the late time and reionization kSZ,
# from Calabrese Hlozek +14
cmbPlot.fkSZLate = lambda l: 0.45 * cmb.fkSZ(l)
cmbPlot.fkSZReio = lambda l: 0.55 * cmb.fkSZ(l)

Nl = 1001
L = np.logspace(np.log10(1.), np.log10(3.6e4), Nl, 10.)

UnlensedTT = np.array(map(lambda l: cmbPlot.funlensedTT(l), L))
LensedCMB = np.array(map(lambda l: cmbPlot.flensedTT(l), L))
CIB = np.array(map(lambda l: cmbPlot.fCIB(l), L))
TSZ = np.array(map(lambda l: cmbPlot.ftSZ(l), L))
KSZLate = np.array(map(lambda l: cmbPlot.fkSZLate(l), L))
KSZReio = np.array(map(lambda l: cmbPlot.fkSZReio(l), L))
TSZ_CIB = np.array(map(lambda l: cmbPlot.ftSZ_CIB(l), L))
RadioPS = np.array(map(lambda l: cmbPlot.fradioPoisson(l), L))
DetectorNoise = np.array(map(lambda l: cmbPlot.fdetectorNoise(l), L))
Total = np.array(map(lambda l: cmbPlot.ftotalTT(l), L))


# debeamed Cl
#fig=plt.figure(1, figsize=(9, 7))
fig=plt.figure(1)
ax=plt.subplot(111)
#
factor = L*(L+1.)/(2.*np.pi)
ax.loglog(L, factor * abs(LensedCMB), 'gray', lw=2, label=r'CMB')
ax.loglog(L, factor * CIB, c=cCib, lw=2, label=r'CIB')
ax.loglog(L, factor * TSZ, c=cTsz, lw=2, label=r'tSZ')
ax.loglog(L, factor * KSZLate, c=cKszLate, lw=2, label=r'kSZ late')
ax.loglog(L, factor * KSZReio, c=cKszReio, lw=2, label=r'kSZ reio')
ax.loglog(L, factor * np.abs(TSZ_CIB), 'c', lw=2, label=r'$|$tSZ$\times$CIB$|$')
ax.loglog(L, factor * RadioPS, 'y', lw=2, label=r'radio PS')
ax.loglog(L, factor * DetectorNoise, 'k--', lw=2, label=r'det. noise')
ax.loglog(L, factor * Total, 'k', lw=2, label=r'total')
#
ax.grid()
ax.legend(loc='center left', fontsize='x-small', labelspacing=0.1)
ax.set_xlim((35., 1.e4))
#ax.set_ylim((1.e-8, 1.e1))
ax.set_ylim((5.e-2, 1.e4))
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell(\ell+1)C_\ell / (2\pi)$ [$(\mu K)^2$]')
#
fig.savefig(pathFig+"cl_foregrounds.pdf", bbox_inches='tight')
fig.clf()
#plt.show()



##################################################################################
##################################################################################
# Lensing foreground bias to CMB lensing: analytical

import cmb_lensing_rec
imp.reload(cmb_lensing_rec)
from cmb_lensing_rec import *


# lensing noise for QE
cmbLensRec = CMBLensRec(cmb, nProc=3)

# Normalizations and lensing noise autos for Q_XY
#cmbLensRec.saveNormQ(pol=True)
cmbLensRec.loadNormQ()
#cmbLensRec.plotNoiseQAuto(pol=True)

# Lensing noise crosses for Q_XY and Q_WZ
#cmbLensRec.saveNoiseCov()
cmbLensRec.loadNoiseCov()
#cmbLensRec.plotNoiseQ(pol=True)

# Normalizations for shear, dilation
#cmbLensRec.saveNormSD()
cmbLensRec.loadNormSD()

# Lensing noises for shear, dilation and QE (test)
#cmbLensRec.saveNoiseQSD()
cmbLensRec.loadNoiseQSD()
#cmbLensRec.plotNoiseQSD()


##################################################################################
# Primary bias

# Response to foreground lensing
#cmbLensRec.saveResponseLensedForeground()
cmbLensRec.loadResponseLensedForeground()
#cmbLensRec.plotResponseLensedForeground()


##################################################################################
# Secondary bias: QE, shear, mag

'''


# use same ells as Nishant
Lsec = np.array([1.390144417910510466e+00,
               2.474858587384993758e+00,
               4.405963113357286964e+00,
               7.843886941749206088e+00,
               1.398822509939085457e+01,
               2.561875184571284336e+01,
               4.458203893325971023e+01,
               8.043406382909739705e+01,
               1.432189839983497279e+02,
               2.558277572968322602e+02,
               4.565886925289308920e+02,
               8.124089972056070792e+02,
               1.446123378920690129e+03,
               2.574491456291714712e+03,
               4.583296068620806182e+03,
               7.177894145556705553e+03])

# 24 for edison, 32 for cori
nProc = min(32, len(Lsec))  #24
#L = np.logspace(np.log10(10.), np.log10(5.e3), nProc-1, 10.)



# Choose estimator: q,s,d
if sys.argv[1]=='q':
   XY = 'TT'
elif sys.argv[1]=='s':
   XY = 'sTT'
elif sys.argv[1]=='d':
   XY = 'dTT'

# Choose foreground
if sys.argv[2]=='cib':
   # CIB
   f = lambda l: cmbLensRec.secondaryLensedForegroundBias(cmb.fCIB, p2d_ciblenscmblens.fPinterp, l, XY)
elif sys.argv[2]=='tsz':
   # tSZ
   f = lambda l: cmbLensRec.secondaryLensedForegroundBias(cmb.ftSZ, p2d_tszlenscmblens.fPinterp, l, XY)
elif sys.argv[2]=='kszlate':
   # kSZ late
   f = lambda l: cmbLensRec.secondaryLensedForegroundBias(cmb.fkSZLate, p2d_kszlenscmblens.fPinterp, l, XY)
elif sys.argv[2]=='kszreio':
   # kSZ reio
   f = lambda l: cmbLensRec.secondaryLensedForegroundBias(cmb.fkSZReio, p2d_kszreiolenscmblens.fPinterp, l, XY)
elif sys.argv[2]=='radiops':
   # radio PS
   f = lambda l: cmbLensRec.secondaryLensedForegroundBias(cmb.fradioPoisson, p2d_radiopslenscmblens.fPinterp, l, XY)
#
elif sys.argv[2]=='tszxcib':
   # tSZxCIB
   fCkkcmb = lambda l: p2d_ciblenscmblens.fPinterp(l) + p2d_tszlenscmblens.fPinterp(l)
   f = lambda l: cmbLensRec.secondaryLensedForegroundBias(cmb.ftSZ_CIB, fCkkcmb, l, XY)


#tStart = time()
#nCalls = np.int(1.e3)
#for i in range(nCalls):
##   cmb.ftSZ_CIB(3000.)
##   cmb.fCIB(3000.)
##   cmb.funlensedTT(3000.)
#   cmb.fprime(143.e9, 143.e9, 2.1, 9.7)
#tStop = time()
#print "took", (tStop - tStart)/nCalls, "sec"


# parallel evaluation
pool = Pool(ncpus=nProc)
tStart = time()
result = np.array(pool.map(f, Lsec))
#result = np.array(map(f, Lsec))
tStop = time()
print "took", (tStop - tStart)/60., "min"

# get the relative bias, and its uncertainty
result[:,0] /= np.array(map(p2d_cmblens.fPinterp, Lsec))
result[:,1] /= np.array(map(p2d_cmblens.fPinterp, Lsec))

print Lsec
print result

# save the result
data = np.zeros((len(Lsec), 3))
data[:,0] = Lsec
data[:,1] = result[:,0]
data[:,2] = result[:,1]
#np.savetxt("test_secondary.txt", data)
#
np.savetxt(pathOut+sys.argv[1]+"_sec_"+sys.argv[2]+"_lmaxt"+str(int(lMaxT))+".txt", data)



#fig=plt.figure(0)
#ax=fig.add_subplot(111)
##
#ax.errorbar(L, result[:,0], yerr=result[:,1], c='b', fmt='-')
#ax.errorbar(L, -result[:,0], yerr=result[:,1], c='b', fmt='--')
##
#ax.plot(L, np.abs(result[:,1]/result[:,0]), 'r')
##
#ax.set_xscale('log', nonposx='clip')
#ax.set_yscale('log', nonposy='clip')
##ax.set_ylim((9.e-6, 1.01))
#ax.grid()
##
#fig.savefig("./test_secondary.pdf", bbox_inches='tight')
#fig.clf()

'''




##################################################################################
##################################################################################
# Lensed foreground bias to CMB lensing: FFT

# map object
nX = 300#600#1200
nY = 300#600#1200
size = 10.#20.  # degrees
baseMap = FlatMap(nX=nX, nY=nY, sizeX=size*np.pi/180., sizeY=size*np.pi/180.)


##################################################################################
# Compute SNR on amplitude of lensing power spectrum for Q, D, S

fSky = 0.4

# Forecast noise power spectrum
fNq = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.ftotalTT, lMin=lMin, lMax=lMaxT, test=False)
fNd = baseMap.forecastN0KappaDilation(cmb.funlensedTT, cmb.ftotalTT, lMin=lMin, lMax=lMaxT, test=False)
fNs = baseMap.forecastN0KappaShear(cmb.funlensedTT, cmb.ftotalTT, lMin=lMin, lMax=lMaxT, test=False)


# Compute total SNR on amplitude of lensing power spectrum, including cosmic variance
def integrand(lnl, fNl, fSky=1.):
   l = np.exp(lnl)
   result = l**2
   result *= p2d_cmblens.fPinterp(l)**2 / (p2d_cmblens.fPinterp(l) + fNl(l))**2  # including cosmic var
   result *= fSky
   return result
# Q
f = lambda lnl: integrand(lnl, fNq, fSky=fSky)
snrQ = integrate.quad(f, np.log(lMin), np.log(2.*lMaxT), epsabs=0., epsrel=1.e-3)[0]
snrQ = np.sqrt(snrQ)
# S
f = lambda lnl: integrand(lnl, fNs, fSky=fSky)
snrS = integrate.quad(f, np.log(lMin), np.log(2.*lMaxT), epsabs=0., epsrel=1.e-3)[0]
snrS = np.sqrt(snrS)
# D
f = lambda lnl: integrand(lnl, fNd, fSky=fSky)
snrD = integrate.quad(f, np.log(lMin), np.log(2.*lMaxT), epsabs=0., epsrel=1.e-3)[0]
snrD = np.sqrt(snrD)
print "SNR in Q, S, D:", snrQ, snrS, snrD


# Compute total SNR on amplitude of cross spectrum, including cosmic variance
def integrand(lnl, fNl, fSky=1.):
   l = np.exp(lnl)
   result = 2. * l**2
   result *= p2d_cmblenslsstgold.fPinterp(l)**2
   result /= p2d_cmblenslsstgold.fPinterp(l)**2 + p2d_lsstgold.fPtotinterp(l)*(p2d_cmblens.fPinterp(l) + fNl(l))  # including cosmic var
   result *= fSky
   return result
# Q
f = lambda lnl: integrand(lnl, fNq, fSky=fSky)
#snrCrossQ = integrate.quad(f, np.log(lMin), np.log(2.*lMaxT), epsabs=0., epsrel=1.e-3)[0]
snrCrossQ = integrate.quad(f, np.log(lMin), np.log(1.e3), epsabs=0., epsrel=1.e-3)[0]
snrCrossQ = np.sqrt(snrCrossQ)
# S
f = lambda lnl: integrand(lnl, fNs, fSky=fSky)
#snrCrossS = integrate.quad(f, np.log(lMin), np.log(2.*lMaxT), epsabs=0., epsrel=1.e-3)[0]
snrCrossS = integrate.quad(f, np.log(lMin), np.log(1.e3), epsabs=0., epsrel=1.e-3)[0]
snrCrossS = np.sqrt(snrCrossS)
# D
f = lambda lnl: integrand(lnl, fNd, fSky=fSky)
#snrCrossD = integrate.quad(f, np.log(lMin), np.log(2.*lMaxT), epsabs=0., epsrel=1.e-3)[0]
snrCrossD = integrate.quad(f, np.log(lMin), np.log(1.e3), epsabs=0., epsrel=1.e-3)[0]
snrCrossD = np.sqrt(snrCrossD)
print "SNR in Q, S, D:", snrCrossQ, snrCrossS, snrCrossD




##################################################################################
# Primary bias: QE

# Compute the multiplicative biases with FFT
fm_CMB = baseMap.forecastMultBiasLensedForegrounds(cmb.funlensedTT, cmb.ftotalTT, cmb.funlensedTT, lMin=lMin, lMax=lMaxT, test=False)
fm_CIB = baseMap.forecastMultBiasLensedForegrounds(cmb.funlensedTT, cmb.ftotalTT, cmb.fCIB, lMin=lMin, lMax=lMaxT, test=False)
fm_kSZLate= baseMap.forecastMultBiasLensedForegrounds(cmb.funlensedTT, cmb.ftotalTT, cmb.fkSZLate, lMin=lMin, lMax=lMaxT, test=False)
fm_kSZReio= baseMap.forecastMultBiasLensedForegrounds(cmb.funlensedTT, cmb.ftotalTT, cmb.fkSZReio, lMin=lMin, lMax=lMaxT, test=False)
fm_tSZ= baseMap.forecastMultBiasLensedForegrounds(cmb.funlensedTT, cmb.ftotalTT, cmb.ftSZ, lMin=lMin, lMax=lMaxT, test=False)
fm_radioPS= baseMap.forecastMultBiasLensedForegrounds(cmb.funlensedTT, cmb.ftotalTT, cmb.fradioPoisson, lMin=lMin, lMax=lMaxT, test=False)
#
fm_tSZxCIB = baseMap.forecastMultBiasLensedForegrounds(cmb.funlensedTT, cmb.ftotalTT, cmb.ftSZ_CIB, lMin=lMin, lMax=lMaxT, test=False)

# Evaluate FFT calculation for plotting
L = np.genfromtxt("./input/Lc.txt")
mCMB = fm_CMB(L)
mCIB = fm_CIB(L)
mkSZLate = fm_kSZLate(L)
mkSZReio = fm_kSZReio(L)
mtSZ = fm_tSZ(L)
mradioPS = fm_radioPS(L)
#
mtSZxCIB = fm_tSZxCIB(L)

# Evaluate analytical calculation for plotting
mCMBAna = cmbLensRec.fResponseLensedFg['TTcmb'](L)
mCIBAna = cmbLensRec.fResponseLensedFg['TTcib'](L)
mkSZLateAna = 0.45 * cmbLensRec.fResponseLensedFg['TTksz'](L)
mkSZReioAna = 0.55 * cmbLensRec.fResponseLensedFg['TTksz'](L)
mtSZAna = cmbLensRec.fResponseLensedFg['TTtsz'](L)
mradioPSAna = cmbLensRec.fResponseLensedFg['TTradiops'](L)
#
mtSZxCIBAna = cmbLensRec.fResponseLensedFg['TTtszxcib'](L)


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
#ax.loglog(L, mCMB, c=cCmb, label=r'CMB')
ax.loglog(L, mCIB, c=cCib, label=r'CIB')
ax.loglog(L, -mCIB, c=cCib, ls='--')
ax.loglog(L, mkSZLate, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mkSZLate, c=cKszLate, ls='--')
ax.loglog(L, mkSZReio, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mkSZReio, c=cKszReio, ls='--')
ax.loglog(L, mtSZ, c=cTsz, label=r'tSZ')
ax.loglog(L, -mtSZ, c=cTsz, ls='--')
ax.loglog(L, mradioPS, c=cRadiops, label=r'radioPS')
ax.loglog(L, -mradioPS, c=cRadiops, ls='--')
#
ax.loglog(L, mtSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mtSZxCIB, c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Lensing response $\mathcal{R}_L^f$')
ax.set_title(r'QE')
#
fig.savefig(pathFig+"q_lensing_responses.pdf", bbox_inches='tight')
fig.clf()
#plt.show()


fig=plt.figure(1)
ax=fig.add_subplot(111)
#
ax.loglog(L, mCMB, c=cCmb, label=r'CMB FFT')
ax.loglog(L, mCMBAna, c=cCmb, lw=1., label=r'CMB Ana')
#
ax.loglog(L, mCIB, c=cCib, label=r'CIB FFT')
ax.loglog(L, -mCIB, c=cCib, ls='--')
ax.loglog(L, mCIBAna, c=cCib, lw=1., label=r'CIB Ana')
ax.loglog(L, -mCIBAna, c=cCib, lw=1., ls='--')
#
ax.loglog(L, mkSZLate, c=cKszLate, label=r'kSZ late FFT')
ax.loglog(L, -mkSZLate, c=cKszLate, ls='--')
ax.loglog(L, mkSZLateAna, c=cKszLate, lw=1., label=r'kSZ late Ana')
ax.loglog(L, -mkSZLateAna, c=cKszLate, lw=1., ls='--')
#
ax.loglog(L, mkSZReio, c=cKszReio, label=r'kSZ reio FFT')
ax.loglog(L, -mkSZReio, c=cKszReio, ls='--')
ax.loglog(L, mkSZReioAna, c=cKszReio, lw=1., label=r'kSZ reio Ana')
ax.loglog(L, -mkSZReioAna, c=cKszReio, lw=1., ls='--')
#
ax.loglog(L, mtSZ, c=cTsz, label=r'tSZ FFT')
ax.loglog(L, -mtSZ, c=cTsz, ls='--')
ax.loglog(L, mtSZAna, c=cTsz, lw=1., label=r'tSZ Ana')
ax.loglog(L, -mtSZAna, c=cTsz, lw=1., ls='--')
#
ax.loglog(L, mradioPS, c=cRadiops, label=r'radioPS FFT')
ax.loglog(L, -mradioPS, c=cRadiops, ls='--')
ax.loglog(L, mradioPSAna, c=cRadiops, lw=1., label=r'radioPS Ana')
ax.loglog(L, -mradioPSAna, c=cRadiops, lw=1., ls='--')
#
ax.loglog(L, mtSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB FFT')
ax.loglog(L, -mtSZxCIB, c=cTszxCib, ls='--')
ax.loglog(L, mtSZxCIBAna, c=cTszxCib, lw=1., label=r'tSZ$\times$CIB Ana')
ax.loglog(L, -mtSZxCIBAna, c=cTszxCib, lw=1., ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Lensing response $\mathcal{R}_L^f$')
ax.set_title(r'QE')
#
fig.savefig(pathFig+"q_lensing_responses_compare_ana.pdf", bbox_inches='tight')
fig.clf()
#plt.show()



fig=plt.figure(2)
ax=fig.add_subplot(111)
#
ax.fill_between(L, np.ones_like(L)/snrQ, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax.loglog(L, mCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mCIB * 2. * rCib, c=cCib, label=r'CIB')
ax.loglog(L, -mCIB * 2. * rCib, c=cCib, ls='--')
ax.loglog(L, mkSZReio * 2. * rKszReio, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mkSZReio * 2. * rKszReio, c=cKszReio, ls='--')
ax.loglog(L, mkSZLate * 2. * rKszLate, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mkSZLate * 2. * rKszLate, c=cKszLate, ls='--')
ax.loglog(L, mtSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax.loglog(L, -mtSZ * 2. * rTsz, c=cTsz, ls='--')
ax.loglog(L, mradioPS * 2. * rRadiops, c=cRadiops, label=r'Radio PS')
ax.loglog(L, -mradioPS * 2. * rRadiops, c=cRadiops, ls='--')
#
ax.loglog(L, mtSZxCIB * 2. * (rCib + rTsz), c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mtSZxCIB * 2. * (rCib + rTsz), c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Primary relative bias on $C_L^{\kappa_\text{CMB}}$')
ax.set_title(r'QE')
#
fig.savefig(pathFig+"q_primary_bias.pdf", bbox_inches='tight')
fig.clf()
#plt.show()


fig=plt.figure(3)
ax=fig.add_subplot(111)
#
ax.fill_between(L, np.ones_like(L)/snrCrossQ, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax.loglog(L, mCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mCIB * 2. * rCibCross, c=cCib, label=r'CIB')
ax.loglog(L, -mCIB * 2. * rCibCross, c=cCib, ls='--')
ax.loglog(L, mkSZReio * 2. * rKszReioCross, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mkSZReio * 2. * rKszReioCross, c=cKszReio, ls='--')
ax.loglog(L, mkSZLate * 2. * rKszLateCross, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mkSZLate * 2. * rKszLateCross, c=cKszLate, ls='--')
ax.loglog(L, mtSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax.loglog(L, -mtSZ * 2. * rTsz, c=cTsz, ls='--')
ax.loglog(L, mradioPS * 2. * rRadiopsCross, c=cRadiops, label=r'Radio PS')
ax.loglog(L, -mradioPS * 2. * rRadiopsCross, c=cRadiops, ls='--')
#
ax.loglog(L, mtSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mtSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Relative bias on $C_L^{g \kappa_\text{CMB}}$')
ax.set_title(r'QE')
#
fig.savefig(pathFig+"q_bias_cross.pdf", bbox_inches='tight')
fig.clf()
#plt.show()


##################################################################################
# secondary bias: QE

'''
# read secondary biases
data = np.genfromtxt(pathOut+"q_sec_cib_lmaxt3500.txt")
Lsec = data[:,0]
secCIB = data[:,1]
sSecCIB = data[:,2]
fsecCIB = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"q_sec_kszreio_lmaxt3500.txt")
#seckSZReio = data[:,1]
#sSeckSZReio = data[:,2]
#fseckSZReio = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"q_sec_kszlate_lmaxt3500.txt")
#seckSZLate = data[:,1]
#sSeckSZLate = data[:,2]
#fseckSZLate = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"q_sec_tsz_lmaxt3500.txt")
#sectSZ = data[:,1]
#sSectSZ = data[:,2]
#fsectSZ = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"q_sec_radiops_lmaxt3500.txt")
#secradioPS = data[:,1]
#sSecradioPS = data[:,2]
#fsecradioPS = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#
data = np.genfromtxt(pathOut+"q_sec_tszxcib_lmaxt3500.txt")
sectSZxCIB = data[:,1]
sSectSZxCIB = data[:,2]
fsectSZxCIB = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)


fig=plt.figure(4)
ax=fig.add_subplot(111)
#
ax.fill_between(Lsec, np.ones_like(Lsec)/snrQ, edgecolor='', facecolor='gray', alpha=0.3)
#
ax.errorbar(Lsec, secCIB, sSecCIB, c=cCib, label=r'CIB')
ax.errorbar(Lsec, -secCIB, -sSecCIB, c=cCib, ls='--')
#ax.errorbar(Lsec, seckSZReio, sSeckSZReio, c=cKszReio, label=r'kSZ reio')
#ax.errorbar(Lsec, -seckSZReio, -sSeckSZReio, c=cKszReio, ls='--')
#ax.errorbar(Lsec, seckSZLate, sSeckSZLate, c=cKszLate, label=r'kSZ late')
#ax.errorbar(Lsec, -seckSZLate, -sSeckSZLate, c=cKszLate, ls='--')
#ax.errorbar(Lsec, sectSZ, sSectSZ, c=cTsz, label=r'tSZ')
#ax.errorbar(Lsec, -sectSZ, -sSectSZ, c=cTsz, ls='--')
#ax.errorbar(Lsec, secradioPS, sSecradioPS, c=cRadiops, label=r'Radio PS')
#ax.errorbar(Lsec, -secradioPS, -sSecradioPS, c=cRadiops, ls='--')
#
ax.errorbar(Lsec, sectSZxCIB, sSectSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB')
ax.errorbar(Lsec, -sectSZxCIB, -sSectSZxCIB, c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Secondary relative bias on $C_L^{\kappa_\text{CMB}}$')
ax.set_title(r'QE')
#
#fig.savefig(pathFig+"q_secondary_bias.pdf", bbox_inches='tight')
#fig.clf()
plt.show()


fig=plt.figure(5)
ax=fig.add_subplot(111)
#
ax.fill_between(L, np.ones_like(L)/snrQ, edgecolor='', facecolor='gray', alpha=0.3)
#
ax.loglog(L, (fsecCIB(L) + mCIB*2.*rCib), c=cCib, label=r'CIB')
ax.loglog(L, -(fsecCIB(L) + mCIB*2.*rCib), c=cCib, ls='--')
ax.loglog(L, (fseckSZReio(L) + mkSZReio*2.*rKszReio), c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -(fseckSZReio(L) + mkSZReio*2.*rKszReio), c=cKszReio, ls='--')
ax.loglog(L, (fseckSZLate(L) + mkSZLate*2.*rKszLate), c=cKszLate, label=r'kSZ late')
ax.loglog(L, -(fseckSZLate(L) + mkSZLate*2.*rKszLate), c=cKszLate, ls='--')
ax.loglog(L, (fsectSZ(L) + mtSZ*2.*rTsz), c=cTsz, label=r'tSZ')
ax.loglog(L, -(fsectSZ(L) + mtSZ*2.*rTsz), c=cTsz, ls='--')
ax.loglog(L, (fsecradioPS(L) + mradioPS*2.*rRadiops), c=cRadiops, label=r'Radio PS')
ax.loglog(L, -(fsecradioPS(L) + mradioPS*2.*rRadiops), c=cRadiops, ls='--')
#
ax.loglog(L, (fsectSZxCIB(L) + mtSZxCIB*2.*(rCib+rTsz)), c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -(fsectSZxCIB(L) + mtSZxCIB*2.*(rCib+rTsz)), c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Prim.+Sec. relative bias on $C_L^{\kappa_\text{CMB}}$')
ax.set_title(r'QE')
#
fig.savefig(pathFig+"q_primsec_bias.pdf", bbox_inches='tight')
fig.clf()
#plt.show()
'''


##################################################################################
# Primary bias: Dilation

fmD_CMB = baseMap.forecastMultBiasLensedForegroundsDilation(cmb.funlensedTT, cmb.ftotalTT, cmb.funlensedTT, lMin=lMin, lMax=lMaxT, test=False)
fmD_CIB = baseMap.forecastMultBiasLensedForegroundsDilation(cmb.funlensedTT, cmb.ftotalTT, cmb.fCIB, lMin=lMin, lMax=lMaxT, test=False)
fmD_kSZLate = baseMap.forecastMultBiasLensedForegroundsDilation(cmb.funlensedTT, cmb.ftotalTT, cmb.fkSZLate, lMin=lMin, lMax=lMaxT, test=False)
fmD_kSZReio = baseMap.forecastMultBiasLensedForegroundsDilation(cmb.funlensedTT, cmb.ftotalTT, cmb.fkSZReio, lMin=lMin, lMax=lMaxT, test=False)
fmD_tSZ= baseMap.forecastMultBiasLensedForegroundsDilation(cmb.funlensedTT, cmb.ftotalTT, cmb.ftSZ, lMin=lMin, lMax=lMaxT, test=False)
fmD_radioPS= baseMap.forecastMultBiasLensedForegroundsDilation(cmb.funlensedTT, cmb.ftotalTT, cmb.fradioPoisson, lMin=lMin, lMax=lMaxT, test=False)
#
fmD_tSZxCIB = baseMap.forecastMultBiasLensedForegroundsDilation(cmb.funlensedTT, cmb.ftotalTT, cmb.ftSZ_CIB, lMin=lMin, lMax=lMaxT, test=False)


# Evaluate FFT calculation for plotting
L = np.genfromtxt("./input/Lc.txt")
mDCMB = fmD_CMB(L)
mDCIB = fmD_CIB(L)
mDkSZLate = fmD_kSZLate(L)
mDkSZReio = fmD_kSZReio(L)
mDtSZ = fmD_tSZ(L)
mDradioPS = fmD_radioPS(L)
#
mDtSZxCIB = fmD_tSZxCIB(L)

# Evaluate analytical calculation for comparison
mDCMBAna = cmbLensRec.fResponseLensedFg['dTTcmb'](L)
mDCIBAna = cmbLensRec.fResponseLensedFg['dTTcib'](L)
mDkSZLateAna = 0.45 * cmbLensRec.fResponseLensedFg['dTTksz'](L)
mDkSZReioAna = 0.55 * cmbLensRec.fResponseLensedFg['dTTksz'](L)
mDtSZAna = cmbLensRec.fResponseLensedFg['dTTtsz'](L)
mDradioPSAna = cmbLensRec.fResponseLensedFg['dTTradiops'](L)
#
mDtSZxCIBAna = cmbLensRec.fResponseLensedFg['dTTtszxcib'](L)


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
#ax.loglog(L, mDCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mDCIB, c=cCib, label=r'CIB')
ax.loglog(L, -mDCIB, c=cCib, ls='--')
ax.loglog(L, mDkSZLate, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mDkSZLate, c=cKszLate, ls='--')
ax.loglog(L, mDkSZReio, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mDkSZReio, c=cKszReio, ls='--')
ax.loglog(L, mDtSZ, c=cTsz, label=r'tSZ')
ax.loglog(L, -mDtSZ, c=cTsz, ls='--')
ax.loglog(L, mDradioPS, c=cRadiops, label=r'radioPS')
ax.loglog(L, -mDradioPS, c=cRadiops, ls='--')
#
ax.loglog(L, mDtSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mDtSZxCIB, c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Lensing response $\mathcal{R}_L^f$')
ax.set_title(r'Magnification')
#
fig.savefig(pathFig+"d_lensing_responses.pdf", bbox_inches='tight')
fig.clf()
#plt.show()




fig=plt.figure(1)
ax=fig.add_subplot(111)
#
ax.loglog(L, mDCMB, c=cCmb, label=r'CMB FFT')
ax.loglog(L, mDCMBAna, c=cCmb, lw=1., label=r'CMB Ana')
#
ax.loglog(L, mDCIB, c=cCib, label=r'CIB FFT')
ax.loglog(L, -mDCIB, c=cCib, ls='--')
ax.loglog(L, mDCIBAna, c=cCib, lw=1., label=r'CIB Ana')
ax.loglog(L, -mDCIBAna, c=cCib, lw=1., ls='--')
#
ax.loglog(L, mDkSZLate, c=cKszLate, label=r'kSZ late FFT')
ax.loglog(L, -mDkSZLate, c=cKszLate, ls='--')
ax.loglog(L, mDkSZLateAna, c=cKszLate, lw=1., label=r'kSZ late Ana')
ax.loglog(L, -mDkSZLateAna, c=cKszLate, lw=1., ls='--')
#
ax.loglog(L, mDkSZReio, c=cKszReio, label=r'kSZ reio FFT')
ax.loglog(L, -mDkSZReio, c=cKszReio, ls='--')
ax.loglog(L, mDkSZReioAna, c=cKszReio, lw=1., label=r'kSZ reio Ana')
ax.loglog(L, -mDkSZReioAna, c=cKszReio, lw=1., ls='--')
#
ax.loglog(L, mDtSZ, c=cTsz, label=r'tSZ FFT')
ax.loglog(L, -mDtSZ, c=cTsz, ls='--')
ax.loglog(L, mDtSZAna, c=cTsz, lw=1., label=r'tSZ Ana')
ax.loglog(L, -mDtSZAna, c=cTsz, lw=1., ls='--')
#
ax.loglog(L, mDradioPS, c=cRadiops, label=r'radioPS FFT')
ax.loglog(L, -mDradioPS, c=cRadiops, ls='--')
ax.loglog(L, mDradioPSAna, c=cRadiops, lw=1., label=r'radioPS Ana')
ax.loglog(L, -mDradioPSAna, c=cRadiops, lw=1., ls='--')
#
ax.loglog(L, mDtSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB FFT')
ax.loglog(L, -mDtSZxCIB, c=cTszxCib, ls='--')
ax.loglog(L, mDtSZxCIBAna, c=cTszxCib, lw=1., label=r'tSZ$\times$CIB Ana')
ax.loglog(L, -mDtSZxCIBAna, c=cTszxCib, lw=1., ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Dilation lensing response $\mathcal{R}_L^f$')
ax.set_title(r'QE')
#
fig.savefig(pathFig+"d_lensing_responses_compare_ana.pdf", bbox_inches='tight')
fig.clf()
#plt.show()



fig=plt.figure(1)
ax=fig.add_subplot(111)
#
ax.fill_between(L, np.ones_like(L)/snrD, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax.loglog(L, mDCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mDCIB * 2. * rCib, c=cCib, label=r'CIB')
ax.loglog(L, -mDCIB * 2. * rCib, c=cCib, ls='--')
ax.loglog(L, mDkSZReio * 2. * rKszReio, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mDkSZReio * 2. * rKszReio, c=cKszReio, ls='--')
ax.loglog(L, mDkSZLate * 2. * rKszLate, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mDkSZLate * 2. * rKszLate, c=cKszLate, ls='--')
ax.loglog(L, mDtSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax.loglog(L, -mDtSZ * 2. * rTsz, c=cTsz, ls='--')
ax.loglog(L, mDradioPS * 2. * rRadiops, c=cRadiops, label=r'Radio PS')
ax.loglog(L, -mDradioPS * 2. * rRadiops, c=cRadiops, ls='--')
#
ax.loglog(L, mDtSZxCIB * 2. * (rCib + rTsz), c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mDtSZxCIB * 2. * (rCib + rTsz), c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Primary relative bias on $C_L^{\kappa_\text{CMB}}$')
ax.set_title(r'Magnification')
#
fig.savefig(pathFig+"d_primary_bias.pdf", bbox_inches='tight')
fig.clf()
#plt.show()


fig=plt.figure(3)
ax=fig.add_subplot(111)
#
ax.fill_between(L, np.ones_like(L)/snrCrossD, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax.loglog(L, mDCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mDCIB * 2. * rCibCross, c=cCib, label=r'CIB')
ax.loglog(L, -mDCIB * 2. * rCibCross, c=cCib, ls='--')
ax.loglog(L, mDkSZReio * 2. * rKszReioCross, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mDkSZReio * 2. * rKszReioCross, c=cKszReio, ls='--')
ax.loglog(L, mDkSZLate * 2. * rKszLateCross, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mDkSZLate * 2. * rKszLateCross, c=cKszLate, ls='--')
ax.loglog(L, mDtSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax.loglog(L, -mDtSZ * 2. * rTsz, c=cTsz, ls='--')
ax.loglog(L, mDradioPS * 2. * rRadiopsCross, c=cRadiops, label=r'Radio PS')
ax.loglog(L, -mDradioPS * 2. * rRadiopsCross, c=cRadiops, ls='--')
#
ax.loglog(L, mDtSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mDtSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Relative bias on $C_L^{g \kappa_\text{CMB}}$')
ax.set_title(r'Magnification')
#
fig.savefig(pathFig+"d_bias_cross.pdf", bbox_inches='tight')
fig.clf()
#plt.show()


##################################################################################
# secondary bias: dilation

'''
# read secondary biases
data = np.genfromtxt(pathOut+"d_sec_cib_lmaxt3500.txt")
Lsec = data[:,0]
secDCIB = data[:,1]
sSecDCIB = data[:,2]
fsecDCIB = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"d_sec_kszreio_lmaxt3500.txt")
#secDkSZReio = data[:,1]
#sSecDkSZReio = data[:,2]
#fsecDkSZReio = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"d_sec_kszlate_lmaxt3500.txt")
#secDkSZLate = data[:,1]
#sSecDkSZLate = data[:,2]
#fsecDkSZLate = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"d_sec_tsz_lmaxt3500.txt")
#sectSSZ = data[:,1]
#sSecDtSZ = data[:,2]
#fsecDtSZ = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"d_sec_radiops_lmaxt3500.txt")
#secDradioPS = data[:,1]
#sSecDradioPS = data[:,2]
#fsecDradioPS = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#
data = np.genfromtxt(pathOut+"d_sec_tszxcib_lmaxt3500.txt")
secDtSZxCIB = data[:,1]
sSecDtSZxCIB = data[:,2]
fsecDtSZxCIB = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)


fig=plt.figure(4)
ax=fig.add_subplot(111)
#
ax.fill_between(Lsec, np.ones_like(Lsec)/snrQ, edgecolor='', facecolor='gray', alpha=0.3)
#
ax.errorbar(Lsec, secDCIB, sSecDCIB, c=cCib, label=r'CIB')
ax.errorbar(Lsec, -secDCIB, -sSecDCIB, c=cCib, ls='--')
#ax.errorbar(Lsec, secDkSZReio, sSecDkSZReio, c=cKszReio, label=r'kSZ reio')
#ax.errorbar(Lsec, -secDkSZReio, -sSecDkSZReio, c=cKszReio, ls='--')
#ax.errorbar(Lsec, secDkSZLate, sSecDkSZLate, c=cKszLate, label=r'kSZ late')
#ax.errorbar(Lsec, -secDkSZLate, -sSecDkSZLate, c=cKszLate, ls='--')
#ax.errorbar(Lsec, secDtSZ, sSecDtSZ, c=cTsz, label=r'tSZ')
#ax.errorbar(Lsec, -secDtSZ, -sSecDtSZ, c=cTsz, ls='--')
#ax.errorbar(Lsec, secDradioPS, sSecDradioPS, c=cRadiops, label=r'Radio PS')
#ax.errorbar(Lsec, -secDradioPS, -sSecDradioPS, c=cRadiops, ls='--')
#
ax.errorbar(Lsec, secDtSZxCIB, sSecDtSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB')
ax.errorbar(Lsec, -secDtSZxCIB, -sSecDtSZxCIB, c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Secondary relative bias on $C_L^{\kappa_\text{CMB}}$')
ax.set_title('dilation')
#
#fig.savefig(pathFig+"d_secondary_bias.pdf", bbox_inches='tight')
#fig.clf()
plt.show()


#fig=plt.figure(5)
#ax=fig.add_subplot(111)
##
#ax.fill_between(L, np.ones_like(L)/snrQ, edgecolor='', facecolor='gray', alpha=0.3)
##
#ax.loglog(L, (fsecDCIB(L) + mDCIB*2.*rCib), c=cCib, label=r'CIB')
#ax.loglog(L, -(fsecDCIB(L) + mDCIB*2.*rCib), c=cCib, ls='--')
#ax.loglog(L, (fsecDkSZReio(L) + mDkSZReio*2.*rKszReio), c=cKszReio, label=r'kSZ reio')
#ax.loglog(L, -(fsecDkSZReio(L) + mDkSZReio*2.*rKszReio), c=cKszReio, ls='--')
#ax.loglog(L, (fsecDkSZLate(L) + mDkSZLate*2.*rKszLate), c=cKszLate, label=r'kSZ late')
#ax.loglog(L, -(fsecDkSZLate(L) + mDkSZLate*2.*rKszLate), c=cKszLate, ls='--')
#ax.loglog(L, (fsecDtSZ(L) + mDtSZ*2.*rTsz), c=cTsz, label=r'tSZ')
#ax.loglog(L, -(fsecDtSZ(L) + mDtSZ*2.*rTsz), c=cTsz, ls='--')
#ax.loglog(L, (fsecDradioPS(L) + mDradioPS*2.*rRadiops), c=cRadiops, label=r'Radio PS')
#ax.loglog(L, -(fsecDradioPS(L) + mDradioPS*2.*rRadiops), c=cRadiops, ls='--')
##
#ax.loglog(L, (fsecDtSZxCIB(L) + mDtSZxCIB*2.*(rCib+rTsz)), c=cTszxCib, label=r'tSZ$\times$CIB')
#ax.loglog(L, -(fsecDtSZxCIB(L) + mDtSZxCIB*2.*(rCib+rTsz)), c=cTszxCib, ls='--')
##
#ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
#ax.set_xlim((30., 6.5e3))
#ax.set_ylim((1.e-4, 1.))
#ax.set_xlabel(r'$L$')
#ax.set_ylabel(r'Prim.+Sec. relative bias on $C_L^{\kappa_\text{CMB}}$')
#ax.set_title(r'Shear')
##
#fig.savefig(pathFig+"d_primsec_bias.pdf", bbox_inches='tight')
##fig.clf()
#plt.show()
'''


##################################################################################
# Primary bias: Shear

fmS_CMB = baseMap.forecastMultBiasLensedForegroundsShear(cmb.funlensedTT, cmb.ftotalTT, cmb.funlensedTT, lMin=lMin, lMax=lMaxT, test=False)
fmS_CIB = baseMap.forecastMultBiasLensedForegroundsShear(cmb.funlensedTT, cmb.ftotalTT, cmb.fCIB, lMin=lMin, lMax=lMaxT, test=False)
fmS_kSZLate = baseMap.forecastMultBiasLensedForegroundsShear(cmb.funlensedTT, cmb.ftotalTT, cmb.fkSZLate, lMin=lMin, lMax=lMaxT, test=False)
fmS_kSZReio = baseMap.forecastMultBiasLensedForegroundsShear(cmb.funlensedTT, cmb.ftotalTT, cmb.fkSZReio, lMin=lMin, lMax=lMaxT, test=False)
fmS_tSZ= baseMap.forecastMultBiasLensedForegroundsShear(cmb.funlensedTT, cmb.ftotalTT, cmb.ftSZ, lMin=lMin, lMax=lMaxT, test=False)
fmS_radioPS= baseMap.forecastMultBiasLensedForegroundsShear(cmb.funlensedTT, cmb.ftotalTT, cmb.fradioPoisson, lMin=lMin, lMax=lMaxT, test=False)
#
fmS_tSZxCIB = baseMap.forecastMultBiasLensedForegroundsShear(cmb.funlensedTT, cmb.ftotalTT, cmb.ftSZ_CIB, lMin=lMin, lMax=lMaxT, test=False)

# Evaluate FFT claculation for plotting
L = np.genfromtxt("./input/Lc.txt")
mSCMB = fmS_CMB(L)
mSCIB = fmS_CIB(L)
mSkSZLate = fmS_kSZLate(L)
mSkSZReio = fmS_kSZReio(L)
mStSZ = fmS_tSZ(L)
mSradioPS = fmS_radioPS(L)
#
mStSZxCIB = fmS_tSZxCIB(L)


# Evaluate analytical calculation for comparison
mSCMBAna = cmbLensRec.fResponseLensedFg['sTTcmb'](L)
mSCIBAna = cmbLensRec.fResponseLensedFg['sTTcib'](L)
mSkSZLateAna = 0.45 * cmbLensRec.fResponseLensedFg['sTTksz'](L)
mSkSZReioAna = 0.55 * cmbLensRec.fResponseLensedFg['sTTksz'](L)
mStSZAna = cmbLensRec.fResponseLensedFg['sTTtsz'](L)
mSradioPSAna = cmbLensRec.fResponseLensedFg['sTTradiops'](L)
#
mStSZxCIBAna = cmbLensRec.fResponseLensedFg['sTTtszxcib'](L)



fig=plt.figure(0)
ax=fig.add_subplot(111)
#
#ax.loglog(L, mSCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mSCIB, c=cCib, label=r'CIB')
ax.loglog(L, -mSCIB, c=cCib, ls='--')
ax.loglog(L, mSkSZLate, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mSkSZLate, c=cKszLate, ls='--')
ax.loglog(L, mSkSZReio, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mSkSZReio, c=cKszReio, ls='--')
ax.loglog(L, mStSZ, c=cTsz, label=r'tSZ')
ax.loglog(L, -mStSZ, c=cTsz, ls='--')
ax.loglog(L, mSradioPS, c=cRadiops, label=r'radioPS')
ax.loglog(L, -mSradioPS, c=cRadiops, ls='--')
#
ax.loglog(L, mStSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mStSZxCIB, c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Lensing response $\mathcal{R}_L^f$')
ax.set_title(r'Shear')
#
fig.savefig(pathFig+"s_lensing_responses.pdf", bbox_inches='tight')
fig.clf()
#plt.show()



fig=plt.figure(1)
ax=fig.add_subplot(111)
#
ax.loglog(L, mSCMB, c=cCmb, label=r'CMB FFT')
ax.loglog(L, mSCMBAna, c=cCmb, lw=1., label=r'CMB Ana')
#
ax.loglog(L, mSCIB, c=cCib, label=r'CIB FFT')
ax.loglog(L, -mSCIB, c=cCib, ls='--')
ax.loglog(L, mSCIBAna, c=cCib, lw=1., label=r'CIB Ana')
ax.loglog(L, -mSCIBAna, c=cCib, lw=1., ls='--')
#
ax.loglog(L, mSkSZLate, c=cKszLate, label=r'kSZ late FFT')
ax.loglog(L, -mSkSZLate, c=cKszLate, ls='--')
ax.loglog(L, mSkSZLateAna, c=cKszLate, lw=1., label=r'kSZ late Ana')
ax.loglog(L, -mSkSZLateAna, c=cKszLate, lw=1., ls='--')
#
ax.loglog(L, mSkSZReio, c=cKszReio, label=r'kSZ reio FFT')
ax.loglog(L, -mSkSZReio, c=cKszReio, ls='--')
ax.loglog(L, mSkSZReioAna, c=cKszReio, lw=1., label=r'kSZ reio Ana')
ax.loglog(L, -mSkSZReioAna, c=cKszReio, lw=1., ls='--')
#
ax.loglog(L, mStSZ, c=cTsz, label=r'tSZ FFT')
ax.loglog(L, -mStSZ, c=cTsz, ls='--')
ax.loglog(L, mStSZAna, c=cTsz, lw=1., label=r'tSZ Ana')
ax.loglog(L, -mStSZAna, c=cTsz, lw=1., ls='--')
#
ax.loglog(L, mSradioPS, c=cRadiops, label=r'radioPS FFT')
ax.loglog(L, -mSradioPS, c=cRadiops, ls='--')
ax.loglog(L, mSradioPSAna, c=cRadiops, lw=1., label=r'radioPS Ana')
ax.loglog(L, -mSradioPSAna, c=cRadiops, lw=1., ls='--')
#
ax.loglog(L, mStSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB FFT')
ax.loglog(L, -mStSZxCIB, c=cTszxCib, ls='--')
ax.loglog(L, mStSZxCIBAna, c=cTszxCib, lw=1., label=r'tSZ$\times$CIB Ana')
ax.loglog(L, -mStSZxCIBAna, c=cTszxCib, lw=1., ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Shear lensing response $\mathcal{R}_L^f$')
ax.set_title(r'QE')
#
fig.savefig(pathFig+"s_lensing_responses_compare_ana.pdf", bbox_inches='tight')
fig.clf()
#plt.show()





fig=plt.figure(1)
ax=fig.add_subplot(111)
#
ax.fill_between(L, np.ones_like(L)/snrS, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax.loglog(L, mSCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mSCIB * 2. * rCib, c=cCib, label=r'CIB')
ax.loglog(L, -mSCIB * 2. * rCib, c=cCib, ls='--')
ax.loglog(L, mSkSZReio * 2. * rKszReio, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mSkSZReio * 2. * rKszReio, c=cKszReio, ls='--')
ax.loglog(L, mSkSZLate * 2. * rKszLate, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mSkSZLate * 2. * rKszLate, c=cKszLate, ls='--')
ax.loglog(L, mStSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax.loglog(L, -mStSZ * 2. * rTsz, c=cTsz, ls='--')
ax.loglog(L, mSradioPS * 2. * rRadiops, c=cRadiops, label=r'Radio PS')
ax.loglog(L, -mSradioPS * 2. * rRadiops, c=cRadiops, ls='--')
#
ax.loglog(L, mStSZxCIB * 2. * (rCib + rTsz), c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mStSZxCIB * 2. * (rCib + rTsz), c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Primary relative bias on $C_L^{\kappa_\text{CMB}}$')
ax.set_title(r'Shear')
#
fig.savefig(pathFig+"s_primary_bias.pdf", bbox_inches='tight')
fig.clf()
#plt.show()


fig=plt.figure(3)
ax=fig.add_subplot(111)
#
ax.fill_between(L, np.ones_like(L)/snrCrossS, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax.loglog(L, mSCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mSCIB * 2. * rCibCross, c=cCib, label=r'CIB')
ax.loglog(L, -mSCIB * 2. * rCibCross, c=cCib, ls='--')
ax.loglog(L, mSkSZReio * 2. * rKszReioCross, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mSkSZReio * 2. * rKszReioCross, c=cKszReio, ls='--')
ax.loglog(L, mSkSZLate * 2. * rKszLateCross, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mSkSZLate * 2. * rKszLateCross, c=cKszLate, ls='--')
ax.loglog(L, mStSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax.loglog(L, -mStSZ * 2. * rTsz, c=cTsz, ls='--')
ax.loglog(L, mSradioPS * 2. * rRadiopsCross, c=cRadiops, label=r'Radio PS')
ax.loglog(L, -mSradioPS * 2. * rRadiopsCross, c=cRadiops, ls='--')
#
ax.loglog(L, mStSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mStSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Relative bias on $C_L^{g \kappa_\text{CMB}}$')
ax.set_title(r'Shear')
#
fig.savefig(pathFig+"s_bias_cross.pdf", bbox_inches='tight')
fig.clf()
#plt.show()



##################################################################################
# secondary bias: shear

'''
# read secondary biases
data = np.genfromtxt(pathOut+"s_sec_cib_lmaxt3500.txt")
Lsec = data[:,0]
secSCIB = data[:,1]
sSecSCIB = data[:,2]
fsecSCIB = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"s_sec_kszreio_lmaxt3500.txt")
#secSkSZReio = data[:,1]
#sSecSkSZReio = data[:,2]
#fsecSkSZReio = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"s_sec_kszlate_lmaxt3500.txt")
#secSkSZLate = data[:,1]
#sSecSkSZLate = data[:,2]
#fsecSkSZLate = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"s_sec_tsz_lmaxt3500.txt")
#sectSSZ = data[:,1]
#sSecStSZ = data[:,2]
#fsecStSZ = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#data = np.genfromtxt(pathOut+"s_sec_radiops_lmaxt3500.txt")
#secSradioPS = data[:,1]
#sSecSradioPS = data[:,2]
#fsecSradioPS = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
#
data = np.genfromtxt(pathOut+"s_sec_tszxcib_lmaxt3500.txt")
secStSZxCIB = data[:,1]
sSecStSZxCIB = data[:,2]
fsecStSZxCIB = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)


fig=plt.figure(4)
ax=fig.add_subplot(111)
#
ax.fill_between(Lsec, np.ones_like(Lsec)/snrQ, edgecolor='', facecolor='gray', alpha=0.3)
#
ax.errorbar(Lsec, secSCIB, sSecSCIB, c=cCib, label=r'CIB')
ax.errorbar(Lsec, -secSCIB, -sSecSCIB, c=cCib, ls='--')
#ax.errorbar(Lsec, secSkSZReio, sSecSkSZReio, c=cKszReio, label=r'kSZ reio')
#ax.errorbar(Lsec, -secSkSZReio, -sSecSkSZReio, c=cKszReio, ls='--')
#ax.errorbar(Lsec, secSkSZLate, sSecSkSZLate, c=cKszLate, label=r'kSZ late')
#ax.errorbar(Lsec, -secSkSZLate, -sSecSkSZLate, c=cKszLate, ls='--')
#ax.errorbar(Lsec, secStSZ, sSecStSZ, c=cTsz, label=r'tSZ')
#ax.errorbar(Lsec, -secStSZ, -sSecStSZ, c=cTsz, ls='--')
#ax.errorbar(Lsec, secSradioPS, sSecSradioPS, c=cRadiops, label=r'Radio PS')
#ax.errorbar(Lsec, -secSradioPS, -sSecSradioPS, c=cRadiops, ls='--')
#
ax.errorbar(Lsec, secStSZxCIB, sSecStSZxCIB, c=cTszxCib, label=r'tSZ$\times$CIB')
ax.errorbar(Lsec, -secStSZxCIB, -sSecStSZxCIB, c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'Secondary relative bias on $C_L^{\kappa_\text{CMB}}$')
ax.set_title('shear')
#
#fig.savefig(pathFig+"s_secondary_bias.pdf", bbox_inches='tight')
#fig.clf()
plt.show()


#fig=plt.figure(5)
#ax=fig.add_subplot(111)
##
#ax.fill_between(L, np.ones_like(L)/snrQ, edgecolor='', facecolor='gray', alpha=0.3)
##
#ax.loglog(L, (fsecSCIB(L) + mSCIB*2.*rCib), c=cCib, label=r'CIB')
#ax.loglog(L, -(fsecSCIB(L) + mSCIB*2.*rCib), c=cCib, ls='--')
#ax.loglog(L, (fsecSkSZReio(L) + mSkSZReio*2.*rKszReio), c=cKszReio, label=r'kSZ reio')
#ax.loglog(L, -(fsecSkSZReio(L) + mSkSZReio*2.*rKszReio), c=cKszReio, ls='--')
#ax.loglog(L, (fsecSkSZLate(L) + mSkSZLate*2.*rKszLate), c=cKszLate, label=r'kSZ late')
#ax.loglog(L, -(fsecSkSZLate(L) + mSkSZLate*2.*rKszLate), c=cKszLate, ls='--')
#ax.loglog(L, (fsecStSZ(L) + mStSZ*2.*rTsz), c=cTsz, label=r'tSZ')
#ax.loglog(L, -(fsecStSZ(L) + mStSZ*2.*rTsz), c=cTsz, ls='--')
#ax.loglog(L, (fsecSradioPS(L) + mSradioPS*2.*rRadiops), c=cRadiops, label=r'Radio PS')
#ax.loglog(L, -(fsecSradioPS(L) + mSradioPS*2.*rRadiops), c=cRadiops, ls='--')
##
#ax.loglog(L, (fsecStSZxCIB(L) + mStSZxCIB*2.*(rCib+rTsz)), c=cTszxCib, label=r'tSZ$\times$CIB')
#ax.loglog(L, -(fsecStSZxCIB(L) + mStSZxCIB*2.*(rCib+rTsz)), c=cTszxCib, ls='--')
##
#ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
#ax.set_xlim((30., 6.5e3))
#ax.set_ylim((1.e-4, 1.))
#ax.set_xlabel(r'$L$')
#ax.set_ylabel(r'Prim.+Sec. relative bias on $C_L^{\kappa_\text{CMB}}$')
#ax.set_title(r'Shear')
##
#fig.savefig(pathFig+"s_primsec_bias.pdf", bbox_inches='tight')
##fig.clf()
#plt.show()
'''


##################################################################################
# bias in cross: joint plot for Q, S, D


fig=plt.figure(1)
fig.set_size_inches(6,13.5)

ax=fig.add_subplot(3,1,1)
#
ax.fill_between(L, np.ones_like(L)/snrCrossQ, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax.loglog(L, mCMB, c=cCmb, label=r'CMB (test)')
ax.loglog(L, mCIB * 2. * rCibCross, c=cCib, label=r'CIB')
ax.loglog(L, -mCIB * 2. * rCibCross, c=cCib, ls='--')
ax.loglog(L, mkSZReio * 2. * rKszReioCross, c=cKszReio, label=r'kSZ reio')
ax.loglog(L, -mkSZReio * 2. * rKszReioCross, c=cKszReio, ls='--')
ax.loglog(L, mkSZLate * 2. * rKszLateCross, c=cKszLate, label=r'kSZ late')
ax.loglog(L, -mkSZLate * 2. * rKszLateCross, c=cKszLate, ls='--')
ax.loglog(L, mtSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax.loglog(L, -mtSZ * 2. * rTsz, c=cTsz, ls='--')
ax.loglog(L, mradioPS * 2. * rRadiopsCross, c=cRadiops, label=r'Radio PS')
ax.loglog(L, -mradioPS * 2. * rRadiopsCross, c=cRadiops, ls='--')
#
ax.loglog(L, mtSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, label=r'tSZ$\times$CIB')
ax.loglog(L, -mtSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, ls='--')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlim((30., 6.5e3))
ax.set_ylim((1.e-4, 1.))
#ax.set_xlabel(r'$L$')
#ax.set_ylabel(r'Relative bias on $C_L^{g \kappa_\text{CMB}}$')
ax.set_title(r'QE')


ax2=fig.add_subplot(3,1,2)
#
ax2.fill_between(L, np.ones_like(L)/snrCrossS, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax2.loglog(L, mSCMB, c=cCmb, label=r'CMB (test)')
ax2.loglog(L, mSCIB * 2. * rCibCross, c=cCib, label=r'CIB')
ax2.loglog(L, -mSCIB * 2. * rCibCross, c=cCib, ls='--')
ax2.loglog(L, mSkSZReio * 2. * rKszReioCross, c=cKszReio, label=r'kSZ reio')
ax2.loglog(L, -mSkSZReio * 2. * rKszReioCross, c=cKszReio, ls='--')
ax2.loglog(L, mSkSZLate * 2. * rKszLateCross, c=cKszLate, label=r'kSZ late')
ax2.loglog(L, -mSkSZLate * 2. * rKszLateCross, c=cKszLate, ls='--')
ax2.loglog(L, mStSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax2.loglog(L, -mStSZ * 2. * rTsz, c=cTsz, ls='--')
ax2.loglog(L, mSradioPS * 2. * rRadiopsCross, c=cRadiops, label=r'Radio PS')
ax2.loglog(L, -mSradioPS * 2. * rRadiopsCross, c=cRadiops, ls='--')
#
ax2.loglog(L, mStSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, label=r'tSZ$\times$CIB')
ax2.loglog(L, -mStSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, ls='--')
#
#ax2.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax2.set_xlim((30., 6.5e3))
ax2.set_ylim((1.e-4, 1.))
#ax2.set_xlabel(r'$L$')
ax2.set_ylabel(r'Relative bias on $C_L^{g \kappa_\text{CMB}}$')
ax2.set_title(r'Shear')


ax3=fig.add_subplot(3,1,3)
#
ax3.fill_between(L, np.ones_like(L)/snrCrossD, edgecolor='', facecolor='gray', alpha=0.3)
#
#ax3.loglog(L, mDCMB, c=cCmb, label=r'CMB (test)')
ax3.loglog(L, mDCIB * 2. * rCibCross, c=cCib, label=r'CIB')
ax3.loglog(L, -mDCIB * 2. * rCibCross, c=cCib, ls='--')
ax3.loglog(L, mDkSZReio * 2. * rKszReioCross, c=cKszReio, label=r'kSZ reio')
ax3.loglog(L, -mDkSZReio * 2. * rKszReioCross, c=cKszReio, ls='--')
ax3.loglog(L, mDkSZLate * 2. * rKszLateCross, c=cKszLate, label=r'kSZ late')
ax3.loglog(L, -mDkSZLate * 2. * rKszLateCross, c=cKszLate, ls='--')
ax3.loglog(L, mDtSZ * 2. * rTsz, c=cTsz, label=r'tSZ')
ax3.loglog(L, -mDtSZ * 2. * rTsz, c=cTsz, ls='--')
ax3.loglog(L, mDradioPS * 2. * rRadiopsCross, c=cRadiops, label=r'Radio PS')
ax3.loglog(L, -mDradioPS * 2. * rRadiopsCross, c=cRadiops, ls='--')
#
ax3.loglog(L, mDtSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, label=r'tSZ$\times$CIB')
ax3.loglog(L, -mDtSZxCIB * 2. * (rCibCross + rTszCross), c=cTszxCib, ls='--')
#
#ax3.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax3.set_xlim((30., 6.5e3))
ax3.set_ylim((1.e-4, 1.))
ax3.set_xlabel(r'$L$')
#ax3.set_ylabel(r'Relative bias on $C_L^{g \kappa_\text{CMB}}$')
ax3.set_title(r'Magnification')

plt.tight_layout()
plt.subplots_adjust(wspace=.35, hspace=.35)
#
fig.savefig(pathFig+"qsd_bias_cross.pdf", bbox_inches='tight')
fig.clf()
#plt.show()

