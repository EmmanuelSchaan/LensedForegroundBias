from headers import *
###############################################################################

class CMBLensRec(object):
   """CMB lensing estimators
   Computes:
   - normalizations for Q_XY, shear and dilation, multipole estimators
   - lensing noise power spectra and covariances for the various estimators
   - bias due to lensed foregrounds for the various TT estimators
   (- bias due to foreground trispectrum for Q_TT)
   """

   def __init__(self, CMB, nProc=1):
      self.CMB = CMB
      self.nProc = nProc

      # bounds for ell integrals
      self.lMin = self.CMB.lMin
      self.lMax = max(self.CMB.lMaxT, self.CMB.lMaxP)
      
      # values of ell to compute the reconstruction noise
#      self.L = np.genfromtxt("./input/Lc.txt") # center of the bins for l
      self.L = np.logspace(np.log10(1.), np.log10(2.*self.lMax+1.), 201, 10.)
      self.Nl = len(self.L)
      
      # output file path
      self.directory = "./output/cmblensrec/"+str(self.CMB)
      # create folder if needed
      if not os.path.exists(self.directory):
         os.makedirs(self.directory)

      # Normalizations and noise power spectra
      self.pathNoiseQ = self.directory+"/noise_q_XY.txt"
      self.pathNormSD = self.directory+"/norm_sd_TT.txt"
      self.pathNoiseQSD = self.directory+"/noise_qsd_TT.txt"
      
      # Lensed foreground bias
      self.pathResponseLensed = self.directory+"/lensed_foreground_response_qsd.txt"



   ###############################################################################
   # Geometry
   # L = l1 + l2
   # phi1 = (L, l1)
   # phi2 = (L, l2)
   # phi21 = (l1, l2) = phi2 - phi1


   def l2(self, L, l1, phi1):
      """Returns the modulus of l2=L-l1 given L, l1 and phi1 = (L, l1)
      """
      return np.sqrt(L**2 + l1**2 - 2.*L*l1*np.cos(phi1))

   def phi21(self, L, l1, phi1):
      """Returns phi21 = (l1, l2) given L, l1 and phi1 = (L, l1)
      """
      x = L*np.cos(-phi1) - l1 # l2*cos(phi21) + l1 = L*cos(-phi1)
      y = L*np.sin(-phi1)  # l2*sin(phi21) = L*np.sin(-phi1)
      result = np.arctan2(y,x)   # = 2.*np.arctan(y/(x+sqrt(x**2+y**2)))
      return result

   def phi2(self, L, l1, phi1):
      """Returns phi2 = (L, l2) given L, l1 and phi1 = (L, l1)
      """
      return self.phi21(L, l1, phi1) + phi1


   def l1l2Max(self, XY):
      """Returns l1Max = l2Max, for the estimator Q_XY.
      For TE and TB, I discard the range between lMaxT and lMaxP (conservative / suboptimal).
      """
      if XY=='TT' or XY=='sTT' or XY=='dTT':
         return self.CMB.lMaxT
      elif XY=='TE' or XY=='TB':
         return min(self.CMB.lMaxT, self.CMB.lMaxP)
      elif XY=='EE' or XY=='EB' or XY=='BB':
         return self.CMB.lMaxP



#
#   # theta_min for the l1 integral
#   # so that l2 > lMin
#   def thetaMin(self, L, lnl1):
#      l1 = np.log(lnl1)
#      if (abs(L-l1)<self.lMin):
#         theta_min = np.arccos((L**2+l1**2-self.lMin**2) / (2.*L*l1))
#      else:
#         theta_min = 0.
#      return theta_min
#
#   # theta_max for the l1 integral
#   # so that l2 < lMax
#   def thetaMax(self, L, lnl1):
#      l1 = np.log(lnl1)
#      if (l1>self.lMax-L):
#         theta_max = np.arccos((L**2+l1**2-self.lMax**2) / (2.*L*l1))
#      else:
#         theta_max = np.pi
#      return theta_max

   
   ###############################################################################
   # Lensing response functions

   def f_XY(self, L, l1, phi1, XY, fCfg=None):
      """Lensing response functions, such that
      < X_l1 Y_{L-l1} > = f_XY(L, l1, phi1) * kappa
      phi1 = (L, l1)
      """
      # geometry
      l2 = self.l2(L, l1, phi1)
      phi21 = self.phi21(L, l1, phi1)
      phi2 = self.phi2(L, l1, phi1)
  
      # responses from Hu Okamoto 02
      if XY=='TT' or XY=='sTT' or XY=='dTT':
         # response of a foreground quadratic pair to lensing,
         # useful for lensed foreground bias
         if fCfg is not None:
            result = fCfg(l1) * L*l1*np.cos(phi1)
            result += fCfg(l2) * L*l2*np.cos(phi2)
         # standard lensing response
         else:
            result = self.CMB.funlensedTT(l1) * L*l1*np.cos(phi1)
            result += self.CMB.funlensedTT(l2) * L*l2*np.cos(phi2)
      elif XY=='TE':
         # typo in Hu & Okamoto 2002: cos(2phi) and not cos(phi)!
         result = self.CMB.funlensedTE(l1) * L*l1*np.cos(phi1) * np.cos(2.*phi21)
         result += self.CMB.funlensedTE(l2) * L*l2*np.cos(phi2)
      elif XY=='TB':
         result = self.CMB.funlensedTE(l1) * L*l1*np.cos(phi1) * np.sin(2.*phi21)
      elif XY=='EE':
         result = self.CMB.funlensedEE(l1) * L*l1*np.cos(phi1)
         result += self.CMB.funlensedEE(l2) * L*l2*np.cos(phi2)
         result *= np.cos(2.*phi21)
      elif XY=='EB':
         result = self.CMB.funlensedEE(l1) * L*l1*np.cos(phi1)
         result -= self.CMB.funlensedBB(l2) * L*l2*np.cos(phi2)
         result *= np.sin(2.*phi21)
      elif XY=='BB':
         result = self.CMB.funlensedBB(l1) * L*l1*np.cos(phi1)
         result += self.CMB.funlensedBB(l2) * L*l2*np.cos(phi2)
         result *= np.cos(2.*phi21)

      # convert from f^phi to f^kappa
      result *= 2. / L**2
      return result



   ###############################################################################
   # Lensing weights, polar coord

   def F_XY(self, L, l1, phi1, XY):
      """Lensing weights for the estimator Q_XY.
      These are the same for kappa and phi, since they only determine the relative weighting of the various X and Y multipoles.
      The response function above determines whether Q_XY is in terms of kappa or phi.
      phi1 = (L, l1)
      """
      # geometry
      l2 = self.l2(L, l1, phi1)
      phi21 = self.phi21(L, l1, phi1)
      phi2 = self.phi2(L, l1, phi1)
      
      # QE weights from Hu Okamoto 02
      if XY=='TT':
         result = self.f_XY(L, l1, phi1, XY) / self.CMB.ftotalTT(l1) / self.CMB.ftotalTT(l2) / 2.
      if XY=='EE':
         result = self.f_XY(L, l1, phi1, XY) / self.CMB.ftotalEE(l1) / self.CMB.ftotalEE(l2) / 2.
      if XY=='BB':
         result = self.f_XY(L, l1, phi1, XY) / self.CMB.ftotalBB(l1) / self.CMB.ftotalBB(l2) / 2.
      elif XY=='TB':
         result = self.f_XY(L, l1, phi1, XY) / self.CMB.ftotalTT(l1) / self.CMB.ftotalBB(l2)
      elif XY=='EB':
         result = self.f_XY(L, l1, phi1, XY) / self.CMB.ftotalEE(l1) / self.CMB.ftotalBB(l2)
      elif XY=='TE':
         numerator = self.CMB.ftotalEE(l1) * self.CMB.ftotalTT(l2) * self.f_XY(L, l1, phi1, XY)
         numerator -= self.CMB.ftotalTE(l1) * self.CMB.ftotalTE(l2) * self.f_XY(L, l2, phi2, XY)
         denom = self.CMB.ftotalTT(l1)*self.CMB.ftotalTT(l2) * self.CMB.ftotalEE(l1)*self.CMB.ftotalEE(l2)
         denom -= ( self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2) )**2
         result = numerator / denom
      
      # Shear estimator
      elif XY=='sTT':
         def fdLnC0dLnl(l):
            e = 0.01
            lup = l*(1.+e)
            ldown = l*(1.-e)
            result = self.CMB.funlensedTT(lup) / self.CMB.funlensedTT(ldown)
            result = np.log(result) / (2.*e)
            return result
         result = self.CMB.funlensedTT(l1) * fdLnC0dLnl(l1)
         result /= 2. * self.CMB.ftotalTT(l1)**2
         result *= np.cos(2. * phi1)

      # Dilation estimator
      elif XY=='dTT':
         def fdLnl2C0dLnl(l):
            e = 0.01
            lup = l*(1.+e)
            ldown = l*(1.-e)
            result = lup**2 * self.CMB.funlensedTT(lup)
            result /= ldown**2 * self.CMB.funlensedTT(ldown)
            result = np.log(result) / (2.*e)
            return result
         result = self.CMB.funlensedTT(l1) * fdLnl2C0dLnl(l1)
         result /= 2. * self.CMB.ftotalTT(l1)**2
      
      if not np.isfinite(result):
         result = 0.
      return result


   def F_XY_sym(self, L, l1, phi1, XY):
      """Symmetrized version in l1, l2.
      """
      # geometry
      l2 = self.l2(L, l1, phi1)
      phi21 = self.phi21(L, l1, phi1)
      phi2 = self.phi2(L, l1, phi1)
      return self.F_XY_sym(L, l1, phi1, XY) + self.F_XY_sym(L, l2, phi2, XY)


   ###############################################################################
   # Normalization of Q_XY estimators, ie noise for QE


   def norm_XY(self, L, XY, fCfg=None):
      """Normalization of the Q_XY estimator, in terms of kappa (not phi):
      norm = 1 / \in F * f
      such that:
      Q_XY = norm * \int F * TT
      For the QE, this coincides with the noise power spectrum of Q_XY
      """
      # integration bounds
      lMin = self.CMB.lMin
      lMax = self.l1l2Max(XY)
      if L>2.*lMax:
         return 0.

      def integrand(x):
         phi1 = x[1]
         l1 = np.exp(x[0])
         # geometry
         l2 = self.l2(L, l1, phi1)
         phi21 = self.phi21(L, l1, phi1)
         phi2 = self.phi2(L, l1, phi1)
         # check integration bounds
         if l1<lMin or l1>lMax or l2<lMin or l2>lMax:
            return 0.
         result = self.f_XY(L, l1, phi1, XY, fCfg) * self.F_XY(L, l1, phi1, XY)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first call to this function, initialize integrator dictionary
      if not hasattr(self.norm_XY.__func__, "integ"):
         self.norm_XY.__func__.integ = {}
      # if first time doing XY, initialize the integrator for XY
      if not self.norm_XY.integ.has_key(XY):
         self.norm_XY.integ[XY] = vegas.Integrator([[np.log(lMin), np.log(lMax)], [0., np.pi]])
         self.norm_XY.integ[XY](integrand, nitn=8, neval=1000)
      result = self.norm_XY.integ[XY](integrand, nitn=1, neval=5000)
      result = 1. / result.mean
      
      if not np.isfinite(result):
         result = 0.
      return result


   ###############################################################################
   # Save/load normalizations for QE

   def saveNormQ(self, pol=True):
      """Save the normalizations of the various estimators Q_XY, in terms of kappa.
      """
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      # ell values
      data = np.zeros((self.Nl, 17))
      data[:,0] = np.copy(self.L)
      
      # save normalizations of Q_XY estimators
      if pol:
         est = ['TT', 'TE', 'TB', 'EE', 'EB']   # 'BB' is zero if there is no unlensed B modes
      else:
         est = ['TT']
      nEst = len(est)
      for iEst in range(nEst):
         XY = est[iEst]
         print "Computing normalization for " + XY
         f = lambda l: self.norm_XY(l, XY)
         data[:,iEst+1] = np.array(pool.map(f, self.L))
#         data[:,iEst+1] = np.array(map(f, self.L))
      # save everything
      np.savetxt(self.pathNoiseQ, data)


   def loadNormQ(self):
      """Read the estimator normalizations,
      i.e. the noise power spectra for QE.
      """
      print "Loading normalizations"
      self.fnorm = {}
      self.fN_k = {}
      self.fN_phi = {}
      self.fN_d = {}
      # Read in normalizations
      data = np.genfromtxt(self.pathNoiseQ)
      L = data[:,0]
      
      # Load all estimators, even if they weren't computed
      est = ['TT', 'TE', 'TB', 'EE', 'EB']
      nEst = len(est)
      for iEst in range(nEst):
         XY = est[iEst]
         
         norm = data[:,iEst+1].copy()
         self.fnorm[XY] = interp1d(L, norm, kind='linear', bounds_error=False, fill_value=0.)
         self.fN_k[XY] = interp1d(L, norm, kind='linear', bounds_error=False, fill_value=0.)
         self.fN_phi[XY] = interp1d(L, norm * 4./L**4, kind='linear', bounds_error=False, fill_value=0.)
         self.fN_d[XY] = interp1d(L, norm * 4./L**2, kind='linear', bounds_error=False, fill_value=0.)



   def plotNoiseQAuto(self, fPkappa=None, pol=True):
      # diagonal covariances
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # signal
      if fPkappa is None:
         # read l^4 / (2pi) * C_l^phiphi for the Planck cosmology
         data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
         L = data[:,0]
         Pphi = data[:, 5] * (2.*np.pi) / L**4
         Pkappa = Pphi * L**4/4
         ax.plot(L, Pkappa, 'k-', lw=3, label=r'signal')
      else:
         Pkappa = np.array(map(fPkappa, self.L))
         ax.plot(self.L, Pkappa, 'k-', lw=3, label=r'signal')
      #
      # noise
      if pol:
         est = ['TT', 'TE', 'TB', 'EE', 'EB']
      else:
         est = ['TT']
      for iEst in range(len(est)):
         XY = est[iEst]
         ax.plot(self.L, self.fN_k[XY](self.L), c=plt.cm.rainbow(iEst/4.), lw=1.5, label=XY)
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='mask')
      ax.set_xlabel(r'$L$', fontsize=24)
      ax.set_ylabel(r'$C_L^\kappa$', fontsize=24)
      ax.set_ylim((3.e-11, 1.e-1))
      ax.set_xlim((10., 4.e4))
      #ax.set_title(r'Noise in lensing deflection reconstruction ('+self.CMB.name+')')
      #
      #path = "./figures/cmblensrec/"+str(self.CMB)+"/full_recnoise_lmax"+str(int(self.lMax))+".pdf"
      #path = "/Users/Emmanuel/Desktop/cmblensrec_atmnoise.pdf"
      #path = "./figures/cmblensrec/summaries_s4/"+str(self.CMB)+".pdf"
      #fig.savefig(path, bbox_inches='tight')
      #fig.clf()
      
      plt.show()
      

   ###############################################################################
   # Noise covariances for the Q_XY


   def noiseCov_XY_WZ(self, L, XY, WZ):
      """Noise cov of the Q_XY and Q_WZ estimators, in terms of kappa (not phi),
      such that
      """
      # integration bounds
      lMin = self.CMB.lMin
      lMax = min(self.l1l2Max(XY), self.l1l2Max(WZ))
      if L>2.*lMax:
         return 0.

      # Some estimators have no covariance
      if (XY=='TT')*(WZ=='TB') or (XY=='TT')*(WZ=='EB') or (XY=='TE')*(WZ=='TB') or (XY=='TE')*(WZ=='EB') or (XY=='TB')*(WZ=='EE') or (XY=='EE')*(WZ=='EB'):
         return  0.

      def integrand(x):
         phi1 = x[1]
         l1 = np.exp(x[0])
         # geometry
         l2 = self.l2(L, l1, phi1)
         phi21 = self.phi21(L, l1, phi1)
         phi2 = self.phi2(L, l1, phi1)
         # check integration bounds
         if l1<lMin or l1>lMax or l2<lMin or l2>lMax:
            return 0.
         
         if XY=='TT' and WZ=='TE':
            result = self.F_XY(L, l1, phi1, WZ)*self.CMB.ftotalTT(l1)*self.CMB.ftotalTE(l2)
            result += self.F_XY(L, l2, phi2, WZ)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTT(l2)
         if XY=='TT' and WZ=='EE':
            result = self.F_XY(L, l1, phi1, WZ)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2)
            result += self.F_XY(L, l2, phi2, WZ)*self.CMB.ftotalTE(l1)*self.CMB.ftotalTE(l2)
         if XY=='TE' and WZ=='EE':
            result = self.F_XY(L, l1, phi1, WZ)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEE(l2)
            result += self.F_XY(L, l2, phi2, WZ)*self.CMB.ftotalTE(l1)*self.CMB.ftotalEE(l2)
         if XY=='TB' and WZ=='EB':
            result = self.F_XY(L, l1, phi1, WZ)*self.CMB.ftotalTE(l1)*self.CMB.ftotalBB(l2)

         result *= self.F_XY(L, l1, phi1, XY)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result

      # if first call to this function, initialize integrator dictionary
      if not hasattr(self.noiseCov_XY_WZ.__func__, "integ"):
         self.noiseCov_XY_WZ.__func__.integ = {}
      # if first time doing XY, initialize the integrator for XY
      if not self.noiseCov_XY_WZ.integ.has_key(XY+WZ):
         self.noiseCov_XY_WZ.integ[XY+WZ] = vegas.Integrator([[np.log(lMin), np.log(lMax)], [0., np.pi]])
         self.noiseCov_XY_WZ.integ[XY+WZ](integrand, nitn=8, neval=1000)
      result = self.noiseCov_XY_WZ.integ[XY+WZ](integrand, nitn=1, neval=5000)
      result = result.mean
      
      # Multiply by the normalizations
      result *= self.fnorm[XY](L) * self.fnorm[WZ](L)
      if not np.isfinite(result):
         result = 0.
      return result


   ###############################################################################
   # Save/load noise covariances for QE, and min. var. noise

   def saveNoiseCov(self):
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      
      # copy the noise autos in the file, for convenience,
      # before adding the crosses
      data = np.genfromtxt(self.pathNoiseQ)
      
      # Adding non-diagonal covariances
      N_XY_WZ = {}
      est = ['TT', 'TE', 'TB', 'EE', 'EB']   # 'BB' is zero and has no noise cov
      nEst = len(est)
      counter = 6 # don't overwrite ell, 'TT', 'TE', 'TB', 'EE', 'EB'
      for iEst1 in range(nEst):
         XY = est[iEst1]
         N_XY_WZ[XY+XY] = self.fnorm[XY]
         for iEst2 in range(iEst1+1, nEst):
            WZ = est[iEst2]
            print "Compute noise cov " + XY + "x" + WZ
            f = lambda l: self.noiseCov_XY_WZ(l, XY, WZ)
            N_XY_WZ[XY+WZ] = np.array(pool.map(f, self.L))
            data[:, counter] = N_XY_WZ[XY+WZ]
            counter += 1

      # variance of mv estimator
      Nmv = np.zeros(self.Nl)
      # for each ell, create the matrix of estimators
      for iL in range(self.Nl):
         N = np.zeros((nEst, nEst))
         # generate the matrix of estimators
         for iEst1 in range(nEst):
            XY = est[iEst1]
            # diagonal element
            N[iEst1, iEst1] = self.fN_k[XY](self.L[iL])
            # off-diagonal elements
            for iEst2 in range(iEst1+1, nEst):
               WZ = est[iEst2]
               N[iEst1, iEst2] = N[iEst2, iEst1] = N_XY_WZ[XY+WZ][iL]
         # invert the matrix
         try:
            Inv = np.linalg.inv(N)
            Nmv[iL] = 1./np.sum(Inv)
         except:
            pass
      data[:,16] = Nmv
      
      # save everything to file
      np.savetxt(self.pathNoiseQ, data)
      return



   def loadNoiseCov(self):
      print "Loading noise covariances and mv noise"
      self.fN_k_XY_WZ = {}
      self.fN_phi_XY_WZ = {}
      self.fN_d_XY_WZ = {}
      # read the file
      data = np.genfromtxt(self.pathNoiseQ)
      L = data[:,0]

      # interpolate the noise covariances
      est = ['TT', 'TE', 'TB', 'EE', 'EB']   # 'BB' is zero and has no noise cov
      nEst = len(est)
      counter = 6 # don't re-interpolate ell, 'TT', 'TE', 'TB', 'EE', 'EB'
      for iEst1 in range(nEst):
         XY = est[iEst1]
         for iEst2 in range(iEst1+1, nEst):
            WZ = est[iEst2]
            cov = data[:, counter].copy()
            self.fN_k_XY_WZ[XY+WZ] = interp1d(L, cov, kind='linear', bounds_error=False, fill_value=0.)
            self.fN_phi_XY_WZ[XY+WZ] = interp1d(L, cov * 4./L**4, kind='linear', bounds_error=False, fill_value=0.)
            self.fN_d_XY_WZ[XY+WZ] = interp1d(L, cov * 4./L**2, kind='linear', bounds_error=False, fill_value=0.)
            counter += 1

      # interpolate the minimum variance estimator noise
      cov = data[:, -1].copy()
      self.fN_k['mv'] = interp1d(L, cov, kind='linear', bounds_error=False, fill_value=0.)
      self.fN_phi['mv'] = interp1d(L, cov * 4./L**4, kind='linear', bounds_error=False, fill_value=0.)
      self.fN_d['mv'] = interp1d(L, cov * 4./L**2, kind='linear', bounds_error=False, fill_value=0.)


   ###############################################################################
   # plots


   def plotNoiseQ(self, fPkappa=None, pol=True):
   
      # diagonal covariances
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # signal
      if fPkappa is None:
         # read l^4 / (2pi) * C_l^phiphi for the Planck cosmology
         data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
         L = data[:,0]
         Pphi = data[:, 5] * (2.*np.pi) / L**4
         Pkappa = Pphi * L**4/4
         ax.plot(L, Pkappa, 'k-', lw=3, label=r'signal')
      else:
         Pkappa = np.array(map(fPkappa, self.L))
         ax.plot(self.L, Pkappa, 'k-', lw=3, label=r'signal')
      #
      # Noise
      if pol:
         est = ['TT', 'TE', 'TB', 'EE', 'EB']
      else:
         est = ['TT']
      for iEst in range(len(est)):
         XY = est[iEst]
         ax.plot(self.L, self.fN_k[XY](self.L), c=plt.cm.rainbow(iEst/4.), lw=1.5, label=XY)
      #
      # Min. var if requested
      if pol:
         ax.plot(self.L, self.fN_k['mv'](self.L), 'k--', lw=1.5, label=r'min. var.')
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='mask')
      ax.set_xlabel(r'$L$', fontsize=24)
      ax.set_ylabel(r'$N_L^\kappa$', fontsize=24)
      ax.set_ylim((3.e-11, 1.e-1))
      ax.set_xlim((10., 4.e4))
      #
      #path = "./figures/cmblensrec/"+str(self.CMB)+"/full_recnoise_lmax"+str(int(self.lMax))+".pdf"
      #path = "/Users/Emmanuel/Desktop/cmblensrec_atmnoise.pdf"
      #path = "./figures/cmblensrec/summaries_s4/"+str(self.CMB)+".pdf"
      #fig.savefig(path, bbox_inches='tight')
      #fig.clf()

      if pol:
         # non-diagonal covariances
         fig=plt.figure(1)
         ax=fig.add_subplot(111)
         #
         ax.loglog(self.L, np.abs(self.fN_k_XY_WZ['TTTE'](self.L)), 'k', lw=2, label=r'TT-TE')
         ax.loglog(self.L, np.abs(self.fN_k_XY_WZ['TTEE'](self.L)), 'r', lw=2, label=r'TT-EE')
         ax.loglog(self.L, np.abs(self.fN_k_XY_WZ['TEEE'](self.L)), 'g', lw=2, label=r'TE-EE')
         ax.loglog(self.L, np.abs(self.fN_k_XY_WZ['TBEB'](self.L)), 'c', lw=2, label=r'TB-EB')
         #
         ax.legend(loc=2)
         ax.set_xlabel(r'$\ell$')
         ax.set_ylabel(r'$| N_\ell^{\kappa \kappa \prime} |$')
         ax.set_title(r'Lensing noise covariances')
         #
         #path = "./figures/cmblensrec/"+str(self.CMB)+"/cross_recnoise_lmax"+str(int(self.lMax))+".pdf"
         #fig.savefig(path, bbox_inches='tight')
      

      plt.show()
   

   # SNR on Cld, including reconstruction noise
   # and cosmic var if requested
   def snrTT(self, XY, fClkappa, cosmicVar=False):
      Clkappa = np.array(map(fClkappa, self.L))
      # convert Clkappa to Cld
      Cld = 4. * Clkappa / self.L**2
      # get rid of artificial zeros in noise
      Noise = self.fN[XY](self.L)
      Noise[Noise==0.] = np.inf
      # compute total SNR
      if cosmicVar:
         Y = self.L * Cld**2 / (Cld + Noise)**2  # w cosmic variance
      else:
         Y = self.L * Cld**2 / Noise**2  # w/o cosmic variance
      snr2 = np.trapz(Y, self.L)
      return np.sqrt(snr2)


   ###############################################################################
   # Save/load normalizations for shear and dilation

   def saveNormSD(self):
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      # ell values
      data = np.zeros((self.Nl, 3))
      data[:,0] = np.copy(self.L)
      
      # compute normalizations
      est = ['sTT', 'dTT']
      nEst = len(est)
      for iEst in range(nEst):
         XY = est[iEst]
         print "Compute normalization for " + XY
         f = lambda l: self.norm_XY(l, XY)
#         data[:,iEst+1] = np.array(pool.map(f, self.L))
         data[:,iEst+1] = np.array(map(f, self.L))
      # save everything
      np.savetxt(self.pathNormSD, data)


   def loadNormSD(self):
      """Read the estimator normalizations,
      i.e. the noise power spectra for QE.
      """
      print "Load normalizations for s, d"
      # Read in normalizations
      data = np.genfromtxt(self.pathNormSD)

      # make sure not to delete the noise spectra for XY different from TT
      if not hasattr(self, "fnorm"):
         self.fnorm = {}
   
      # interpolate normalizations
      est = ['sTT', 'dTT']
      nEst = len(est)
      for iEst in range(nEst):
         XY = est[iEst]
         self.fnorm[XY] = interp1d(data[:,0], data[:,iEst+1], kind='linear', bounds_error=False, fill_value=0.)


   ###############################################################################
   # Noise power spectrum for shear and dilation


   def noiseQSD(self, L, XY):
      """Noise power spectrum of shear/dilation estimators, in terms of kappa.
      Choose est = 'sTT' for shear, 'dTT' for mag, 'TT' for QE (as a test).
      """
      if L>2.*self.CMB.lMaxT:
         return 0.

      def integrand(x):
         l1 = np.exp(x[0])
         phi1 = x[1]
         # geometry
         l2 = self.l2(L, l1, phi1)
         phi21 = self.phi21(L, l1, phi1)
         phi2 = self.phi2(L, l1, phi1)
         # check integration bounds
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
      
         result = self.F_XY(L, l1, phi1, XY)
         result *= self.F_XY(L, l1, phi1, XY) + self.F_XY(L, l2, phi2, XY)
         result *= self.CMB.ftotalTT(l1) * self.CMB.ftotalTT(l2)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result

      # if first call to this function, initialize integrator dictionary
      if not hasattr(self.noiseQSD.__func__, "integ"):
         self.noiseQSD.__func__.integ = {}
      # if first time doing XY, initialize the integrator for XY
      if not self.noiseQSD.integ.has_key(XY):
         self.noiseQSD.integ[XY] = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxT)], [0., np.pi]])
         self.noiseQSD.integ[XY](integrand, nitn=8, neval=1000)
      result = self.noiseQSD.integ[XY](integrand, nitn=1, neval=5000)
      result = result.mean

      result *= self.fnorm[XY](L)**2
      if not np.isfinite(result):
         result = 0.
      return result


   def saveNoiseQSD(self):
      print "Compute lensing noises for q(test), s, d"
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      data = np.zeros((self.Nl, 4))
      data[:,0] = self.L.copy()
      
      print "Noise: QE (test)"
      f = lambda l: self.noiseQSD(l, 'TT')
      data[:,1] = np.array(pool.map(f, self.L))
      print "Noise: shear"
      f = lambda l: self.noiseQSD(l, 'sTT')
      data[:,2] = np.array(pool.map(f, self.L))
      print "Noise: dilation"
      f = lambda l: self.noiseQSD(l, 'dTT')
      data[:,3] = np.array(pool.map(f, self.L))
      np.savetxt(self.pathNoiseQSD, data)

   def loadNoiseQSD(self):
      print "Load lensing noises for q(test), s, d"
      # read noise power spectra
      data = np.genfromtxt(self.pathNoiseQSD)
      # interpolate them
      self.fN_k['qTT'] = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k['sTT'] = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=False, fill_value=0.)
      self.fN_k['dTT'] = interp1d(data[:,0], data[:,3], kind='linear', bounds_error=False, fill_value=0.)


   def plotNoiseQSD(self, fPkappa=None):
   
      # diagonal covariances
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # signal
      if fPkappa is None:
         # read l^4 / (2pi) * C_l^phiphi for the Planck cosmology
         data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
         L = data[:,0]
         Pphi = data[:, 5] * (2.*np.pi) / L**4
         Pkappa = Pphi * L**4/4
         ax.plot(L, Pkappa, 'k-', lw=3, label=r'signal')
      else:
         Pkappa = np.array(map(fPkappa, self.L))
         ax.plot(self.L, Pkappa, 'k-', lw=3, label=r'signal')
      #
      # q(test), s, d
      ax.plot(self.L, self.fN_k['qTT'](self.L), c=plt.cm.rainbow(0.), lw=1.5, label='qTT')
      ax.plot(self.L, self.fN_k['sTT'](self.L), c=plt.cm.rainbow(0.5), lw=1.5, label='sTT')
      ax.plot(self.L, self.fN_k['dTT'](self.L), c=plt.cm.rainbow(1.), lw=1.5, label='dTT')
      #
      ax.plot(self.L, self.fN_k['TT'](self.L), 'k--', lw=1., label='qTT: check')
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='mask')
      ax.set_xlabel(r'$L$', fontsize=24)
      ax.set_ylabel(r'$N_L^\kappa$', fontsize=24)
      ax.set_ylim((3.e-11, 1.e-1))
      ax.set_xlim((10., 4.e4))
      #
      #path = "./figures/cmblensrec/"+str(self.CMB)+"/full_recnoise_lmax"+str(int(self.lMax))+".pdf"
      #path = "/Users/Emmanuel/Desktop/cmblensrec_atmnoise.pdf"
      #path = "./figures/cmblensrec/summaries_s4/"+str(self.CMB)+".pdf"
      #fig.savefig(path, bbox_inches='tight')
      #fig.clf()
      
      plt.show()



   ###############################################################################
   # Response of Q,S,D to a lensed foreground,
   # useful for primary lensed foreground bias
   # and lensed foreground bias in cross-correlation


   def saveResponseLensedForeground(self):
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)

      # ell values to evaluate
      data = np.zeros((self.Nl, 19))
      data[:,0] = self.L.copy()

      # estimators to consider
      est = ['TT', 'sTT', 'dTT']

      # foregrounds to evaluate
      fg = ['cmb', 'cib', 'ksz', 'tsz', 'radiops', 'tszxcib']
      fCfg = [self.CMB.funlensedTT, self.CMB.fCIB, self.CMB.fkSZ, self.CMB.ftSZ, self.CMB.fradioPoisson, self.CMB.ftSZ_CIB]
      
      # loop over estimators
      counter = 1
      for iEst in range(len(est)):
         XY = est[iEst]
         print "Resp. of " + XY + " to lensed fg:",
         # loop over foregrounds
         for iFg in range(len(fg)):
            print fg[iFg],
            f = lambda l: self.fnorm[XY](l) / self.norm_XY(l, XY, fCfg=fCfg[iFg])
            data[:, counter] = np.array(pool.map(f, self.L))
            counter += 1
         print ""
      # save everything
      np.savetxt(self.pathResponseLensed, data)



   def loadResponseLensedForeground(self):
      print "Load responses to lensed fg"
      # create dictionary of responses to lensed foregrounds
      self.fResponseLensedFg = {}
      # read file
      data = np.genfromtxt(self.pathResponseLensed)

      # estimators to consider
      est = ['TT', 'sTT', 'dTT']

      # foregrounds to evaluate
      fg = ['cmb', 'cib', 'ksz', 'tsz', 'radiops', 'tszxcib']
      fCfg = [self.CMB.funlensedTT, self.CMB.fCIB, self.CMB.fkSZ, self.CMB.ftSZ, self.CMB.fradioPoisson, self.CMB.ftSZ_CIB]
      
      # loop over estimators
      counter = 1
      for iEst in range(len(est)):
         XY = est[iEst]
         # loop over foregrounds
         for iFg in range(len(fg)):
            # interpolate
            self.fResponseLensedFg[XY+fg[iFg]] = interp1d(data[:,0], data[:, counter], kind='linear', bounds_error=False, fill_value=0.)
            counter += 1




   def plotResponseLensedForeground(self):
      # estimators to consider
      est = ['TT', 'sTT', 'dTT']
      # foregrounds to evaluate
      fg = ['cmb', 'cib', 'ksz', 'tsz', 'radiops', 'tszxcib']
      colors = ['k', 'g', 'r', 'b', 'y', 'c']
      
      # loop over estimators
      for iEst in range(len(est)):
         XY = est[iEst]
         
         # create plot
         fig=plt.figure(iEst)
         ax=fig.add_subplot(111)
         #
         # loop over foregrounds
         for iFg in range(len(fg)):
            ax.plot(self.L, self.fResponseLensedFg[XY+fg[iFg]](self.L), c=colors[iFg], label=fg[iFg])
            ax.plot(self.L, -self.fResponseLensedFg[XY+fg[iFg]](self.L), ls='--', c=colors[iFg])
         #
         ax.set_xscale('log', nonposx='clip')
         ax.set_yscale('log', nonposy='clip')
         ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
         #ax.set_ylim((1.e-4, 1.))
         ax.set_xlabel(r'$L$')
         ax.set_ylabel(r'Response of '+XY+' to lensed fg')

      plt.show()



   ###############################################################################
   ###############################################################################
   # Lensing weights, cartesian coord
   # used for secondary lensed foreground bias calculation


   def F_XY_cart(self, l1, l2, XY):
      """Lensing weights for the estimator Q_XY.
      These are the same for kappa and phi, since they only determine the relative weighting of the various X and Y multipoles.
      The response function above determines whether Q_XY is in terms of kappa or phi.
      l1, l2 are 2d vectors.
      Enforces the lMin, lMax boundaries
      """
      # geometry
      L = l1 + l2
      Lnorm = np.sqrt(np.sum(L**2))
      l1norm = np.sqrt(np.sum(l1**2))
      l2norm = np.sqrt(np.sum(l2**2))
      # integration bounds
      if (l1norm<self.CMB.lMin) or (l1norm>self.CMB.lMaxT) or (l2norm<self.CMB.lMin) or (l2norm>self.CMB.lMaxT):
         return 0.
      
      # QE weights from Hu Okamoto 02
      if XY=='TT':
         result = self.CMB.funlensedTT(l1norm) * np.dot(L, l1)
         result += self.CMB.funlensedTT(l2norm) * np.dot(L, l2)
         # convert from f^phi to f^kappa
         result *= 2. / Lnorm**2
         result /= self.CMB.ftotalTT(l1norm)
         result /= self.CMB.ftotalTT(l2norm)
         result /= 2.
      
      # Shear estimator
      elif XY=='sTT':
         def fdLnC0dLnl(l):
            e = 0.01
            lup = l*(1.+e)
            ldown = l*(1.-e)
            result = self.CMB.funlensedTT(lup) / self.CMB.funlensedTT(ldown)
            result = np.log(result) / (2.*e)
            return result
         # cos(2 * theta_{L, l1})
         result = (L[0]**2 - L[1]**2) * (l1[0]**2 - l1[1]**2)
         result += 4. * L[0] * L[1] * l1[0] * l1[1]
         result /= Lnorm**2 * l1norm**2
         result *= self.CMB.funlensedTT(l1norm)
         result *= fdLnC0dLnl(l1norm)
         result /= self.CMB.ftotalTT(l1norm)**2
         result /= 2.

      # Dilation estimator
      elif XY=='dTT':
         def fdLnl2C0dLnl(l):
            e = 0.01
            lup = l*(1.+e)
            ldown = l*(1.-e)
            result = lup**2 * self.CMB.funlensedTT(lup)
            result /= ldown**2 * self.CMB.funlensedTT(ldown)
            result = np.log(result) / (2.*e)
            return result
         result = self.CMB.funlensedTT(l1norm)
         result *= fdLnl2C0dLnl(l1norm)
         result /= self.CMB.ftotalTT(l1norm)**2
         result /= 2.
      
      if not np.isfinite(result):
         result = 0.
      return result



   def F_XY_cart_sym(self, l1, l2, XY):
      """Symmetrized version in l1, l2.
      """
      return 0.5 * (self.F_XY_cart(l1, l2, XY) + self.F_XY_cart(l2, l1, XY))


   def alpha_cart(self, l1, l2):
      """Coupling kernel between unlensed map and convergence map,
      to produce the first order lensed map.
      l1, l2 are 2d vectors.
      Quantity is not symmetric in l1,l2, as expected.
      No lmin, lmax boundaries to enforce.
      """
      l1norm = np.sqrt(np.sum(l1**2))
      if l1norm==0.:
         result = 0.
      else:
         result = -2. * np.dot(l1, l2) / l1norm**2
      return result
   

   ###############################################################################
   # Secondary lensed foreground bias for Q,S,D
   

   def secondaryLensedForegroundBias(self, fCf, fCkkf, L0, XY):
      """Computes the secondary lensed foreground bias to CMB lensing, at multipole L0.
      Gives the bias on C_ell^kappa_CMB.
      Assumes that the QE uses the unlensed CMB power in the lensing response (numerator).
      fCf: unlensed foreground power spectrum
      fCkkf: cross-power spectrum of kappa_CMB and kappa_foreground
      Choose est = 'sTT' for shear, 'dTT' for mag, 'TT' for QE.
      """
      if L0>2.*self.CMB.lMaxT:
         return 0., 0.
      # Make L0 a 2d vector, along x-axis
      L0 = np.array([L0, 0.])

      def integrand(pars):
         lnLnorm = pars[0]
         thetaL = pars[1]
         lnlnorm = pars[2]
         thetal = pars[3]
         L = np.array([np.exp(lnLnorm) * np.cos(thetaL), np.exp(lnLnorm) * np.sin(thetaL)])
         l = np.array([np.exp(lnlnorm) * np.cos(thetal), np.exp(lnlnorm) * np.sin(thetal)])

         # factors in common between term 1 and term 2
         result = self.F_XY_cart_sym(l, L0-l, XY)
         result *= self.F_XY_cart_sym(l-L-L0, L-l, XY)
         result *= self.alpha_cart(L, l-L)
         result *= fCkkf(np.sqrt(np.sum(L**2)))

         # Term 1: L0
         term1 = self.alpha_cart(-L, l-L0)
         term1 *= self.CMB.funlensedTT(np.sqrt(np.sum((L0-l)**2)))
         term1 *= fCf(np.sqrt(np.sum((l-L)**2)))

         # Term 2: L0
         term2 = self.alpha_cart(-L, L0+L-l)
         term2 *= self.CMB.funlensedTT(np.sqrt(np.sum((l-L)**2)))
         term2 *= fCf(np.sqrt(np.sum((l-L-L0)**2)))

         # factor from symmetries
         result *= 8. * (term1 + term2)
         # factors of pi from Fourier convention
         result /= (2.*np.pi)**4
         # normalize properly for the kappa quadratic estimator squared
         result *= self.fnorm[XY](L0[0])**2
#         # convert from phi squared to kappa squared
#         result *= (-L0[0]**2/2.)**2
         # jacobian for polar coordinates, with log norm
         result *= np.exp(lnLnorm)**2
         result *= np.exp(lnlnorm)**2
         # compensate for halving the integration domain
         result *= 2.
         return result


      # if first call to this function, initialize integrator dictionary
      if not hasattr(self.secondaryLensedForegroundBias.__func__, "integ"):
         self.secondaryLensedForegroundBias.__func__.integ = {}
      # if first time doing XY, initialize the integrator for XY
      if not self.secondaryLensedForegroundBias.integ.has_key(XY):
         
         print "wah"
         self.secondaryLensedForegroundBias.integ[XY] = vegas.Integrator([[np.log(1.), np.log(2.*self.CMB.lMaxT)], # L-l is limited: goes into QE
                                                                              [0., np.pi],   # keep only half the domain (symmetry)
                                                                              [np.log(self.CMB.lMin), np.log(self.CMB.lMaxT)],   # l is limited: goes into QE
                                                                              [0., 2.*np.pi]])

#         self.secondaryLensedForegroundBias.integ[XY](integrand, nitn=10, neval=1e6)
#      print "hoohoo"
#      result = self.secondaryLensedForegroundBias.integ[XY](integrand, nitn=30, neval=6e6)
#      print "weehee"

#         self.secondaryLensedForegroundBias.integ[XY](integrand, nitn=7, neval=1e5) # 7h33m run: 20, 1e5
#      print "hoohoo"
#      result = self.secondaryLensedForegroundBias.integ[XY](integrand, nitn=15, neval=1e5) # 7h33m run: 60, 5e5
#      print "weehee"

#         self.secondaryLensedForegroundBias.integ[XY](integrand, nitn=7, neval=1e2) # 7h33m run: 20, 1e5
#      print "hoohoo"
#      result = self.secondaryLensedForegroundBias.integ[XY](integrand, nitn=15, neval=1e2) # 7h33m run: 60, 5e5
#      print "weehee"

         self.secondaryLensedForegroundBias.integ[XY](integrand, nitn=10, neval=1e6)
      print "hoohoo"
      result = self.secondaryLensedForegroundBias.integ[XY](integrand, nitn=30, neval=6e6)
      print "weehee"


#      print L0[0], result.sdev/result.mean
#      print result.summary()
      return result.mean, result.sdev

   
   
   
   
   












   ###############################################################################
   ###############################################################################
   # Multipole lensing estimators
   
   ###############################################################################
   # Azimuthal multipole moments of the lensing response f_TT,
   # with theta = (L, l), l1 = L/2 + l and l2 = L/2 - l.
   # Needed for shear/dilation estimators, in the squeezed limit L<<l,
   # and in the general (not squeezed) limit.


   def config_multipole(self, L, l, t):
      '''Returns l1, l2 and phi21=(l1, l2), phi1=(L, l1), phi2=(L,l2)
      given L, l and theta=(L,l),
      where:
      l1 = L/2 + l,
      l2 = L/2 - l.
      This configuration is relevant for azimuthal multipoles, especially in the squeezed limit L<<l.
      '''
      l1 = np.sqrt(l**2 + L**2/4. + l*L*np.cos(t)) # np.sqrt((L/2.+l*np.cos(t))**2 + (l*np.sin(t))**2)
      l2 = np.sqrt(l**2 + L**2/4. - l*L*np.cos(t)) # np.sqrt((L/2.-l*np.cos(t))**2 + (l*np.sin(t))**2)

      x1 = L/2. + l*np.cos(t)
      y1 = l*np.sin(t)
      phi1 = np.arctan2(y1,x1)

      x2 = L/2. - l*np.cos(t)
      y2 = -l*np.sin(t)
      phi2 = np.arctan2(y2,x2)

      phi21 = phi2 - phi1
      return l1, l2, phi21, phi1, phi2


   def f_TT_multipole_squeezed(self, L, l, m=0):
      '''Expected limit of the m-th multipole moment of f_TT when L<<l,
      to first order in L/l.
      Should match f_TT_multipole_interp in this limit.
      '''
      # derivative of the unlensed power spectrum
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = self.CMB.funlensedTT(lup) / self.CMB.funlensedTT(ldown)
         result = np.log(result) / (2.*e)
         return result
      
      if m==0:
         result = fdLnC0dLnl(l) + 2.  # = dln(l^2C0)/dlnl
      elif m==2:
         result = fdLnC0dLnl(l)
      else:
         result = 0. # I haven't computed the other multipoles in the squeezed limit
      result *= - self.CMB.funlensedTT(l)
#      result *= L**2/2. # convert from f_kappa to f_phi
      return result


   def f_TT_multipole(self, L, l, m=0):
      '''m-th multipole moment of f_TT(l1, l2):
      \int dtheta/(2pi) f_TT(l1, l2) * cos(m theta),
      where:
      l1 = L/2 + l,
      l2 = L/2 - l.
      No consideration of lmin and lmax here, so that all the theta integrals
      cover the full circle. This is somewhat sketchy.
      '''
      def integrand(t):
         l1, l2, phi21, phi1, phi2 = self.config_multipole(L, l, t)
#         if l1<self.CMB.lMin or l1>self.CMB.lMaxT or l2<self.CMB.lMin or l2>self.CMB.lMaxT:
#            print "nope"
#            result = 0.
#         else:
         result = self.f_XY(L, l1, phi1, XY='TT')
         if m>0:
            result *= 2. * np.cos(m*t)
         return result / (np.pi) # because half the angular integration domain
      
      result = integrate.quad(integrand, 0., np.pi, epsabs=0, epsrel=1.e-3)[0]
      return result

   ###############################################################################
   # Interpolate the azimuthal multipole moments of the lensing response f_TT.

   def save_f_TT_multipole(self, m=0):
      '''Precompute f_TT_multipole(L, l, m),
      for speed.
      '''
      NL = 501
      Nl = 501
      Ell = np.logspace(np.log10(10.), np.log10(2.*self.CMB.lMaxT), NL, 10.)
      ell = np.logspace(np.log10(10.), np.log10(self.CMB.lMaxT), Nl, 10.)
      table = np.zeros((NL, Nl))
      print "precompute the "+str(m)+"-th multipole of f_TT"
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      for iL in range(NL):
         f = lambda il: self.f_TT_multipole(Ell[iL], ell[il], m)
         table[iL, :] = np.array(pool.map(f, range(Nl)))
         print "- done "+str(iL+1)+" of "+str(NL)
      # save the table
      np.savetxt(self.directory+"/Llong_fTTmultipole_m"+str(m)+".txt", Ell)
      np.savetxt(self.directory+"/lshort_fTTmultipole_m"+str(m)+".txt", ell)
      np.savetxt(self.directory+"/fTTmultipole_m"+str(m)+".txt", table)
      return


   def save_f_TT_multipoles(self):
      "Compute f_TT_m, m = ",
      M = [0, 2, 4, 6, 8]
      for m in M:
         print m
         self.save_f_TT_multipole(m)
      print ""


   def load_f_TT_multipole(self, m):
      # read files
      Ell = np.genfromtxt(self.directory+"/Llong_fTTmultipole_m"+str(m)+".txt").copy()
      ell = np.genfromtxt(self.directory+"/lshort_fTTmultipole_m"+str(m)+".txt").copy()
      table = np.genfromtxt(self.directory+"/fTTmultipole_m"+str(m)+".txt").copy()
      # interpolate
      interp = RectBivariateSpline(np.log10(Ell), np.log10(ell), table, kx=1, ky=1, s=0)
      f = lambda Lnew, lnew: (Lnew>=Ell.min() and Lnew<=Ell.max()) * (lnew>=ell.min() and lnew<=ell.max()) * interp(np.log10(Lnew), np.log10(lnew))[0,0]
      return f


   def load_f_TT_multipoles(self):
      print "Load f_TT_multipoles: m=",
      self.f_TT_multipole_interp = {}
#      M = [0, 2, 4, 6, 8]
      M = [0, 2]
      for m in M:
         print m,
         self.f_TT_multipole_interp[m] = self.load_f_TT_multipole(m)
      print ""

   def plot_f_TT_multipoles(self):
      print "Plotting f_TT_m"
      # l, L values to show
      NL = 501
      Nl = 501
      Ell = np.logspace(np.log10(10.), np.log10(2.*self.CMB.lMaxT), NL, 10.)
      ell = np.logspace(np.log10(10.), np.log10(self.CMB.lMaxT), Nl, 10.)

      # multipoles to show
      M = [0, 2]  #[0, 2, 4, 6, 8]
      for m in M:
         fig=plt.figure(m)
         ax=fig.add_subplot(111)
         #
         for iL in range(0, len(Ell), 50):
#         for iL in range(NL):
            L = Ell[iL]
            f = lambda l: self.f_TT_multipole_interp[m](L, l)
            Finterp = np.array(map(f, ell))
            ax.loglog(ell, np.abs(Finterp), c=plt.cm.winter(iL*1./len(Ell)), label=r'$L=$'+str(np.int(L)))
            #
            # theory multipole, to first order in L/l
            f = lambda l: self.f_TT_multipole_squeezed(L,l,m)
            Ftheory = np.array(map(f, ell))
            ax.loglog(ell, np.abs(Ftheory), c=plt.cm.winter(iL*1./len(Ell)), ls='--')
         #
         ax.loglog([], [], c=plt.cm.winter(0.), label=r'non-perturbative')
         ax.loglog([], [], c=plt.cm.winter(0.), ls='--', label=r'squeezed')
         ax.legend(loc=3)
         ax.set_xlabel(r'$\ell$')
         ax.set_ylabel(r'$f^{m}(L, \ell)$')
         ax.set_title(r'Multipole '+str(m))

         plt.show()



   ###############################################################################
   # Noise of the lensing estimator from the m-th multipole moment of f_TT


   def N_k_TT_m(self, L, m=0, optimal=True):
      """Noise power spectrum of kappa from TT,
      using only the m-th multipole moment of the lensing response f_TT.
      Here theta=(L,l), with:
      l1 = L/2 + l,
      l2 = L/2 - l.
      This is not the same convention as in my standard QE integrals.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      
      # integrand
      def integrand(lnl):
         l = np.exp(lnl)
         
         # choose l bounds so that the theta integral can cover the full circle
         # otherwise, the multipole estimators will be biased
         if (np.abs(l-L/2)<self.CMB.lMin) or (l+L/2>self.CMB.lMaxT):
            result = 0.
         
         else:
            result = self.f_TT_multipole_interp[m](L, l)**2
            result *= l**2 / (2.*np.pi)
            if m>0:
               result /= 4.

            # use the optimal noise weighting: angular average of the Cl^total
            if optimal:
               # angular integrand
               def f(t):
                  l1, l2, phi21, phi1, phi2 = self.config_multipole(L, l, t)
                  result = self.CMB.ftotalTT(l1) * self.CMB.ftotalTT(l2)
                  result *= np.cos(m*t)**2
                  result /= 2.*np.pi
                  result *= 2.
                  result *= 2.   # because integrating over half the domain
                  return result
               # compute angular integral
               integral = integrate.quad(f, 0., np.pi, epsabs=0, epsrel=1.e-3)[0]
            # else use the suboptimal intuitive noise
            else:
               integral = 2. * self.CMB.ftotalTT(l)**2
            result /= integral
   
         if not np.isfinite(result):
            result = 0.
         return result
      
      result = integrate.quad(integrand, np.log(1.), np.log(self.CMB.lMaxT), epsabs=0, epsrel=1.e-3)[0]
      try:  # gives error when dividing by zero
         result = 1. / result
      except:
         pass
#      result = (L**2/2.)**2 / result
      if not np.isfinite(result):
         result = 0.
      print "- done L="+str(L), result
      return result


   def N_k_TT_test(self, L):
      """Noise power spectrum of kappa from TT:
      should recover the standard Hu & Okamoto expression.
      This is a good test for my angular conversions.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      
      # integrand
      def integrand(x):
         l = np.exp(x[0])
         t = x[1]
         l1, l2, phi21, phi1, phi2 = self.config_multipole(L, l, t)

         if (l1<self.CMB.lMin or l1>self.CMB.lMaxT) or (l2<self.CMB.lMin or l2>self.CMB.lMaxT):
            result = 0.
         else:
            result = self.f_XY(L, l1, phi1, XY='TT')
            result = result**2
            result /= 2. * self.CMB.ftotalTT(l1) * self.CMB.ftotalTT(l2)
            result *= l**2 / (2.*np.pi)**2
            result *= 2.   # because half of the angular integration domain
         if not np.isfinite(result):
            print "problem:", L, l, t, l1, l2, phi21
            result = 0.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.N_k_TT_test.__func__, "integ"):
         self.N_k_TT_test.__func__.integ = vegas.Integrator([[np.log(1.), np.log(self.CMB.lMaxT)], [0., np.pi]])
         self.N_k_TT_test.integ(integrand, nitn=8, neval=1000)
      result = self.N_k_TT_test.integ(integrand, nitn=1, neval=5000)
      result = 1. / result.mean
#      result = (L**2/2.)**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      print "- done L="+str(L)
      return result



   def save_N_k_TT_multipoles(self):
      path = self.directory+"/Nk_TT_m.txt"
      data = np.zeros((self.Nl, 9))
      data[:,0] = self.L.copy()
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      
      print "Compute QE noise (test)"
      data[:,1] = np.array(pool.map(self.N_k_TT_test, self.L))

#      M = [0, 2, 4, 6, 8]
      M = [0, 2]
      print "Compute noise: m=",
      for iM in range(len(M)):
         m = M[iM]
         print m,
         f = lambda L: self.N_k_TT_m(L, m=m)
#         data[:, iM+2] = np.array(pool.map(f, self.L))
         data[:, iM+2] = np.array(map(f, self.L))
         print ""
   
      # save everything
      np.savetxt(path, data)


   def load_N_k_TT_multipoles(self):
      print "Load lensing noise multipole estimators"
      self.fN_k_TT_m = {}
      # read file
      path = self.directory+"/Nk_TT_m.txt"
      data = np.genfromtxt(path)
      
      # QE noise (test)
      self.fN_k_TT_m[-1] = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      
      # multipole estimators
#      M = [0, 2, 4, 6, 8]
      M = [0, 2]
      for iM in range(len(M)):
         m = M[iM]
         self.fN_k_TT_m[m] = interp1d(data[:,0], data[:,iM+2], kind='linear', bounds_error=False, fill_value=0.)
   

   def plotNoiseMultipoles(self, fPkappa=None):
   
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # signal
      if fPkappa is None:
         # read C_l^phiphi for the Planck cosmology
         data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
         L = data[:,0]
         Pphi = data[:, 5] * (2.*np.pi) / L**4
         ax.plot(L, L**4 / 4. * Pphi, 'k-', lw=3, label=r'signal')
      else:
         Pkappa = np.array(map(fPkappa, self.L))
         ax.plot(self.L, Pkappa * 4./(2.*np.pi), 'k-', lw=3, label=r'signal')
      #
      # multipole estimators
      M = [0, 2]
      N_comb = 0.
      for iM in range(len(M)):
         m = M[iM]
         ax.plot(self.L, self.fN_k_TT_m[m](self.L), c=plt.cm.winter(iM/(len(M)-1.)), label=r'm='+str(m))
      #
      # combined multipole estimators, as if they were independent
      N_comb = 0.
      for iM in range(len(M)):
         m = M[iM]
         N_comb += 1. / self.fN_k_TT_m[m](self.L)
      N_comb = 1./N_comb
      ax.plot(self.L, N_comb, c='gray', ls='--', label=r'naive combination')
      #
      # QE and test for QE
      ax.plot(self.L, self.fN_k['TT'](self.L), 'k', label=r'QE')
      ax.plot(self.L, self.fN_k_TT_m[-1](self.L), 'k--', label=r'QE (test)')
      #
      ax.legend(loc=2, labelspacing=0.)
      ax.set_xscale('log')
      ax.set_yscale('log', nonposy='mask')
      ax.set_xlabel(r'$L$', fontsize=24)
      ax.set_ylabel(r'$C_L^\kappa$', fontsize=24)

      plt.show()



   ###############################################################################
   ###############################################################################
   # show which T_l contribute to kappa_L


   def snr2Density_TT(self, L, l1):
      """"snr^2 density" = d(1/N_L^kappa)/dlnl1
      showing which l1 contribute to kappa_L.
      Normalized such that the integral wrt dlnl1 yields 1/N_L^kappa.
      This breaks the symmetry between l1 and l2,
      because l1 is fixed while l2 varies within l1-L and L1+L
      """

      # make sure L is within reconstruction range
      if L>2.*self.CMB.lMaxT:
         return 0.
      # make sure l1 is within the map range
      if l1<self.CMB.lMin or l1>self.CMB.lMaxT:
         return 0.

      # integrand
      def integrand(phi1):
         # geometry
         l2 = self.l2(L, l1, phi1)
         phi21 = self.phi21(L, l1, phi1)
         phi2 = self.phi2(L, l1, phi1)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         result = self.f_XY(L, l1, phi1, XY='TT') * self.F_XY(L, l1, phi1, XY='TT')
         result *= l1**2   # integrand wrt dlnl1 * dphi
         result /= (2.*np.pi)**2
         result *= 2.   # from halving the integration domain
         return result

      result = integrate.quad(integrand, 0., np.pi, epsabs=0, epsrel=1.e-3)[0]
      return result


   def snr2DensitySymLowL_TT(self, L, l):
      """"snr^2 density" = d(1/N_L^kappa)/dlnl
      showing which l contribute to kappa_L,
      ONLY IN THE REGIME L << l.
      Normalized such that the integral wrt dlnl yields 1/N_L^kappa.
      Here l = (l1+l2)/2, and L = l1-l2,
      which makes l1 and l2 symmetric:
      they both vary between l-L/2 and l+L/2.
      This decomposition is inspired by the shear/dilation estimators
      """
      # make sure L is within reconstruction range
      if L>2.*self.CMB.lMaxT:
         return 0.

      # integrand: theta is the angle between L = (l1-l2)/2 and l = (l1+l2)/2.
      # We integrate over theta, at fixed l and L, ie varying l1 and l2 accordingly
      def integrand(theta):
         # derivatives of the unlensed power spectrum
         a = 1.+1.e-3
         dlnCldlnl = np.log(self.CMB.funlensedTT(l*a) / self.CMB.funlensedTT(l*a))
         dlnCldlnl /= 2.*np.log(a)
         dlnl2Cldlnl = dlnCldlnl + 2.
         # get l1 and l2
         l1, l2, phi21, phi1, phi2 = self.config_multipole(L, l, theta)
         if l1<self.CMB.lMin or l1>self.CMB.lMaxT:
            return 0.
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         # compute the integrand
         result = dlnl2Cldlnl + np.cos(2.*theta) * dlnCldlnl
         result *= self.CMB.funlensedTT(l)
         result = result**2
         result /= self.CMB.ftotalTT(l1)
         result /= self.CMB.ftotalTT(l2)
         result /= 2.
         result *= l**2   # integrand wrt dlnl1 * dphi
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      result = integrate.quad(integrand, 0., np.pi, epsabs=0, epsrel=1.e-3)[0]
      return result



   def plotSnr2DensitySymLowL_TT_lines(self):
      """"snr^2 density" = d(1/N_L^kappa)/dlnl
      showing which l contribute to kappa_L,
      ONLY IN THE REGIME L << l.
      Normalized such that the integral wrt dlnl yields 1/N_L^kappa.
      Here l = (l1+l2)/2, and L = l1-l2,
      which makes l1 and l2 symmetric:
      they both vary between l-L/2 and l+L/2.
      This decomposition is inspired by the shear/dilation estimators
      """
      # ell values for phi
      #L = np.logspace(np.log10(10.), np.log10(1.e4), 5, 10.)
      #L = np.array([10., 1.e2, 5.e2])
      L = [100.]
      
      # ell values for T
      L1 = np.logspace(np.log10(10.), np.log10(6.e4), 501, 10.)

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      for iL in range(len(L)):
         f = lambda l1: self.snr2DensitySymLowL_TT(L[iL], l1)
         snr2Density = np.array(map(f, L1))
         # expected value if only counting number of modes
         dNLkappadlnl_theory = L1**2 / (np.pi)# * L[iL]**4

         # save the values for future use
         data = np.zeros((len(L1), 3))
         data[:,0] = L1.copy()
         data[:,1] = snr2Density
         data[:,2] = dNLkappadlnl_theory
         path = self.directory+"/dsnr2_dlnl_L"+intExpForm(L[iL])+".txt"
         print "saving to "+path
         np.savetxt(path, data)
         
         # normalize both, for plot
         dNLkappadlnl_theory /= np.max(snr2Density)
         snr2Density /= np.max(snr2Density)#[np.where(L1 <= 5.e3)]) # just to rescale in plot
         #ax.plot(L1, snr2Density, lw=2, c=plt.cm.jet(1.*iL/len(L)), label=r'$L_\phi=$'+str(int(L[iL])))
         ax.plot(L1, snr2Density, lw=2, c='r', label=r'$L_\phi=$'+str(int(L[iL])))
         ax.plot(L1, dNLkappadlnl_theory, 'r', ls='--', lw=1)
      #
      ax.legend(loc=1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_xlabel(r'$\ell_T$')
      ax.set_ylabel(r'$d\text{SNR}(\phi_L)^2/d\text{ln}\ell_T$')
      #ax.set_xlim((1., 5.e3))
      ax.set_ylim((0., 1.1))
      #
      path = "./figures/cmblensrec/dsnr2_dl_test.pdf"
      fig.savefig(path, bbox_inches='tight')
      
      plt.show()

      return L1, snr2Density
   
   
   
   def plotSnr2Density_TT_color(self):
   
      # multipoles of phi
      nL = 101  #51
      lnLMin = np.log10(1.)
      lnLMax = np.log10(self.lMax*(1.-1.e-1))   #np.log10(2.*self.lMax*(1.-1.e-3))
      dlnL = (lnLMax-lnLMin)/nL
      lnL = np.linspace(lnLMin, lnLMax, nL)
      lnLEdges = np.linspace(lnLMin-0.5*dlnL, lnLMax+0.5*dlnL, nL+1)
      L = 10.**lnL
      LEdges = 10.**lnLEdges

      # multipoles of T
      nL1 = 101  #51
      lnL1Min = np.log10(self.lMin*(1.+1.e-1))
      lnL1Max = np.log10(self.lMax*(1.-1.e-1))
      dlnL1 = (lnL1Max-lnL1Min)/nL1
      lnL1 = np.linspace(lnL1Min, lnL1Max, nL1)
      lnL1Edges = np.linspace(lnL1Min-0.5*dlnL1, lnL1Max+0.5*dlnL1, nL1+1)
      L1 = 10.**lnL1
      L1Edges = 10.**lnL1Edges
   
      # compute
      dSNR2dl = np.zeros((nL1, nL))
      for iL in range(nL):
         l = L[iL]
         for iL1 in range(nL1):
            l1 = L1[iL1]
            dSNR2dl[iL1, iL] = self.snr2Density_TT(l, l1)
         # normalize so that int dl1 dSNR2/dl1 = 1
         #dSNR2dl[:,iL] /= 1./self.A_TT(l)
   
      # make the color plot
      LL1,LL = np.meshgrid(L1Edges, LEdges, indexing='ij')
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      cp=ax.pcolormesh(LL1, LL, np.log10(dSNR2dl + 1.e-10), linewidth=0, rasterized=True, cmap=plt.cm.YlOrRd)
      #cp.set_clim(0., 1.)
      fig.colorbar(cp)
      #
      ax.set_xscale('log')
      ax.set_yscale('log')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$L$')
      
      plt.show()
   
   
   ###############################################################################
   ###############################################################################
   # for a source field defined by dCl/dz = f(l, z),
   # compute the effective source redshift distribution,
   # appropriate for the quadratic estimator,
   # analogous to the dn/dz of galaxy lensing.
   
   def fz_TT(self, l1, l2, phi, z):
      result = self.CMB.fdCldz(l1, z) * l1*(l1 + l2*np.cos(phi))
      result += self.CMB.fdCldz(l2, z) * l2*(l2 + l1*np.cos(phi))
      return result
   
   
   # here l is the multipole of kappa considered
   # gives the effective source distribution at z
   def sourceDist_TT(self, L, z):
      if L>2.*self.CMB.lMaxT:
         return 0.
      
      def integrand(x):
         l1 = np.exp(x[0])
         phi1 = x[1]
         # geometry
         l2 = self.l2(L, l1, phi1)
         phi21 = self.phi21(L, l1, phi1)
         phi2 = self.phi2(L, l1, phi1)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         result = self.fz_TT(l1, l2, phi21, z) * self.F(L, l1, phi1)
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.   # by symmetry, we integrate over half the area
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.sourceDist_TT.__func__, "integ"):
         self.sourceDist_TT.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxT)], [0., np.pi]])
         self.sourceDist_TT.integ(integrand, nitn=8, neval=1000)
      result = self.sourceDist_TT.integ(integrand, nitn=1, neval=5000)
      result = L**2 / result.mean
      if not np.isfinite(result):
         result = 0.
      print L, z, result
      return result
   
   
   def plotSourceDist_TT(self):
   
      # multipoles of phi
      nL = 3  #51
      lnLMin = np.log10(1.)
      lnLMax = np.log10(2.*self.lMax*(1.-1.e-3))
      dlnL = (lnLMax-lnLMin)/nL
      lnL = np.linspace(lnLMin, lnLMax, nL)
      lnLEdges = np.linspace(lnLMin-0.5*dlnL, lnLMax+0.5*dlnL, nL+1)
      L = 10.**lnL
      LEdges = 10.**lnLEdges
      
      # redshifts
      nZ = 3  #51
      zMin = 1.e-3
      zMax = 5.
      dZ = (zMax-zMin)/nZ
      Z = np.linspace(zMin, zMax, nZ)
      zEdges = np.linspace(zMin-0.5*dZ, zMax+0.5*dZ, nZ+1)
      
      # compute!!!
      SourceDistTT = np.zeros((nZ, nL))
      for iZ in range(nZ):
         z = Z[iZ]
         for iL in range(nL):
            l = L[iL]
            SourceDistTT[iZ, iL] = self.sourceDist_TT(l, z)
      
      print SourceDistTT
      
      # make the color plot
      ZZ,LL = np.meshgrid(zEdges, LEdges, indexing='ij')
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      cp=ax.pcolormesh(ZZ, LL, SourceDistTT, linewidth=0, rasterized=True, cmap=plt.cm.YlOrRd_r)
      #cp.set_clim(0., 1.)
      fig.colorbar(cp)
      #
      #ax.set_xscale('log')
      #ax.set_yscale('log')
      ax.set_xlabel(r'$z$')
      ax.set_ylabel(r'$L$')
      
      plt.show()
   
   
   
   
   ###############################################################################
   ###############################################################################
   # Order of magnitude for the effect of the unlensed trispectrum on the lensing noise
   # e.g., effect of CIB trispectrum on CIB lensing noise
   # This is really T / CC.
   # !!! not really used
   
   # L is the ell for phi
   def trispectrumCorrection(self, fTnondiag, L=20.):
      """multiplicative correction due to the trispectrum,
      i.e. T_{l,L-l,l,L-l} / C_l C_{L-l},
      for L given as input,
      and l in [1, lMaxT].
      """
      def f(arg):
         # here the arguments expected are |l|, |L-l|
         l1 = arg[0]
         l2 = arg[1]
         cc = self.CMB.ftotalTT(l1)*self.CMB.ftotalTT(l2)
         t = fTnondiag(l1, l2)
         return t/cc
      
      # initialize the array of l
      lx = np.linspace(1., self.CMB.lMaxT, 11)
      ly = np.linspace(1., self.CMB.lMaxT, 11)
      lx, ly = np.meshgrid(lx, ly, indexing='ij')
      #
      l = np.sqrt(lx**2 + ly**2)
      Lminusl = np.sqrt( (L-lx)**2 + ly**2 )
      Arg = np.array([(l.flatten()[i], Lminusl.flatten()[i]) for i in range(len(l.flatten()))])
      #
      Result = np.array(map(f, Arg))
      Result.reshape(np.shape(l))
      return lx, ly, Result
   
   
   def plotTrispectrumCorrection(self, fTnondiag, L=20.):
      lx, ly, Result = self.trispectrumCorrection(fTnondiag, L=20.)
      plt.pcolormesh(lx, ly, np.log(Result))
      plt.colorbar()
      plt.show()
      return lx, ly, result


   ###############################################################################
   ###############################################################################
   # Effect of white trispectrum on lensing noise
   # e.g., effect of CIB trispectrum on CIB lensing
   # here I assume that the trispectra of T, Q, U are white,
   # i.e. the trispectra of E and B are non-white.


   def relativeNoiseWhiteTrispec_TT(self, L):
      """Factor such that:
      N_L^\kappa = N_L^{0 \kappa} * (1 + factor * Trispectrum),
      where the trispectrum is assumed white.
      In other words, factor * Trispectrum is the relative increase in lensing noise
      due to the unlensed white trispectrum.
      """
      if L>2.*self.CMB.lMaxT:
         return 0.
      # integrand
      def integrand(x):
         l1 = np.exp(x[0])
         phi1 = x[1]
         # geometry
         l2 = self.l2(L, l1, phi1)
         phi21 = self.phi21(L, l1, phi1)
         phi2 = self.phi2(L, l1, phi1)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxT:
            return 0.
         result = self.F_XY(L, l1, phi1, XY='TT')
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.relativeNoiseWhiteTrispec_TT.__func__, "integ"):
         self.relativeNoiseWhiteTrispec_TT.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.relativeNoiseWhiteTrispec_TT.integ(integrand, nitn=8, neval=1000)
      result = self.relativeNoiseWhiteTrispec_TT.integ(integrand, nitn=1, neval=5000)

      result = result.mean**2
      result *= self.fN_phi_TT(L)
      if not np.isfinite(result):
         result = 0.
      return result



   def relativeNoiseWhiteTrispec_EB(self, L):
      """Factor such that:
      N_L^\kappa = N_L^{0 \kappa} * (1 + factor * Trispectrum),
      where the trispectrum is assumed white.
      In other words, factor * Trispectrum is the relative increase in lensing noise
      due to the unlensed white trispectrum.
      Here, we assume the trispectrum of Q and U are white and uncorrelated,
      so that the E and B trispectra are non-white.
      !!! Gives weird result in polarization: very large and noisy. Cancellations?
      """
      if L>2.*self.CMB.lMaxP:
         return 0.
      # integrand
      def integrand(x):
         l1 = np.exp(x[0])
         phi1 = x[1]
         # geometry
         l2 = self.l2(L, l1, phi1)
         phi21 = self.phi21(L, l1, phi1)
         phi2 = self.phi2(L, l1, phi1)
         if l2<self.CMB.lMin or l2>self.CMB.lMaxP:
            return 0.
         result = self.F_XY(L, l1, phi1, XY='EB')
         result *= l1**2
         result /= (2.*np.pi)**2
         result *= 2.
         # extra factors because the trispectrum of Q and U are white,
         # so the trispectrum EBEB isn't white:
         result *= l1**2 * np.cos(2.*phi1)
#         result *= l1**2 * (np.cos(phi1)**2 - np.sin(phi1)**2)
         result *= 2. * (L - l1 * np.cos(phi1)) * (- l1 * np.sin(phi1))
#         result *= 2. * l2 * np.cos(phi2) * l2 * np.sin(phi2)
         return result
      
      # if first time, initialize integrator
      if not hasattr(self.relativeNoiseWhiteTrispec_EB.__func__, "integ"):
         self.relativeNoiseWhiteTrispec_EB.__func__.integ = vegas.Integrator([[np.log(self.CMB.lMin), np.log(self.CMB.lMaxP)], [0., np.pi]])
         self.relativeNoiseWhiteTrispec_EB.integ(integrand, nitn=10, neval=10000)
#      result = self.relativeNoiseWhiteTrispec_EB.integ(integrand, nitn=50, neval=5000)
      result = self.relativeNoiseWhiteTrispec_EB.integ(integrand, nitn=10, neval=1000)

      result = result.mean**2
      result *= self.fN_phi_EB(L)
      # extra factor 2, due to conversion from Q/U trispectra to EBEB trispectrum
      result *= 2.
      if not np.isfinite(result):
         result = 0.
      return result


   def saveRelativeNoiseWhiteTrispec(self):
      path = self.directory+"/relative_noise_white_trispectrum.txt"
      data = np.zeros((self.Nl, 3))
      data[:,0] = self.L.copy()
      # parallelize the integral evaluations
      pool = Pool(ncpus=self.nProc)
      print "relative noise from trispectrum: TT"
      data[:,1] = np.array(pool.map(self.relativeNoiseWhiteTrispec_TT, self.L))
      np.savetxt(path, data)
      print "relative noise from trispectrum: EB"
      data[:,2] = np.array(pool.map(self.relativeNoiseWhiteTrispec_EB, self.L))
      np.savetxt(path, data)


   def loadRelativeNoiseWhiteTrispec(self):
      path = self.directory+"/relative_noise_white_trispectrum.txt"
      data = np.genfromtxt(path)
      self.fRelativeNoiseWhiteTrispec_TT = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      self.fRelativeNoiseWhiteTrispec_EB = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=False, fill_value=0.)


   def plotRelativeNoiseWhiteTrispec(self):
      TT = self.fRelativeNoiseWhiteTrispec_TT(self.L)
      EB = self.fRelativeNoiseWhiteTrispec_EB(self.L)
      N_TT_G = self.fN_k_TT(self.L)
      N_EB_G = self.fN_k_EB(self.L)
      
      # Q and U trispectra from polarized point sources, in (muK)^4 sr^-3,
      # according to CMBPol white paper, Eq 20-22,
      # for a flux cut of 5mJy instead of 200mJy
      Trispec = 3.5e-20
      
      print "TT"
      print TT
      print "EB"
      print EB

      # conversion factor
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.loglog(self.L, TT, c='k', label=r'TT')
      ax.loglog(self.L, EB, c='b', label=r'EB')
      #
      ax.legend(loc=1)
      #ax.set_ylim((1.e-4, 1.))
      ax.set_xlabel(r'$L$')
      ax.set_ylabel(r'$N^{0, \text{NG}} / N^{0, \text{G}} / \mathcal{T}$')
      
      # compare noises
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
#      ax.loglog(self.L, N_TT_G, 'k--', label=r'TT G')
#      ax.loglog(self.L, TT * N_TT_G, 'k--', label=r'TT PS')
      #
      ax.loglog(self.L, N_EB_G, 'b', label=r'EB')
      ax.loglog(self.L, EB * N_EB_G * Trispec, 'b--', label=r'EB PS')
      #
      ax.legend(loc=1)
      #ax.set_ylim((1.e-4, 1.))
      ax.set_xlabel(r'$L$')
      ax.set_ylabel(r'N^{0 \; \kappa}$')

      plt.show()
































































