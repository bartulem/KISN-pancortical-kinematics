"""
Created on Mon Jan 31 11:07:27 2022

@author: claudia

"""
#IO
import sys
import os
import csv
import pickle

#Math
import numpy as np 
import scipy 
from scipy.interpolate import LSQUnivariateSpline 
import scipy.io
import math
import vg

#Plot
import matplotlib.pyplot as plt
from matplotlib import cm

#Time-frequency
import pycwt as wavelet
from pycwt.helpers import find 

#Machine learning
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import manifold
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

#Extra
import time

class BehaviouralDecomposition():
    """
    A class used to perform behavioural clustering and classification

    Attributes
    ----------
    
    training_data_file : str
        fullpath to the raw data .pkl file used for training
    test_data_file : str
        fullpath to the test data .pkl file with raw postural features to be 
        classified
    training_data_labels_and_PCA_file : str
        fullpath to the .pkl file of labels and PCA features to be used for 
        classification
    splines_order : int
        order of splines to use for detrending the postural time-series
    dspline : int
        distance between splines knots in number of time bins
    captureframerate : float
        tracking frequency (Hz)
    frequency_upper_bound : float
        upper bound for the frequency (Hz) range used in the time-frequency 
        decomposition 
    num_frequencies : int
        number of frequencies for spectral decomposition
    dj : float
        spacing in logarithmic scale between scales (inverse of frequency) for
        spectral decomposition
    outputdir : str
        fullpath to output directory
    

    Methods
    -------
    
    time_frequency_analysis()
    
    PCA():

    TSNE_embedding():
        
    wathershed_segmentation():
        
    classification():
    
    """
    
    def __init__(self):

        self.training_data_files = None
        self.testing_data_file = None
        self.test_data_file = None
        self.test_data_file  = dict()
        self.training_data_labels_and_PCA = dict()
        self.test_data_time_frequencied = dict()
        self.splines_order = 3
        self.dspline = 120
        self.captureframerate = 120
        self.frequency_upper_bound = 20
        self.num_frequencies=18
        self.dj=1./3
        self.outputdir = ''
        self.s_max=1./self.frequency_upper_bound*2**(self.num_frequencies*self.dj)
        self.trainS = None
        self.trainPCA = None
        self.trainsubPCA = None
        self.eigenvectors = None
        self.pervals = [20, 40, 70, 100, 200,300]
        self.widths=[0.1,1.,2.]
        self.dict_labels = None
        self.chosen_width=1.
        self.chosen_perplexity=200
        self.testS = None
        self.SKIPPING=int(self.captureframerate)
        self.features_list=['exp0','exp1','exp2','speed2','BackPitch','BackAzimuth',
                            'NeckElevation']
        self.N_features = len(self.features_list)
    

    def time_frequency_analysis(self,plot_inputvariables_dist=False,
                                plot_detrendedvariables=False,
                                plot_inversetransform=False,
                                plot_power_hist=False,
                                plot_time_frequency=False,
                                results_dir=None,mother=wavelet.Morlet(6),
                                SIGLEV=0.95,training=False):
    
        '''Perform time-frequency analysis with parameteers specified as attributes 
        of self. Does hypotesis testing on power for each behavioural covariate
        using a 1st order autoregressive model and plots intermediate results 
        for detrending 
       
        '''

    
        ## Sanity checks 
        
        if results_dir==None:
            results_dir=self.outputdir
        
        if  self.frequency_upper_bound>self.captureframerate/2.:
            print('The upper bound of frequencies for spectral decomposition '
                  'is higher than the Nyqvist frequency. Set frequency_upper_bound '
                  'to a value smaller than %4.1f'%(self.captureframerate/2.))
        
        x=input('By the number of scales and resolution the lowest frequency in '  
              'the spectral decomposition will be %3.3f. Check compatibility ' 
              'with detrending. Continue? y/n'%(1./self.s_max))
        if not x=='y':
            return
        
        ## Load tracking data
        
        print(self.testing_data_file)
        
        if training==True:
            list_raw_data=self.training_data_files
            S_beh_all=np.zeros((1,(self.num_frequencies+1)*self.N_features))
        else:
            list_raw_data=[self.testing_data_file]
        
        print(list_raw_data)
        
        for fl in list_raw_data:
        
            try:  
                with open(fl,'rb') as infile:
                    covariates = pickle.load(infile)
                infile.close()
            except:
                print("Raw data file should be a .pkl file")
                return
            
            cov_keys=covariates.keys()
            for n in cov_keys:
                if n!='file_info':
                    print(n+' ;Size='+str(np.shape(np.array(covariates[n]))))
                    
            # Extract relevant tracked features
            
            relevant_features=['framerate','speeds','ego3_rotm','back_ang',
                               'sorted_point_data']
            
            
            for feat in relevant_features:
                 if feat not in cov_keys: 
                     print(feat+' is missing in tracking data')
                     return
            
            
            
            speeds=np.array(covariates['speeds'])
            rotmat_ego3=np.array(covariates['ego3_rotm'])
            Orig_Time=np.shape(rotmat_ego3)[0]
            
            # Derive features list for time-frequency analysis from tracked
            
           
            self.N_features=int(len(self.features_list))# tot behavioral covariates
            
            ego3q=np.zeros((Orig_Time,3))
            for tt in range(Orig_Time):
                       ego3q[tt,:]=rot2expmap(rotmat_ego3[tt,:,:])
    
            Raw_variables=np.zeros((Orig_Time,len(self.features_list)))
            Raw_variables[:,0:3]=ego3q
            Raw_variables[:,4:6]=covariates['back_ang']
            Raw_variables[:,3]=speeds[:,2]
            Raw_variables[:,6]=covariates['sorted_point_data'][:,4,2]
            
            # z-score each feature 
            
            data=np.zeros((Orig_Time,len(self.features_list)))
            for n in range(len(self.features_list)):
                    new_var=Raw_variables[:,n]
                    std_cov=np.nanstd(new_var)
                    mean_cov=np.nanmean(new_var)
                    new_var=(new_var-mean_cov)#/std_cov
                    data[:,n]=new_var
                    
            # Remove NaNs
            
            temp_indices=np.array(list(range(Orig_Time)))
            used_indices=temp_indices[~np.isnan(data[temp_indices,:]).any(axis=1)]
            data=data[used_indices,:]
            Time=len(used_indices)
            
            # Outputmat initialization
            timefreq = np.zeros((self.N_features,self.num_frequencies,Orig_Time))
            timefreq[:] = np.nan
            
            
            
            ## Spectrogram generation
            
            smootheddata = np.zeros(( self.N_features, Time))
            smooth = np.zeros(( self.N_features, Time))
            iwaves = np.zeros(( self.N_features, Time))
            timefrequencies = []
            t_freq_sig_thr_95 = []
            t_freq_sig_thr_99 = []
            scales = []
            
            splknots = np.arange(self.dspline / 2.0, Time - self.dspline / 2.0 
                                    + 2, self.dspline)
            x=np.array(list(range(Time)))
            t=x/self.captureframerate
            
                
            for feat in range(self.N_features):
                
                varname=self.features_list[feat]
                dat = data[:,feat]
                dat=dat-np.nanmean(dat)
                print('The maximum of the variable '+ varname + ' is ' + 
                      str(np.nanmax(dat[:])))
                print('The minimum of the variable '+ varname + ' is ' + 
                      str(np.nanmin(dat[:])))
                print('The mean of the variable '+ varname + ' is ' + 
                      str(np.nanmean(dat[:])))
                print('The standard deviation of the variable '+ varname + ' is ' + 
                      str(np.nanstd(dat[:])))
                
                if (plot_inputvariables_dist):
                    
                      # Histogram of the feature "feat"
                      
                      fig=plt.figure() 
                      plt.hist(dat,40) 
                      plt.yscale('log') 
                      plt.xlabel('z-scored '+varname)
                      plt.ylabel('# time bins')
                      plt.title('Distribution %s'%varname)
                      resname=results_dir+'Distribution_'+varname
                      resname=os.path.join(results_dir,resname)
                      plt.savefig(resname)
                      plt.close(fig)
           
                # Detrending
                
                spl = LSQUnivariateSpline(x=x, y=dat, t=splknots, 
                                          k=self.splines_order) 
                smoothed = spl(x)
                smootheddata[feat, :] = smoothed 
                dat_notrend = dat - smoothed
                std = np.nanstd(dat_notrend)  # Standard deviation
                var = std ** 2  # Variance
                dat_norm = dat_notrend/ std  # Normalized dataset
                 
                if (plot_detrendedvariables):
                    
                    # Plot the data, the spline smoothed data (first subplot) and 
                    # the residual (second subplot)
                      
                    sel=np.array(range(1200))
                    fig=plt.figure()
                    plt.clf()
                    plt.subplot(2,1,1)
                    plt.title('dspline ' + varname + '; timebins %d-%d'%(sel[0],
                                                                         sel[-1]))
                    line1=plt.plot(t[sel],dat[sel],label='original')
                    line2=plt.plot(t[sel],smoothed[sel],label='smoothed')
                    plt.ylabel('z-scored %s'%varname)
                    plt.legend(loc='best')
                    ax=plt.subplot(2,1,2)
                    plt.plot(t[sel],dat_notrend[sel])
                    ax.set_xlabel('time (s) ')
                    ax.set_ylabel('z-scored %s'%varname)
                    plt.legend(['Residual data'])
                    resname=results_dir+'Detrend_%s'%(varname)
                    resname=os.path.join(results_dir,resname)
                    plt.savefig(resname)
                    plt.close(fig)
            
                  
            
                ## Time series analysis with 'mother' wavelet transform
             
                wave,scales,freqs,coi,fft,fftfreqs=wavelet.cwt(dat_norm,
                                                               1./self.captureframerate,
                                                               self.dj/2.,
                                                               1./self.frequency_upper_bound
                                                               ,self.num_frequencies*2-1,
                                                               wavelet=mother )#1./self.frequency_range[1], J, mother) # dt= time step; dj= scale step; 1./self.frequency_range[1]=lower bound of the scale, the default corresponds to the Nyquist frequency?
                iwave = wavelet.icwt(wave, scales, 1./self.captureframerate, self.dj,
                                     mother)* std # inverse wave transform * standard deviation?
                iwaves[feat, :] = iwave
                print('These are the scales in seconds:')
                print(scales)
                
                
                if (plot_inversetransform):
                    # Plot the original data and the inverse wavelet transform in two small intervals
                    fig,axs=plt.subplots(2,1,sharey=True)
                    axs[0].set_title('{}: Wavelet transform ({})'.format(varname,mother.name))
                    axs[0].plot(t[:900],dat_notrend[:900], '-', color='blue')
                    axs[0].plot(t[:900],iwave[:900], '-', color='red')
                    axs[0].set_ylabel('z-scored {}'.format(varname))
                    axs[1].plot(t[:200],dat_notrend[:200], '-', color='blue',label='detrended data')
                    axs[1].plot(t[:200],iwave[:200], '-', color='red',label='inverse transform')
                    axs[1].set_xlabel('time (s)')
                    axs[1].set_ylabel('z-scored {}'.format(varname))
                    axs[1].legend()
                  
                    resname=results_dir+'Wavelet_transform_FirstSeconds_%s'%(varname)
                    resname=os.path.join(results_dir,resname)
                    plt.savefig(resname)
                    plt.close(fig)
            
                # Hypothesis testing against 1st order autoregressive process
                power = (np.abs(wave)) ** 2
                fft_power = np.abs(fft) ** 2
                period = 1 / freqs # freqs are the fourier frequencies corresponding to the scales
                power /= scales[:, None] 
            
                alpha, _, _ = wavelet.ar1(dat_notrend)  # Lag-1 autocorrelation for red noise
                signif, fft_theor = wavelet.significance(1.0, 
                                                         1./self.captureframerate,
                                                         scales, 0, alpha, 
                                                         significance_level=SIGLEV,
                                                         wavelet=mother)
                sig95 = np.ones([1, Time]) * signif[:, None] 
                sig95 = power / sig95 
                glbl_power = power.mean(axis=1) # Mean over time of the power
                dof = Time - scales  # Correction for padding at edges (dof=degrees of freedom)
                glbl_signif, tmp = wavelet.significance(var, 1./self.captureframerate,
                                                        scales, 1, alpha, 
                                                        significance_level=SIGLEV, 
                                                        dof=dof, 
                                                        wavelet=mother)
            
                # Smoothing of power across scales, Torrence and Compo (1998)
                Cdelta = mother.cdelta 
                scale_avg = (scales * np.ones((Time, 1))).transpose() 
                scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
                std_power=np.nanstd(np.ravel(power))
                bin_edges=np.linspace(0.,2.*std_power,40)
                
                
                if (plot_power_hist):
                    fig,axs=plt.subplots(round(self.num_frequencies/3.),1,sharex=True,sharey=True) 
                  
                for i in range(self.num_frequencies):
            
                    sel = find(abs(period-period[2*i]) < 0.05) 
                    if(len(sel)<1): 
                      continue
                  
                    #Scale-averaged wavelet power (average performed in the sel range)
    
                    scale_avg_i = var * self.dj * 1./self.captureframerate / Cdelta *scale_avg[sel, :].sum(axis=0) 
                    
                    if(len(timefrequencies[:])<1): # Initialize results matrices
                      timefrequencies = np.zeros(( len(smootheddata[:,0]), self.num_frequencies, 
                                                  len(smootheddata[0,:]) )) 
                      t_freq_sig_thr_95 = np.zeros(( len(smootheddata[:,0]), self.num_frequencies ))
                      t_freq_sig_thr_99 = np.zeros(( len(smootheddata[:,0]), self.num_frequencies ))
                   
                    timefrequencies[feat,i,:] = scale_avg_i+0.
                    t_freq_sig_thr_95[feat,i], tmp = wavelet.significance(var,1./self.captureframerate, 
                                                                          scales, 2, alpha, significance_level=0.95,
                                                                          dof=[scales[sel[0]], scales[sel[-1]]],
                                                                          wavelet=mother)
                    t_freq_sig_thr_99[feat,i], tmp = wavelet.significance(var, 1./self.captureframerate,
                                                                          scales, 2, alpha, significance_level=0.99,
                                                                          dof=[scales[sel[0]], scales[sel[-1]]], 
                                                                          wavelet=mother)
                    
                    if (plot_power_hist):
                        if(np.mod(2*i,3)==0):
                            idx=int(i/3)
                            axs[idx].hist(power[:,2*i],bin_edges)#scale_avg_i,bin_edges) 
                            axs[idx].set_yscale('log') 
                            axs[idx].set_title('Period %2.3f s'%(scales[2*i]),fontsize=12)
                            
                        
                        
                tf=timefrequencies[feat,:,:] 
                
                print('The maximum of the scale averaged power for %s is %2.8f'%(varname,np.nanmax(tf[:])))
                print('The mean of the scale averaged power for %s is %2.8f'%(varname,np.nanmean(tf[:])))
                print('The std of the scale averaged power for %s is %2.8f'%(varname,np.nanstd(tf[:])))
                    
                if (plot_power_hist):
                # Plot power histogram at every 3rd frequency
                    axs[idx].set_xlabel('power')
                    axs[int(self.num_frequencies/6.)].set_ylabel('# time bins')
                    resname='Power_'+varname
                    resname=os.path.join(results_dir,resname)
                    plt.suptitle(varname)
                    plt.tight_layout()
                    plt.savefig(resname)
                    plt.close(fig)
                
                if (plot_time_frequency):  
                # Plot time-frequency summary
                    self.plt_time_frequency(feat,varname,t,period,power,coi,dat_notrend,
                                         iwave,sig95,var,fft_power,fftfreqs,glbl_signif,
                                         fft_theor,results_dir,mother,glbl_power)
                      
                #Rescaling features
            
                smoo_sig=smootheddata[feat,:]
                mean_smoo=np.nanmean(smoo_sig)
                smoo_sig=smoo_sig-mean_smoo
                std_smoo=np.nanstd(smoo_sig)
                smoo_sig=smoo_sig/std_smoo
                smooth[feat,:]=smoo_sig[np.newaxis,:]
                power_mat=np.sqrt(tf)
                mean_power=np.nanmean(np.ravel(power_mat))
                power_mat=power_mat-mean_power
                power_mat=power_mat/std_smoo
                timefreq[feat,:,used_indices]=np.transpose(power_mat)
    
            S_beh=[]
            used_id_beh=[]
            
            #Concatenating trend data and power spectrum
            
            for t in range(Time):
    
                valsTF = np.ravel(timefreq[:,:,t])
                valsS = np.ravel(smooth[:,t])
                feat_vec=np.append( valsS, valsTF )
                if (sum(np.isnan(np.ravel(feat_vec)))==0):
                        used_id_beh.append(used_indices[t])
                        S_beh.append(feat_vec)
    
                            
                       
            
            S_beh=np.array(S_beh) 
            
            # Save results in .pkl file
            
            if training==True:
                print(np.shape(S_beh))
                S_beh_all=np.append(S_beh_all,S_beh,axis=0)
            else:
                self.testS=S_beh
                
    
            resname=fl[:-4]
            resname=resname+'_time_frequencied.pkl'
            summay_dict={'smootheddata':smootheddata, 'timefrequencies':timefrequencies,
                         'period':period, 'scales':scales, 'iwaves':iwaves, 
                         'timefrequenciessignificancethreshold95':t_freq_sig_thr_95,
                         'timefrequenciessignificancethreshold99':t_freq_sig_thr_99,
                         'used_indices':used_indices,'Orig_Time':Orig_Time,
                         'features_list':self.features_list,'features_behaviour':S_beh}
            output = open(resname, 'wb')
            pickle.dump(summay_dict, output)
            output.close()
            
        if training==True:
            self.trainS=S_beh_all[1:,:]
        


        
    def PCA(self,plot_explained_variance=True):
        
        '''PCA projection of the full feature matrix self.trainS accounting for 95%
        of the variance
        '''
        
        
        pca = PCA(svd_solver='full')
        pca.fit(self.trainS)
        N_allfeatures=np.shape(self.trainS)[1]
        cum_expl_var=np.zeros(N_allfeatures)
        a=0
        for i in range(N_allfeatures):
            cum_expl_var[i]=cum_expl_var[i-1]+pca.explained_variance_ratio_[i]
            if (a==0) and (cum_expl_var[i]>0.95):
                a=1
                thr=i
                print('PCA component %d has %2.3f projection on signal'%(i,np.sqrt(np.dot(pca.components_[i,0:self.N_features],pca.components_[i,0:self.N_features])/np.dot(pca.components_[i,:],pca.components_[i,:]))))
        print('%d components explain more than 95%% of the variance'%thr)
        print('20 components explain %1.3f of the variance'%cum_expl_var[19])
        
        if (plot_explained_variance):
            plt.figure()
            plt.plot(np.arange(np.shape(self.trainS)[1])+1,cum_expl_var,color='k')
            plt.plot(np.array([thr,thr]),np.array([0,1]),'k--')
            plt.xlabel('PCA component')
            plt.ylabel('explained variance')
            plt.title('Time frequency: PCA (%3d comp>95%%)'%thr)
            plt.savefig(self.outputdir+'PCA.png',dpi=500)
            
        # Select # components for > 95% explained variance    
            
        pca = PCA(svd_solver='full',n_components=thr)
            
        self.trainPCA=pca.fit_transform(self.trainS)
        self.eigenvectors=pca.components_

    def TSNE_embedding(self,plot_TSNE_embeddings=False):
        
        '''TSNE embedding by varying the perplexity parameter
        '''
        ## Parameters settings
        
        ## Perplexity parameter (to be tuned)
        print(self.pervals)
        # The perplexity is related to the number of nearest neighbors that is used in
        # other manifold learning algorithms. Larger datasets usually require a 
        # larger perplexity. Consider selecting a value between 5 and 50. 
        # Different values can result in significanlty different results.

        # other TSNE parameters
        n_components = 2 # number of components for the embedding
        lrate = 200
        earlyexag = 12 #The early exaggeration controls how tight natural clusters
        # in the original space are in the embedded space and how much space there
        # will be between them. For larger values, the space between natural 
        # clusters will be larger in the embedded space. The choice of this 
        # parameter is not very critical. If the cost function increases during 
        # initial optimization, the early exaggeration factor or the learning 
        # rate might be too high.
        
        # Perplexity optimization
        
        self.trainsubPCA=self.trainPCA[np.arange(0,np.shape(self.trainPCA)[0],self.SKIPPING),:]
        dic_TSNEs=dict()
        for ppp in self.pervals:
            t0 = time.time()
            tsne = manifold.TSNE(perplexity=ppp, early_exaggeration=earlyexag, 
                                 learning_rate=lrate,n_iter=5000,n_components=n_components,
                                 init='pca', random_state=0)
            Y = tsne.fit_transform(self.trainsubPCA)
            dic_TSNEs['perplexity='+str(ppp)]=Y
            t1 = time.time()
            # Plot embedding
            if (plot_TSNE_embeddings):
                tit = "TSNE (%.2g sec)" % (t1 - t0)
                name = 'maps_TSNE_%03d'%ppp
                fig = plt.figure(34, figsize=(30,30))
                plt.clf()
                plt.scatter(Y[:, 0],Y[:, 1], marker='.',color='lightcoral')
                plt.title(tit)
                plt.savefig('%s.png'%(self.outputdir+name),dpi=500)

            self.trainTSNEs=dic_TSNEs
        
    def wathershed_segmentation(self):
        
        '''Watershed of TSNE as by varying the perplexity parameter in tSNE
        and the width of the gaussian blur on the tSNE embedding. Clustering metrics
        are plotted.
        '''
        
        ## Watershed parameters
        Npixels=60
        n_widths=len(self.widths)# Smoothing widths of gaussian blur
        
        ## Clustering metrics
        n_pervals=len(self.pervals)
        ch_score=np.zeros((n_pervals,n_widths))
        db_score=np.zeros((n_pervals,n_widths))
        silhouette=np.zeros((n_pervals,n_widths))
        Nclusters=np.zeros((n_pervals,n_widths))
        
        # Loop through perplexity parameter for tSNE and gaussian blur width
        c=0
        self.dict_labels=dict()
        for ppp in self.pervals:
            Y=self.trainTSNEs['perplexity='+str(ppp)]
            YY=np.transpose(Y)
            image=fromCOORtoIMAGE(YY,Npixels)
            smoothedimage=np.zeros((Npixels,Npixels,n_widths)) 
            labels =np.zeros((Npixels,Npixels,n_widths)) 
            dict_WS=dict()
            for wid in range(n_widths):
                smoothedimage[:,:,wid], labels[:,:,wid] = getwatershedimage(image,self.widths[wid])
                labels_points=assign_WSpoints_to_clusters(labels[:,:,wid],YY,Npixels)
                ch_score[c,wid]=metrics.calinski_harabasz_score(self.trainsubPCA, labels_points)
                db_score[c,wid]=metrics.davies_bouldin_score(self.trainsubPCA, labels_points)
                silhouette[c,wid]=metrics.silhouette_score(self.trainsubPCA, labels_points)
                Nclusters[c,wid]=np.nanmax(labels_points)+1
                dict_WS['width=%2.2f'%wid]=labels_points
            self.dict_labels['perplexity='+str(ppp)]=dict_WS
            self.plotfigs(ppp,Y,smoothedimage,labels,ch_score[c,:],db_score[c,:],silhouette[c,:], Nclusters[c,:])
            c+=1
        
        # Plot summary metrics for watershed and tSNE parameters
        plt.close('all')    
        fig,ax=plt.subplots(nrows=4,ncols=1,sharex=True)
        for wid in range(n_widths):
            ax[0].plot(self.pervals,ch_score[:,wid],alpha=(wid+1.)/3.,color='k',label='width='+str(wid))
        ax[0].set_title('CH score')
        ax[0].legend()
        
        for wid in range(n_widths):
            ax[1].plot(self.pervals,db_score[:,wid],alpha=(wid+1.)/3.,color='b',label='width='+str(wid))
        ax[1].set_title('DB score')
        
        for wid in range(n_widths):
            ax[2].plot(self.pervals,silhouette[:,wid],alpha=(wid+1.)/3.,color='r',label='width='+str(wid))
        ax[2].set_title('silhouette')
        
        for wid in range(n_widths):
            ax[3].plot(self.pervals,Nclusters[:,wid],alpha=(wid+1.)/3.,color='g',label='width='+str(wid))
        ax[3].set_title('# clusters')
        plt.xlabel('perplexity for TSNE')
        
        plt.tight_layout() 
        plt.savefig('%sMetrics_watershed_inTSNE'%(self.outputdir))
        plt.close(fig)
            
    
    def classification(self):
        
        '''classsification comment
        '''
        
        try:
            if self.testS==None:
                self.testS=self.trainS+0.
        except:
            print('Out-of-sample classified timepoints')
        
        labels_in_time=self.dict_labels['perplexity='+str(self.chosen_perplexity)]['width=%.2f'%(self.chosen_width)]
        numBEH=int(np.max(labels_in_time)+1)
        xvals=np.array(range(numBEH))

        allTIMES=np.array(range(np.shape(self.testS)[0]))
        NOTnantimepoints=np.isfinite(np.sum(self.testS,axis=1)) 
        timepoints_beh=allTIMES[NOTnantimepoints]
        timepoints_beh=timepoints_beh.astype('int')
        matPCA_X=np.dot(self.testS,np.transpose(self.eigenvectors))

        ## Assign behavioral label
        T=np.shape(matPCA_X)[0]
        newlabels=np.zeros(T)
        newlabels[:]=np.nan
        for i in timepoints_beh:
            vec=matPCA_X[i,:]
            dist=scipy.spatial.distance.cdist(self.trainsubPCA,vec[np.newaxis,:], 'euclidean')
            orig = np.argmin(dist) 
            newlabels[i]=labels_in_time[orig]
        
        return newlabels
    
    
# Ancillary functions inside class

## Plot funcs for time_frequency
    def plt_time_frequency(self,feat,varname,t,period,power,coi,dat_notrend,
                                     iwave,sig95,var,fft_power,fftfreqs,glbl_signif,
                                     fft_theor,results_dir,mother,glbl_power):
                  
                  power*=var
                  fig = plt.figure(figsize=(20, 8)) 
            
                  # First sub-plot, the original time series and inverse wavelet
                  # transform.
                  ax = plt.axes([0.1, 0.75, 0.65, 0.2])
                  ax.plot(t, iwave, '-', linewidth=0.5, color=[0.5, 0.5, 0.5],
                          label='inverse transform')
                  ax.plot(t, dat_notrend, 'k', linewidth=0.5,label='original data')
                  ax.set_title('{}: Wavelet inverse transform'.format(varname))
                  ax.set_ylabel(r'z-scored {}'.format(varname))
                  ax.legend()
            
                  # Second sub-plot, the normalized wavelet power spectrum and significance
                  # level contour lines and cone of influece hatched area. Note that period
                  # scale is logarithmic.
                  bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
                  dl=(np.log2(power.max())-np.log2(power.min()))/8.
                  levels=np.arange(np.log2(power.min()),np.log2(power.max())+dl,dl)
                  #levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
                  #max_pow=power.max()
                  #power = power*16./np.max(np.ravel(power)) # rescale the power between 0 and 16???
                  cnf=bx.contourf(t, np.log2(period), np.log2(power), 
                                  levels, extend='both', 
                                  cmap=plt.cm.viridis)
                  cbar=plt.colorbar(cnf)
                  cbar.set_ticks(levels)
                  cbar.set_ticklabels(np.char.mod('%.1e',2**levels))
                  extent = [t.min(), t.max(), 0, max(period)]
                  bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k',
                             linewidths=0.5, extent=extent,legend='95% sign level')
                  bx.fill(np.concatenate([t, t[-1:] + 1./self.captureframerate,
                                          t[-1:] + 1./self.captureframerate, 
                                          t[:1] - 1./self.captureframerate, 
                                          t[:1] - 1./self.captureframerate]), 
                          np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                                          np.log2(period[-1:]), [1e-9]]),  'k', alpha=0.3, hatch='x')
                  bx.set_title(' {}: Wavelet Power Spectrum ({})'.format(varname, mother.name))
                  bx.set_ylabel('Period (1/Fourier freq) (s) - log2 scale')#, units))
                  bx.set_xlabel('Time (s)')
                  #
                  Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
                  bx.set_yticks(np.log2(Yticks))
                  bx.set_yticklabels(Yticks)
            
                  # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
                  # noise spectra. Note that period scale is logarithmic.
                  cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
                  cx.plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc', linewidth=0.4,label='fft')
                  cx.plot(glbl_signif, np.log2(period), 'k--',linewidth=1,label='95% significance')
                  cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc',label='red noise')
                  cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=0.5,label='avg power over time')
                  cx.set_title(' Global Wavelet Spectrum')
                  cx.set_xlabel(r'Power ')#'.format(units))
                  cx.set_xlim([0, glbl_power.max()*var]) #+var
                  cx.set_ylim(np.log2([period.min(), period.max()]))
                  cx.set_yticks(np.log2(Yticks))
                  cx.set_yticklabels(Yticks)
                  #plt.setp(cx.get_yticklabels(), visible=False)
                  cx.legend()
                  
                  resname=results_dir+'Wavelet_Spectrum_{}'.format(varname)#%(int(np.floor(1./self.frequency_range[1]*100000)), whichy)
                  resname=os.path.join(results_dir,resname)
                  plt.savefig(resname,dpi=500)
                  
                  plt.close(fig)


    def plotfigs(self,perp,Y,smoothedimage,labels,ch_score,db_score,silhouette,num_clusters):

        plt.clf()
        fsize=6
        fig = plt.figure(constrained_layout=True)
        gs= fig.add_gridspec(5, len(self.widths))
        ax1 = fig.add_subplot(gs[0:3,0:3])
        ax1.scatter(Y[:, 0], Y[:, 1], marker='.',s=0.1,alpha=0.5,color='lightcoral')
        ax1.axis('square')
        ax1.axis('off')
        #ax[0].axis('tight')
        ax1.set_title('TSNE; perp='+str(perp))
        for wid in range(len(self.widths)):
            ax = fig.add_subplot(gs[3,wid])
            smi=smoothedimage[:,:,wid]
            ax.imshow(np.transpose( smi ),cmap='inferno',origin='lower')
            #aimage.colorbar()
            ax.axis('equal')
            ax.axis('off')
            ax.set_title('smoothed TSNE, widht='+str(self.widths[wid]),fontsize=8)
            ax2 = fig.add_subplot(gs[4,wid])
            lb=labels[:,:,wid]
            ax2.imshow(np.transpose(lb), cmap=plt.cm.nipy_spectral, interpolation='nearest', origin='lower')
            ax2.axis('equal')
            ax2.axis('off')
            ax2.set_title('watershed; \n CH score=%5.2f; DB score=%5.2f;\n silhouette=%5.2f; #clusters=%d'%(ch_score[wid],db_score[wid],silhouette[wid],num_clusters[wid]),fontsize=fsize)  
        
        plt.tight_layout()
        
        
        bsname='TSNE_ppp%d'%perp
        plt.savefig('%s.png'%(self.outputdir+bsname))
        plt.close(fig)
  
    


# Ancillary functions outside class

def rot2expmap(rot_mat):
    """ Converts rotation matrix to quaternions """
    
    expmap=np.zeros(3)
    if np.sum(np.isfinite(rot_mat))<9:
        expmap[:]=np.nan
    else:
        d = rot_mat - np.transpose(rot_mat)
        if scipy.linalg.norm(d)>0.01:
            r0 = np.zeros(3)
            r0[0] = -d[1, 2]
            r0[1] = d[0, 2]
            r0[2] = -d[0, 1]
            sintheta = scipy.linalg.norm(r0) / 2.
            costheta = (np.trace(rot_mat) - 1.) / 2.
            theta = math.atan2(sintheta, costheta)
            r0=r0/scipy.linalg.norm(r0)
        else:
            eigval,eigvec=scipy.linalg.eig(rot_mat)
            eigval=np.real(eigval)
            r_idx=np.argmin(np.abs(eigval-1))
            r0=np.real(eigvec[:,r_idx])
            theta=vg.angle(r0,np.dot(rot_mat,r0))
        
        theta = np.fmod(theta + 2 * math.pi, 2 * math.pi)  # Remainder after division (modulo operation)
        if theta > math.pi:
            theta = 2 * math.pi - theta
            r0 = -r0
        expmap= r0 * theta
    
    return expmap

def fromCOORtoIMAGE(Y,Npixels):
    mx = min(Y[0,:])
    Mx = max(Y[0,:])
    my = min(Y[1,:])
    My = max(Y[1,:])

    dy = My-my
    my = my-0.05*dy
    My = My+0.05*dy

    dx = Mx-mx
    mx = mx-0.05*dx
    Mx = Mx+0.05*dx

    NN = Npixels
    dy = (My-my)/float(NN)
    dx = (Mx-mx)/float(NN)
    image = np.zeros((NN,NN))

    for i in range(NN-1):
        whichesX = (Y[0,:]>=dx*i+mx)*(Y[0,:]<(dx*(i+1)+mx))
        if(sum(whichesX)<1):
            continue
        for j in range(NN):
            whichesY = (Y[1,:]>=dy*j+my)*(Y[1,:]<(dy*(j+1)+my))
            if(sum(whichesY)<1):
                continue
            image[i,j] = sum(whichesX*whichesY)

    return image

def getwatershedimage(image, stdev):
    
    smoothedimage = cv2.GaussianBlur(image,(0,0), stdev)
    local_maxi = peak_local_max(smoothedimage, indices=False) #, footprint=np.ones((3, 3)))
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-smoothedimage, markers, mask=smoothedimage)
  
    return smoothedimage, labels

def assign_WSpoints_to_clusters(pixel_labels,Y,Npixels):
    mx = min(Y[0,:])
    Mx = max(Y[0,:])
    my = min(Y[1,:])
    My = max(Y[1,:])

    dy = My-my
    my = my-0.05*dy
    My = My+0.05*dy

    dx = Mx-mx
    mx = mx-0.05*dx
    Mx = Mx+0.05*dx

    NN = Npixels
    dy = (My-my)/float(NN)
    dx = (Mx-mx)/float(NN)
    #print(np.shape(Y)[1])
    new_labels_pts = np.zeros(np.shape(Y)[1])
    #print(np.shape(new_labels_pts))
    #print(np.shape(pixel_labels))

    for i in range(NN-1):
        whichesX = (Y[0,:]>=dx*i+mx)*(Y[0,:]<(dx*(i+1)+mx))
        if(sum(whichesX)<1):
            continue
        for j in range(NN):
            whichesY = (Y[1,:]>=dy*j+my)*(Y[1,:]<(dy*(j+1)+my))
            if(sum(whichesY)<1):
                continue
            whiches= whichesX*whichesY
            new_labels_pts[whiches]=pixel_labels[i,j]
            
    return new_labels_pts