'''
Created on Mar 1, 2013

@author: kai.cui

This is the main module to run that calls all other models
This module takes in one argument indicating whether krigging
interpolation should be done: 1: yes. others:N
-----------------------------------------------------------
input: read in the input of for correlated sampling. List of
    inputs are shown in the input module.
subset_correlation_sampling: sampling on subset crude VRG.
krigging_interpolation: interpolate to fine VRG via krigging.
output: save the grid-point level losses to files or database
-----------------------------------------------------------

'''
if __name__ == "__main__":
    import sys
    import scipy as sp, scipy.stats as sps, numpy as np
    import middleware_functions_internal_test as mf
    from copy import deepcopy
    import pandas as pd
    import time
    import math
    import datetime as dt

    '''-----------------------------------------
       READ IN THE INPUTS
    -----------------------------------------'''
    from data_input import alpha, beta, coords,Sigma_vul, spatial_par, N_event,Do
    N_loc=coords.shape[0]
    N_vul=Sigma_vul.shape[0]
    N_event=N_event.ix[0,0]

    
    '''-----------------------------------------
       SAMPLE CRUDE GRID POINTS
    -----------------------------------------'''
        
    corr_spatial=np.exp(-spatial_par.ix[0,0]*Do);
    # Cholesky decomposition for spatial and structure correlation matrix separately
    corr_spatial_cholesky=np.linalg.cholesky(corr_spatial)
    corr_vul_cholesky=np.linalg.cholesky(Sigma_vul)
    
    sample_normal_loc=mf.corr_sampling(corr_spatial_cholesky, corr_vul_cholesky, N_event, N_loc, N_vul)
    sample_sev_loc=sps.norm.cdf(sample_normal_loc)
    #sample_beta_loc=sps.beta.ppf(sample_sev_loc,np.tile(alpha[:(N_loc*N_vul)],(1,N_event)),np.tile(beta[:(N_loc*N_vul)],(1,N_event)))

    
    


       
    '''-----------------------------------------
       INTERPOLATE TO FINE GRID POINTS IF NEEDED
    ------------------------
     '''
    
    if len(sys.argv)<2:
        
        #np.savetxt("./output/sample_normal_krigging.txt",sample_normal_krigging)
        #np.savetxt('./output/sample_sev_complete.csv', np.reshape(sample_sev_loc, (-1,1), order='F'),fmt='%.4g')
        #np.savetxt("./output/sample_beta_complete.txt",sample_beta_loc)


        print '''
                 ======================================
                       NO INTERPOLATION PERFORMED
                 ======================================

              '''
        
    else:

        from data_input import pcoords, Dop
        N_loc_krigging=pcoords.shape[0]
        corr_spatial_cross=np.exp(-spatial_par.ix[0,0]*Dop)

        # calculating the inverse of spatial correlation matrix via cholesky decomposition
        corr_spatial_cholesky_inv=sp.linalg.solve_triangular(corr_spatial_cholesky, np.identity(N_loc),lower=True)
        corr_spatial_inv=np.dot((corr_spatial_cholesky_inv).T, corr_spatial_cholesky_inv)

        corr_vul_cholesky_inv=sp.linalg.solve_triangular(corr_vul_cholesky, np.identity(N_vul),lower=True)

        '''
        # This deepcopy line is needed if the function uses kronmult function instead of kronmult2
        #sample_normal_krigging=deepcopy(sample_normal_loc)
        #mf.krigging(sample_normal_krigging, corr_spatial_inv, corr_spatial_cross,N_vul)
        '''
        sample_sev_krigging=np.zeros((N_loc_krigging*N_vul, N_event))

        N_split=int(math.ceil(N_loc_krigging/1000.0))  #same as math.ceil(N_loc*N_vul/1000.0)
        N_split_step=1000*N_vul;

        '''
        #file_write=open('./output/sample_sev_complete.npy','wb+',100000000)
        file_write=open('./output/sample_sev_complete.csv','a',1000000)
        
        for i in xrange(0,N_split):
            A=np.dot(corr_spatial_cross[:,(1000*i):min(1000*(i+1),N_loc_krigging)].T, corr_spatial_inv)
            
            N_sub=min(N_split_step,N_loc_krigging*N_vul-N_split_step*i)
            sample_sev_krigging=mf.krigging(sample_normal_loc, A, corr_spatial_cross[:,(1000*i):min(1000*(i+1),N_loc_krigging)], corr_spatial_cholesky_inv,N_sub,N_vul,N_event)
            
            sample_sev_krigging=sps.norm.cdf(sample_sev_krigging)

            #np.save(file_write,np.reshape(np.vstack((sample_sev_loc,sample_sev_krigging)),(-1,1)))
            np.savetxt(file_write,sample_sev_krigging,fmt='%.4g')



        #file_write=open('./output/sample_sev_complete.npy','ab',10000000)
        file_runtime=open('./output/runtime_log.txt','a+')
        start=time.time()
        
        for j in xrange(0,1):
            sample_normal_loc=mf.corr_sampling(Do, corr_spatial_cholesky, corr_vul_cholesky, N_event, N_loc, N_vul)
            sample_sev_loc=sps.norm.cdf(sample_normal_loc)
            
            #sample_sev_krigging=np.zeros(N_loc_krigging*N_vul, N_event)
            for i in xrange(0,N_split):
                A=np.dot(corr_spatial_cross[:,(1000*i):min(1000*(i+1),N_loc_krigging)].T, corr_spatial_inv)
                N_sub=min(N_split_step,N_loc_krigging*N_vul-N_split_step*i)
                sample_sev_krigging=mf.krigging(sample_normal_loc, A, corr_spatial_cross[:,(1000*i):min(1000*(i+1),N_loc_krigging)], corr_spatial_cholesky_inv,N_sub,N_vul,N_event)
                #np.save(file_write,np.reshape(np.vstack((sample_sev_loc,sample_sev_krigging)),(-1,1)))
                np.savetxt(file_write,np.reshape(np.vstack((sample_sev_loc,sample_sev_krigging)),(-1,1)),fmt='%.4g')

            np.savetxt(file_runtime, [time.time()-start])

            
        file_write.close()
        file_runtime.close()

        print time.time()-start
        '''
        

        start1=time.time()
        for i in xrange(0,N_split):
            A=np.dot(corr_spatial_cross[:,(1000*i):min(1000*(i+1),N_loc_krigging)].T, corr_spatial_inv)
            N_sub=min(N_split_step,N_loc_krigging*N_vul-N_split_step*i)
            sample_sev_krigging[(N_split_step*i):min(N_split_step*(i+1),N_loc_krigging*N_vul),:]=mf.krigging(sample_normal_loc, A, corr_spatial_cross[:,(1000*i):min(1000*(i+1),N_loc_krigging)], corr_spatial_cholesky_inv,N_sub,N_vul,N_event)

        



        '''
        ------------
        Transform to uniforma and beta
        '''

        sample_sev_krigging=sps.norm.cdf(sample_sev_krigging)
        #file_write=open('./output/sample_sev_complete.npy','rb+',100000000)
        file_write=open('./output/sample_sev_complete000.csv','a+',1000000)
        file_log=open('./output/Log_'+(dt.datetime.now()).strftime("%y%m%d%H%M%S")+'.txt','a+')
        
       
        #np.savetxt(file_write,np.vstack((sample_sev_loc,sample_sev_krigging)).T,fmt='%.4g')
        #np.savetxt("./output/sample_sev_complete.csv",np.reshape(np.vstack((sample_sev_loc,sample_sev_krigging)),(-1,1)),fmt='%.4g')
        np.savetxt(file_write,np.reshape(np.vstack((sample_sev_loc,sample_sev_krigging)),(-1,1), order='F'),fmt='%.4g')
        count=N_event
        
        start=time.time()
        for j in xrange(0,1):
            sample_normal_loc=mf.corr_sampling(Do, corr_spatial_cholesky, corr_vul_cholesky, N_event, N_loc, N_vul)
            sample_sev_loc=sps.norm.cdf(sample_normal_loc)
            
            #sample_sev_krigging=np.zeros(N_loc_krigging*N_vul, N_event)
            for i in xrange(0,N_split):
                A=np.dot(corr_spatial_cross[:,(1000*i):min(1000*(i+1),N_loc_krigging)].T, corr_spatial_inv)
                N_sub=min(N_split_step,N_loc_krigging*N_vul-N_split_step*i)
                sample_sev_krigging[(N_split_step*i):min(N_split_step*(i+1),N_loc_krigging*N_vul),:]=mf.krigging(sample_normal_loc, A, corr_spatial_cross[:,(1000*i):min(1000*(i+1),N_loc_krigging)], corr_spatial_cholesky_inv,N_sub,N_vul,N_event)
                
            sample_sev_krigging=sps.norm.cdf(sample_sev_krigging)

            #np.savetxt(file_write,np.vstack((sample_sev_loc,sample_sev_krigging)).T,fmt='%.4g')
            #np.savetxt(file_write,np.reshape(np.vstack((sample_sev_loc,sample_sev_krigging)),(-1,1)),fmt='%.4g')
            #np.save(file_write,np.reshape(np.vstack((sample_sev_loc,sample_sev_krigging)),(-1,1)))
            np.savetxt(file_write,np.reshape(np.vstack((sample_sev_loc,sample_sev_krigging)),(-1,1), order='F'),fmt='%.4g')
            np.savetxt(file_log, zip([count],[time.time()-start]))

            
            if count%500==0:
                file_write.close()
                file_write=open('./output/sample_sev_complete'+str(count)+'.csv','a+',1000000)

            print count

                
            count=count+N_event
            
                
            
        file_write.close()
        file_log.close()

        print time.time()-start
       

