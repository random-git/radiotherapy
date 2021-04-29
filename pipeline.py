#!/usr/bin/env python
# coding: utf-8




#Author: Cong Zhu
import umap
from sklearn.datasets import load_digits

import pandas as pd
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import os
import seaborn as sns
from xgboost import plot_importance
from matplotlib import pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from statistics import mean
from pandas import DataFrame
import matplotlib.pyplot as plt


# In[239]:


os.chdir("...")
date = "date"
seed = 222
n_try = 2
top20vs = '....'.format(date, seed, n_try)
selected_features = '....'.format(date,seed,n_try)
optimized_parms = '....'.format(date, seed,n_try)


# # Import data (clustered created; check R and notebook)




df_clusters = pd.read_excel('....')





cluster_list = ["cluster_2D_minsamp1_mincluster100"]


model_seed = 222





params = {'max_depth': 9,
          'gamma': 1,
          'n_estimators': 100,
          'subsample': 0.8,
          'eta': 0.01,
          'min_child_weight':1,
          "colsample_bytree":1,
          'colsample_bylevel':1,
          'colsample_bynode ':1,
          'objective':'reg:logistic',
          'eval_metric': 'auc'}



for n in [0,1]:
    for i in [0,1]:
        df = df_clusters[(df_clusters[cluster_method]==n) & (df_clusters["proton"]==i)]

        X = df.iloc[:,2:-2]
        y=df["G4RIL"]
        ratio_try = float(np.sum(y == 0)) / np.sum(y == 1)
        
        print(n,i)
        print(ratio_try)
        print(np.sum(y == 1)/y.shape[0])
        print("-------")


def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio_try = float(np.sum(label == 0)) / np.sum(label == 1)
    if ratio_try > 3:
        ratio = 2
    else:
        ratio = 1
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)



vs_results = {}
for cluster_method in cluster_list:
    n_cluster = df_clusters[cluster_method].unique()
    
    cluster = {}
    for n in n_cluster:
        rt_mod = {}
        for i, mod in zip(range(2),["photon","proton"]):
            print(cluster_method+": "+"cluster"+str(n)+ "- " +mod)

            df = df_clusters[(df_clusters[cluster_method]==n) & (df_clusters["proton"]==i)]

            X = df.iloc[:,2:-2]
            y=df["G4RIL"]
            ratio_try = float(np.sum(y == 0)) / np.sum(y == 1)
            if ratio_try>3:
                ratio = 2
            else:
                ratio = 1.0
            params['scale_pos_weight'] = ratio
            
        
            
            '''top 20 important variables according to SHAP'''
            model_temp = XGBClassifier(**params)
            model_temp.fit(X, y)
            
            shap_values = shap.TreeExplainer(model_temp).shap_values(X)
            shap.summary_plot(shap_values, X,max_display=20)
            
            vals= np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['var_name','importance'])
            feature_importance.sort_values(by=['importance'], ascending=False,inplace=True)
            feature_selected = feature_importance.head(20)

            cv_logloss={}
            
            cv_logloss.update({"fs20":feature_selected})
            '''experiment with different number of feastures, selection based on the ranking of importannce'''
            for j in range(5,20):
                X_selected = X[feature_selected["var_name"][0:j]]

                dtrain = xgb.DMatrix(X_selected, label = y)
                
            
            
                cv = xgb.cv(params,
                    dtrain,
                    num_boost_round=200,
                    seed=model_seed,
                    folds=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=model_seed),
                    metrics={'auc'},
                    early_stopping_rounds=30,
                    fpreproc = fpreproc)
                
                logloss = round(cv['test-auc-mean'].tail(1).ravel()[0],3)
                cv_logloss.update({j:logloss})
            
            rt_mod.update({mod:cv_logloss})
            
        cluster.update({("cluster" + str(n)):rt_mod})
                          
    vs_results.update({cluster_method:cluster})





writer = pd.ExcelWriter(top20vs)
for cluster_method in vs_results.keys():
    for cluster_number in vs_results[cluster_method].keys():
        for rt_mod in vs_results[cluster_method][cluster_number].keys():
            cluster_info = cluster_method.replace("_minsamp1","").replace            ("cluster_","").replace("mincluster","mincl")            + cluster_number.replace("cluster","_c")+"-"+rt_mod
   
            print(cluster_info)
            output_df = DataFrame(vs_results[cluster_method][cluster_number][rt_mod]['fs20'])
            output_df.to_excel(writer, sheet_name=cluster_info)
writer.save()





import copy




vs_results2 = copy.deepcopy(vs_results)


# In[252]:


vs_results2['cluster_2D_minsamp1_mincluster100'].keys()


# In[253]:


vs_results2





writer = pd.ExcelWriter(selected_features)
for cluster_method in vs_results2.keys():
    for cluster_number in vs_results2[cluster_method].keys():
        for rt_mod in vs_results2[cluster_method][cluster_number].keys():
            
            cluster_info = cluster_method.replace("_minsamp1","").replace            ("cluster_","").replace("mincluster","mincl")            + cluster_number.replace("cluster","_c")+"-"+rt_mod
   
            print(cluster_info)
            feature20 = vs_results2[cluster_method][cluster_number][rt_mod]['fs20']
            
            vs_results3 = copy.deepcopy(vs_results2[cluster_method][cluster_number][rt_mod])
            del vs_results3['fs20']
            
            var_index = max(vs_results3, 
                            key = vs_results3.get)
            
            feature_selected = feature20[0:var_index]
            print("Index:"+str(var_index))
            
            output_df = DataFrame(feature_selected)
            output_df.to_excel(writer, sheet_name=cluster_info)
writer.save()





output_df


# ### Bayesian opitmization


xlsx_names = ['2D_mincl100_c1-photon', '2D_mincl100_c1-proton', '2D_mincl100_c0-photon', '2D_mincl100_c0-proton']
c_name_list = ['2D_mincl100_c1_photon', '2D_mincl100_c1_proton', '2D_mincl100_c0_photon', '2D_mincl100_c0_proton']
fs_list = {}
for cluster_name1, cluster_name2 in zip(c_name_list,xlsx_names):
    fs = cluster_name1+"_fs" 
    fs = pd.read_excel(selected_features,sheet_name = cluster_name2)
    fs_list.update({cluster_name1:fs})




fs_list




bo_result={}
for cluster_method in cluster_list:
    cluster = {}
    for cluster_number in vs_results[cluster_method].keys():
        mod = {}
        for rt_mod in vs_results[cluster_method][cluster_number].keys():
            print(cluster_method+":",cluster_number+"-",rt_mod)
            
            if rt_mod=="proton":
                i = 1
            else:
                i = 0
                
            n = int(cluster_number[-1])

            df = df_clusters[(df_clusters[cluster_method]==n) & (df_clusters["proton"]==i)]
            X = df.iloc[:,2:-2]
            y=df["G4RIL"]
            
            '''use selected features for optimization'''
            fs_c_name = "2D_mincl100_c"+str(n)+"_"+rt_mod
            var_selected = fs_list[fs_c_name]["var_name"]
            
            
            print(var_selected)
           
            
            X_selected = X[var_selected]
            dtrain = xgb.DMatrix(X_selected, label = y)
            
            def bo_tune_xgb(max_depth, gamma, n_estimators ,subsample,eta,
                           colsample_bytree,
                           colsample_bylevel,
                            colsample_bynode,min_child_weight):
                params = {'max_depth': int(max_depth),
                          'gamma': gamma,
                          'n_estimators': int(n_estimators),
                          'subsample': subsample,
                          'eta': eta,
                          'min_child_weight':min_child_weight,
                          'colsample_bytree':colsample_bytree,
                          'colsample_bylevel':colsample_bylevel,
                          'colsample_bynode':colsample_bynode,
                              
                          'eval_metric': 'auc'}
               
                cv_result = xgb.cv(params = params, dtrain = dtrain, num_boost_round=200,seed=model_seed,
                    folds=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=model_seed),
                                  early_stopping_rounds=30,
                                  fpreproc=fpreproc)
                
                
               

                return (cv_result['test-auc-mean'].iloc[-1])
         
            xgb_bo = BayesianOptimization(bo_tune_xgb, 
                                            {'max_depth': (3, 15),
                                             'gamma': (0, 2),
                                             'n_estimators':(50,250),
                                             'subsample':(0.3,1),
                                             'eta':(0.05,0.35),
                                             'min_child_weight':(0,4),
                                              'colsample_bytree':(0.3,1),
                                              'colsample_bylevel':(0.3,1),
                                              'colsample_bynode':(0.3,1)
                                            
                                            },
                                         random_state=model_seed)
            
            xgb_bo.maximize(n_iter=60, init_points=5)
            
            params = xgb_bo.max['params']
            
            mod.update({rt_mod:params})
            
        cluster.update({cluster_number:mod})
    
    bo_result.update({cluster_method:cluster})


# ### Show results of BO model


auc_results = []
for cluster_method in bo_result.keys():
    for cluster_number in bo_result[cluster_method].keys():
        for rt_mod in bo_result[cluster_method][cluster_number].keys():
            if rt_mod=="proton":
                i = 1
            else:
                i = 0
                
            n = int(cluster_number[-1])
            
                     
            df = df_clusters[(df_clusters[cluster_method]==n) & (df_clusters["proton"]==i)]
            X = df.iloc[:,2:-2]
            y=df["G4RIL"]
            
            feature20 = vs_results2[cluster_method][cluster_number][rt_mod]['fs20']
 
            vs_results3 = copy.deepcopy(vs_results2[cluster_method][cluster_number][rt_mod])
            del vs_results3['fs20']
            
            var_index = max(vs_results3, 
                            key = vs_results3.get)
            
            feature_selected = feature20[0:var_index]['var_name']
            
            
            X_selected = X[feature_selected]
            dtrain = xgb.DMatrix(X_selected, label = y)
            
            '''select optimized hyperparameters'''
            parms_selected = bo_result[cluster_method][cluster_number][rt_mod]
            
            '''convert max_depth and n_estiamtors to integer'''
            parms_selected['max_depth']= int(parms_selected['max_depth'])
            parms_selected['n_estimators']= int(parms_selected['n_estimators'])
            parms_selected['eval_metric']= 'auc'

            '''repeated stratified k-fold CV'''

  
            results = xgb.cv(params = parms_selected, dtrain = dtrain, num_boost_round=200,seed=model_seed,
                    folds=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=model_seed),
                                  early_stopping_rounds=30,
                                metrics={'auc'},
                                  fpreproc=fpreproc)
            
            
            auc_mean = round(results['test-auc-mean'].tail(1).ravel()[0],3)
            auc_sd = round(results['test-auc-std'].tail(1).ravel()[0],4)
            
            
            cluster_info = cluster_method.replace("_minsamp1","")+"_"+ cluster_number+"-"+rt_mod
            
            print(cluster_info)
            print("AUC: %.2f%% (%.2f%%)" % (auc_mean, auc_sd))

            auc_results_tmp = [cluster_info, auc_mean, auc_sd]
            auc_results +=[auc_results_tmp]
            print("-------------------------------------------")






for cluster_method in bo_result.keys():
    for cluster_number in bo_result[cluster_method].keys():
        for rt_mod in bo_result[cluster_method][cluster_number].keys():
            cluster_info = cluster_method.replace("_minsamp1","")+":"+ cluster_number+"-"+rt_mod
            print(cluster_info)
            print (bo_result[cluster_method][cluster_number][rt_mod])
            print("----------------------------------------------")





date = "..."
seed = 222
n_try = 2
optimized_parms = '....'.format(date, seed,n_try)
model_performance = ".....".format(date, seed,n_try)




writer = pd.ExcelWriter(optimized_parms)
for cluster_method in bo_result.keys():
    for cluster_number in bo_result[cluster_method].keys():
        for rt_mod in bo_result[cluster_method][cluster_number].keys():
            
            cluster_info = cluster_method.replace("_minsamp1","").replace            ("cluster_","").replace("mincluster","mincl")            + cluster_number.replace("cluster","_c")+"-"+rt_mod
            
            output_df = bo_result[cluster_method][cluster_number][rt_mod]
            output_df = pd.DataFrame.from_dict(output_df, orient='index',columns = ['value'])
            
            output_df = DataFrame(output_df)
            output_df.to_excel(writer, sheet_name=cluster_info)
writer.save()   




auc_results2 = DataFrame(auc_results,columns = ["Cluster-Name","Mean AUC","SD AUC"])





auc_results2.to_excel(model_performance,sheet_name = "XGBoost")





df = df_clusters[(df_clusters["cluster_2D_minsamp1_mincluster100"]==0)                              & (df_clusters["proton"]==1)]
var_selected = fs_list['2D_mincl100_c0_proton']["var_name"]

X = df[var_selected]
                
y=df["G4RIL"]

dtrain = xgb.DMatrix(X, label = y)
params_temp = bo_result['cluster_2D_minsamp1_mincluster100']['cluster0']['proton']

'''get corresponding optimized variables'''



params_temp2 = { 
                      'colsample_bylevel':params_temp['colsample_bylevel'],
                      'colsample_bynode':params_temp['colsample_bynode'],
                      'colsample_bytree':params_temp['colsample_bytree'],              
                      'eta' : params_temp['eta'],
                      'gamma' : params_temp['gamma'],
                      'max_depth' : int(params_temp['max_depth']),
                      'min_child_weight': params_temp['min_child_weight'],
                      'n_estimators' : int(params_temp['n_estimators']),
                      'subsample' : params_temp['subsample']
            }

ratio_try = float(np.sum(y == 0)) / np.sum(y == 1)
if ratio_try>3:
    ratio = 2
else:
    ratio = 1.0
params_temp2['scale_pos_weight'] = ratio


model2 = XGBClassifier(**params_temp2)
model2.fit(X, y)
shap_values = shap.TreeExplainer(model2).shap_values(X)
vals= np.abs(shap_values).mean(0)


maxdisplay = 25


plt.rcParams["figure.figsize"] = (14, 25)
plt.rcParams.update({'font.size': 40,"font.weight" : "bold",
                 'axes.titlesize':40,"axes.labelweight":"bold"})
shap.summary_plot(shap_values, X, plot_type="bar",max_display=maxdisplay,show=True)

plt.rcParams["figure.figsize"] = (14, 25)
plt.rcParams.update({'font.size': 40,"font.weight" : "bold",
                 'axes.titlesize':40,"axes.labelweight":"bold"})
shap.summary_plot(shap_values, X,max_display=maxdisplay,show=True)


# # Visualizations



date = "..."
seed = 222
n_try = 2
top20vs = '....'.format(date, seed, n_try)
selected_features = '.....'.format(date,seed,n_try)





xl = pd.ExcelFile(selected_features)
xlsx_names = xl.sheet_names

xgb_params = pd.ExcelFile(optimized_parms)
xgb_params_grpname = xgb_params.sheet_names

c_name_list=[]
for cluster_name in xlsx_names:
    cluster_name2 = cluster_name.replace('-',"_")
    c_name_list +=[cluster_name2]
    
fs_list = {}
for cluster_name1, cluster_name2 in zip(c_name_list,xlsx_names):
    fs = cluster_name1+"_fs" 
    fs = pd.read_excel(selected_features,sheet_name = cluster_name2)
    fs_list.update({cluster_name1:fs})





xgbparams_list = {}
for cluster_name1, cluster_name2 in zip(c_name_list,xgb_params_grpname):
    fs = cluster_name1+"_fs" 
    fs = pd.read_excel(optimized_parms,sheet_name = cluster_name2)
    xgbparams_list.update({cluster_name1:fs})





group_list = ['2D_mincl100_c0_photon', '2D_mincl100_c0_proton', 
                '2D_mincl100_c1_photon', '2D_mincl100_c1_proton']




group_list2 = [['2D_mincl100_c0_photon', '2D_mincl100_c0_proton'], 
                ['2D_mincl100_c1_photon', '2D_mincl100_c1_proton']]




class SHAP_vis:
    def __init__(self,cluster_name,vername):
        self.cluster_name = cluster_name
        self.vername = vername
        
    def plot_shap(self):
        shap_dict = {}
        for cluster in self.cluster_name:
            c_number = int(cluster.split("_")[2][1])
            rt_mod = cluster.split("_")[3]
            if rt_mod=="proton":
                rt_mod2 = 1
            else:
                rt_mod2 = 0
                
            df = df_clusters[(df_clusters["cluster_2D_minsamp1_mincluster100"]==c_number)                              & (df_clusters["proton"]==rt_mod2)]
            var_selected = fs_list[cluster]["var_name"]

            X = df[var_selected]
                
            X = X.rename(columns={'oxaliplatin_concurrent':'Oxaliplatin Concurrent',
                                   'CRT0ALC':'Baseline ALC',
                                    "CRT0NLR": "Baseline NLR",
                                   "CRT0WBC":"Baseline WBC",
                                   "CRT0RBC":"Baseline RBC",
                                   "Total_blood_volume_litres_Nadlerformula":'Total Blood Volume',
                                   "TAXANE_concurrent": "TAXANE Concurrent Chemo",
                                   "Tumor_length": "Tumor Length",
                                "CRT0monocyte_absolute_count_KuL":"Baseline Monocyte Count",
                                  "CRT0eosinophil_absolute_count_KuL":"Baseline Eosinophil Count",
                                 "meanheartdose":"Mean Heart Dose",
                                  "CRT0Hb":"Baseline Hemoglobin",
                                  "CRT0PLC":"Baseline Platelet"
                                 
                                 })
                      
            y=df["G4RIL"]

            dtrain = xgb.DMatrix(X, label = y)
            params_list = xgbparams_list[cluster]['value']
            
            '''get corresponding optimized variables'''

            param_bo = { 
                  'colsample_bylevel':params_list[0],
                  'colsample_bynode':params_list[1],
                  'colsample_bytree':params_list[2],              
                  'eta' : params_list[3],
                  'gamma' : params_list[4],
                  'max_depth' : int(params_list[5]),
                  'min_child_weight': params_list[6],
                  'n_estimators' : int(params_list[7]),
                  'subsample' : params_list[8]}
            
    
            
            ratio_try = float(np.sum(y == 0)) / np.sum(y == 1)
            if ratio_try>3:
                #ratio = min(ratio_try,1.5)
                ratio = 2
            else:
                ratio = 1.0

            param_bo['scale_pos_weight'] = ratio

       

         
            
            
            model2 = XGBClassifier(**param_bo)
            model2.fit(X, y)
            shap_values = shap.TreeExplainer(model2).shap_values(X)
            vals= np.abs(shap_values).mean(0)
            
            shap_values_pd = pd.DataFrame(shap_values)
            neg_val = shap_values_pd[shap_values_pd<0].mean(0)
            pos_val = shap_values_pd[shap_values_pd>=0].mean(0)
            
            feature_importance = pd.DataFrame(list(zip(X.columns, vals,neg_val,pos_val)), 
                                              columns=['var_name','overall importance',
                                                       "negative effect", "positive effect"])
            feature_importance = pd.DataFrame(list(zip(X.columns, vals)), 
                                              columns=['var_name','overall importance'])
                
                
            feature_importance.sort_values(by=['overall importance'], ascending=False,inplace=True)

                
            shap_dict.update({cluster:feature_importance})
            print(cluster)
            maxdisplay = 25
      
            
            plt.rcParams["figure.figsize"] = (14, 25)
            plt.rcParams.update({'font.size': 40,"font.weight" : "bold",
                                 'axes.titlesize':40,"axes.labelweight":"bold"})
            shap.summary_plot(shap_values, X, plot_type="bar",max_display=maxdisplay,show=False)
            plt.savefig('output/SHAP_plots/SHAP_bar_{}_final_{}.png'.format(cluster,self.vername),
                        dpi=400,bbox_inches='tight',transparent = True)
            plt.clf()
            
            
            plt.rcParams["figure.figsize"] = (14, 25)
            plt.rcParams.update({'font.size': 40,"font.weight" : "bold",
                                 'axes.titlesize':40,"axes.labelweight":"bold"})
            shap.summary_plot(shap_values, X,max_display=maxdisplay,show=False)
            plt.savefig('output/SHAP_plots/SHAP_dot_{}_final_{}.png'.format(cluster,self.vername),
                        dpi=400,bbox_inches='tight',transparent = True)
            plt.clf()
            print("----------------------------------------------------")
        #return (shap_dict, shap_values)
        return (shap_dict)
    
    
    def dec_plot(self):
        plt.rcParams["figure.figsize"] = (200, 150)

        for dua_cluster,cluster_rank in zip(self.cluster_name,['cluster1','cluster2']):
            for i, cluster in zip(range(1,3),dua_cluster):

                c_number = int(cluster.split("_")[2][1])
                rt_mod = cluster.split("_")[3]
                if rt_mod=="proton":
                    rt_mod2 = 1
                else:
                    rt_mod2 = 0

                df = df_clusters[(df_clusters["cluster_2D_minsamp1_mincluster100"]==c_number)                                          & (df_clusters["proton"]==rt_mod2)]
                var_selected = fs_list[cluster]["var_name"]

                X = df[var_selected]
 


                X = X.rename(columns={'oxaliplatin_concurrent':'Oxaliplatin Concurrent',
                                               'CRT0ALC':'Baseline ALC',
                                                "CRT0NLR": "Baseline NLR",
                                               "CRT0WBC":"Baseline WBC",
                                               "CRT0RBC":"Baseline RBC",
                                               "Total_blood_volume_litres_Nadlerformula":'Total Blood Volume',
                                               "TAXANE_concurrent": "TAXANE Concurrent Chemo",
                                               "Tumor_length": "Tumor Length",
                                            "CRT0monocyte_absolute_count_KuL":"Baseline Monocyte Count",
                                              "CRT0eosinophil_absolute_count_KuL":"Baseline Eosinophil Count",
                                             "meanheartdose":"Mean Heart Dose",
                                              "CRT0Hb":"Baseline Hemoglobin",
                                              "CRT0PLC":"Baseline Platelet"
                                             })

                y=df["G4RIL"]



                params_list = xgbparams_list[cluster]['value']

                '''get corresponding optimized variables'''
        
                param_bo = { 
                      'colsample_bylevel':params_list[0],
                      'colsample_bynode':params_list[1],
                      'colsample_bytree':params_list[2],              
                      'eta' : params_list[3],
                      'gamma' : params_list[4],
                      'max_depth' : int(params_list[5]),
                      'min_child_weight': params_list[6],
                      'n_estimators' : int(params_list[7]),
                      'subsample' : params_list[8]}

                    
                ratio_try = float(np.sum(y == 0)) / np.sum(y == 1)
                if ratio_try>3:
                    #ratio = min(ratio_try,2)
                    ratio = 2
                else:
                    ratio = 1.0
           
                param_bo['scale_pos_weight'] = ratio




                model = XGBClassifier(**param_bo)
                model.fit(X, y)

                explainer = shap.TreeExplainer(model)
                expected_value = explainer.expected_value
                shap_values2 = shap.TreeExplainer(model).shap_values(X)
                fs_index = np.full((shap_values2.shape[0]), False, dtype=bool)
                if cluster=='2D_mincl100_c0_proton':
                    pt_index=58
                else:
                    pt_index = 5
                fs_index[pt_index] = True
                plt.rcParams.update({'font.size': 15,"font.weight" : "bold",
                                    'axes.titlesize':15,"axes.labelweight":"bold"})


                ax1 = plt.subplot(1,2,1)
                stand = shap.decision_plot(expected_value, shap_values2, X, link='logit', return_objects=True ,show = False)

                ax2 = plt.subplot(1,2,2, sharey = ax1)
                shap.decision_plot(expected_value, shap_values2[fs_index], X[fs_index],feature_order=stand.feature_idx,
                                           link='logit', highlight=0,show = False)


                #ax1.xaxis.set_visible(False)
                ax2.yaxis.set_visible(False)

                plt.tight_layout(pad=3, w_pad=0.9, h_pad=1)
                plt.rcParams.update({'font.size': 5,"font.weight" : "bold",
                                    'axes.titlesize':5,"axes.labelweight":"bold",
                                   'axes.labelsize': 5,'axes.titlesize':5,
                                     'legend.fontsize': 5, 'xtick.labelsize': 5, 'ytick.labelsize': 5,
                                    'figure.subplot.wspace':0.6,'figure.subplot.hspace':0.6})
                plt.savefig('output/SHAP_plots/SHAP_decision_{}_{}.png'.format(cluster,self.vername),
                            dpi=500,transparent = True,
                            bbox_inches='tight')
                plt.clf()  





shap_vis1 = SHAP_vis(group_list,"...")




shap_dict = shap_vis1.plot_shap()





SHAP_vis(group_list2, '....').dec_plot()




plt.rcParams["figure.figsize"] = (24, 41)
plt.rcParams.update({'font.size': 40,"font.weight" : "bold",
                     'axes.titlesize':40,"axes.labelweight":"bold"})



def hist(x, ax=None):
    cm = plt.cm.get_cmap("seismic")
    ax = ax or plt.gca()
    _, bins, patches = ax.hist(x,color="r",bins=30)

    bin_centers = 0.5*(bins[:-1]+bins[1:])
    maxi = np.abs(bin_centers).max()
    norm = plt.Normalize(-maxi,maxi)

    for c, p in zip(bin_centers, patches):
        plt.setp(p, "facecolor", cm(norm(c)))




class counterfact_plot:
    def __init__(self,cluster_name):
        self.cluster_name = cluster_name
    
    def prob_switch(self):
        prob_switch_list = []
        prob_orig_list = []
        df_prob = []
        for cluster in self.cluster_name:
            
            c_number = int(cluster.split("_")[2][1])
            c_modality = cluster.split("_")[3]
            if c_modality == "proton":
                c_modality_index = 1
                c_modality_index2 = 0
            else:
                c_modality_index = 0
                c_modality_index2 = 1
            
            '''get the subgroup data'''
            
            df = df_clusters[(df_clusters["cluster_2D_minsamp1_mincluster100"]==c_number)                              & (df_clusters["proton"]==c_modality_index)]
            
            var_selected = fs_list[cluster]["var_name"]
            
            X = df[var_selected]
            y=df["G4RIL"]
            
            params_list = xgbparams_list[cluster]['value']
            
            '''get corresponding optimized variables'''
       
            
      
            param_bo = { 
                      'colsample_bylevel':params_list[0],
                      'colsample_bynode':params_list[1],
                      'colsample_bytree':params_list[2],              
                      'eta' : params_list[3],
                      'gamma' : params_list[4],
                      'max_depth' : int(params_list[5]),
                      'min_child_weight': params_list[6],
                      'n_estimators' : int(params_list[7]),
                      'subsample' : params_list[8]
            }
            
            ratio_try = float(np.sum(y == 0)) / np.sum(y == 1)
            if ratio_try>3:
        
                ratio = 2
            else:
                ratio = 1.0
            param_bo['scale_pos_weight'] = ratio
        
        
            
            
            model = XGBClassifier(**param_bo)

            model.fit(X, y)
            prob_orig = model.predict_proba(X)[:, 1]
            
            
            
            '''counterfactual: switch modality'''
            orig_mod= cluster.split("_")

            if orig_mod[3]=='proton':
                orig_mod[3] = 'photon'
            else:
                orig_mod[3] = "proton"
     
            switch_mod = "_".join(orig_mod)
            
            '''same patients but with different predictors'''
            '''train the model use the patients with opposing modality in the same cluster'''
            df2 = df_clusters[(df_clusters["cluster_2D_minsamp1_mincluster100"]==c_number)                              & (df_clusters["proton"]==c_modality_index2)]
            
            var_switch = fs_list[switch_mod]["var_name"]
            
            X_opp = df2[var_switch]
            y_opp = df2["G4RIL"]
            
            X_switch = df[var_switch]


            params_list = xgbparams_list[switch_mod]['value']

            '''get corresponding optimized variables'''


    
            param_bo_counter = { 
                      'colsample_bylevel':params_list[0],
                      'colsample_bynode':params_list[1],
                      'colsample_bytree':params_list[2],              
                      'eta' : params_list[3],
                      'gamma' : params_list[4],
                      'max_depth' : int(params_list[5]),
                      'min_child_weight': params_list[6],
                      'n_estimators' : int(params_list[7]),
                      'subsample' : params_list[8]
            }
            
            ratio_try2 = float(np.sum(y_opp == 0)) / np.sum(y_opp == 1)
            if ratio_try2>3:
  
                ratio2 = 2
            else:
                ratio2 = 1.0
            param_bo_counter['scale_pos_weight'] = ratio2


            model2 = XGBClassifier(**param_bo_counter)
            #model2 = XGBClassifier()
            model2.fit(X_opp, y_opp)
            prob_switch = model2.predict_proba(X_switch)[:, 1]
            
            prob_change = (prob_switch - prob_orig)*100
            
            prob_switch_list+=[prob_change]
            prob_orig_list+=[prob_orig]
            
            df["orig_risk"] = prob_orig*100
            df["counter_risk"] = prob_switch*100
            df["risk_change"] = prob_change
            df_prob +=[df]
        return(prob_switch_list,prob_orig_list,df_prob)





cal_prob = counterfact_plot(group_list)




prob_switch,prob_orig_list, df_riskcal = cal_prob.prob_switch()



def dvh_dep(n,i,dvh,vername):
    df = df_clusters[(df_clusters["cluster_2D_minsamp1_mincluster100"]==n)                                  & (df_clusters["proton"]==i)]

    if i == 1:
        rt_mod = "proton"
    else:
        rt_mod = 'photon'

    cluster_number1 = "2D_mincl100_c"+str(n)+"_"+rt_mod
    cluster_number2 = "cluster"+str(n)


    var_selected = fs_list[cluster_number1]["var_name"]

    X = df[var_selected]
    X = X.rename(columns={'oxaliplatin_concurrent':'Oxaliplatin Concurrent',
                                   'CRT0ALC':'Baseline ALC',
                                    "CRT0NLR": "Baseline NLR",
                                   "CRT0WBC":"Baseline WBC",
                                   "CRT0RBC":"Baseline RBC",
                                   "Total_blood_volume_litres_Nadlerformula":'Total Blood Volume',
                                   "TAXANE_concurrent": "TAXANE Concurrent Chemo",
                                   "Tumor_length": "Tumor Length",
                                "CRT0monocyte_absolute_count_KuL":"Baseline Monocyte Count",
                                  "CRT0eosinophil_absolute_count_KuL":"Baseline Eosinophil Count",
                                 "meanheartdose":"Mean Heart Dose",
                                  "CRT0Hb":"Baseline Hemoglobin",
                                  "CRT0PLC":"Baseline Platelet",
                                  "meanspleendose":"Mean Spleen Dose",
                                 
                                 })
    

    y=df["G4RIL"]

    dtrain = xgb.DMatrix(X, label = y)
    params_temp = bo_result['cluster_2D_minsamp1_mincluster100'][cluster_number][rt_mod]

    '''get corresponding optimized variables'''



    params_temp2 = { 
                          'colsample_bylevel':params_temp['colsample_bylevel'],
                          'colsample_bynode':params_temp['colsample_bynode'],
                          'colsample_bytree':params_temp['colsample_bytree'],              
                          'eta' : params_temp['eta'],
                          'gamma' : params_temp['gamma'],
                          'max_depth' : int(params_temp['max_depth']),
                          'min_child_weight': params_temp['min_child_weight'],
                          'n_estimators' : int(params_temp['n_estimators']),
                          'subsample' : params_temp['subsample']
                }

    ratio_try = float(np.sum(y == 0)) / np.sum(y == 1)
    if ratio_try>3:
        #ratio = min(ratio_try,3)
        ratio = 2
    else:
        ratio = 1.0
    params_temp2['scale_pos_weight'] = ratio


    model2 = XGBClassifier(**params_temp2)
    model2.fit(X, y)
    shap_values = shap.TreeExplainer(model2).shap_values(X)
    
    plt.rcParams["figure.figsize"] = (14, 25)
    plt.rcParams.update({'font.size': 40,"font.weight" : "bold",
                        'axes.titlesize':40,"axes.labelweight":"bold"})

    shap.dependence_plot(dvh, shap_values, X, interaction_index=None, show = False) 
    
    plt.savefig('output/SHAP_plots/SHAP_dependence_{}_final_{}.png'.format(dvh, vername),
                        dpi=400,bbox_inches='tight',transparent = True)
    plt.clf()



class scatter_vis:
    def __init__(self,cluster_name,vername):
        self.cluster_name = cluster_name
        self.vername = vername
        
    def plot_dep(self):
        shap_dict = {}
        for cluster in self.cluster_name:
            c_number = int(cluster.split("_")[2][1])
            rt_mod = cluster.split("_")[3]
            if rt_mod=="proton":
                rt_mod2 = 1
            else:
                rt_mod2 = 0
                
            df = df_clusters[(df_clusters["cluster_2D_minsamp1_mincluster100"]==c_number)                              & (df_clusters["proton"]==rt_mod2)]
            var_selected = fs_list[cluster]["var_name"]

            X = df[var_selected]
                
            X = X.rename(columns={'oxaliplatin_concurrent':'Oxaliplatin Concurrent',
                                   'CRT0ALC':'Baseline ALC',
                                    "CRT0NLR": "Baseline NLR",
                                   "CRT0WBC":"Baseline WBC",
                                   "CRT0RBC":"Baseline RBC",
                                   "Total_blood_volume_litres_Nadlerformula":'Total Blood Volume',
                                   "TAXANE_concurrent": "TAXANE Concurrent Chemo",
                                   "Tumor_length": "Tumor Length",
                                "CRT0monocyte_absolute_count_KuL":"Baseline Monocyte Count",
                                  "CRT0eosinophil_absolute_count_KuL":"Baseline Eosinophil Count",
                                 "meanheartdose":"Mean Heart Dose",
                                  "CRT0Hb":"Baseline Hemoglobin",
                                  "CRT0PLC":"Baseline Platelet"
                                 
                                 })
                      
            y=df["G4RIL"]

            dtrain = xgb.DMatrix(X, label = y)
            params_list = xgbparams_list[cluster]['value']
            
            '''get corresponding optimized variables'''

            param_bo = { 
                  'colsample_bylevel':params_list[0],
                  'colsample_bynode':params_list[1],
                  'colsample_bytree':params_list[2],              
                  'eta' : params_list[3],
                  'gamma' : params_list[4],
                  'max_depth' : int(params_list[5]),
                  'min_child_weight': params_list[6],
                  'n_estimators' : int(params_list[7]),
                  'subsample' : params_list[8]}
            
    
            
            ratio_try = float(np.sum(y == 0)) / np.sum(y == 1)
            if ratio_try>3:

                ratio = 2
            else:
                ratio = 1.0

            param_bo['scale_pos_weight'] = ratio

       

         
            
            
            model2 = XGBClassifier(**param_bo)
            model2.fit(X, y)
            shap_values = shap.TreeExplainer(model2).shap_values(X)
            
            
            plt.rcParams["figure.figsize"] = (14, 25)
            plt.rcParams.update({'font.size': 40,"font.weight" : "bold",
                                 'axes.titlesize':40,"axes.labelweight":"bold"})
            shap.dependence_plot('Baseline ALC', shap_values, X, interaction_index=None,show=False)
            
            plt.savefig('output/SHAP_plots/SHAP_dependence_{}_final_{}.png'.format(cluster,self.vername),
                        dpi=400,bbox_inches='tight',transparent = True)
            plt.clf()


        return (shap_dict)





scatter_vis(group_list, ".....").plot_dep()






