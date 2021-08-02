#Author: Cong Zhu
#Purpose: supplemental tables S2.1 - S2.6
library(readxl)


setwd("......")
df = read_excel('.....')



#var_list: list of variables to be tested,e.g., age, BMI, sex, DVH
#data; input data

dec_tab = function(data, var_list){
  dec_sum_list = list()
  n_var = dim(data)[2]
  chratable_cluster = list()
  
  cont_index = c()
  cat_index = c()
  
  
  for (k in c(1:dim(data)[2])){
    if (length(table(data[[k]]))>5){
      cont_index = append(cont_index, k)
    } 
    else{
      cat_index = append(cat_index, k)
    }
    
  }
  
  #summary statistics of categorical variables
  char_sum_list = list()
  for (var_i in var_list){
    chartable = c()
    var_i_loc = grep(var_i, colnames(data))
    cat_index2 = cat_index[!cat_index %in% var_i_loc] 
    for (i in cat_index2){
      
      tb = table(data[[i]], data[[var_i]])
      prop.tb = prop.table(tb,2)
      
      p = round(chisq.test(tb)$p.value,3)
      p = ifelse(p<=0.001, "<0.001",p)
      
      P_value = rep("", dim(tb)[1])
      P_value[1] = p
      
      result_list = list()
      for (j in c(1:dim(tb)[2])){
        col = paste("col",1)
        assign(col, paste(tb[,j],"(", round(prop.tb[,j],3)*100,"%",")"))
        result_list[[j]] = eval(as.name(col))
        
        result_temp = do.call(cbind,result_list)
        
        
        
        
        
      }
      
      tab_row_name = paste(names(data)[i],":",levels(as.factor(data[[i]])))
      tab_col_name = names(table(data[[var_i]]))
      
      
      result_temp = as.data.frame(result_temp)
      names(result_temp) = tab_col_name
      row.names(result_temp) = tab_row_name
      
      chartable = rbind(chartable,cbind(result_temp,P_value))
      
      
    }
    char_sum_list = append(char_sum_list, list(as.data.frame(chartable)))
  }
  names(char_sum_list) = var_list
  
  #summary statistics of continous variables
  cont_sum_list = list()
  for (var_i in var_list){
    var_i_loc = grep(var_i, colnames(data))
    cont_index2 = append(cont_index,var_i_loc)
    
    df_cont = data[,cont_index2]
    
    n_cont_col = dim(df_cont)[2]
    
    
    mean_result = c()
    sd_result = c()
    p_value = c()
    
    for (j in c(1:(n_cont_col-1))){
      mean_1var = aggregate(df_cont[[j]]~ df_cont[[n_cont_col]], df_cont, function(x) mean = mean(x, na.rm = TRUE))
      mean_1var = as.numeric(t(mean_1var)[2,])
      mean_1var = round(mean_1var,2)
      
      mean_result = rbind(mean_result,mean_1var)
      
      
      sd_1var = aggregate(df_cont[[j]]~ df_cont[[n_cont_col]], df_cont, function(x) sd = sd(x, na.rm = TRUE))
      sd_1var = as.numeric(t(sd_1var)[2,])
      sd_1var = round(sd_1var,2)
      
      sd_result = rbind(sd_result,sd_1var)
      
      t_test = t.test(df_cont[[j]]~ df_cont[[n_cont_col]])
      p = t_test$p.value
      
      if (p <0.0001){
        p = "<0.001"
      }
      else{
        p = round(p,3)
      }
      
      
      p_value = rbind(p_value,p)
      
    }
    
    mean_sd_1col = paste(mean_result[,1]," (",sd_result[,1], ")")
    mean_sd_2col = paste(mean_result[,2]," (",sd_result[,2], ")")
    mean_sd_all = cbind(mean_sd_1col, mean_sd_2col,p_value)
    
    mean_sd_all = as.data.frame(mean_sd_all)
    rownames(mean_sd_all) = names(df_cont)[1:(n_cont_col-1)]
    
    header = names(table(df_cont[[n_cont_col]]))
    
    colnames(mean_sd_all) = c(header, "p-value") 
    
    cont_sum_list = append(cont_sum_list, list(mean_sd_all))
    
    
  }
  
  names(cont_sum_list) = var_list
  
  #output tables as excel spreadsheets
  for (var_i in var_list){
    write.xlsx(cont_sum_list[[var_i]],file="....." 
               ,sheetName=var_i
               ,append=T,row.names = T)
  } 
  
  
  for (var_i in var_list){
    write.xlsx(char_sum_list[[var_i]],file="....." 
               ,sheetName=var_i
               ,append=T,row.names = T)
  } 
  
  
}

#list of variables for subgroup comparisons
#e.g., clusters, radiotherapy modality, toxicity status
var_list = c('var_name')



#call the function
dec_tab(df, var_list)
