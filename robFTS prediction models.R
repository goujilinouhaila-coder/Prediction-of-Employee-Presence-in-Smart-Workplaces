##############Packages##############
library(robflreg)
library(readxl)
library(tidyverse)
library(xgboost)
library(caret)
library(ggplot2)
library(lubridate)
library(keras)
library(tensorflow)
library(tensorflow.keras)
library(RSNNS)
library(nnet)
library(caret)
library(dplyr)
library(ggplot2)
#install.packages("openxlsx")
library(openxlsx)
library(fda)                       
library(ftsa) 
library(matrixStats)
library(plsdepot)                  
library(plsgenomics)               
library(expm)  
# install.packages("robflreg")
# install.packages("robustbase")
library(robflreg)



# Import the dataframe of Venue

df_all <- read.xlsx("data/Employee Presence data.xlsx")

#Data cleaning and processing 
 
colnames(df_all)[1] <- c("Date")

df_all$Date <- as.Date(df_all$Date, origin = "1899-12-30")

# overview of the dataframe
DT::datatable(head(df_all, n = 50))


# Convert dates to days of the week (Monday to Friday)
days_week <- weekdays(df_all$Date)

# Define the order of the days of the week, respecting Monday to Friday
order_days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

# Use reorder() to reorder the days of the week
days_week_order <- factor(days_week, levels = order_days)

# Create a new column in your dataframe (or vector) with reorganized days of the week
df_all <- data.frame(df_all, days_week_order)


# Delete weeks where Global is less than 100 for at least one day of the week
df<- df_all %>%
  group_by(Date) %>%   # Group by week
  filter(all(GLOBAL > 100))  # Filter out weeks where Global is always greater than 100

# Now, df_filtered contains the weeks where Global is always greater than or equal to 100 for each day of the week.

df <- df %>%
  arrange(Date) %>%
  group_by(Week = lubridate::week(Date)) %>%
  ungroup()

df$Year <- year(df$Date)
df$Year_a_Week <- paste(df$Year, "Sem", df$Week)

# Filter full weeks (Monday to Friday)
df_completes <- df %>%
  group_by(Year_a_Week) %>%
  filter(all(c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday") %in% days_week)) %>%
  ungroup()


df_completes$Year <- year(df_completes$Date)

df_completes$Year_a_Week <- paste(df_completes$Year, "Sem",df_completes$Week)

# df_completes now only contains complete weeks

## know which weeks have been removed
serie1 <- unique(df$Year_a_Week)
serie2 <- unique(df_completes$Year_a_Week)


weeks_removed <- setdiff(serie1, serie2)
weeks_removed

#### import meteo ####

meteo <- read.xlsx("data/export-paris0.xlsx", sheet = 2)
meteo$DATE <- as.Date(meteo$DATE, origin = "1899-12-30")
colnames(meteo)[1] <- colnames(df_completes)[1]

## import greve ### 
greve <- read.csv("data/archive_greve.csv", sep=";")
greve$Date <- as.Date(greve$Date, format = "%d/%m/%Y")
colnames(greve)[1] <- colnames(df_completes)[1]

### final df ### 
# merge data into a single final df

df_final <- merge(merge(df_completes, meteo, by = "Date", all = FALSE), greve, by = "Date", all =FALSE)

## Add the dataframe of meeting rooms reservations (the script to retrieve and process reservation data is in Processing
# reservation data)

Nbr_Resa <- read.csv("data/MeetingRoom reservations.csv", sep = ",")
colnames(Nbr_Resa)[1] <- "Date"
# Fusionner les tables
df_final <- merge(df_final, Nbr_Resa, by = "Date", all = FALSE)


# Initialization of matrices and result tables
mape_results <- numeric()
r_squared_results <- numeric()
mse_results <- numeric()
mae_results <- numeric()
rmse_results <- numeric()
mape_new_MM <- numeric()
mape_new_MCD <- numeric()
mape_new_S <- numeric()
mape_new_tau <- numeric()
mae_new_MM <- numeric()
mae_new_MCD <- numeric()
mae_new_S <-numeric()
mae_new_tau <- numeric()
rmse_new_MM <- numeric()
rmse_new_MCD <- numeric()
rmse_new_S <- numeric()
rmse_new_tau <- numeric()

predictions_list_MM <- list()
predictions_list_MCD <- list()
predictions_list_S <- list()
predictions_list_tau <- list()
mean_predictions_list <- list()


### Run the model ### 
#####################
## select all the rows of the dataframe without the last two weeks, which represent the last 10 rows 
#### to predict the last week 

#### if, for example, you want to predict more than one week (>5days), 
#### you can reduce the size of the training data by removing, for example, the last 5 rows 
#### corresponding to an additional week, in order to take them into account in the test data 
### to predict the week you've added.


donnees_app <- df_final[1:220, ] 

# Global loop to process data in batches of 5 
for (j in seq(221, 221, by = 5)) {
  # Add 5 lines from df_final to donnees_app
  donnees_app <- rbind(donnees_app, df_final[j:(j+4), ])
  
  # Select the following 5 lines from df_final for test_data
  donnees_test <- df_final[(j+5):((j+5)+4), ]
  
  # Initialization of response, temp, rain, greve matrices
  response <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  temp <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  pluie <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  greve <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  reservations <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  ferie <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  pont_conge <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  holiday <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  p=2
  response_lagged <- list() 
  
  
  # Loop to build response, temp, rain, greve matrices
  for (i in seq(1, nrow(donnees_app), by = 5)) {
    response[(i-1)/5+1, 1:5] <- donnees_app$GLOBAL[i:(i+4)]
    temp[(i-1)/5+1, 1:5] <- donnees_app$Temp[i:(i+4)]
    pluie[(i-1)/5+1, 1:5] <- donnees_app$pluie[i:(i+4)]
    greve[(i-1)/5+1, 1:5] <- donnees_app$Greve_nationale[i:(i+4)]
    reservations[(i-1)/5+1, 1:5] <- donnees_app$Total_reservations[i:(i+4)]
    ferie[(i-1)/5+1, 1:5] <- donnees_app$jour_feriÃ.[i:(i+4)]
    pont_conge[(i-1)/5+1, 1:5] <- donnees_app$pont.congÃ.[i:(i+4)]
    holiday[(i-1)/5+1, 1:5] <- donnees_app$holiday[i:(i+4)]
  }
  mean_lagged <- colMeans(response)
  response_lagged[[1]] <- rbind(mean_lagged, response[-(nrow(donnees_app)/5),])
  
  response_lagged[[p]] <- rbind(mean_lagged, (response_lagged[[p-1]])[-(nrow(donnees_app)/5),])
  
  
  # Exogenous variables
  response_test <- matrix(0, ncol = 5, nrow = 1)  
  temp_test <- matrix(0, ncol = 5, nrow = 1)
  pluie_test <- matrix(0, ncol = 5, nrow = 1)
  greve_test <- matrix(0, ncol = 5, nrow = 1)
  reservations_test <- matrix(0, ncol = 5, nrow = 1)
  ferie_test <- matrix(0, ncol = 5, nrow = 1)
  pont_conge_test <- matrix(0, ncol = 5, nrow = 1)
  holiday_test <- matrix(0, ncol = 5, nrow = 1)
  response_lagged_test <- list()
  
  for (k in seq(1,5,by=5)){
    response_test[(k-1)/5+1,1:5] <- donnees_test$GLOBAL[k:(k+4)]
    temp_test[(k-1)/5+1, 1:5] <- donnees_test$Temp[k:(k+4)]
    pluie_test[(k-1)/5+1, 1:5] <- donnees_test$pluie[k:(k+4)]
    greve_test[(k-1)/5+1, 1:5] <- donnees_test$Greve_nationale[k:(k+4)]
    reservations_test[(k-1)/5+1, 1:5] <- donnees_test$Total_reservations[k:(k+4)]
    ferie_test[(k-1)/5+1, 1:5] <- donnees_test$jour_feriÃ.[k:(k+4)]
    pont_conge_test[(k-1)/5+1, 1:5] <- donnees_test$pont.congÃ.[k:(k+4)]
    holiday_test[(k-1)/5+1, 1:5] <- donnees_test$holiday[k:(k+4)]
  }
  p=2
  response_lagged_test[[1]] <- as.matrix(t(response[(nrow(donnees_app)/5)-1,]))
  response_lagged_test[[p]] <- as.matrix(t(response_lagged[[p-1]][(nrow(donnees_app)/5)-1,]))
  
  #predictor <- list(  as.matrix(response_lagged[[1]]),  as.matrix(temp),  as.matrix(pluie), as.matrix(reservations))
  #predictor_test <- list(as.matrix(response_lagged_test[[1]]),  as.matrix(temp_test), as.matrix(pluie_test), as.matrix(reservations_test))
  
  predictor <- list(  as.matrix(response_lagged[[1]]),  as.matrix(temp),  as.matrix(reservations))
  predictor_test <- list( as.matrix(response_lagged_test[[1]]),as.matrix(temp_test),as.matrix(reservations_test))
  

  # Model training
  fmodel_MM <- rob.ff.reg(Y = response, X = predictor, model = "full", emodel = "robust", 
                          fmodel = "MM", nbasisY = 5, nbasisX = rep(5, length(predictor)))
  
  fmodel_MCD <- rob.ff.reg(Y = response, X = predictor, model = "full", emodel = "robust", 
                           fmodel = "MCD", nbasisY = 5, nbasisX = rep(5, length(predictor)))
  
  fmodel_S <- rob.ff.reg(Y = response, X = predictor, model = "full", emodel = "robust", 
                         fmodel = "S", nbasisY = 5, nbasisX = rep(5, length(predictor)))
  
  fmodel_tau <- rob.ff.reg(Y = response, X = predictor, model = "full", emodel = "robust", 
                           fmodel = "tau", nbasisY = 5, nbasisX =rep(5, length(predictor)))
  
  # Prédiction
  # predictions_MM <- rbind(predictions_MM, predict_ff_regression(fmodel_MM, Xnew = predictor_test))
  # predictions_MCD <- rbind(predictions_MCD, predict_ff_regression(fmodel_MCD, Xnew = predictor_test))
  # predictions_S <- rbind(predictions_S , predict_ff_regression(fmodel_S, Xnew = predictor_test))
  # predictions_tau <- rbind(predictions_tau, predict_ff_regression(fmodel_tau, Xnew = predictor_test))
  # 
  predictions_MM <-  predict_ff_regression(fmodel_MM, Xnew = predictor_test)
  predictions_MCD <- predict_ff_regression(fmodel_MCD, Xnew = predictor_test)
  predictions_S <-  predict_ff_regression(fmodel_S, Xnew = predictor_test)
  predictions_tau <- predict_ff_regression(fmodel_tau, Xnew = predictor_test)

  predictions_list_MM[[length(predictions_list_MM) + 1]] <- predictions_MM
  predictions_list_MCD[[length(predictions_list_MCD) + 1]] <- predictions_MCD
  predictions_list_S[[length(predictions_list_S) + 1]] <- predictions_S
  predictions_list_tau[[length(predictions_list_tau) + 1]] <- predictions_tau
  
  
  # Create a matrix of predictions
  predictions_matrix <- cbind(t(predictions_MM), t(predictions_MCD),
                              t(predictions_S), t(predictions_tau))
  
  # Calculate the average for each observation
  mean_predictions <- rowMeans(predictions_matrix)
  
  mean_predictions_list[[length(mean_predictions_list) + 1]] <- mean_predictions 
  
  
  
  # Calculation of MAPE prediction error
  mape_new_MM <- c(mape_new_MM , (mean(abs((predictions_MM - donnees_test$GLOBAL) / donnees_test$GLOBAL)) * 100))
  mape_new_MCD <- c(mape_new_MCD,(mean(abs((predictions_MCD - donnees_test$GLOBAL) / donnees_test$GLOBAL)) * 100))
  mape_new_S <- c(mape_new_S,(mean(abs((predictions_S - donnees_test$GLOBAL) / donnees_test$GLOBAL)) * 100))
  mape_new_tau <- c(mape_new_tau,(mean(abs((predictions_tau - donnees_test$GLOBAL) / donnees_test$GLOBAL)) * 100))
  
  # Storage of the smallest mape_new
  #mape_new_i <- min(mape_new_MM, mape_new_MCD, mape_new_S, mape_new_tau)
  
  # Calculation of MAE prediction error
  mae_new_MM <- c(mae_new_MM , mean(abs(predictions_MM - donnees_test$GLOBAL)))
  mae_new_MCD <- c(mae_new_MCD, mean(abs(predictions_MCD - donnees_test$GLOBAL)))
  mae_new_S <- c(mae_new_S, mean(abs(predictions_S - donnees_test$GLOBAL)))
  mae_new_tau <- c(mae_new_tau, mean(abs(predictions_tau - donnees_test$GLOBAL)))
  
  # Calculation of RMSE prediction error
  rmse_new_MM <- c(rmse_new_MM, sqrt(mean((predictions_MM - donnees_test$GLOBAL)^2)))
  rmse_new_MCD <- c(rmse_new_MCD, sqrt(mean((predictions_MCD - donnees_test$GLOBAL)^2)))
  rmse_new_S <- c(rmse_new_S, sqrt(mean((predictions_S - donnees_test$GLOBAL)^2)))
  rmse_new_tau <- c(rmse_new_tau, sqrt(mean((predictions_tau - donnees_test$GLOBAL)^2)))
  
  ## calculate residuals ### 
  residuals <- mean_predictions - donnees_test$GLOBAL
  mape_new_i <- mean(abs(residuals / donnees_test$GLOBAL)) * 100
  mape_results <- c(mape_results, mape_new_i)
  mae <- mean(abs(residuals))
  mae_results <- c(mae_results, mae)
  mse <- mean(residuals^2)
  mse_results <- c(mse_results, mse)
  rmse <- sqrt(mse)
  rmse_results <- c(rmse_results, rmse)
  r_squared <- 1 - sum(residuals^2) / sum((donnees_test$GLOBAL - mean(donnees_test$GLOBAL))^2)
  r_squared_results <- c(r_squared_results, r_squared)
  
}


## print the mean of the mape of each estimator 

mean(mape_new_MCD)
#mape_new_MCD
mean(mape_new_MM)
#mape_new_MM
mean(mape_new_S)
#mape_new_S
mean(mape_new_tau)
#mape_new_tau
mean(mape_results)

## print the mean of the rmse of each estimator 

mean(rmse_new_MCD)
mean(rmse_new_MM)
mean(rmse_new_S)
mean(rmse_new_tau)
mean(rmse_results)

## print the mean of the mae of each estimator 

mean(mae_new_MCD)
mean(mae_new_MM)
mean(mae_new_S)
mean(mae_new_tau)
mean(mae_results)




## Print the predictions 

mean_predictions_list
predictions_list_tau
predictions_list_MM
predictions_list_S
predictions_list_MCD

