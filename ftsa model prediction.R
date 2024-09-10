### load librairies
library(ftsa)
library(fda)
library(rainbow)
library(openxlsx)

# Import
df_all <- read.xlsx("data/Employee Presence data.xlsx")

colnames(df_all)[1] <- c("Date")


df_all$Date <- as.Date(df_all$Date, origin = "1899-12-30")

#df$Date <- format(df$Date, format = "%d/%m/%Y")


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

df <- df_all %>%
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

#### test ftsa on df_final without removing weeks when a day contains less than 100p.######

####################################################################################################
#### a loop that explores all the data and adds a week after each iteration #######################
####################################################################################################


data <- df_final
mape_results <- numeric()
mae_results <- numeric()
donnees_app <- data[1:235,] 

# Global loop to process data in batches of 5
for (j in seq(236, 241, by = 5)) {
  # Add 5 lines from df_final to donnees_app
  donnees_app <- rbind(donnees_app, data[j:(j+4), ])
  func_app <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  for (i in seq(1, nrow(donnees_app), by = 5)) {
    func_app[(i-1)/5+1, 1:5] <- donnees_app$GLOBAL[i:(i+4)]
  }
  colnames(func_app) <- c("1","2","3","4","5")
  n <- nrow(func_app)
  rownames(func_app) <- 1:n
  donnees_app_fts <- fts(1:5, t(func_app))
  
  # Select the next 5 lines of df_final for test_data
  donnees_test <- data[(j+5):((j+5)+4), 2]
  
  # Model training
  mod <- ftsm(donnees_app_fts,order=1,method = "rapca",weight = FALSE)
  res <- forecast.ftsm(mod,h=1,method = "arima")
  res.vec <- as.vector(res$mean$y)
  mape <- mean(abs(donnees_test-res.vec)*100/donnees_test)
  mape_results <- c(mape_results, mape)
  mae <- mean(abs(donnees_test-res.vec))
  mae_results <- c(mae_results, mae) 

}

mape_results
mean(mape_results)
mae_results
mean(mae_results)


### with df_final without days with less than 100p.  #### 

# Import
df_all <- read.xlsx("data/Employee Presence data.xlsx")

colnames(df_all)[1] <- c("Date")


df_all$Date <- as.Date(df_all$Date, origin = "1899-12-30")

#df$Date <- format(df$Date, format = "%d/%m/%Y")


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

mape_results_ets <- numeric()
mae_results_ets <- numeric()
rmse_results_ets <- numeric()
predictions_list_ets <- list()

mape_results_arima <- numeric()
mae_results_arima <- numeric()
rmse_results_arima <- numeric()
predictions_list_arima <- list()

mape_results_rwdrift <- numeric()
mae_results_rwdrift <- numeric()
rmse_results_rwdrift <- numeric()
predictions_list_rwdrift <- list()

mape_results_rw <- numeric()
mae_results_rw <- numeric()
rmse_results_rw <- numeric()
predictions_list_rw <- list()

mape_results_arfima <- numeric()
mae_results_arfima <- numeric()
rmse_results_arfima <- numeric()
predictions_list_arfima <- list()




donnees_app <- df_final[1:220,] 

# Global loop to process data in batches of 5
for (j in seq(221, 221, by = 5)) {
  # Add 5 lines from df_final to donnees_app
  donnees_app <- rbind(donnees_app, df_final[j:(j+4), ])
  func_app <- matrix(0, ncol = 5, nrow = nrow(donnees_app)/5)
  for (i in seq(1, nrow(donnees_app), by = 5)) {
    func_app[(i-1)/5+1, 1:5] <- donnees_app$GLOBAL[i:(i+4)]
  }
  colnames(func_app) <- c("1","2","3","4","5")
  n <- nrow(func_app)
  rownames(func_app) <- 1:n
  donnees_app_fts <- fts(1:5, t(func_app))
  
  # Select the following 5 lines from df_final for test_data
  donnees_test <- df_final[(j+5):((j+5)+4), 2]

  
  # Model training
  mod <- ftsm(donnees_app_fts,order=1,method = "rapca",weight = FALSE)
  ## ets 
  res.ets <- forecast.ftsm(mod,h=1,method = "ets")
  res.vec.ets <- as.vector(res.ets$mean$y)
  mape.ets <- mean(abs(donnees_test-res.vec.ets)*100/donnees_test)
  mape_results_ets <- c(mape_results_ets, mape.ets)
  mae.ets <- mean(abs(donnees_test-res.vec.ets))
  mae_results_ets <- c(mae_results_ets, mae.ets)
  rmse.ets <- sqrt(mean((res.vec.ets - donnees_test)^2))
  rmse_results_ets<- c(rmse_results_ets, rmse.ets)
  
  ### stockage des prédictions ### 
  predictions_list_ets[[length(predictions_list_ets) + 1]] <- res.vec.ets
  
  ### arima
  res.arima <- forecast.ftsm(mod,h=1,method = "arima")
  res.vec.arima <- as.vector(res.arima$mean$y)
  mape.arima<- mean(abs(donnees_test-res.vec.arima)*100/donnees_test)
  mape_results_arima <- c(mape_results_arima, mape.arima)
  mae.arima <- mean(abs(donnees_test-res.vec.arima))
  mae_results_arima <- c(mae_results_arima, mae.arima)
  rmse.arima <- sqrt(mean((res.vec.arima - donnees_test)^2))
  rmse_results_arima<- c(rmse_results_arima, rmse.arima)
  
  ### stockage des prédictions ### 
  predictions_list_arima[[length(predictions_list_arima) + 1]] <- res.vec.arima
  
  ### rwdrift
  res.rwdrift <- forecast.ftsm(mod,h=1,method = "rwdrift")
  res.vec.rwdrift <- as.vector(res.rwdrift$mean$y)
  mape.rwdrift<- mean(abs(donnees_test-res.vec.rwdrift)*100/donnees_test)
  mape_results_rwdrift <- c(mape_results_rwdrift, mape.rwdrift)
  mae.rwdrift <- mean(abs(donnees_test-res.vec.rwdrift))
  mae_results_rwdrift<- c(mae_results_rwdrift, mae.rwdrift)
  rmse.rwdrift <- sqrt(mean((res.vec.rwdrift - donnees_test)^2))
  rmse_results_rwdrift <- c(rmse_results_rwdrift, rmse.rwdrift)
  
  ### stockage des prédictions ### 
  predictions_list_rwdrift[[length(predictions_list_rwdrift) + 1]] <- res.vec.rwdrift
  
  ### rw 
  res.rw <- forecast.ftsm(mod,h=1,method = "rw")
  res.vec.rw <- as.vector(res.rwdrift$mean$y)
  mape.rw<- mean(abs(donnees_test-res.vec.rw)*100/donnees_test)
  mape_results_rw <- c(mape_results_rw, mape.rw)
  mae.rw <- mean(abs(donnees_test-res.vec.rw))
  mae_results_rw<- c(mae_results_rw, mae.rw)
  rmse.rw <- sqrt(mean((res.vec.rw - donnees_test)^2))
  rmse_results_rw <- c(rmse_results_rw, rmse.rw)
  
  ### stockage des prédictions ### 
  predictions_list_rw[[length(predictions_list_rw) + 1]] <- res.vec.rw
  
  ### arfima
  res.arfima <- forecast.ftsm(mod,h=1,method = "arfima")
  res.vec.arfima <- as.vector(res.arfima$mean$y)
  mape.arfima<- mean(abs(donnees_test-res.vec.arfima)*100/donnees_test)
  mape_results_arfima <- c(mape_results_arfima, mape.arfima)
  mae.arfima <- mean(abs(donnees_test-res.vec.arfima))
  mae_results_arfima<- c(mae_results_arfima, mae.arfima)
  rmse.arfima <- sqrt(mean((res.vec.arfima - donnees_test)^2))
  rmse_results_arfima <- c(rmse_results_arfima, rmse.arfima)
  
  ### stockage des prédictions ### 
  predictions_list_arfima[[length(predictions_list_arfima) + 1]] <- res.vec.arfima
}


mean(mape_results_ets)
mean(mae_results_ets)
mean(rmse_results_ets)

mean(mape_results_arima)
mean(mae_results_arima)
mean(rmse_results_arima)

mean(mape_results_rwdrift)
mean(mae_results_rwdrift)
mean(rmse_results_rwdrift)

mean(mape_results_rw)
mean(mae_results_rw)
mean(rmse_results_rw)

mean(mape_results_arfima)
mean(mae_results_arfima)
mean(rmse_results_arfima)


predictions_list_ets
predictions_list_arima
predictions_list_rwdrift
predictions_list_rw
predictions_list_arfima




