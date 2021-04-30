library(mgcv)
setwd("~/nfl-runpass/")

### Load data and create ratio between score margin and time
data <- read.csv("datasets/nflmodeldatafirst.csv")
nfldata <- data[data$ydstogo == 10,]
nfldata$margin_time_ratio <- nfldata$margin/(nfldata$game_seconds_remaining + 1)

set.seed(143)

### Create training & test datasets
test.id = sample(seq(1,nrow(nfldata), 1), floor(nrow(nfldata)*0.3))
nfldata_test = nfldata[test.id,]
nfldata_train = nfldata[-test.id,]

### Initial model: logistic GLM
logitmodel <- glm(posteam_won ~ margin + game_seconds_remaining + margin_time_ratio + yardline_100, data = nfldata, family = "binomial")

### RMSE function
RMSE = function(y,yhat){
  SSE = sum((y-yhat)^2, na.rm = T) 
  return(sqrt(SSE/length(y)))
}

### Evaluating GLM Model
summary(logitmodel)

RMSE(nfldata_test$posteam_won, predict(logitmodel, newdata = nfldata_test, type = "response"))
RMSE(nfldata_train$posteam_won, predict(logitmodel, newdata = nfldata_train, type = "response"))

library(MLmetrics)
print(LogLoss(logitmodel$fitted.values, nfldata_train$posteam_won))

### GAM model and relevant stats for evaluation
model <- mgcv::bam(posteam_won ~ s(margin) + s(game_seconds_remaining) + s(margin_time_ratio) + s(yardline_100), data = nfldata_train, family = "binomial")

summary(model)

RMSE(nfldata_test$posteam_won, predict(model, newdata = nfldata_test, type = "response"))
RMSE(nfldata_train$posteam_won, predict(model, newdata = nfldata_train, type = "response"))

print(LogLoss(model$fitted.values, nfldata_train$posteam_won))

### Create calibration plot
calibPlot <- function(model, trainData){
  trainCopy = trainData
  trainCopy$predictedVals <- predict(model, newdata = trainData)
  xPoints = rep(NA, 10)
  yPoints = rep(NA, 10)
  for(i in 1:10){
    xPoints[i] = mean(trainCopy$predictedVals[trainCopy$predictedVals <= 0.1*i & trainCopy$predictedVals >= 0.1*(i-1)], na.rm = TRUE)
    yPoints[i] = mean(trainCopy$posteam_won[trainCopy$predictedVals <= 0.1*i & trainCopy$predictedVals >= 0.1*(i-1)], na.rm = TRUE)
  }
  print(xPoints)
  print(yPoints)
  plot(x = xPoints, y = yPoints, main = "Calibration Plot", xlab = "Predicted Probability", ylab = "Actual Proportion of Wins", ylim = c(0, 1))
}

calibPlot(model, nfldata_train)

# Model assumptions
gam.check(model)

### Getting punt distance and field goal probability data
punt_fg_data <- read.csv("datasets/punt_fg_probs.csv")
fg_prob <- punt_fg_data$fg
punt_dist <- punt_fg_data$punt

for(i in 1:length(punt_dist)){
  if(punt_dist[i] == 0){
    punt_dist[i] <- NA
  }
}

run_prob_data <- read.csv("datasets/run_probs.csv")
run_prob <- run_prob_data$X0

pass_prob_data <- read.csv("datasets/pass_probs.csv")
pass_prob <- pass_prob_data$X0

### Getting run play time and pass play time data
run_time_data <- read.csv("datasets/runtime.csv")
pass_time_data <- read.csv("datasets/passtime.csv")
run_times <- run_time_data$X0
pass_times <- pass_time_data$X0

### Getting interception probability, pick six probability, and interception return yards
intercept_calcs <- read.csv("datasets/intercept_probs.csv")
intercept_probs <- intercept_calcs$prob
intercept_td_probs <- intercept_calcs$td_prob
intercept_return <- intercept_calcs$return

### Getting yards gained on failed pass/run attempts
pass_fail_yds_data <- read.csv("datasets/pass_fails.csv")
run_fail_yds_data <- read.csv("datasets/run_fails.csv")

### Function to generate fourth down action resulting in best win probability 
f <- function(ydline, ydstogo, margin_curr, time_left){
  # Where we'll store our win probabilities for each action
  predictions <- c(0, 0, 0, 0)
  
  ## Punts
  if(is.na(punt_dist[ydline])){
    # If no data available for a yard line, assume a touchback
    x_punt <- data.frame("margin" = -margin_curr, "game_seconds_remaining" = time_left, "margin_time_ratio" = (-margin_curr)/(time_left+1), "yardline_100" = 80)
    predictions[1] = 1 - mgcv::predict.bam(model, newdata = x_punt, type = "response")[1]
  }
  else{
    # Otherwise, use the available average punt distance data
    x_punt <- data.frame("margin" = -margin_curr, "game_seconds_remaining" = time_left, "margin_time_ratio" = (-margin_curr)/(time_left+1), "yardline_100" = 100-(ydline-punt_dist[ydline]))
    predictions[1] = 1 - mgcv::predict.bam(model, newdata = x_punt, type = "response")[1]
  }
  
  ## Field Goals
  
  # In case of success, add 3 to the margin and have yardline be 75; then get win probability from perspective of the defending team and subtract from 1
  x_fg_success <- data.frame("margin" = -margin_curr - 3, "game_seconds_remaining" = time_left, "margin_time_ratio" = (-margin_curr - 3)/(time_left+1), "yardline_100" = 75)
  fg_success_wp <- 1 - mgcv::predict.bam(model, newdata = x_fg_success, type = "response")[1]
  
  # In case of failure, margin stays the same, and defending team gets the ball at the spot of the kick
  x_fg_fail <- data.frame("margin" = -margin_curr, "game_seconds_remaining" = time_left, "margin_time_ratio" = (-margin_curr)/(time_left+1), "yardline_100" = min(80, 100-(ydline+8)))
  fg_fail_wp <- 1 - mgcv::predict.bam(model, newdata = x_fg_fail, type = "response")[1]
  
  # Use LOTP to get final probability
  predictions[2] = fg_prob[ydline]*(fg_success_wp)+(1-fg_prob[ydline])*(fg_fail_wp)
  
  ## Run Plays
  if(ydstogo >= ydline){
    # If in a 4th & Goal situation and success, add expected touchdown points to margin, subtract time taken by run play from time remaining, and give ball to defending team at 75 yard line
    # Calculate WP in perspective of defending team and subtract from 1
    x_run_success <- data.frame("margin" = -margin_curr - 6.970960661826777, "game_seconds_remaining" = time_left - run_times[ydstogo], "margin_time_ratio" = (-margin_curr-6.970960661826777)/(time_left - run_times[ydstogo]+1), "yardline_100" = 75)
    run_success_wp <- 1 - mgcv::predict.bam(model, newdata = x_run_success, type = "response")[1]
  }
  else{
    # If not in 4th & Goal and success, time remaining goes down by expected amount, and yard line decreases
    # Calculate WP in perspective of team in possession
    x_run_success <- data.frame("margin" = margin_curr, "game_seconds_remaining" = time_left - run_times[ydstogo], "margin_time_ratio" = (margin_curr)/(time_left - run_times[ydstogo]+1), "yardline_100" = ydline-ydstogo)
    run_success_wp <- mgcv::predict.bam(model, newdata = x_run_success, type = "response")[1]
  }
  
  # If run play attempt fails, give ball to defending team at yard line based on expected number of yard gained on failed run play
  # Calculate WP in perspective of defending team and subtract from 1
  x_run_fail <- data.frame("margin" = -margin_curr, "game_seconds_remaining" = time_left - run_times[ydstogo], "margin_time_ratio" = (-margin_curr)/(time_left - run_times[ydstogo]+1), "yardline_100" = 100-(ydline - run_fail_yds_data[ydstogo, ydline+1]))
  run_fail_wp <- 1-mgcv::predict.bam(model, newdata = x_run_fail, type = "response")[1]
  
  # Use LOTP to get final probability
  predictions[3] = run_prob[ydstogo]*(run_success_wp)+(1-run_prob[ydstogo])*(run_fail_wp)
  
  ## Pass Plays
  if(ydstogo >= ydline){
    # If in a 4th & Goal situation and success, add expected touchdown points to margin, subtract time taken by pass play from time remaining, and give ball to defending team at 75 yard line
    # Calculate WP in perspective of defending team and subtract from 1
    x_pass_success <- data.frame("margin" = -margin_curr - 6.970960661826777, "game_seconds_remaining" = time_left - pass_times[ydstogo], "margin_time_ratio" = (-margin_curr-6.970960661826777)/(time_left- pass_times[ydstogo]+1), "yardline_100" = 75)
    pass_success_wp <- 1 - mgcv::predict.bam(model, newdata = x_pass_success, type = "response")[1]
  }
  else{
    # If not in 4th & Goal and success, time remaining goes down by expected amount, and yard line decreases
    # Calculate WP in perspective of team in possession
    x_pass_success <- data.frame("margin" = margin_curr, "game_seconds_remaining" = time_left- pass_times[ydstogo], "margin_time_ratio" = (margin_curr)/(time_left- pass_times[ydstogo]+1), "yardline_100" = ydline-ydstogo)
    pass_success_wp <- mgcv::predict.bam(model, newdata = x_pass_success, type = "response")[1]
  }
  
  # If pass attempt fails with no interception, give ball to defending team at yardline based on expected number of yards gained by failed pass play
  # Calculate WP in perspective of defending team and subtract from 1
  x_pass_fail <- data.frame("margin" = -margin_curr, "game_seconds_remaining" = time_left- pass_times[ydstogo], "margin_time_ratio" = (-margin_curr)/(time_left- pass_times[ydstogo]+1), "yardline_100" = 100-(ydline - pass_fail_yds_data[ydstogo, ydline+1]))
  pass_fail_wp <- 1-mgcv::predict.bam(model, newdata = x_pass_fail, type = "response")[1]
  
  # If interception occurs with no pick six, give ball to defending team at yardline based on expected number of interception return yards
  # Calculate WP in perspective of defending team and subtract from 1
  x_pass_intercept <- data.frame("margin" = -margin_curr, "game_seconds_remaining" = time_left- pass_times[ydstogo], "margin_time_ratio" = (-margin_curr)/(time_left- pass_times[ydstogo]+1), "yardline_100" = max(min(100-ydline - intercept_return[ydline], 99), 1))
  pass_intercept_wp <- 1-mgcv::predict.bam(model, newdata = x_pass_intercept, type = "response")[1]
  
  # If pick six occurs, subtract expected touchdown points from margin and give ball to team in possession at 75
  x_pick_six <- data.frame("margin" = margin_curr - 6.970960661826777, "game_seconds_remaining" = time_left- pass_times[ydstogo], "margin_time_ratio" = (margin_curr - 6.970960661826777)/(time_left- pass_times[ydstogo]+1), "yardline_100" = 75)
  pick_six_wp <- mgcv::predict.bam(model, newdata = x_pick_six, type = "response")[1]
  
  # Use LOTP to get final probability
  predictions[4] = pass_prob[ydstogo]*(pass_success_wp)+(1-pass_prob[ydstogo] - intercept_probs[ydline])*(pass_fail_wp) + intercept_probs[ydline]*pass_intercept_wp + intercept_td_probs[ydline]*pick_six_wp
  
  # Check whether the highest win probability and second-highest win probability are extremely close and indicate whether they are
  if(max(predictions) - max(predictions[-which.max(predictions)]) >= 0.005){
    return(which.max(predictions))
  }
  else{
    return(which.max(predictions) + 5)
  }
}

### Code for generating heatmaps
test <- c()
for(ydline in 1:99){
  for(ydtogo in 1:15){
    print(ydline)
    print(ydtogo)
    # third input is score margin for 4th down team
    # fourth is seconds remaining
    # -7, 300, 600, 1200
    # -14, 300, 600
    test = c(test, f(ydline, ydtogo, 14, 300))
  }
}

testMat <- matrix(test, nrow = 15)
for(i in 1:99){
  for(j in 1:15){
    if(i < j){
      testMat[j, i] = 5
    }
    if((10-j)+i>99){
      testMat[j, i] = 5
    }
  }
}
testReverted <- as.numeric(testMat)

library(ggplot2)

col <- c("red", "blue", "green", "yellow", "black", "#ffcccb", "#add8e6", "#90ee90", "#ffffcc")
colnames(testMat) <- seq(1, 99, by = 1)
df <- data.frame(ydtogo = rep(1:15, 99), ydline = rep(99:1, each = 15), decision = testReverted)
p<-ggplot(df,aes(x=max(ydline)+1-ydline,y=ydtogo))+
  #tile layer
  geom_tile(aes(fill=factor(decision))) +
  #setting the color
  scale_fill_manual(
    "Play Type",
    values=col,
    breaks=c("1","2","3","4","5", "6", "7", "8", "9"),
    labels=c("Punt (strong)","Field Goal (strong)", "Run (strong)", "Pass (strong)","Impossible", "Punt (weak)", "Field Goal (weak)", "Run (weak)", "Pass (weak)")) +
  scale_y_reverse() +
  ggtitle("NFL Fourth Down, 5 Min. Remaining \n and Score Margin +14") +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Yardline") + 
  ylab("Yards to Go")

p
