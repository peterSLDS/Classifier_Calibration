## Chapter 4: Application


rm(list=ls())
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(ggplot2)
library(randomForest)
library(gbm)
library(e1071)
library(dplyr)
source("functions_calibration.r")


# 4.0) Load and Split Data -------------------------------------------------

df = read.csv("data_simul.csv")
colnames(df) = c("X", "F1", "F2", "F3", "F4", "F5", "Target")
df$Target = as.factor(df$Target)

# Split data randomly in three distinct sets
frac_train = nrow(df)*0.6
frac_calib = nrow(df)*0.2
test_calib = nrow(df)*0.2
set.seed(231122)
split = split(df, sample(rep(1:3, 
                             times = c(frac_train, frac_calib, test_calib))))
train_df = split$`1`
calib_df = split$`2`
test_df = split$`3`

# Check that no observation is used more than once
idx = c(train_df$X, calib_df$X, test_df$X)
sum(duplicated(idx))

data = lapply(list(train_df, calib_df, test_df), function(x){x[,-1]} )



# 4.1) Assessing Calibration ----------------------------------------------

target = "Target"
features <- names(train_df)
features <- features[! features %in% c(target)]
formula = as.formula(paste(target, paste(features, collapse=" + "), sep=" ~ "))

rf_classifier = classifier_train_func(classifier_model = "RF")
gbm_classifier = classifier_train_func(classifier_model = "GBM")
nb_classifier = classifier_train_func(classifier_model = "NB")
lr_classifier = classifier_train_func(classifier_model = "LR")

# Assessing calibration of unseen data: calibration data set
rf_scored = classifier_score_func(classifier_model = "RF")
gbm_scored = classifier_score_func(classifier_model = "GBM")
nb_scored = classifier_score_func(classifier_model = "NB")
lr_scored = classifier_score_func(classifier_model = "LR")

# Binning and Counting
rf_bc = bin_func(data = rf_scored, nbins = 10, eqdist = FALSE)
gbm_bc = bin_func(data = gbm_scored, nbins = 10, eqdist = FALSE)
nb_bc = bin_func(data = nb_scored, nbins = 10, eqdist = FALSE)
lr_bc = bin_func(data = lr_scored, nbins = 10, eqdist = FALSE)

ggplot()+
  geom_point(aes(x = rf_bc$scores_mean, y = rf_bc$label_mean))+
  geom_line(aes(x = rf_bc$scores_mean, y = rf_bc$label_mean))+
  geom_errorbar(aes(x = rf_bc$scores_mean,
                    ymin = rf_bc$label_mean-rf_bc$label_sd,
                    ymax = rf_bc$label_mean+rf_bc$label_sd))

ggplot()+
  geom_point(aes(x = rf_bc$scores_mean, y = rf_bc$label_mean, color = "Random Forest"))+
  geom_line(aes(x = rf_bc$scores_mean, y = rf_bc$label_mean, color = "Random Forest"))+
  geom_point(aes(x = gbm_bc$scores_mean, y = gbm_bc$label_mean, color = "GBM"))+
  geom_line(aes(x = gbm_bc$scores_mean, y = gbm_bc$label_mean, color = "GBM"))+
  geom_point(aes(x = nb_bc$scores_mean, y = nb_bc$label_mean, color = "Naive Bayes"))+
  geom_line(aes(x = nb_bc$scores_mean, y = nb_bc$label_mean, color = "Naive Bayes"))+
  geom_point(aes(x = lr_bc$scores_mean, y = lr_bc$label_mean, color = "Logistic"))+
  geom_line(aes(x = lr_bc$scores_mean, y = lr_bc$label_mean, color = "Logistic"))+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Mean Predicted Score")+
  ylab("Event Frequency")+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#006400", "#6495ED"))
  
# Sigmoid Scaling
rf_sig = sigmoid_fit_func(scores_uncal_label = rf_scored)
rf_sig_fit = rf_sig$preds_sigmoid
gbm_sig = sigmoid_fit_func(scores_uncal_label = gbm_scored)
gbm_sig_fit = gbm_sig$preds_sigmoid
nb_sig = sigmoid_fit_func(scores_uncal_label = nb_scored)
nb_sig_fit = nb_sig$preds_sigmoid
lr_sig = sigmoid_fit_func(scores_uncal_label = lr_scored)
lr_sig_fit = lr_sig$preds_sigmoid

ggplot()+
  geom_line(aes(x = rf_sig_fit$scores, y = rf_sig_fit$probs, color = "Random Forest"))+
  geom_line(aes(x = gbm_sig_fit$scores, y = gbm_sig_fit$probs, color = "GBM"))+
  geom_line(aes(x = nb_sig_fit$scores, y = nb_sig_fit$probs, color = "Naive Bayes"))+
  geom_line(aes(x = lr_sig_fit$scores, y = lr_sig_fit$probs, color = "Logistic"))+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Predicted Score")+
  ylab("Fitted Probability")+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#006400", "#6495ED"))

# Isotonic Fit
rf_iso = isotonic_fit_func(scores_uncal_label = rf_scored)
rf_iso_fit = rf_iso$preds_iso
gbm_iso = isotonic_fit_func(scores_uncal_label = gbm_scored)
gbm_iso_fit = gbm_iso$preds_iso
nb_iso = isotonic_fit_func(scores_uncal_label = nb_scored)
nb_iso_fit = nb_iso$preds_iso
lr_iso = isotonic_fit_func(scores_uncal_label = lr_scored)
lr_iso_fit = lr_iso$preds_iso

ggplot()+
  geom_line(aes(x = rf_iso_fit$scores, y = rf_iso_fit$probs, color = "Random Forest"))+
  geom_line(aes(x = gbm_iso_fit$scores, y = gbm_iso_fit$probs, color = "GBM"))+
  geom_line(aes(x = nb_iso_fit$scores, y = nb_iso_fit$probs, color = "Naive Bayes"))+
  geom_line(aes(x = lr_iso_fit$scores, y = lr_iso_fit$probs, color = "Logistic"))+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Predicted Score")+
  ylab("Fitted Probability")+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#006400", "#6495ED"))

# Score Decomposition
rf_decomp = score_decomp_func(x = rf_iso_fit$scores, y_hat = rf_iso_fit$probs, y_obs = rf_iso_fit$label)
gbm_decomp = score_decomp_func(x = gbm_iso_fit$scores, y_hat = gbm_iso_fit$probs, y_obs = gbm_iso_fit$label)
nb_decomp = score_decomp_func(x = nb_iso_fit$scores, y_hat = nb_iso_fit$probs, y_obs = nb_iso_fit$label)
lr_decomp = score_decomp_func(x = lr_iso_fit$scores, y_hat = lr_iso_fit$probs, y_obs = lr_iso_fit$label)

decomp = data.frame(rbind(rf = rf_decomp[,c("S_x", "MCB", "DSC", "UNC")],
                          gbm = gbm_decomp[,c("S_x", "MCB", "DSC", "UNC")],
                          nb = nb_decomp[,c("S_x", "MCB", "DSC", "UNC")],
                          lr = lr_decomp[,c("S_x", "MCB", "DSC", "UNC")]))
decomp





# ROC Curves

rf_perf = calc_auc(scores = rf_scored$scores, label = rf_scored$label)
rf_auc = rf_perf$auc
rf_roc = rf_perf$roc
gbm_perf = calc_auc(scores = gbm_scored$scores, label = gbm_scored$label)
gbm_auc = gbm_perf$auc
gbm_roc = gbm_perf$roc
nb_perf = calc_auc(scores = nb_scored$scores, label = nb_scored$label)
nb_auc = nb_perf$auc
nb_roc = nb_perf$roc
lr_perf = calc_auc(scores = lr_scored$scores, label = lr_scored$label)
lr_auc = lr_perf$auc
lr_roc = lr_perf$roc

ggplot()+
  geom_line(aes(x = rf_roc@x.values[[1]], rf_roc@y.values[[1]], color = "Random Forest"))+
  geom_line(aes(x = gbm_roc@x.values[[1]], gbm_roc@y.values[[1]], color = "GBM"))+
  geom_line(aes(x = nb_roc@x.values[[1]], nb_roc@y.values[[1]], color = "Naive Bayes"))+
  geom_line(aes(x = lr_roc@x.values[[1]], lr_roc@y.values[[1]], color = "Logistic"))+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#006400", "#6495ED"))+
  xlab("False Positive Rate")+
  ylab("True Positive Rate")+
  scale_x_continuous(limits = c(0, 0.25))+
  scale_y_continuous(limits = c(0.5, 1))+
  geom_label(aes(label = paste0("AUC", "\n", "RF: ", round(rf_auc, 4), "\n", "GBM: ", round(gbm_auc, 4), "\n",
                                "NB: ", round(nb_auc, 4), "\n", "Logistic: ", round(lr_auc, 4))), 
             x = 0.2, y = 0.7, color = "yellow", fill = "black", size = 6)

ggplot()+
  geom_line(aes(x = rf_roc@x.values[[1]], rf_roc@y.values[[1]], color = "Random Forest"))+
  geom_line(aes(x = gbm_roc@x.values[[1]], gbm_roc@y.values[[1]], color = "GBM"))+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#006400", "#6495ED"))+
  xlab("False Positive Rate")+
  ylab("True Positive Rate")

# Uncertainty
library(reliabilitydiag)
r_rf = reliabilitydiag(x = rf_scored$scores, y = rf_scored$label, region.position = "diagonal", region.method = "resampling")$x
r_gbm = reliabilitydiag(x = gbm_scored$scores, y = gbm_scored$label, region.position = "diagonal", region.method = "resampling")$x
r_nb = reliabilitydiag(x = nb_scored$scores, y = nb_scored$label, region.position = "diagonal", region.method = "resampling")$x
r_lr = reliabilitydiag(x = lr_scored$scores, y = lr_scored$label, region.position = "diagonal", region.method = "resampling")$x


uncert_plot(data = r_rf, scores = rf_scored$scores)
uncert_plot(data = r_gbm, scores = gbm_scored$scores)
uncert_plot(data = r_nb, scores = nb_scored$scores)
uncert_plot(data = r_lr, scores = lr_scored$scores)



# 4.2) Re-Calibration -----------------------------------------------------


## 4.2.1) Train-Calibration-Test Split -------------------------------------

rf_test = score_test_func(classifier_model = "RF", classifier = rf_classifier, 
                          calibrator_sigmoid = rf_sig$calibrator_sigmoid,
                          iso_func_in = rf_iso$iso_func)
gbm_test = score_test_func(classifier_model = "GBM", classifier = gbm_classifier,
                           calibrator_sigmoid = gbm_sig$calibrator_sigmoid,
                           iso_func = gbm_iso$iso_func)
nb_test = score_test_func(classifier_model = "NB", classifier = nb_classifier,
                          calibrator_sigmoid = nb_sig$calibrator_sigmoid,
                          iso_func = nb_iso$iso_func)

test_assess_func = function(data){
  r_raw = summary(reliabilitydiag(x = data$scores_uncal, y = data$label,
                          region.method = "resampling", region.position = "diagonal"))
  r_sig = summary(reliabilitydiag(x = data$probs_sig, y = data$label,
                          region.method = "resampling", region.position = "diagonal"))
  r_iso = summary(reliabilitydiag(x = data$probs_iso, y = data$label,
                  region.method = "resampling", region.position = "diagonal"))
  result = as.data.frame(rbind(uncal = r_raw[,-1],
                               sigmoid = r_sig[,-1],
                               isotonic = r_iso[,-1]))
  return(result)
}
test_assess_func(data = rf_test)
test_assess_func(data = gbm_test)
test_assess_func(data = nb_test)
# DSC doesn't change when using sigmoid scaling because ordering is identical, with isotonic
# multiple units can get same post-calib score and thus change DSC

# Graphical analysis
ggplot()+
  geom_line(aes(x = rf_test$scores_uncal, y = rf_test$probs_iso, color = "Random Forest"))+
  geom_line(aes(x = gbm_test$scores_uncal, y = gbm_test$probs_iso, color = "GBM"))+
  geom_line(aes(x = nb_test$scores_uncal, y = nb_test$probs_iso, color = "Naive Bayes"))+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Predicted Score")+
  ylab("Fitted Probability")+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#6495ED"))

# 4.2.2) Cross-Validation -------------------------------------------------

train_frac_cv = nrow(df)*0.8
test_frac_cv = nrow(df)*0.2
set.seed(231122)
split = split(df, sample(rep(1:2, times = c(train_frac_cv, test_frac_cv))))
train_df_cv = split$`1`
test_df_cv = split$`2`

# Check that no observation is used more than once
idx = c(train_df_cv$X, test_df_cv$X)
sum(duplicated(idx))

data = lapply(list(train_df_cv, test_df_cv), function(x){x[,-1]} )
train_df_cv = data[[1]]
test_df_cv = data[[2]]

id = sample(rep(1:5, each = nrow(train_df_cv)/5))
hist(id)

train_cv = cbind(train_df_cv, id)

target = "Target"
features <- names(train_df_cv)
features <- features[! features %in% c(target)]
formula = as.formula(paste(target, paste(features, collapse=" + "), sep=" ~ "))

# Train Classification Models and Sigmoid/Isotonic Calibrator
rf_cv_train = train_cv_func(classifier_model = "RF", train_cv = train_cv)
gbm_cv_train = train_cv_func(classifier_model = "GBM", train_cv = train_cv)
nb_cv_train = train_cv_func(classifier_model = "NB", train_cv = train_cv)

# Run unseen test data through models

rf_test_cv_result = test_cv_func(classifier_model = "RF", Classifier_list = rf_cv_train$Classifier_list,
                              Sigmoid_list = rf_cv_train$Sigmoid_list, Isotonic_list = rf_cv_train$Isotonic_list)
gbm_test_cv_result = test_cv_func(classifier_model = "GBM", Classifier_list = gbm_cv_train$Classifier_list,
                                 Sigmoid_list = gbm_cv_train$Sigmoid_list, Isotonic_list = gbm_cv_train$Isotonic_list)
nb_test_cv_result = test_cv_func(classifier_model = "NB", Classifier_list = nb_cv_train$Classifier_list,
                                 Sigmoid_list = nb_cv_train$Sigmoid_list, Isotonic_list = nb_cv_train$Isotonic_list)

test_assess_func(data = rf_test_cv_result)
test_assess_func(data = gbm_test_cv_result)
test_assess_func(data = nb_test_cv_result)

# Graphical analysis
ggplot()+
  geom_line(aes(x = rf_test_cv_result$scores_uncal, y = rf_test_cv_result$probs_iso, color = "Random Forest"))+
  geom_line(aes(x = gbm_test_cv_result$scores_uncal, y = gbm_test_cv_result$probs_iso, color = "GBM"))+
  geom_line(aes(x = nb_test_cv_result$scores_uncal, y = nb_test_cv_result$probs_iso, color = "Naive Bayes"))+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Predicted Score")+
  ylab("Fitted Probability")+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#6495ED"))


# 4.3) Optimization -------------------------------------------------------

library(gramEvol)
library(parallel)

eval_func = function(gen, train, calib, test, target, model_type, metric){
  # Train Classifier
  set.seed(1234)
  classifier = randomForest::randomForest(formula, data = train, 
                                          mtry = gen[1], 
                                          ntree = gen[2], 
                                          #maxnodes = 10, 
                                          #nodesize = 3,
                                          importance = FALSE)
  
  # Score Calibration data set on classifier
  scores_uncal = predict(rf_classifier, newdata = calib, type = "prob")[,2]
  
  # Train Isotonic regression: no hyper parameters needed
  rf_iso = isotonic_fit_func(scores_uncal_label = rf_scored)
  rf_iso_fit = rf_iso$preds_iso
  iso_func_in = rf_iso$iso_func
  
  # Score Test Data on classifier
  scores_uncalib_test = predict(classifier, newdata = test, type = "prob")[,2]
  recalib_iso_test = iso_func_in(scores_uncalib_test)
  
  calibs_test = as.data.frame(cbind(label = as.numeric(test$Target)-1,
                                    scores_uncal = scores_uncalib_test,
                                    probs_iso = recalib_iso_test))
  calibs_test = calibs_test[order(calibs_test$scores_uncal),]
  
  # Score Decomposition to obtain MCB
  MCB = score_decomp_func(x = calibs_test$scores_uncal,
                          y_obs = calibs_test$label,
                          y_hat = calibs_test$probs_iso)[,"MCB"]
  # Return MCB as objective to minimize
  return(abs(MCB))
  
}

monitor_func = function(result){
  cat("Best of gen: ", min(result$best$cost), "Params: ", result$best$genome, "\n")
}

system.time({
gen_res_rf = GeneticAlg.int(genomeLen = 2, genomeMin = c(1, 50), genomeMax = c(4, 500),
                            popSize = 10, iterations = 4, mutationChance = 0.2, plapply = mclapply,
                            allowrepeat = TRUE, monitorFunc = monitor_func,
                            evalFunc = function(x) eval_func(x, train = train_df,
                                                             calib = calib_df,
                                                             test = test_df,
                                                             target = "Target", 
                                                             model_type = "RF", 
                                                             metric = "brier"))
})










