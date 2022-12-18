## Chapter 4: Application


rm(list=ls())
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(ggplot2)
library(randomForest)
library(gbm)
library(e1071)
library(dplyr)
library(tidyr)
library(ggpubr)
source("functions_calibration.r")


# 4.0) Load and Split Data -------------------------------------------------

data_source = "UCI"

if (data_source == "simul"){
  df = read.csv("data_simul.csv")
  colnames(df) = c("X", "F1", "F2", "F3", "F4", "F5", "Target")
  df$Target = as.factor(df$Target)
} else if (data_source == "UCI"){
  #df = read.csv("australian_credit_clean.csv")
  df = get_uci()
  colnames(df) = c("X", "Target", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14")
  df$Target = as.factor(df$Target)
} else if (data_source == "matlab"){
  df = read.csv("matlab_credit.csv")
  colnames(df) = c("X", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "Target")
  df$Target = as.factor(df$Target)
}

# df = read.csv("data_simul.csv")
# colnames(df) = c("X", "F1", "F2", "F3", "F4", "F5", "Target")
# df$Target = as.factor(df$Target)

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
train_df = data[[1]]
calib_df = data[[2]]
test_df = data[[3]]


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

ecd = ggplot()+
  stat_ecdf(aes(x = rf_scored$scores, color = "Random Forest"))+
  stat_ecdf(aes(x = gbm_scored$scores, color = "GBM"))+
  stat_ecdf(aes(x = nb_scored$scores, color = "Naive Bayes"))+
  stat_ecdf(aes(x = lr_scored$scores, color = "Logistic"))+
  xlab("Predicted Score")+
  ylab("Cumulative Density")+
  theme(legend.position = "bottom", text = element_text(size = 12))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#006400", "#6495ED"))
  
  
# Binning and Counting
rf_bc = bin_func(data = rf_scored, nbins = 10, eqdist = FALSE)
gbm_bc = bin_func(data = gbm_scored, nbins = 10, eqdist = FALSE)
nb_bc = bin_func(data = nb_scored, nbins = 10, eqdist = FALSE)
lr_bc = bin_func(data = lr_scored, nbins = 10, eqdist = FALSE)

bc_err = ggplot()+
  geom_point(aes(x = rf_bc$scores_mean, y = rf_bc$label_mean), color = "#6495ED", size = 2)+
  geom_line(aes(x = rf_bc$scores_mean, y = rf_bc$label_mean), color = "#6495ED", size = 1)+
  geom_errorbar(aes(x = rf_bc$scores_mean,
                    ymin = rf_bc$label_mean-rf_bc$label_sd,
                    ymax = rf_bc$label_mean+rf_bc$label_sd),
                color = '#a10028', size = 1)+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Mean Predicted Score")+
  ylab("Event Frequency")+
  theme(legend.position = "bottom", text = element_text(size = 12))

bc_plot = ggplot()+
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
  theme(legend.position = "bottom", text = element_text(size = 12))+
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

sig_plot = ggplot()+
  geom_line(aes(x = rf_sig_fit$scores, y = rf_sig_fit$probs, color = "Random Forest"))+
  geom_line(aes(x = gbm_sig_fit$scores, y = gbm_sig_fit$probs, color = "GBM"))+
  geom_line(aes(x = nb_sig_fit$scores, y = nb_sig_fit$probs, color = "Naive Bayes"))+
  geom_line(aes(x = lr_sig_fit$scores, y = lr_sig_fit$probs, color = "Logistic"))+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Predicted Score")+
  ylab("Fitted Probability")+
  theme(legend.position = "bottom", text = element_text(size = 12))+
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

iso_plot = ggplot()+
  geom_line(aes(x = rf_iso_fit$scores, y = rf_iso_fit$probs, color = "Random Forest"))+
  geom_line(aes(x = gbm_iso_fit$scores, y = gbm_iso_fit$probs, color = "GBM"))+
  geom_line(aes(x = nb_iso_fit$scores, y = nb_iso_fit$probs, color = "Naive Bayes"))+
  geom_line(aes(x = lr_iso_fit$scores, y = lr_iso_fit$probs, color = "Logistic"))+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Predicted Score")+
  ylab("Fitted Probability")+
  theme(legend.position = "bottom", text = element_text(size = 12))+
  scale_color_manual('Classifier', values=c('#9932CC', "#DC143C", "#006400", "#6495ED"))


#ggarrange(ecd, bc_plot, sig_plot, iso_plot, common.legend = TRUE)

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

library(stargazer)
#stargazer(as.matrix(decomp))



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
  #scale_x_continuous(limits = c(0, 0.25))+
  #scale_y_continuous(limits = c(0.5, 1))+
  geom_label(aes(label = paste0("AUC", "\n", "RF: ", round(rf_auc, 4), "\n", "GBM: ", round(gbm_auc, 4), "\n",
                                "NB: ", round(nb_auc, 4), "\n", "Logistic: ", round(lr_auc, 4))), 
             x = 0.7, y = 0.2, color = "yellow", fill = "black", size = 6)

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
r_rf_conf = reliabilitydiag(x = rf_scored$scores, y = rf_scored$label, region.position = "estimate", region.method = "resampling")$x
r_gbm = reliabilitydiag(x = gbm_scored$scores, y = gbm_scored$label, region.position = "diagonal", region.method = "resampling")$x
r_nb = reliabilitydiag(x = nb_scored$scores, y = nb_scored$label, region.position = "diagonal", region.method = "resampling")$x
r_lr = reliabilitydiag(x = lr_scored$scores, y = lr_scored$label, region.position = "diagonal", region.method = "resampling")$x


uncert_plot(data = r_rf, scores = rf_scored$scores, fit_col = "#6495ED")
uncert_plot(data = r_rf_conf, scores = rf_scored$scores, fit_col = "#6495ED")
uncert_plot(data = r_gbm, scores = gbm_scored$scores, fit_col = '#9932CC')
uncert_plot(data = r_nb, scores = nb_scored$scores, fit_col = "#DC143C")
uncert_plot(data = r_lr, scores = lr_scored$scores, fit_col = "#006400")



# 4.2) Re-Calibration -----------------------------------------------------


## 4.2.1) Train-Calibration-Test Split -------------------------------------

rf_test = score_test_func(classifier_model = "RF", classifier = rf_classifier, 
                          calibrator_sigmoid = rf_sig$calibrator_sigmoid,
                          iso_func_in = rf_iso$iso_func, test_df = test_df)
gbm_test = score_test_func(classifier_model = "GBM", classifier = gbm_classifier,
                           calibrator_sigmoid = gbm_sig$calibrator_sigmoid,
                           iso_func = gbm_iso$iso_func, test_df = test_df)
nb_test = score_test_func(classifier_model = "NB", classifier = nb_classifier,
                          calibrator_sigmoid = nb_sig$calibrator_sigmoid,
                          iso_func = nb_iso$iso_func, test_df = test_df)


test_assess_func(data = rf_test)
#stargazer(as.matrix(test_assess_func(data = rf_test)), digits = 4)
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

#train_frac_cv = nrow(df)*0.8
#test_frac_cv = nrow(df)*0.2
#set.seed(231122)
#split = split(df, sample(rep(1:2, times = c(train_frac_cv, test_frac_cv))))
#train_df_cv = split$`1`
#test_df_cv = split$`2`
train_df_cv = rbind(train_df, calib_df)
test_df_cv = test_df

if (data_source == "simul"){
  set.seed(231122)
  id = sample(rep(1:5, each = nrow(train_df_cv)/5))
  hist(id)
} else if (data_source == "UCI"){
  set.seed(231122)
  id = sample(rep(1:5, each = ceiling(nrow(train_df_cv)/5)))
  id = id[1:nrow(train_df_cv)]
  hist(id)
}
#id = sample(rep(1:5, each = nrow(train_df_cv)/5))
#hist(id)

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
                              Sigmoid_list = rf_cv_train$Sigmoid_list, Isotonic_list = rf_cv_train$Isotonic_list,
                              test_df = test_df_cv)
gbm_test_cv_result = test_cv_func(classifier_model = "GBM", Classifier_list = gbm_cv_train$Classifier_list,
                                 Sigmoid_list = gbm_cv_train$Sigmoid_list, Isotonic_list = gbm_cv_train$Isotonic_list,
                                 test_df = test_df_cv)
nb_test_cv_result = test_cv_func(classifier_model = "NB", Classifier_list = nb_cv_train$Classifier_list,
                                 Sigmoid_list = nb_cv_train$Sigmoid_list, Isotonic_list = nb_cv_train$Isotonic_list,
                                 test_df = test_df_cv)

test_assess_func(data = rf_test_cv_result)
#stargazer(as.matrix(test_assess_func(data = rf_test_cv_result)), digits = 4)
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

# RF Three-fold split vs CV
ggplot()+
  geom_line(aes(x = rf_test_cv_result$scores_uncal, y = rf_test_cv_result$probs_iso, color = "Cross-Validation"), size=1.1)+
  geom_line(aes(x = rf_test$scores_uncal, y = rf_test$probs_iso, color = "Simple Split"), size=1.1)+
  geom_point(aes(x = rf_test$scores_uncal, y = rf_test$label), color = "#7a8b8b")+
  geom_abline(intercept = 0, slope = 1, linetype="dashed")+
  xlab("Re-calibrated Score")+
  ylab("Isotonic Fit")+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Re-Calibration Method', values=c("#2a5a83", "#ff700f"))

# ROC Comparison
perf_simple = calc_auc(scores = rf_test$probs_iso, label = rf_test$label)
perf_cv = calc_auc(scores = rf_test_cv_result$probs_iso, label = rf_test_cv_result$label)

perf_simple$auc
perf_cv$auc

ggplot()+
  geom_line(aes(x = perf_simple$roc@x.values[[1]], perf_simple$roc@y.values[[1]], color = "Simple Split"))+
  geom_line(aes(x = perf_cv$roc@x.values[[1]], perf_cv$roc@y.values[[1]], color = "Cross-Validation"))+
  theme(legend.position = "bottom", text = element_text(size = 16))+
  scale_color_manual('Re-Calibration Method', values=c("#2a5a83", "#ff700f"))+
  xlab("False Positive Rate")+
  ylab("True Positive Rate")

#sum(perf_simple$ro@y.values[[1]] - perf_cv$roc@y.values[[1]])

# 4.3) Optimization -------------------------------------------------------


# 4.3.1) Grid Search ------------------------------------------------------

train_df_gbm = train_df
train_df_gbm$Target = as.numeric(train_df_gbm$Target)-1

MCB_GS = matrix(ncol = 4)
MCB_GS = array(dim = c(3,3,3))

n.trees_GS = c(100, 200, 300)
interaction_GS = c(1, 3, 5)
shrinkage_GS = c(0.001, 0.01, 0.1)

mtry_GS = c(1, 3, 5)
ntree_GS = c(100, 200, 300)
maxnodes_GS = c(5, 10, 15)

for (i in c(1:3)) {
  for(j in c(1:3)){
    for (k in c(1:3)) {
      set.seed(1234)
      classifier_GS = randomForest::randomForest(formula, data = train_df, 
                                                 mtry = mtry_GS[i], 
                                                 ntree = ntree_GS[j], 
                                                 maxnodes = maxnodes_GS[k], 
                                                 nodesize = 3,
                                                 importance = FALSE)
      # classifier_GS = gbm::gbm(formula = formula, data = train_df_gbm,
      #                       distribution = "bernoulli", 
      #                       n.trees = n.trees_GS[i],
      #                       interaction.depth = interaction_GS[j], 
      #                       shrinkage = shrinkage_GS[k],
      #                       n.minobsinnode = 3)
      # Score Calibration data set on classifier
      scores_uncal_GS = predict(classifier_GS, newdata = calib_df, type = "prob")[,2]
      
      scores_uncal_label_GS = as.data.frame(cbind(scores = scores_uncal_GS,
                                                  label  = as.numeric(calib_df$Target)-1))
      
      
      # Train Isotonic regression: no hyper parameters needed
      rf_iso_GS = isotonic_fit_func(scores_uncal_label = scores_uncal_label_GS)
      rf_iso_fit_GS = rf_iso_GS$preds_iso
      iso_func_in_GS = rf_iso_GS$iso_func
      
      # Score Test Data on classifier
      scores_uncalib_test_GS = predict(classifier_GS, newdata = test_df, type = "prob")[,2]
      recalib_iso_test_GS = iso_func_in_GS(scores_uncalib_test_GS)
      
      calibs_test = as.data.frame(cbind(label = as.numeric(test_df$Target)-1,
                                        scores_uncal = scores_uncalib_test_GS,
                                        probs_iso = recalib_iso_test_GS))
      calibs_test = calibs_test[order(calibs_test$scores_uncal),]
      
      # Score Decomposition to obtain MCB
      r_GS = summary(reliabilitydiag(x = calibs_test$probs_iso, y = calibs_test$label))
      MCB_GS[i, j, k] = r_GS$miscalibration
    }
  }
}

idx_min_GS = which(MCB_GS == min(MCB_GS), arr.ind = TRUE)


# 4.3.2) Genetic Optimization ---------------------------------------------

library(gramEvol)
library(parallel)

eval_func = function(gen, train, calib, test, model_type, metric){
  # Train Classifier
  set.seed(1234)
  classifier_GA = randomForest::randomForest(formula, data = train, 
                                          mtry = gen[1], 
                                          ntree = gen[2], 
                                          maxnodes = gen[3], 
                                          nodesize = 3,
                                          importance = FALSE)
  
  # Score Calibration data set on classifier
  scores_uncal_GA = predict(classifier_GA, newdata = calib, type = "prob")[,2]
  
  scores_uncal_label_GA = as.data.frame(cbind(scores = scores_uncal_GA,
                                           label  = as.numeric(calib$Target)-1))
  
  # Train Isotonic regression: no hyper parameters needed
  rf_iso_GA = isotonic_fit_func(scores_uncal_label = scores_uncal_label_GA)
  rf_iso_fit_GA = rf_iso_GA$preds_iso
  iso_func_in_GA = rf_iso_GA$iso_func
  
  # Score Test Data on classifier
  scores_uncalib_test_GA = predict(classifier_GA, newdata = test, type = "prob")[,2]
  recalib_iso_test_GA = iso_func_in_GA(scores_uncalib_test_GA)
  
  calibs_test = as.data.frame(cbind(label = as.numeric(test$Target)-1,
                                    scores_uncal = scores_uncalib_test_GA,
                                    probs_iso = recalib_iso_test_GA))
  calibs_test = calibs_test[order(calibs_test$scores_uncal),]
  
  # Score Decomposition to obtain MCB
  r_GS = summary(reliabilitydiag(x = calibs_test$probs_iso, y = calibs_test$label))
  MCB = r_GS$miscalibration
  negDSC = -r_GS$discrimination
  if(metric == "MCB"){
    # Return MCB as objective to minimize
    return(MCB)
  }else if(metric == "DSC"){
    # Return negative DSC as objective to minimize, as DSC should be maximized
    return(negDSC)
  }
  
  
}

monitor_func = function(result){
  cat("Best of gen: ", min(result$best$cost), "Params: ", result$best$genome, "\n")
}

system.time({
  set.seed(231122)
  gen_res_rf = GeneticAlg.int(genomeLen = 3, genomeMin = c(1, 50, 1), genomeMax = c(4, 500, 15),
                              popSize = 9, iterations = 5, mutationChance = 0.2, plapply = mclapply,
                              allowrepeat = TRUE, monitorFunc = monitor_func,
                              evalFunc = function(x) eval_func(x, train = train_df,
                                                               calib = calib_df,
                                                               test = test_df, 
                                                               model_type = "RF", 
                                                               metric = "MCB"))
})

system.time({
set.seed(231122)
gen_res_rf = GeneticAlg.int(genomeLen = 3, genomeMin = c(1, 50, 1), genomeMax = c(4, 500, 15),
                            popSize = 9, iterations = 5, mutationChance = 0.2, plapply = mclapply,
                            allowrepeat = TRUE, monitorFunc = monitor_func,
                            evalFunc = function(x) eval_func(x, train = train_df,
                                                             calib = calib_df,
                                                             test = test_df, 
                                                             model_type = "RF", 
                                                             metric = "DSC"))
})










