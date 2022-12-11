## Functions

# Binning the data
library(dplyr)
library(tidyr)
bin_func = function(data, nbins, eqdist){
  if(eqdist == TRUE){
    breaks = seq(from = 0, to = 1, by = (1/nbins))
    labels = as.character(seq(from = 1, to = nbins, by = 1))
    row.bin = cut(data[,"scores"], breaks = breaks, labels = labels)
    data.bin = as.data.frame(cbind(data, row.bin))
    colnames(data.bin) = c("scores", "label", "bin")
    data.bin$scores = round(data.bin$scores, 4)
    data.bin$bin = ifelse(data.bin$scores == 0, 1, data.bin$bin)
    #return(data.bin)
  }else{
    data.order = data[order(data[,"scores"], decreasing = FALSE),]
    data.bin = data.order %>% group_by((row_number()-1) %/% (n()/nbins)) 
    colnames(data.bin) = c("scores", "label", "bin")
    data.bin$scores = round(data.bin$scores, 4)
    data.bin$bin = data.bin$bin + 1
    #return(data.bin)
  }
  bin.collapse = data.bin %>%
    group_by(bin) %>%
    summarise(
      count = n(),
      scores_mean = mean(scores),
      label_mean = mean(label),
      label_sd = sd(label)
    ) %>%
    mutate(se = label_sd / count)
    #summarise_at(vars("scores", "label"), list(mean=mean, sd=sd, n=n))
  return(bin.collapse)
}


score_decomp_func = function(x, y_obs, y_hat){
  r = mean(y_obs)
  S_x = mean((x-y_obs)^2)
  S_c = mean((y_hat - y_obs)^2)
  S_r = mean((y_obs - r)^2)
  
  MCB = S_x - S_c
  DSC = S_r - S_c
  UNC = S_r
  decomp = cbind(MCB, DSC, UNC, S_x, S_c, r)
  
  return(decomp)
}


classifier_train_func = function(classifier_model){
  if(classifier_model == "RF"){
    set.seed(1234)
    classifier = randomForest::randomForest(formula, data = train_df, 
                                            mtry = sqrt(length(features)), 
                                            ntree = 300, 
                                            maxnodes = 10, 
                                            nodesize = 3,
                                            importance = FALSE)
  }else if(classifier_model == "GBM"){
    set.seed(1234)
    train_df_gbm = train_df
    train_df_gbm$Target = as.numeric(train_df_gbm$Target)-1
    classifier = gbm::gbm(formula = formula, data = train_df_gbm,
                          distribution = "bernoulli", n.trees = 300,
                          interaction.depth = 5, n.minobsinnode = 3)
  }else if(classifier_model == "NB"){
    set.seed(1234)
    classifier = e1071::naiveBayes(formula = formula, data = train_df)
  }else if (classifier_model == "LR"){
    set.seed(1234)
    classifier = glm(formula = formula, data = train_df, family = "binomial")
  }
  return(classifier)
}


classifier_score_func = function(classifier_model){
  if (classifier_model == "RF"){
    scores_uncal = predict(rf_classifier, newdata = calib_df, type = "prob")[,2]
  } else if (classifier_model == "GBM"){
    scores_uncal = predict(gbm_classifier, newdata = calib_df, type = "response")
  } else if (classifier_model == "NB"){
    scores_uncal = predict(nb_classifier, newdata = calib_df, type = "raw")[,2]
  } else if (classifier_model == "LR"){
    scores_uncal = predict(lr_classifier, newdata = calib_df, type = "response")
  }
  
  # Match Scores with true labels
  scores_uncal_label = as.data.frame(cbind(scores = scores_uncal,
                                           label  = as.numeric(calib_df$Target)-1))
  return(scores_uncal_label)
}

sigmoid_fit_func = function(scores_uncal_label){
  set.seed(1234)
  calibrator_sigmoid = glm(label ~ scores, data = scores_uncal_label, family = "binomial")
  
  preds_sigmoid = as.data.frame(cbind(scores = scores_uncal_label$scores, 
                                      probs = predict(calibrator_sigmoid,  type = "response")))
  preds_sigmoid = preds_sigmoid[order(preds_sigmoid$scores), ]
  return(list(preds_sigmoid = preds_sigmoid,
              calibrator_sigmoid = calibrator_sigmoid))
}

isotonic_fit_func = function(scores_uncal_label){
  scores_uncal_label_sorted = scores_uncal_label[order(scores_uncal_label$scores),]
  iso = isoreg(x = scores_uncal_label_sorted$scores, y = scores_uncal_label_sorted$label)
  x_func = c(0, iso$x, 1)
  y_func = c(0, iso$yf, 1) 
  xy_func = as.data.frame(cbind(x_func, y_func))
  iso_func = approxfun(xy_func$x_func, xy_func$y_func, ties=min)
  
  preds_iso = as.data.frame(cbind(scores = scores_uncal_label$scores, 
                                  probs = iso_func(scores_uncal_label$scores),
                                  label = scores_uncal_label$label))
  preds_iso = preds_iso[order(preds_iso$scores),]
  return(list(preds_iso = preds_iso,
              iso_func = iso_func))
}

uncert_plot = function(data, scores){
  ggplot()+
    geom_ribbon(aes(x=data$regions$x, ymin=data$regions$lower, 
                    ymax=data$regions$upper, 
                    fill = 'Uncertainty'), alpha = 0.6)+
    geom_point(aes(x=sort(scores), y=data$cases$CEP_pav, color='Iso Fit'), size = 2)+
    geom_line(aes(x=sort(scores), y=data$cases$CEP_pav, color='Iso Fit'), size = 1)+
    geom_abline(intercept = 0, slope = 1, linetype="dashed")+
    scale_color_manual('Legend', values=c('steelblue'))+
    scale_fill_manual('Legend', values=c('#a10028'))+
    theme(legend.position = "none", text = element_text(size = 16))+
    xlab("Predicted Score")+
    ylab("Fitted Probability")
}

calc_auc = function(scores, label){
  library(ROCR)
  predroc = prediction(scores, label)
  perf = performance(predroc, "tpr", "fpr")
  auc = performance(predroc, "tpr", "fpr", measure = "auc")
  auc = auc@y.values[[1]]
  return(list(roc = perf, auc = auc))
}

score_test_func = function(classifier_model, classifier, calibrator_sigmoid, iso_func_in){
  if (classifier_model == "RF"){
    scores_uncalib_test = predict(classifier, newdata = test_df, type = "prob")[,2]
  } else if (classifier_model == "GBM"){
    scores_uncalib_test = predict(classifier, newdata = test_df, type = "response")
  } else if (classifier_model == "NB"){
    scores_uncalib_test = predict(classifier, newdata = test_df, type = "raw")[,2]
  } else if (classifier_model == "LR"){
    scores_uncalib_test = predict(classifier, newdata = test_df, type = "response")} 
  
  recalib_sigmoid_test = predict(calibrator_sigmoid, 
                                 newdata = as.data.frame(cbind(scores = scores_uncalib_test)),
                                 type = "response")
  recalib_iso_test = iso_func_in(scores_uncalib_test)
  
  calibs_test = as.data.frame(cbind(label = as.numeric(test_df$Target)-1,
                                    scores_uncal = scores_uncalib_test,
                                    probs_sig = recalib_sigmoid_test,
                                    probs_iso = recalib_iso_test))
  calibs_test = calibs_test[order(calibs_test$scores_uncal),]
  return(calibs_test)
}

train_cv_func = function(classifier_model, train_cv){
  
  Classifier_list = list()
  Sigmoid_list = list()
  Isotonic_list = list()
  
  for (i in 1:5) { 
    # Split data into training folds
    train_classifier = train_cv[which(train_cv$id == i), -ncol(train_cv)]
    train_calibrator = train_cv[which(train_cv$id != i), -ncol(train_cv)]
    
    # Train Classifier
    if(classifier_model == "RF"){
      set.seed(1234)
      classifier = randomForest::randomForest(formula, data = train_classifier, 
                                              mtry = sqrt(length(features)), 
                                              ntree = 300, 
                                              maxnodes = 10, 
                                              nodesize = 3,
                                              importance = FALSE)
    }else if(classifier_model == "GBM"){
      set.seed(1234)
      train_classifier_gbm = train_classifier
      train_classifier_gbm$Target = as.numeric(train_classifier_gbm$Target)-1
      classifier = gbm::gbm(formula = formula, data = train_classifier_gbm,
                            distribution = "bernoulli", n.trees = 300,
                            interaction.depth = 5, n.minobsinnode = 3)
    }else if(classifier_model == "NB"){
      set.seed(1234)
      classifier = e1071::naiveBayes(formula = formula, data = train_classifier)
    }
    
    # Store each Classifier in a list
    Classifier_list[[paste0("Classifier_", i)]] = classifier
    
    
    # Obtain scores for calibrator from classifier model
    if (classifier_model == "RF"){
      scores_uncal = predict(classifier, newdata = train_calibrator, type = "prob")[,2]
    } else if (classifier_model == "GBM"){
      scores_uncal = predict(classifier, newdata = train_calibrator, type = "response")
    } else if (classifier_model == "NB"){
      scores_uncal = predict(classifier, newdata = train_calibrator, type = "raw")[,2]
    }
    
    # Match Scores with true labels
    scores_uncal_label = as.data.frame(cbind(scores = scores_uncal,
                                             label = as.numeric(train_calibrator$Target)-1))
    
    # Sigmoid Scaling
    set.seed(1234)
    calibrator_sigmoid = glm(label ~ scores_uncal, data = scores_uncal_label, family = "binomial")
    
    Sigmoid_list[[paste0("Sigmoid_", i)]] = calibrator_sigmoid
    
    # Isotonic Rescaling
    scores_uncal_label_sorted = scores_uncal_label[order(scores_uncal_label$scores),]
    iso = isoreg(x = scores_uncal_label_sorted$scores, y = scores_uncal_label_sorted$label)
    x_func = c(0, iso$x, 1)
    y_func = c(0, iso$yf, 1) 
    xy_func = as.data.frame(cbind(x_func, y_func))
    iso_func = approxfun(xy_func$x_func, xy_func$y_func, ties=min)
    
    Isotonic_list[[paste0("Isotonic_", i)]] = iso_func
    
  }
  return(list(Classifier_list = Classifier_list,
              Sigmoid_list = Sigmoid_list,
              Isotonic_list = Isotonic_list))
}


test_cv_func = function(classifier_model, Classifier_list,
                        Sigmoid_list, Isotonic_list){
  # Obtain average un-calibrated scores
  uncal_scores = lapply(Classifier_list, function(x){
    #predict(x, newdata = test_df, type = "prob")[,2]
    if (classifier_model == "RF"){
      predict(x, newdata = test_df, type = "prob")[,2]
    } else if (classifier_model == "GBM"){
      predict(x, newdata = test_df, type = "response")
    } else if (classifier_model == "NB"){
      predict(x, newdata = test_df, type = "raw")[,2]
    }
  })
  uncal_avg_scores = cbind(uncal_scores$Classifier_1,
                           uncal_scores$Classifier_2,
                           uncal_scores$Classifier_3,
                           uncal_scores$Classifier_4,
                           uncal_scores$Classifier_5)
  
  uncal_avg_scores = apply(uncal_avg_scores, 1, mean)
  
  sigmoid_scores = lapply(Sigmoid_list, function(x){
    predict(x, newdata = as.data.frame(cbind(scores_uncal = uncal_avg_scores)), type = "response")
  })
  sigmoid_avg_probs = cbind(sigmoid_scores$Sigmoid_1,
                            sigmoid_scores$Sigmoid_2,
                            sigmoid_scores$Sigmoid_3,
                            sigmoid_scores$Sigmoid_4,
                            sigmoid_scores$Sigmoid_5)
  
  sigmoid_avg_probs = apply(sigmoid_avg_probs, 1, mean)
  
  isotonic_scores = lapply(Isotonic_list, function(x){
    x(uncal_avg_scores)
  })
  
  isotonic_avg_probs = cbind(isotonic_scores$Isotonic_1,
                             isotonic_scores$Isotonic_2,
                             isotonic_scores$Isotonic_3,
                             isotonic_scores$Isotonic_4,
                             isotonic_scores$Isotonic_5)
  isotonic_avg_probs = apply(isotonic_avg_probs, 1, mean)
  
  calibs_test = as.data.frame(cbind(label = as.numeric(test_df$Target)-1,
                                    scores_uncal = uncal_avg_scores,
                                    probs_sig = sigmoid_avg_probs,
                                    probs_iso = isotonic_avg_probs))
  calibs_test = calibs_test[order(calibs_test$scores_uncal),]
  
  return(calibs_test)
}





