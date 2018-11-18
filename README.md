# kaggle-plasticc

Contains some experimental files for plasticc competition on kaggle.

### Currently Done:
1. Preprocessing for train objects.

### Next:
1. Get loss weights of each classes based on weights posted in kaggle discussions.
2. Set aside validation set and prepare code for k-fold CV.
3. Create full training pytorch pipeline.
If Validation Loss less than 0.55/0.6, then:
1. Preprocessing for test set and submit for LB score.
2. Try different augmentation techniques to increase number of dataset.
3. Potentially split galactic and extra-galactic models?
4. Try some feature engineering with XGBoost, LightGBM, NN and blend.
