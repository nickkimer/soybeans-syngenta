# soybeans-syngenta

Commercialization of Soybean Seed Varieties

## Motivation
The goal of this analysis was to help a biotech company called Syngenta select the soybean varieties that will be most productive upon commercialization. Syngenta products soybeans and commercializes them each year. Each soybean variety that makes it to the market has gone through a very selective process and passed a series of tests. Soybeans go through classes much like students in a school. Each year they are tested and only some will pass; the rest will be removed from consideration. After several years of testing, the soybean varieties reach a graduation year and a select percentage will be selected to finally be commercialized. Syngenta aims to commercialize only the one that will produce the best results for consumers. Therefore the main aim to this study is to use analytics to help Syngenta choose the best varieties and minimize false positives (seeds that test well and perform poorly). 

## Data

The data provided for this analysis is provided by Syngenta and includes information on the past four classes of seeds. A training set is given that consists of the experimental data from 2009 - 2013 that includes the location of testing, the variety and breeding family of the seed, experiment number, replication number, among many other descriptive variables. 

Additionally, the check variety is the elite commercial variety that is used as a benchmark that measures experimental variety performance. The last predictor is relative maturity which reflects differences in amount of time it takes individual varieties to reach physiological maturity. According to Syngenta, late maturing varieties have greater yields than early maturing varieties which makes it an important effect in the analyses. The actual sales volume data for each of these years is also provided. 

Secondly, an evaluation data set that includes information about the varieties that were tested in 2014 is provided. This set details the varieties from which to make the selections from. The limit of 15% results in selecting no more than 5 of the 38 varieties listed in this set. 

## Process

### Deep Learning Architecture
A deep learning modeling structure was chosen to classify the varieties by which one provided the best yields. The training set was the data from the class of 2013 up to the final testing year of 2013 and the test set was the final year of 2014 which was used to validate the model that was built. This modeling structure examined relationships between our variables and assign weights to those relationships. The variables that were explored were all of the ones available in the dataset. Those weights will be reassigned until an optimized loss function is found. The loss function chosen was cross entropy because this it penalizes false positives; this fits well with the business objective which is to allow less varieties through that test well but perform poorly. 

The loss function to use was an available tuning option but also the choice of many other parameters to choose from such as the number of hidden layers and how many nodes are in those layers. There are too many possibilities to do an exhaustive search to find the globally optimal solution. Iterating through, tuning the models, and doing a grid search for the best model in the selection were hypothesized to achieve the best local results. The resulting selected model is the one that has the best and most consistent training and testing AUC and MSE values. The hidden layer parameters that created the best model were with two hidden layers between 5 and 10 nodes.

### Mixed Effects Model
In forecasting a prediction for the number of bags sold for the selected varieties from the output of the deep learning model, a mixed effects model approach was used. A mixed model refers to the use of both fixed and random effects in the same analysis; random effects have levels that are not of primary interest, but are representative of a random selection from a much larger set of levels. The levels identified were at the individual variety level, the family level, and by location level. Starting with the initial fixed effects yield and relative maturity, the likelihood ratio test was done in determining whether or not the added random effect in question was contributing to the model.

The final mixed effects model had the fixed effect terms Yield and Relative Maturity with the random effects Location and Family. With the best model in terms of prediction selected by comparing all the options for fixed and random effects utilizing the likelihood ratio test and thinking about the blocking effects, it was used to predict the number of bags sold for the varieties selected from the deep learning model. 

## Conclusions
The conclusions reached from this conducted analysis are that (1) the deep learning approach to modeling the classification part of the problem achieved an AUC of 0.7392 and which almost beat the baseline standard set at 75% from previous research and (2) predictions for the numbers of bags sold with the selected varieties were done using a mixed effects model to account for the multi-level problem and random effects from location and family. 

While the hypothesis’ objective was not met in terms of the AUC, it was very close to the baseline set at 75% AUC. In context, the AUC was a relatively successful measure for this deep learning model considering that climate effects and other standardization efforts were not made in the dataset provided as opposed to other studies such as by Johnson in 2014. Additionally, the predictions for the number of bags sold was done utilizing the model that was identified to be the best in terms of being able to model the levels of the problem including at the individual variety, family, and location levels. The predictions didn’t seem to deviate very far from the other varieties in the same family which provided a somewhat safe or conservative prediction in that regard. This approach could be improved upon by using unsupervised clustering approaches to identify trends in the data or provide insights into underlying patterns between varieties or families that might have gone undetected in this research. 

