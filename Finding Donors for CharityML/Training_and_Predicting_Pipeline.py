### Creating a Training and Predicting Pipeline
''' - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
    - Fit the learner to the sampled training data and record the training time.
    - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
    - Record the total prediction time.
    - Calculate the accuracy score for both the training subset and testing set.
    - Calculate the F-score for both the training subset and testing set.
    - Make sure that you set the `beta` parameter! '''


# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score,accuracy_score
from time import time
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):

    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time


    # Calculate the training time
    results['train_time'] = end-start
    #print('train_time:',results['train_time'])
    # Get the predictions on the test set(X_test),
    # then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end-start
    #print('pred_time:',results['pred_time'])
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)

    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test,predictions_test)

    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300],predictions_train,beta=.5)

    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=.5)

    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    # Return the results
    return results

############################################################
############################################################

# function to interprets the result
# function to interprets the result
def getresult(clflist):
    '''
    input:
    - clflist need to be a list of instantiated classifiers
    e.g: getresult([rfclf,adaclf,bagclf])
    '''
    results={}
    for clf in clflist:
        clf_name=clf.__class__.__name__
        results[clf_name]={}
        # 50% of training data
        scores=train_predict(clf,int(len(X_train)*.5), X_train, y_train, X_test, y_test)
        for k,v in scores.items():
            results[clf_name][k]=v
    print('\n')

    for k, v in results.items():
        print(k+':')
        for kk,vv in v.items():
            print(kk,vv)
        print('\n')
