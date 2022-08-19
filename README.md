### Develop and deploy an inference pipeline in AWS Sagemaker
This tutorial has the below objectives :
1. Train a machine learning model with a newsgroup [dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) which should be able to categorize
   a news summary from 20 different news categories ([link to python notebook](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/analysis/news_group_classification.ipynb)).
   
2. Create an inference pipeline in AWS Sagemaker and deploy it on sklearn [docker container](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-scikit-learn-spark.html), ([source code folder](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/tree/master/newsgroup_classifier)).


### Building the model

> Prerequisites: 
>  1. Create a python venv, conda env with jupyter notebook. 
>  2. `pip install -r requirements.txt` (Run the requirements.txt file present [here](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/tree/master/analysis/requirements.txt))

* We will load the dataset using the sklearn `fetch_20newsgroups` function


```python
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
```


```python
# There are about 11k train data and 7.5k test data (emails containing the new summary).
print(f" Train data contains {len([newsgroups_train.target_names[index] for index in newsgroups_train.target])} data points")
print(f" Test data contains {len([newsgroups_train.target_names[index] for index in newsgroups_test.target])} data points")
```

     Train data contains 11314 data points
     Test data contains 7532 data points
    

* View one sample data


```python
# Data
print(newsgroups_train['data'][0])
```

    From: lerxst@wam.umd.edu (where's my thing)
    Subject: WHAT car is this!?
    Nntp-Posting-Host: rac3.wam.umd.edu
    Organization: University of Maryland, College Park
    Lines: 15
    
     I was wondering if anyone out there could enlighten me on this car I saw
    the other day. It was a 2-door sports car, looked to be from the late 60s/
    early 70s. It was called a Bricklin. The doors were really small. In addition,
    the front bumper was separate from the rest of the body. This is 
    all I know. If anyone can tellme a model name, engine specs, years
    of production, where this car is made, history, or whatever info you
    have on this funky looking car, please e-mail.
    
    Thanks,
    - IL
       ---- brought to you by your neighborhood Lerxst ----
    
    
    
    
    
    

* Now we will create data frames and do some preprocessing like removing whitespaces, removing header from each datapoint


```python
train_df = pd.DataFrame({'data':['\n'.join(list(filter(lambda x: x != '', data.split('\n')[5:]))) 
                                 for data in newsgroups_train['data']],
                         'data_subject':['\n'.join(list(filter(lambda x: 'subject' in x.lower(), data.split('\n'))))[9:]
                                         for data in newsgroups_train['data']],
                         'label_names':[newsgroups_train.target_names[index] for index in newsgroups_train.target],
                         'label':newsgroups_train.target})
test_df = pd.DataFrame({'data':['\n'.join(list(filter(lambda x: x != '', data.split('\n')[5:]))) 
                                 for data in newsgroups_test['data']],
                         'data_subject':['\n'.join(list(filter(lambda x: 'subject' in x.lower(), data.split('\n'))))[9:]
                                         for data in newsgroups_test['data']],
                         'label_names':[newsgroups_test.target_names[index] for index in newsgroups_test.target],
                         'label':newsgroups_test.target})
```


```python
train_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>data_subject</th>
      <th>label_names</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I was wondering if anyone out there could enl...</td>
      <td>WHAT car is this!?</td>
      <td>rec.autos</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Organization: University of Washington\nLines:...</td>
      <td>SI Clock Poll - Final Call</td>
      <td>comp.sys.mac.hardware</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>well folks, my mac plus finally gave up the gh...</td>
      <td>PB questions...\nthis is a real subjective que...</td>
      <td>comp.sys.mac.hardware</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NNTP-Posting-Host: amber.ssd.csd.harris.com\nX...</td>
      <td>Re: Weitek P9000 ?</td>
      <td>comp.graphics</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>From article &lt;C5owCB.n3p@world.std.com&gt;, by to...</td>
      <td>Re: Shuttle Launch Question</td>
      <td>sci.space</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



* Till now we have just prepared the data to be suitable for futher preprocessing steps, now we will will use it to remove the email id, stopwords, create tokens(divide the entire news summary into tokens(words)) .
> In order add more words to stopwords list in nltk use `stopwords_list.extend(['wordA', 'wordB'])` .


```python
stopwords_list = stopwords.words('english')
def remove_stopwords(sent):
    final_sent = ''
    for word in sent.split(' '):
        if word not in stopwords_list:
            final_sent += word
            final_sent += ' '
    final_sent = re.sub(r'[\w\.-]+@[\w\.-]+', ' ',  final_sent) # remove all email addresses from each datapoint
    return final_sent    
```


```python
train_df['data'] = train_df.data.apply(lambda x: x.strip().lower()) # removed whitespace and lowercase
train_df['data'] = train_df.data.apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x)) # removed special chars
train_df['data'] = train_df.data.apply(remove_stopwords) # removed stopwords
train_df['data_tokens'] = train_df.data.apply(tokenize.word_tokenize)


test_df['data'] = test_df.data.apply(lambda x: x.strip().lower()) # removed whitespace and lowercase
test_df['data'] = test_df.data.apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x)) # removed special chars
test_df['data'] = test_df.data.apply(remove_stopwords) # removed stopwords
test_df['data_tokens'] = test_df.data.apply(tokenize.word_tokenize)
```

> Saving the data for AWS Sagemaker, the generated files will be uploaded to s3 storage using this [script](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/newsgroup_classifier/upload_train_data_s3.py)


```python
train_df.to_csv('../newsgroup_classifier/data/train.csv')
test_df.to_csv('../newsgroup_classifier/data/test.csv')
```

* We will now try to visualize the data to get some insights


```python
# Downloading punkt for tokenization , execute this cell only for  1st run only
nltk.download('punkt')
```


```python
tokenize_train_data = []
for data in train_df.data.values:
    tokenize_train_data += tokenize.word_tokenize(data)
print(len(tokenize_train_data))
```

    1897170
    


```python
freq_words = defaultdict(int)
for data in tokenize_train_data:
    freq_words[data]+=1

f_dist = dict(freq_words)
f_dist_dict = dict([(m, n) for m, n in f_dist.items() if n > 3 and len(m) > 2 and n < 100])
f_dist_dict = sorted(f_dist_dict.items(), key= lambda kv: kv[1], reverse=True)
print("Top 10 most frequent words")
print(f_dist_dict[:10])
print("--------------------------")
print("Top 10 least frequent words")
print(f_dist_dict[-10:])
```

    Top 10 most frequent words
    [('corner', 99), ('constant', 99), ('wonderful', 99), ('observations', 99), ('survey', 99), ('roman', 99), ('oracle', 99), ('ama', 99), ('screw', 99), ('diego', 99)]
    --------------------------
    Top 10 least frequent words
    [('stds', 4), ('positivity', 4), ('prevalences', 4), ('lilac', 4), ('calloway', 4), ('snijpunten', 4), ('jackman', 4), ('melpar', 4), ('chineham', 4), ('critz', 4)]
    

* Create a wordcloud to see some visualization of word frequency


```python
wrd = []
for kv in f_dist_dict:
    wrd += ((kv[0]+' ')*kv[1]).split()
len(wrd)
```




    461680




```python
words_cloud_str = ' '.join(wrd)
wordcloud_newsgrp = WordCloud(background_color="white", max_font_size=5000, max_words=30000).generate(words_cloud_str)

plt.figure(figsize = (50,50))
plt.imshow(wordcloud_newsgrp, interpolation='bilinear')
plt.axis("off")
plt.show()
```


    
![png](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/analysis/assets/output_20_0.png)
    


* Create word vectors: TD-IDF word vectors

> Checking how tf-idf matrix is created with sci-kit learn


```python
len(tokenize_train_data)
```




    1897170




```python
vct = TfidfVectorizer()
sample_text = ['this is sent1 and this is sent3 ', 'let\'s see how tf idf will perform', 'let\'s see how tf idf will perform' ]
vctrs = vct.fit_transform(sample_text)
vctrs
```




    <3x12 sparse matrix of type '<class 'numpy.float64'>'
    	with 19 stored elements in Compressed Sparse Row format>




```python
sample_text = ['this is sent1 and this is sent3', 'let\'s see how tf idf will perform', 'let\'s see how tf idf will perform' ]
a = []
for i in sample_text:
    a += i.split()
len(set(a))
```




    12




```python
print(f"Total number of non-zero elements per sentence (per row in sparse matrix) are : \033[35m{vctrs.nnz / float(vctrs.shape[0])}")
```

    Total number of non-zero elements per sentence (per row in sparse matrix) are : [35m6.333333333333333
    

> Working on train data


```python
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(train_df.data.values)
vectors
```




    <11314x85964 sparse matrix of type '<class 'numpy.float64'>'
    	with 1149626 stored elements in Compressed Sparse Row format>




```python
print(f"Total number of non-zero elements per sentence (per row in sparse matrix) are : \033[35m{vectors.nnz / float(vectors.shape[0])}")
```

    Total number of non-zero elements per sentence (per row in sparse matrix) are : [35m101.61092451829592
    


```python
import sys
print(f"Memory size occupied by the matrix : \033[35m{sys.getsizeof(vectors)} bytes")
```

    Memory size occupied by the matrix : [35m48 bytes
    


```python
vectors_test = vectorizer.transform(test_df.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, train_df.label.values)
pred = clf.predict(vectors_test)
pred
```




    array([ 7, 11,  0, ...,  9, 12, 15])



* Evaluating the model


```python
metrics.f1_score(test_df.label.values, pred, average='macro')
```




    0.8118865539222201




```python
metrics.accuracy_score(y_pred=pred, y_true=test_df.label.values)
```




    0.8187732342007435




```python
fig, ax = plt.subplots(figsize = (20,20))
disp = metrics.plot_confusion_matrix(clf, vectors_test, test_df.label.values,
                                 cmap=plt.cm.Blues,
                                 display_labels=[train_df.label_names[train_df.label == result_label].values[0]
                                                 for result_label in range(20)],
                                 normalize=None, ax=ax)
disp.ax_.set_title("Confusion matrix", fontsize=18)
plt.rcParams.update({'font.size': 19})
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Predicted labels", fontsize=18)
plt.ylabel("True labels", fontsize=18)
plt.show()
```


    
![png](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/analysis/assets/output_35_0.png)
    


#### Since this is a multiclass classification so the one vs all ( whether the output is a specific class or not) approach is taken while plotting the confusion matrix, for example there are 241 instances where alt.atheism category is correctly classifies as alt.atheism (True Positive) while the same class is 23 times misinterpreted as talk.religion.misc (False Negative) and talk.religion.misc is 35 times mis classified as alt.atheism (False Positive)


```python
clf_1vrest = OneVsRestClassifier(MultinomialNB(alpha=.01))
clf_1vrest.fit(vectors, train_df.label.values)
pred_prob = clf_1vrest.predict_proba(vectors_test)
fpr = {}
tpr = {}
thresh ={}

n_class = 20

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(test_df.label.values, pred_prob[:,i], pos_label=i)
    
# plotting 
plt.figure(figsize=(20, 20))
for i in range(n_class): 
    r=random.random()
    g=random.random()
    b=random.random()
    plt.plot(fpr[i], tpr[i], linestyle='--',c=(r, g, b),
             label=f'Class {train_df.label_names[train_df.label == i].values[0]} vs Rest, AUC : {metrics.auc(fpr[i], tpr[i]):0.2f}')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC.png',dpi=300)
```


    
![png](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/analysis/assets/output_37_0.png)
    


* The True positive rate (Recall) VS False positive rate (1- specificity) also known as the ROC curve suggests that the model has classified almost every category correctly. 


```python
clf_1vrest = OneVsRestClassifier(MultinomialNB(alpha=.01))
clf_1vrest.fit(vectors, train_df.label.values)
pred_prob = clf_1vrest.predict_proba(vectors_test)
pr = {}
recall = {}
thresh ={}

n_class = 20

for i in range(n_class):    
    pr[i], recall[i], thresh[i] = metrics.precision_recall_curve(test_df.label.values, pred_prob[:,i], pos_label=i)
    
# plotting 
plt.figure(figsize=(20, 20))
for i in range(n_class): 
    r=random.random()
    g=random.random()
    b=random.random()
    plt.plot(pr[i], recall[i], linestyle='--',c=(r, g, b), 
             label=f'Class {train_df.label_names[train_df.label == i].values[0]} vs Rest, AUC:{metrics.auc(recall[i], pr[i]):0.2f}')
plt.title('Multiclass PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.savefig('Multiclass PRCurve.png',dpi=300)
```


    
![png](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/analysis/assets/output_39_0.png)
    


* From the PR curve it looks there are chances of class imbalance for those categories which has lower Area Under the Curve, so the model could be a bit biased towards the other categories which has greater AUC. This could be avoided by including more data for those categories with less AUC.



### Deploying the model
So till now we have got the training data preprocessed and we know the model with which we have to train with i.e. the MultiNomial NaiveBayes model. We will now prepare some driver scripts which will make use of the python AWS SDK - [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).

> __Prerequisites__:
>  1. Amazon AWS account.
>  2. Generate and save the Access key and secret access key from AWS IAM for accessing AWS services from python boto3 library.
>  3. Set the python environment variables - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION (by dafault it is : us-east-1)

Before executing the script you will need to create a role with these two policies attached - 1. AmazonS3FullAccess, 2. AmazonSageMakerFullAccess :
1. Log in to AWS Console and navigate to IAM.
2. Click on ` Create new role `.
   ![image](https://user-images.githubusercontent.com/11462012/122663665-f8e76000-d1b9-11eb-9d38-c0d9dda48653.png)
3. Select use case as SageMaker and then Click on `Next: Permissions`
   ![image](https://user-images.githubusercontent.com/11462012/122663719-321fd000-d1ba-11eb-8a68-b6f6109f4c2d.png)
4. Search and select these two roles : 1. AmazonS3FullAccess, 2. AmazonSageMakerFullAccess
5. Click on `Next: Tags `, then give the role some name and review and save it.
Now the roles has been created and you can update the same in `exec_train_job_aws_sm.py` file line # 7. 

* Deploy the preprocessed csv files generated above to S3 by running `newsgroup_classifier/upload_train_data_s3.py` script, the [script](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/newsgroup_classifier/upload_train_data_s3.py) is pretty much straight forward you just need to create a s3 bucket and call upload function to upload the file you have in you local to the s3 bucket.
* We will now prepare a training [script](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/newsgroup_classifier/exec_train_job_aws_sm.py) which will train the model using the sklearn container in AWS Sagemaker:
  1. Create a parser object to collect the environment variables that are in the default AWS Scikit-learn Docker container:
      ```
       parser = argparse.ArgumentParser()
       parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
       parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
       parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
       parser.add_argument('--source_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
       args = parser.parse_args()
      ```
      You can check out the [documentation](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#prepare-a-scikit-learn-training-script) for description of these environment variables
  2. Below is the directory structure when the runtime environment is created in Sagemaker:\
     /opt/ml/
            input/
                 config/
                 data/
            output/
            failure/
  3. Read the csv files:
     ```
     train_df = pd.read_csv(os.path.join(args.train,'train.csv'),
                                    index_col=0, engine="python")
     ```
  4. Convert the raw text data to its tf-idf vector form :
     ```
       vectorizer = TfidfVectorizer()
       vectors = vectorizer.fit_transform(train_df.data.values)
       train_X = vectors
       train_Y = train_df.label.values
     ```
     > We will need this vector format so we will save it in `output` folder after training is completed
  5. Fitting the model :
     ```
        model = MultinomialNB(alpha=.01)
        model.fit(train_X, train_Y)
     ```
  6. Saving the model:
     ```
        joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
     ```
  7. Saving the vectorizer using pickle:
     ```
        pickle.dump(vectorizer.vocabulary_, open("/opt/ml/model/tf_idf_vocab.pkl","wb"))
     ```
  8. We will also need a label mapping containing the mapping of numeric labels vs the textual labels so we will read it from the csv file and save it using pickle:
     ```
        label_dict = {}
        for i in range(20):
           label_dict[i] = train_df.label_names[train_df.label == i].values[0]
        with open("/opt/ml/model/label_json.json", 'w') as file:
           json.dump(label_dict, file) 
     ```
  9. Now we are done with the training script preparation, but still we will need to define few functions for loading, processing the output which will be required while testing out model.

```
def model_fn(model_dir):
    """
    This function gets invoked by sagemaker to load the saved model
    :param model_dir: Path where the model was saved
    :return: sklearn.naive_bayes.MultinomialNB
    """
    print(f"Fetching and loading the model from {model_dir}")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
    
    
def input_fn(request_body, request_content_type):
    """
    This function gets invoked by sagemaker to format the input,
    in this case the input string (request_body) get converted into
    its tf-idf vector form
    :param request_body: Input while invoking the endpoint
    :param request_content_type: request content type
    :return: scipy.sparse.csr.csr_matrix
    """
    print(f"Processing the input format")
    if request_content_type == 'text/csv':
        print(f"Loading a tf-idf vocab from pickle file")
        vectorizer = TfidfVectorizer(vocabulary=pickle.load(
            open('/opt/ml/model/tf_idf_vocab.pkl',"rb")))
        vector = vectorizer.fit_transform([request_body])
        print("Vectorizer created")
        return vector
    else:
        raise ValueError("This model only supports text/csv input")
        
    def predict_fn(input_data, model):
    """
    This function gets invoked by sagemaker to get the prediction output
    using the trained model
    :param input_data: output of input_fn function
    :param model: model returned by model_fn function
    :return: numpy.int64 or numpy.int32
    """
    print("Predicting output")
    sample_output = model.predict(input_data)
    return sample_output


def output_fn(prediction, content_type):
    """
    This function gets invoked by sagemaker to take the output of
    predict_fn and convert it to as per the mapping obtained from the json file.
    This json file is create during the training job execution.
    :param prediction: output from predict_fn function
    :param content_type: content type
    :return: str
    """
    print(f"Output predicted is {prediction}, decoding the label")
    with open("/opt/ml/model/label_json.json", "r") as file:
        label_dict = json.load(file)
    output = label_dict[str(prediction[0])]
    return output

```

Now we are all set to execute the [driver script](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/newsgroup_classifier/exec_train_job_aws_sm.py) for training 

On completion of this script an endpoint will get created :
* On success you will see the below in console:
  ![image](https://user-images.githubusercontent.com/11462012/122664875-92fed680-d1c1-11eb-902a-32f2b51a1981.png) 

> Copy this end point name and update it in the variable `endpoint_name` at line # 23 in [`exec_endpoint_aws_sm.py`](https://github.com/prabhupad26/aws-sm-inference-pipline-tutorial/blob/master/newsgroup_classifier/exec_endpoint_aws_sm.py) script then run it.
 
* During training you will see this job getting created in AWS SM :
  ![image](https://user-images.githubusercontent.com/11462012/122664576-ecfe9c80-d1bf-11eb-9997-68aec5a88c2f.png)
* Once the job is completed a new end point will get created here :
  ![image](https://user-images.githubusercontent.com/11462012/122664598-0dc6f200-d1c0-11eb-9218-c6a4812f155a.png)

Once the end point is create now you can test you model by running the `exec_endpoint_aws_sm.py` script, you can validate the output with the expected results.
