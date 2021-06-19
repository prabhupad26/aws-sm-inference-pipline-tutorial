import os
import argparse
from sklearn.externals import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json


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


if __name__ == '__main__':
    # Create a parser object to collect the environment variables that are in the
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--source_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load the train data and convert it to its tf-idf vector format
    train_df = pd.read_csv(os.path.join(args.train,'train.csv'),
                                    index_col=0, engine="python")
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(train_df.data.values)
    train_X = vectors
    train_Y = train_df.label.values

    # Train the model using the fit method
    model = MultinomialNB(alpha=.01)
    model.fit(train_X, train_Y)

    # Below code ensure artifacts create post training are stored in s3

    # Save the model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # Save the tf-idf vocab vector to a pickle file
    pickle.dump(vectorizer.vocabulary_, open("/opt/ml/model/tf_idf_vocab.pkl","wb"))
    print("Successfully created model and vocab")

    # Save the json file containing the mapping of news category names to its label on
    # which the model is trained on
    label_dict = {}
    for i in range(20):
        label_dict[i] = train_df.label_names[train_df.label == i].values[0]
    with open("/opt/ml/model/label_json.json", 'w') as file:
        json.dump(label_dict, file)
    print("Successfully create label dictionary json")
