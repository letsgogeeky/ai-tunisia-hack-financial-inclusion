import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split


def count_plot(dataset, hue, related_to):
    sns.set(style='whitegrid', color_codes=True)
    sns.set(rc={'figure.figsize': (30.7, 8.27)})
    sns.countplot(related_to, data=dataset, hue=hue)
    sns.despine(offset=10, trim=True)
    plt.show()


def encode_column_label(data_source, column_name):
    column = data_source[column_name].tolist()

    label_encoder = LabelEncoder()
    label_encoder.fit(column)
    result = label_encoder.transform(column)

    return result


def encode_column(data_source, column_name):
    column = data_source[column_name].tolist()

    hot_encoder = OneHotEncoder()
    hot_encoder.fit(column)
    result = hot_encoder.transform(column)

    return result


data = pd.read_csv('../data/Train_v2.csv')

# print(data.groupby('job_type').size())
#
# count_plot(data, 'bank_account', 'education_level')
#
# print(data.dtypes)
#
# print(data['education_level'].unique())

le = LabelEncoder()

data['bank_account'] = le.fit_transform(data['bank_account'])
data['job_type'] = le.fit_transform(data['job_type'])

data['cellphone_access'] = le.fit_transform(data['cellphone_access'])

data['gender_of_respondent'] = le.fit_transform(data['gender_of_respondent'])

data['relationship_with_head'] = le.fit_transform(data['relationship_with_head'])

data['education_level'] = le.fit_transform(data['education_level'])

data['marital_status'] = le.fit_transform(data['marital_status'])
data['country'] = le.fit_transform(data['country'])

data['location_type'] = le.fit_transform(data['location_type'])

cols = ['job_type', 'cellphone_access',
        'relationship_with_head', 'education_level',
         'household_size', 'location_type']

dataset = data[cols]
target = data['bank_account']

print(cols)

def apply_model(model, data_train, data_test, target_train, target_test):
    model = model()

    pred = model.fit(data_train, target_train).predict(data_test)

    print(model.__class__, accuracy_score(target_test, pred, normalize=True))


data_train, data_test, target_train, target_test = train_test_split(dataset, target, test_size=0.30, random_state=10)


from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


apply_model(GaussianNB, data_train, data_test, target_train, target_test)

apply_model(MultinomialNB, data_train, data_test, target_train, target_test)


apply_model(LogisticRegression, data_train, data_test, target_train, target_test)

ln = LinearSVC(random_state=0)

pred = ln.fit(data_train, target_train).predict(data_test)
print("LinearSVC accuracy : ", accuracy_score(target_test, pred, normalize=True))

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(data_train, target_train)
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ", accuracy_score(target_test, pred))

from sklearn.svm import SVC

svc = SVC(gamma='auto')

svc.fit(data_train, target_train)
pred = svc.predict(data_test)

submission_data = pd.read_csv('../data/Test_v2.csv')

submission_data['job_type'] = le.fit_transform(submission_data['job_type'])

submission_data['cellphone_access'] = le.fit_transform(submission_data['cellphone_access'])

submission_data['gender_of_respondent'] = le.fit_transform(submission_data['gender_of_respondent'])

submission_data['relationship_with_head'] = le.fit_transform(submission_data['relationship_with_head'])

submission_data['education_level'] = le.fit_transform(submission_data['education_level'])

submission_data['marital_status'] = le.fit_transform(submission_data['marital_status'])
submission_data['country'] = le.fit_transform(submission_data['country'])

submission_data['location_type'] = le.fit_transform(submission_data['location_type'])

filtered_data = submission_data[cols]
submission_pred = svc.predict(filtered_data)

submission_hint = svc.predict(filtered_data[:10])

print(submission_hint)

print("Support vector machine", accuracy_score(target_test, pred))




# bank_account = encode_column_label(data, 'bank_account')
#
# job_type = encode_column(data, 'job_type') # important
#
# location_type = encode_column(data, 'location_type')
#
# cellphone_access = encode_column(data, 'cellphone_access') # important
#
# gender_of_respondent = encode_column(data, 'gender_of_respondent') # has correlation
#
# relationship_with_head = encode_column(data, 'relationship_with_head') # important
#
# marital_status = encode_column(data, 'marital_status') # not strong
#
# education_level = encode_column(data, 'education_level') # important

# data['job_type_code'] = pd.Series(job_type)

# print(data.head(10))
#

test = pd.read_csv('../data/Test_v2.csv')
test['uniqueid'] = test['uniqueid'] + ' x ' + test['country']
print(test['uniqueid'])

submission = pd.DataFrame(columns=['uniqueid', 'bank_account'], data=test)

submission.bank_account = submission_pred

submission.to_csv('../data/SubmissionFile.csv', index=False)



