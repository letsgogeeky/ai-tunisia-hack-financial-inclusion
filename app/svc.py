import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

features = ['job_type', 'cellphone_access',
            'relationship_with_head', 'education_level',
            'household_size', 'location_type', 'country']


def prepare_data(file_path):
    data = pd.read_csv(file_path)

    le = LabelEncoder()
    if 'bank_account' in data:
        data['bank_account'] = le.fit_transform(data['bank_account'])
    data['job_type'] = le.fit_transform(data['job_type'])

    data['cellphone_access'] = le.fit_transform(data['cellphone_access'])

    data['gender_of_respondent'] = le.fit_transform(data['gender_of_respondent'])

    data['relationship_with_head'] = le.fit_transform(data['relationship_with_head'])

    data['education_level'] = le.fit_transform(data['education_level'])

    data['marital_status'] = le.fit_transform(data['marital_status'])
    data['country'] = le.fit_transform(data['country'])

    data['location_type'] = le.fit_transform(data['location_type'])

    return data


def split_data_and_target(data):
    dataset = data[features]
    target = data['bank_account']

    return dataset, target


def apply_svc(data, target):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30)

    svc = SVC(gamma='auto', kernel='rbf', C=5)

    svc.fit(data_train, target_train)
    pred = svc.predict(data_test)
    print("Support vector machine", accuracy_score(target_test, pred))
    return svc


def evaluate_for_submission(model):
    submission_data = prepare_data('../data/Test_v2.csv')
    filtered_data = submission_data[features]
    submission_pred = model.predict(filtered_data)
    return submission_pred


def write_submission_data(predicted):
    test = pd.read_csv('../data/Test_v2.csv')
    test['uniqueid'] = test['uniqueid'] + ' x ' + test['country']

    submission = pd.DataFrame(columns=['uniqueid', 'bank_account'], data=test)

    submission.bank_account = predicted

    submission.to_csv('../data/SubmissionFile.csv', index=False)


def __main__():
    result_data = prepare_data('../data/Train_v2.csv')  # data
    data, target = split_data_and_target(result_data)
    model = apply_svc(data, target)
    predicted = evaluate_for_submission(model)
    write_submission_data(predicted)


__main__()






