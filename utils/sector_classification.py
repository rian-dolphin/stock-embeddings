import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score  # , top_k_accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def get_sector_score(
    embedding_matrix,
    sectors,
    classifier=SVC(kernel="rbf", probability=True),
    smote=True,
):
    # embedding_matrix = model.embeddings.weight.detach().numpy()

    # -- Cross validation approach

    accuracy_list = []
    # accuracy_list_top_k = []
    # k = 3
    f1_list = []
    recall_list = []
    precision_list = []

    X = embedding_matrix
    y = np.expand_dims(np.array(sectors), axis=1)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(X, y.flatten())):
        X_train = X[train_index]
        y_train = y[
            train_index
        ]  # Based on your code, you might need a ravel call here, but I would look into how you're generating your y
        X_test = X[test_index]
        y_test = y[test_index]  # See comment on ravel and  y_train
        classifier = classifier
        if smote:
            sm = SMOTE()
            X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
            classifier.fit(X_train_oversampled, y_train_oversampled)
        else:
            classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        accuracy_list.append(classifier.score(X_test, y_test))
        # accuracy_list_top_k.append(top_k_accuracy_score(y_test, classifier.predict_proba(X_test),k=k))
        f1_list.append(f1_score(y_test, y_pred, average="weighted"))
        recall_list.append(recall_score(y_test, y_pred, average="weighted"))
        precision_list.append(precision_score(y_test, y_pred, average="weighted"))

    print(f"Precision Score: {np.round(np.mean(precision_list),2)}")
    print(f"Recall Score: {np.round(np.mean(recall_list),2)}")
    print(f"F1 Score: {np.round(np.mean(f1_list),2)}")
    print(f"Accuracy Score: {np.round(np.mean(accuracy_list),2)}")
    # print(f'Accuracy Score Top-{k}: {np.round(np.mean(accuracy_list_top_k),2)}')
    print(f"Accuracy Score: {np.round(np.mean(accuracy_list),2)}")
    # print(f'Accuracy Score Top-{k}: {np.round(np.mean(accuracy_list_top_k),2)}')
