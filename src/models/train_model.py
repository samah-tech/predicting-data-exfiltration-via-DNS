def training_model(raw_data_path):
    from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import classification_report
    import pickle
    data = pd.read_csv(raw_data_path)
    print(data.head())
 
    data=data.drop(columns=['timestamp'])
    data=data.drop(columns=['longest_word'])
    data=data.drop(columns=['sld'])
    
    y = data['Label']
    x=data.loc[:, data.columns != 'Label']

    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
    #___________________________________RandomForestClassifier_______________________________________________
    clf = RandomForestClassifier(random_state=0 ,n_estimators=250 ,max_depth=26,bootstrap='true')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy for RandomForestClassifier is :',accuracy*100)
    f1_score = f1_score(y_test, y_pred)
    print('f1_score for RandomForestClassifier is :' ,f1_score )
    print("Classification report:\n", classification_report(y_test, y_pred))
    confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(clf,X_test,y_test)

    pickle.dump(clf,open("D:\\cyber\\ass_2\\CS_ASS\\assignment2-samah-tech\\models\\saved_model",'wb'))


