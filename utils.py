import numpy as np

from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import LabelEncoder

############################# DATA PRE-PROCESSING ##############################
def clean_data(df):
    """Perform pre-processing and encoding on input data.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature matrix with shape (n_samples, n_features).
    
    Returns
    -------
    X : ndarray of shape (n_resamples, n_encoded_features)
        Cleaned and encoded feature matrix.
    y : ndarray of shape (n_resamples)
        Class identities of resampled data.
    class_labels : list[str]
        List of class labels with label index corresponding to label encoding.

    Notes
    -----
    Pre-processing comprises removing non-feature columns (protein name and symbol), removing rows which
    have no data, and then performing class rebalancing for nominal data using SMOTEN, and finally
    applying a one-hot encoding on the data.
    """

    # DROP NON-FEATURE COLUMNS
    df.drop(['ID (REMOVE)', 'SYMBOL', 'HGNC approved name'], axis=1, inplace=True)
    df.replace('#', np.nan, inplace=True)

    # REMOVE ROWS WITHOUT DATA
    to_check = df.columns[5:] # remaining cols: 'Modification' , 'Protein complex', 'Target molecule', 'Target entity', 'Product'
    df.dropna(how='all', subset=to_check, inplace=True)
    df.replace(np.nan, '-', inplace = True)

    # SMOTEN CLASS REBALANCING
    X_ = df.iloc[:, 1:].astype('category')
    y_ = df.iloc[:, 0]
    le = LabelEncoder()
    y_categorical = le.fit_transform(y_)
    class_labels = list(le.classes_) # used for plotting
    X_res, y = SMOTEN(random_state=42).fit_resample(X_, y_categorical)

    # ONE-HOT ENCODING
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X = encoder.fit_transform(X_res)

    return X, y, class_labels
