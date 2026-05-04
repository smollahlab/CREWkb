import numpy as np
from sklearn.preprocessing import LabelEncoder

_RANDOM_STATE = 42
_CATEGORICAL_COLS = [
    'Pfam domains', 'HGNC gene family tag', 'HGNC gene family description', 
    'Function', 'Modification', 'Protein complex', 'Target molecule',
    'Target entity', 'Product'
]

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
        List of class labels with index corresponding to class encoding.

    Notes
    -----
    Pre-processing comprises removing non-feature columns (protein name and symbol), 
    removing rows which have no data, and encoding class labels to a categorical variable
    taking on values from 0 to `n`-1 where `n` is the number of classes.
    .
    """
    df = df.rename(columns={'REW (Convert to Numbers)': 'REW'})
    df = df.replace(['#', np.nan], '')

    # DROP NON-FEATURE COLUMNS
    df = df.drop(['ID (REMOVE)', 'SYMBOL', 'HGNC approved name'], axis=1)

    # REMOVE ROWS WITHOUT DATA
    to_check = df.columns[5:] # remaining cols: 'Modification' , 'Protein complex', 'Target molecule', 'Target entity', 'Product'
    df = df.dropna(how='all', subset=to_check)

    # SEPARATE FEATURES FROM LABELS
    X = df.iloc[:, 1:].astype('category')
    y_ = df.iloc[:, 0]
    le = LabelEncoder()
    y = le.fit_transform(y_)
    class_labels = list(le.classes_)

    return X, y, class_labels
