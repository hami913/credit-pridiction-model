# UCI credit dataset and provenance

The uploaded credit project ZIP did **not** contain a dataset file. Its notebook
referenced `/content/UCI_Credit_Card.csv` and its scripts referenced a local Windows
download. To make this project complete, the named **Default of Credit Card
Clients** dataset was retrieved directly from the UCI Machine Learning Repository.

- `raw/UCI_Credit_Card.xls`: the original UCI workbook bytes, with a portable filename.
- `processed/UCI_Credit_Card.csv`: all 30,000 rows, ID, 23 raw predictors, and target.
  The only schema change is renaming `default payment next month` to the original
  project's `default.payment.next.month`. Values and row order are preserved.
- `dataset_provenance.json`: source URL, license, hashes, transformations, and limitations.

The file contains 23,364 non-default and 6,636 default outcomes, verified from the
retrieved workbook. No Iris data is included. No model was trained on this restored
copy during integration. The absent user's local CSV cannot be hash-compared, and
no claim of byte identity to that unavailable file is made.

Feature engineering happens in `src/preprocessing.py` and the unchanged
`src/feature_engineering.py`: 23 inputs plus 15 derived columns = 38 model input
columns. The fitted preprocessing inside the original model produces 41 classifier
columns. This is separate from the 25 columns in the canonical dataset.

## Attribution

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C55S3H

Dataset page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

License: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/

The bundled dataset is supplied for this educational project, with attribution
and the target-column/format changes stated above.
