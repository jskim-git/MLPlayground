import os
from pathlib import Path

import numpy as np
import pandas as pd

project_path = Path(__file__).resolve().parents[2]
data_path = os.path.join(project_path, 'data')
raw_path = os.path.join(data_path, 'raw')

excel_path = os.path.join(raw_path, '20171221회의_CDW 건증코딩북_추출원칙_적용.xlsx')


def load_dataset():
    if os.path.isfile(os.path.join(raw_path, 'abd_raw.pkl')):
        print("Pickle file exists...")
        print("Loading to dataframe...")
        df = pd.read_pickle(os.path.join(raw_path, 'abd_raw.pkl'))
    else:
        print("No pickle file found...")
        print("Reading from Excel file...")
        df = pd.read_excel(excel_path, sheet_name=2)
        print("Creating pickle file for future use...")
        df.to_pickle(os.path.join(raw_path, 'abd_raw.pkl'))

    # Drop non-unique columns, or unused columns
    df = df.drop(["ORD_CD", "FATTY_DEG_K", "ABSONO"], axis=1)

    # Length of Primary Key == Size of dataframe  => Remove Primary Key
    len(df["IPTN_NO"].unique())
    df = df.drop(["IPTN_NO"], axis=1)

    # Tidy up diagnosis (Remove line separators? eg. \n \r), remove non-English language?
    diagnosis = df.IPTN_CNCS_CNTE

    iptn = []
    for i in range(len(diagnosis)):
        d = str(diagnosis[i])
        d = d.replace("\r", " ").replace("\n", " ")
        d = " ".join(d.split())
        # d = re.sub(r'(?<=[.,)])(?=[^\s])', r' ', d)
        word = d.encode('utf-8').decode('ascii', 'ignore')
        word = word.strip()
        remove = ['.', ' ', ',', '']
        if word not in remove:
            iptn.append(d)
        else:
            iptn.append(np.nan)

    df["IPTN_CNCS_CNTE"] = iptn
    df = df[df.IPTN_CNCS_CNTE == df.IPTN_CNCS_CNTE]  # Remove empty diagnosis
    df = df[df["FATTY_DEG"] != -1]  # Remove -1 category

    return df
