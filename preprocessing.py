import pandas as pd
from Bio import SeqIO
from glob import glob
import os



def feature_cleaning(df):
    
    df.columns[0] = "ena project" #TODO make foolproof

    # remove NaN-only lines and columns
    df.dropna(axis="index", subset=df.columns[1:], how="all", inplace=True)
    df.dropna(axis="columns", how="all", inplace=True)

    # casefold column and index names
    df.rename(str.casefold, axis="columns", inplace=True)
    df.index.rename(df.index.name.casefold(), inplace=True)

    # TODO some tests/molecules have multiple names 


def load_asts(folder="../SA-ast"):
    # looks for _AST.txt files in folder and loads them into a dataframe list
    # due to inconsistent feature names, features are assumed to be ordered as
    # such: ENA Project ID, Fastq URLs, Run accession, antibiotic_1, ...,
    # antibiotic_r
    assert os.path.exists(folder)
    asts = []
    origin = {}
    for filename in glob(os.path.join(folder, "*_AST.txt")):
        # todo add usecols by index
        cols = pd.read_csv(filename, sep="\t", header=0, nrows=0).columns
        cols = cols.delete(1)  # remove URLs, decreases loading times

        df = pd.read_csv(filename,
                         sep="\t",
                         header=0,
                         usecols=cols,
                         skip_blank_lines=True,
                         index_col=1)  # use run accession as index

        # skip NaN-only tables
        if df.iloc.drop(columns=0).isna().all(axis=None):
            continue
        
        df.columns[0] = "ena project"

        # remove NaN-only lines and columns
        df.dropna(axis="index", subset=df.columns[1:], how="all", inplace=True)
        df.dropna(axis="columns", how="all", inplace=True)
        # capitalize column and index names
        df.rename(str.casefold, axis="columns", inplace=True)
        df.index.rename(df.index.name.casefold(), inplace=True)

        feature_cleaning(df)

        asts.append(df)

        basename = os.path.basename(filename)
        for idx in df.index:
            origin[idx]=basename
    return pd.concat(asts), origin  # TODO there are duplicates

def test_features(asts):
    
    feature_counts = asts.isna().sum()
    n_features = asts.shape[0]

if __name__ == "__main__":
    asts, origin = load_asts()
    test_features(asts)
    exit()
    # test_repeating_projects()
    # test_repeating_bacteria() (runs)
    # test_bacteria_contigs_v_ast()
    # test_label_count()
