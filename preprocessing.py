"""
Basic preprocessing pipeline for the AST dataset.
"""

import os
from glob import glob
import numpy as np
import pandas as pd
from Bio import SeqIO

# tests that induce resistance on each other
equivalent_features = {
    'ena projects': 'ena project',
    'methicillin/oxacillin': 'methicilin',
    'oxacillin': 'methicilin',
    'clindamycin': 'constitutive clindamycin',
    'rifampin': 'rifampicin'
}

# intermediate responses that represent a small subset of samples
discarded_responses = ['B', 'I', 'G']


def feature_cleaning(df):
    """Cleans feature names in-place.

    Parameters
    ----------
    df : DataFrame
        DataFrame to clean.
    """
    # casefold column names
    df.rename(str.casefold, axis="columns", inplace=True)

    # rename index name
    df.index = df.index.rename("run accession")
    # make features consistent
    df.rename(columns=equivalent_features, inplace=True)

    # remove the Oxford identifier
    df.drop(columns="oxford comid", axis="columns", errors='ignore', inplace=True)


def result_cleaning(df):
    """Cleans test results in-place.

    Parameters
    ----------
    df : DataFrame
        DataFrame to clean.
    """
    # remove NaN-only columns
    # NB: we need to do it before replacing otherwise Series.str.replace errors out
    df.dropna(axis="columns", how="all", inplace=True)

    for col in df.columns.drop("ena project"):
        df.loc[:, col] = df.loc[:, col].str.replace(" ", "", regex=False)

    df.replace("", np.NaN, inplace=True)
    # remove NaN-only lines and columns
    df.dropna(axis="index", subset=df.columns.drop("ena project"), how="all", inplace=True)
    df.dropna(axis="columns", how="all", inplace=True)


def load_asts(folder="../SA-ast"):
    """Loads anti-biotic sensitivity files into a DataFrame with some preprocessing.

    Looks for `"*_AST.txt"` files in `folder`. Due to inconsistent feature names,
    features are assumed to be ordered as such:
    ENA Project ID, Fastq URLs, Run accession, antibiotic_1, ..., antibiotic_r

    Parameters
    ----------
    folder : path-like, optional
        The folder in which the AST files are contained, by default "../SA-ast"

    Returns
    -------
    DataFrame
        A table containing the data from the AST files.
    """
    assert os.path.exists(folder)
    asts = []
    for filename in glob(os.path.join(folder, "*_AST.txt")):
        # todo add usecols by index
        cols = pd.read_csv(filename, sep="\t", header=0, nrows=0).columns
        cols = cols.delete(1)  # remove URLs, decreases loading times

        df = pd.read_csv(filename,
                         sep="\t",
                         header=0,
                         usecols=cols,
                         na_values=['NT', 0, '-'] + discarded_responses,
                         skip_blank_lines=True,
                         index_col=1)  # use run accession as index

        # skip NaN-only tables
        if df.iloc[:, 1:].isna().all(axis=None):
            continue

        feature_cleaning(df)

        result_cleaning(df)

        # add filename to DataFrame
        basename = os.path.basename(filename)
        df.insert(loc=1, column="ast_filename", value=basename)
        asts.append(df)

    # NB: the following also asserts that the index (i.e. run accession) is a unique identifier
    return pd.concat(asts, verify_integrity=True)


def find_contigs(asts, folder="../SA-contigs", filter=True, dropna=True):
    """Inserts a "contig_path" column to the input DataFrame, with filtering logic.
    New: paths are now relative to the specified folder, not your current working directory.

    Parameters
    ----------
    asts : DataFrame
        The table to modify
    folder : str, optional
        The folder to look for the contigs in, by default "../SA-contigs"
    filter : boolean, optional
        Whether to remove rows whose contig files cannot be found, True by default
    dropna : boolean, optional
        Whether to drop columns that become NaN after filtering, True by default
    """
    regex = os.path.join(folder, "**", "*_contigs.fasta")
    contig_paths = glob(regex, recursive=True)
    contig_paths = [os.path.relpath(path, folder) for path in contig_paths]
    contig_runs = [os.path.basename(path)[:-14] for path in contig_paths]
    paths = pd.Series(contig_paths, index=contig_runs)
    paths.reindex(asts.index)
    asts.insert(value=paths, loc=1, column="contig_path")
    if filter:
        asts.dropna(axis="index", subset=["contig_path"], how='any', inplace=True)
        if dropna:
            asts.dropna(axis="columns", how="all", inplace=True)


def find_contig_path(folder, run_accession):
    """Looks up a contig folder. Note that the folder must have the following specific structure:
    if e.g. looking for `ERR029342`, we will check if folder `<folder>/ERR029/ERR029342*` exists.

    Parameters
    ----------
    folder : path-like
        The SA-contigs folder to look inside of.
    run_accession : str
        The run accession identifier to look up.

    Returns
    -------
    str
        A path to the requested contigs folder.
    """
    contig_folder = glob(os.path.join(
        folder, f"{run_accession[:7]}", "{run}*"))
    if len(contig_folder) == 0:
        return np.NaN
    else:
        return contig_folder[0]


def unique_test_results(asts):
    df = asts.drop("ena project", axis=1, errors='ignore')
    df = df.drop("ast_filename", axis=1, errors='ignore')
    df = df.drop("contig_path", axis=1, errors='ignore')
    return np.unique(df.values.flatten().astype("str"))


def test_features(asts):
    na_counts = asts.isna().sum().sort_values()
    n_lines = asts.shape[0]
    df = pd.DataFrame({
        "na counts": na_counts,
        "na ratios": na_counts / n_lines
    })
    return df


def main():
    """
    Expects files to be located in "../SA-ast" and "../SA-contigs" respectively.
    Currently requires AST files (but not contigs) to be at the root of "SA-contigs".
    Outputs the preprocessed `ast.csv` to the contigs folder.
    """
    asts = load_asts()
    find_contigs(asts)
    print("Feature summary:")
    print(test_features(asts))
    out_filename = "../SA-contigs/ast.csv"
    print(f"Exporting to {out_filename}")
    asts.to_csv(out_filename)


if __name__ == "__main__":
    main()
    exit()
