import pandas as pd
from Bio import SeqIO
from glob import glob
import os
import numpy as np


def feature_cleaning(df):
    """Cleans feature names in-place.

    Parameters
    ----------
    df : DataFrame
        DataFrame to clean.
    """
    # remove NaN-only lines and columns
    df.dropna(axis="index", subset=df.columns[1:], how="all", inplace=True)
    df.dropna(axis="columns", how="all", inplace=True)

    # casefold column names
    df.rename(str.casefold, axis="columns", inplace=True)

    # rename index and project names
    df.index = df.index.rename("run accession")
    df.rename(columns={"ena projects": "ena project"}, inplace=True)

    # TODO some tests/molecules have multiple names


def load_asts(folder="../SA-ast"):
    """Loads anti-biotic sensitivity files into a DataFrame. 

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
                         skip_blank_lines=True,
                         index_col=1)  # use run accession as index

        # skip NaN-only tables
        if df.iloc[:, 1:].isna().all(axis=None):
            continue

        feature_cleaning(df)

        # add filename to DataFrame
        basename = os.path.basename(filename)
        df.insert(loc=1, column="ast_filename", value=basename)
        asts.append(df)

    # NB: the following also asserts that the index (i.e. run accession) is a unique identifier
    return pd.concat(asts, verify_integrity=True)


def filter_contigs(asts, folder="../SA-contigs", dropna=True):
    """Returns the DataFrame filtered on the condition that the contig files pointed to 
    run accession is contained in folder.

    Parameters
    ----------
    asts : DataFrame
        The table to filter.
    folder : str, optional
        The folder to look for the contigs in, by default "../SA-contigs"
    dropna : boolean, optional
        Whether to drop columns that become NaN due to filtering, True by default

    Returns
    -------
    DataFrame
        The filtered table.
    """
    regex = os.path.join(folder, "**", "*_contigs.fasta")
    contig_list = glob(regex, recursive=True)
    contig_list = [os.path.basename(path)[:-14] for path in contig_list]
    exists = asts.index.isin(contig_list)
    filtered = asts.iloc[exists, :]
    if dropna:
        return filtered.dropna(axis="columns", how="all")
    return filtered


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


def test_features(asts):
    na_counts = asts.isna().sum().sort_values()
    n_lines = asts.shape[0]
    df = pd.DataFrame({
        "na counts": na_counts,
        "na ratios": na_counts/n_lines
    })
    return df

def main():
    asts = load_asts()
    asts = filter_contigs(asts)
    print("Feature summary:")
    print(test_features(asts))
    os.makedirs("../out", exist_ok=True)
    out_filename = "../out/ast.csv"
    print(f"Exporting to {out_filename}...")
    asts.to_csv(out_filename)

if __name__ == "__main__":
    main()
    exit()
