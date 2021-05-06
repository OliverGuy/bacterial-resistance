"""Use parsers to generate usable data from individual contig files.

When called, parsers return a (potentially nested) list of integers 
representing the contig using the following arguments:

id : hashable
    A unique identifier for the contig, e.g. its run accession.
fullpath : path-like
    The path to the contig file.

See the `parsers` dict.
"""
import re

from Bio.SeqIO.FastaIO import SimpleFastaParser

from itertools import islice

nucleotides = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
length_regex = re.compile(".*_length_(\\d*)_.*")


class ContigMaxParser:
    def __init__(self):
        self.max_entries = {}
        self.ndims = 1

    def __call__(self, *args):
        return self.parse(*args)

    def parse(self, id, fullpath):
        """Parses a contig into a list.

        Parameters
        ----------
        id : hashable
            A unique identifier for the contig, e.g. its run accession.
        fullpath : path-like
            The path to the contig file.

        Returns
        -------
        list of ints
            A list of ints representing the maximum-length sequence in the
            contig file.
        """
        if id not in self.max_entries:
            self.max_entries[id] = self.__compute_longest_contig(fullpath)
        with open(fullpath) as handle:
            generator = SimpleFastaParser(handle)
            _, seq = next(islice(generator, self.max_entries[id], None))
        res = [nucleotides[nucl] for nucl in seq]
        return res

    def __compute_longest_contig(self, fullpath):
        max_len = 0
        seq_key = None

        with open(fullpath) as handle:
            for idx, (name, sequence) in enumerate(SimpleFastaParser(handle)):
                # try to infer the sequence length from the name
                m = length_regex.match(name)
                if m:
                    l = int(m.group(1))
                else:
                    l = len(sequence)
                if l > max_len:
                    max_len = l
                    seq_key = idx
        return seq_key


class ContigFullParser:
    """Returns the whole contig as a list of lists"""

    def __init__(self):
        self.ndims = 2

    def __call__(self, *args):
        return self.parse(*args)

    def parse(self, id, fullpath):
        """Parses a contig into a list of lists.

        Parameters
        ----------
        id : hashable
            A unique identifier for the contig, e.g. its run accession.
        fullpath : path-like
            The path to the contig file.

        Returns
        -------
        list of list of ints
            All of the sequences in the contigs.
        """
        with open(fullpath) as handle:
            res = [[nucleotides[nucl] for nucl in sequence]
                   for _, sequence in SimpleFastaParser(handle)]
        return res


class contigCutParser:
    """
    Returns the contig with long sequences split into several nodes

    Parameters
    ----------
    max_length : int, default: 1000
        The max length of the split sequences
    max_nodes : int, default: 300
        The max number of nodes in the sequence; all additional nodes 
        will be cropped out, starting with the shortest sequences.
    """

    def __init__(self, max_length=1000, max_nodes=300):
        self.ndims = 2
        self.max_length = max_length
        self.max_nodes = max_nodes

    def __call__(self, *args):
        return self.parse(*args)

    def parse(self, id, fullpath):
        """Parses a contig into a list of lists.

        Parameters
        ----------
        id : hashable
            A unique identifier for the contig, e.g. its run accession.
        fullpath : path-like
            The path to the contig file.

        Returns
        -------
        list of list of ints
            All of the sequences in the contigs.
        """
        with open(fullpath) as handle:
            res = []
            for _, sequence in SimpleFastaParser(handle):
                for i, nucl in enumerate(sequence):
                    if i % self.max_length == 0:  # the current node is too long
                        res.append([])
                    res[-1].append(nucleotides[nucl])

        if len(res) > self.max_nodes:  # too many nodes
            res.sort(key=len, reverse=True)
            return res[:self.max_nodes]
        return res


parsers = {
    "max": ContigMaxParser,
    "full": ContigFullParser,
    "cut": contigCutParser
}
