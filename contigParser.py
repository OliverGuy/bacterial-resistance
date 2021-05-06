"""Use parsers to generate usable data from individual contig files.

Parsers notably implement the `parse` method, which returns a list
of integers representing the contig using the following arguments:

id : hashable
    A unique identifier for the contig, e.g. its run accession.
fullpath : path-like
    The path to the contig file.
"""
import re

from Bio.SeqIO.FastaIO import SimpleFastaParser

from itertools import islice

nucleotides = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
length_regex = re.compile(".*_length_(\\d*)_.*")


class ContigMaxParser:
    def __init__(self):
        self.max_entries = {}

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


parsers = {"max": ContigMaxParser}
