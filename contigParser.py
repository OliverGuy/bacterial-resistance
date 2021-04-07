"""Use parsers to generate usable data from individual contig files.

Parsers notably implement the `parse` method, which returns a list
of integers representing the contig using the following arguments:

id : hashable
    A unique identifier for the contig, e.g. its run accession.
fullpath : path-like
    The path to the contig file.
"""
import re

from Bio import SeqIO

from itertools import islice

nucleotides = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}


class ContigMaxParser:
    def __init__(self):
        self.regex = re.compile(".*_length_(\d*)_.*")
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
        generator = SeqIO.parse(fullpath, "fasta")
        seq = next(islice(generator, self.max_entries[id], None))
        res = [nucleotides[nucl] for nucl in seq]
        return res

    def __compute_longest_contig(self, fullpath):
        max_len = 0
        seq_key = None
        records = SeqIO.parse(fullpath, "fasta")

        # iterating the index returns sequence ids as strings
        for id, record in enumerate(records):
            # try to infer the sequence length from the name
            m = self.regex.match(record.id)
            if m:
                l = int(m.group(1))
            else:
                l = len(record)
            if l > max_len:
                max_len = l
                seq_key = id
        assert seq_key is not None
        return seq_key


parsers = {"max": ContigMaxParser}
