from .column import MaskedColumn
import astropy.table
import astropy.utils
import h5py
import os

class Table(astropy.table.Table):

    def __init__(self, *args, **kwargs):
        super(Table, self).__init__(*args, **kwargs)

        self.MaskedColumn = MaskedColumn
        if self._masked:
            self._column_class = MaskedColumn

    @staticmethod
    def read_hdf5(*args, **kwargs):
        try:
            self = Table(astropy.table.Table.read(*args, **kwargs, \
                    path="table"), masked=True)
            self.mask = astropy.table.Table.read(*args, **kwargs, path="mask")
        except OSError:
            self = Table(astropy.table.Table.read(*args, **kwargs, \
                    path="table"))

        for col in self.colnames:
            if 'S' in self[col].dtype.str:
                self.replace_column(col, self[col].astype(str))

        return self

    def write_hdf5(self, *args, **kwargs):
        for col in self.colnames:
            if 'U' in self[col].dtype.str:
                self.replace_column(col, self[col].astype(bytes))

        self.write(*args, **kwargs, path='table', overwrite=True)

        for col in self.colnames:
            if 'S' in self[col].dtype.str:
                self.replace_column(col, self[col].astype(str))

        if self.masked:
            self.mask.write(*args, **kwargs, path='mask', append=True)

        return
