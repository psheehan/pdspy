import astropy.table

class Table(astropy.table.Table):

    def read_hdf5(self, *args, **kwargs):
        return

    def write_hdf5(self, *args, **kwargs):
        for col in self.colnames:
            if 'U' in self[col].dtype.str:
                self.replace_column(col, self[col].astype(bytes))

        self.write(*args, **kwargs)

        for col in self.colnames:
            if 'S' in self[col].dtype.str:
                self.replace_column(col, self[col].astype(str))
        return
