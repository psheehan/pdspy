import astropy.table

class MaskedColumn(astropy.table.MaskedColumn):

    def __getitem__(self, item):
        x = super(MaskedColumn, self).__getitem__(item)

        try:
            if x.mask:
                return '--'
        except:
            return x
