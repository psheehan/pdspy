import numpy
import astropy
import astropy.coordinates

def match_source_lists(list1, list2, tol=0.3, table_names=['1', '2']):

    coord1 = astropy.coordinates.SkyCoord(list1["ra"].tolist(), \
            list1['dec'].tolist(), 'icrs')
    coord2 = astropy.coordinates.SkyCoord(list2["ra"].tolist(), \
            list2["dec"].tolist(), 'icrs')

    idx, d2d, d3d = astropy.coordinates.match_coordinates_sky(coord1,coord2)

    dra = (coord1.ra - coord2[idx].ra).arcsec
    ddec = (coord1.ra - coord2[idx].ra).arcsec

    d = (dra**2 + ddec**2)**0.5

    list1["id"] = numpy.arange(len(list1)) + len(list2)
    list2["id"] = numpy.arange(len(list2))
    list1["id"][d < tol] = idx[d < tol]
    
    joined_list = astropy.table.join(list1, list2, join_type='outer', \
            keys='id', table_names=table_names, \
            uniq_col_name='{col_name}_{table_name}')

    del list1["id"]
    del list2["id"]
    del joined_list["id"]

    return joined_list
