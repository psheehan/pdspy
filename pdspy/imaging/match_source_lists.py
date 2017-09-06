import numpy
import astropy
import astropy.coordinates

#def match_source_lists(*lists, tol=0.3, table_names=['1', '2']):
def match_source_lists(tol=0.3, table_names=['1', '2'], *lists):

    for i in range(len(lists)-1):
        if i == 0:
            list1 = lists[i]
        else:
            list1 = joined_list
        list2 = lists[i+1]

        coord1 = astropy.coordinates.SkyCoord(list1["ra"].tolist(), \
                list1['dec'].tolist(), frame='icrs')
        coord2 = astropy.coordinates.SkyCoord(list2["ra"].tolist(), \
                list2["dec"].tolist(), frame='icrs')

        idx, d2d, d3d = astropy.coordinates.match_coordinates_sky(coord1,coord2)

        ids, counts = numpy.unique(idx[d2d.arcsec < tol], return_counts=True)
        for j in range(len(ids)):
            if counts[j] > 1:
                min_d2d = min(d2d[idx == ids[j]].arcsec)
                d2d[(d2d.arcsec != min_d2d) & (idx == ids[j])] = \
                        astropy.coordinates.Angle("0d00m{0:f}s".format(2*tol))

        list1["id"] = numpy.arange(len(list1)) + len(list2)
        list2["id"] = numpy.arange(len(list2))
        list1["id"][d2d.arcsec < tol] = idx[d2d.arcsec < tol]

        if i == 0:
            names = table_names[0:2]
        else:
            names = ["",table_names[i+1]]
        
        joined_list = astropy.table.join(list1, list2, join_type='outer', \
                keys='id', table_names=names, \
                uniq_col_name='{col_name}_{table_name}')

        del list1["id"]
        del list2["id"]
        del joined_list["id"]

        if i == 0:
            joined_list["ra"] = numpy.where(\
                    joined_list["ra_"+names[0]].mask, \
                    joined_list["ra_"+names[1]], joined_list["ra_"+names[0]])
            joined_list["dec"] = numpy.where(\
                    joined_list["dec_"+names[0]].mask, \
                    joined_list["dec_"+names[1]], joined_list["dec_"+names[0]])
        else:
            for name in list2.colnames:
                if name in joined_list.colnames:
                    joined_list.rename_column(name, name+"_"+names[1])

            joined_list["ra"] = numpy.where(\
                    joined_list["ra_"].mask, joined_list["ra_"+names[1]], \
                    joined_list["ra_"])
            joined_list["dec"] = numpy.where(\
                    joined_list["dec_"].mask, joined_list["dec_"+names[1]], \
                    joined_list["dec_"])

            del joined_list["ra_"]
            del joined_list["dec_"]

    del joined_list["ra"]
    del joined_list["dec"]

    return joined_list
