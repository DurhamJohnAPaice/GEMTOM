import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
from astropy import units as u

# filename = "../Data/tns_search_BG_Transients_20240415.csv"

def GEM_to_TOM(filename):
    data = pd.read_csv(filename, header=0, comment='#', index_col=False)

    c = SkyCoord(data.RA, data.DEC, frame='icrs', unit=(u.hourangle, u.deg))

    names = data.Name

    s           = pd.Series(['SIDEREAL'])
    sidereal    = s.repeat(len(names))
    sidereal    = sidereal.set_axis(range(len(names)))

    ra  = c.ra.value
    dec = c.dec.value

    lng = c.galactic.l.value
    lat = c.galactic.b.value

    s       = pd.Series(['Public'])
    groups  = s.repeat(len(names))
    groups  = groups.set_axis(range(len(names)))

    gemtom_dataframe = pd.DataFrame({'name' : names, 'type' : sidereal, 'ra' : ra , 'dec' : dec, 'galactic_lng' : lng, 'galactic_lat' : lat, 'groups' : groups})

    return gemtom_dataframe

    # gemtom_dataframe.to_csv("../Data/GEMTOM_test.csv", index=False)


# Name
# Type SIDEREAL
# RA
# Dec
# Gal_Lon galactic_lng
# Gal_Lat galactic_lat
# Groups "Public, Private"
