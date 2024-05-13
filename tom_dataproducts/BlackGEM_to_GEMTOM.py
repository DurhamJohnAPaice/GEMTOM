import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
from astropy import units as u
from django.contrib import messages

# filename = "../Data/tns_search_BG_Transients_20240415.csv"

def column_name_handler(data, request):

    ## Deal with column names - all lower case!
    for column_name in data.columns:
        data = data.rename(columns={column_name : column_name.lower()})

    return data


def GEM_to_TOM(data, request):

    ## Deal with column names - all lower case!
    for column_name in data.columns:
        if \
        (column_name.lower() == "name") or \
        (column_name.lower() == "ra") or \
        (column_name.lower() == "dec") or \
        (column_name.lower() == "groups") or \
        (column_name.lower() == "type"):
            data = data.rename(columns={column_name : column_name.lower()})

    ## TEST
    print("\n\n\n\n\n\n\n\n\n")
    print(data.columns)
    print(data.iloc[0])

    ## Names
    names = data.name

    ## RA/Dec
    try:
        c = SkyCoord(data.ra, data.dec, frame='icrs', unit=(u.hourangle, u.deg))
    except Exception as e:
        raise ValueError("An error occured while processing your targets: \n   > " + str(e) + " \n Do you have the right values and number of columns?")

    ra  = c.ra.value
    dec = c.dec.value

    lng = c.galactic.l.value
    lat = c.galactic.b.value

    ## Assemble the dataframe
    gemtom_dataframe = pd.DataFrame({
        'name' : names,
        'ra' : ra ,
        'dec' : dec,
        'galactic_lng' : lng,
        'galactic_lat' : lat,
    })

    ## Type
    if 'type' not in data.columns:
        s           = pd.Series(['SIDEREAL'])
        sidereal    = s.repeat(len(names))
        sidereal    = sidereal.set_axis(range(len(names)))
        gemtom_dataframe["type"] = sidereal

    ## Handle incorrect type
    else:
        num = 0
        for this_type, this_name in zip(data.type, data.name):
            num += 1
            if this_type != 'SIDEREAL' and this_type != 'NON_SIDEREAL':
                raise ValueError("Error in Target #" + str(num) + " (" + this_name + "): 'type' must be either SIDEREAL or NON_SIDEREAL (case-sensitive).")
                # messages.warning(request, "Target #" + str(num) + " (" + this_name + "): \
                # Warning! Type is something other than SIDEREAL or NON_SIDEREAL. Delete, fix, and re-import.")

    ## Groups
    if 'groups' not in data.columns:
        s       = pd.Series(['Public'])
        groups  = s.repeat(len(names))
        groups  = groups.set_axis(range(len(names)))
        gemtom_dataframe["groups"] = groups

    ## Get rid of the original name/ra/dec columns
    data = data.drop(columns=['name'])
    data = data.drop(columns=['ra'])
    data = data.drop(columns=['dec'])

    gemtom_dataframe = pd.concat([gemtom_dataframe, data], axis=1).reindex(gemtom_dataframe.index)

    return gemtom_dataframe

    # gemtom_dataframe.to_csv("../Data/GEMTOM_test.csv", index=False)


# Name
# Type SIDEREAL
# RA
# Dec
# Gal_Lon galactic_lng
# Gal_Lat galactic_lat
# Groups "Public, Private"
