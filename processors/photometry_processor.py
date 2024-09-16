import mimetypes

from astropy import units
from astropy.io import ascii as astropy_ascii
from astropy.time import Time, TimezoneInfo
import numpy as np

from tom_dataproducts.data_processor import DataProcessor
from tom_dataproducts.exceptions import InvalidFileFormatException

class PhotometryProcessor(DataProcessor):
    def process_data(self, data_product):
        """
        Routes a photometry processing call to a method specific to a file-format.

        :param data_product: Photometric DataProduct which will be processed into the specified format for database
        ingestion
        :type data_product: DataProduct

        :returns: python list of 2-tuples, each with a timestamp and corresponding data
        :rtype: list
        """

        mimetype = mimetypes.guess_type(data_product.data.path)[0]
        if mimetype in self.PLAINTEXT_MIMETYPES:
            photometry = self._process_photometry_from_plaintext(data_product)
            return [(datum.pop('timestamp'), datum, datum.pop('source', '')) for datum in photometry]
        else:
            raise InvalidFileFormatException('Unsupported file type')

    def _process_photometry_from_plaintext(self, data_product):
        """
        Processes the photometric data from a plaintext file into a list of dicts. File is read using astropy as
        specified in the below documentation. The file is expected to be a multi-column delimited file, with headers for
        time, magnitude, filter, and error.
        # http://docs.astropy.org/en/stable/io/ascii/read.html

        :param data_product: Photometric DataProduct which will be processed into a list of dicts
        :type data_product: DataProduct

        :returns: python list containing the photometric data from the DataProduct
        :rtype: list
        """

        print("Running photometry processor...")
        photometry = []

        data = astropy_ascii.read(data_product.data.path)
        if len(data) < 1:
            raise InvalidFileFormatException('Empty table or invalid file type')

        ## Set all column names to lowercase
        for column_name in data.colnames:
            data[column_name].name = column_name.lower()

        ## --- Deal with column names ---
        ## Step 1: Time...
        if ('time' not in data.colnames) and ('mjd' not in data.colnames) and ('jd' not in data.colnames) and ('hjd' not in data.colnames):
            print("Bad time column!")
            raise InvalidFileFormatException("No time column found in file; Photometry requires a time column with the name 'time', 'mjd', 'hjd', or 'jd'.")
        ## Step 2: Magnitude...
        if 'mag' in data.colnames: data['mag'].name = 'magnitude'
        if 'magnitude' not in data.colnames: raise InvalidFileFormatException("No 'magnitude' column found in file; Photometry only supports magnitude.")
        ## Step 2: Error...
        if 'mag_err' in data.colnames: data['mag_err'].name = 'error'
        if 'magnitude_error' in data.colnames and 'error' not in data.colnames:
            data['magnitude_error'].name ='error'

        ## Remove superfluous columns:
        for column_name in data.colnames:
            if column_name not in ['time', 'mjd', 'hjd', 'jd', 'telescope', 'mag', 'magnitude', 'error', 'limit', 'source', 'filter']:
                data.remove_column(column_name)

        print("Recognised columns:", data.colnames)

        ## If Telescope, Filter, and Source columns aren't present, then create and fill in.
        if 'telescope' not in data.colnames:
            s           = ['Unknown Telescope']
            s           *= len(data)
            data["telescope"] = s

        if 'filter' not in data.colnames:
            s           = ['Unknown Filter']
            s           *= len(data)
            data["filter"] = s

        if 'source' not in data.colnames:
            s           = ['Unknown Source']
            s           *= len(data)
            data["source"] = s

        if 'limit' not in data.colnames:
            s           = ['']
            s           *= len(data)
            data["limit"] = s

        for datum in data:
            ## If the Magnitude value is invalid, just skip the whole datum.
            if (datum['magnitude'] == '99.990'):
                continue

            ## For the Time value, make sure it's in the right format.
            if 'time' in datum.colnames:
                if float(datum['time']) > 2400000:
                    time = Time(float(datum['time']), format='jd')
                else:
                    time = Time(float(datum['time']), format='mjd')
            if 'mjd' in datum.colnames:
                time = Time(float(datum['mjd']), format='mjd')
            if 'jd' in datum.colnames:
                time = Time(float(datum['jd']), format='jd')
            if 'hjd' in datum.colnames:
                time = Time(float(datum['hjd']), format='jd')

            ## Check that every row has a valid magnitude or limit value.
            if np.ma.is_masked(datum['magnitude']) and 'limit' not in datum.colnames:
                raise InvalidFileFormatException("One or more Magnitude values missing. Please check and re-upload.")

            ## If the magnitude shows an upper limit, remove.
            datum_magnitude = str(datum['magnitude'])
            # print(datum_magnitude)
            if ('>' in datum_magnitude) or ('<' in datum_magnitude):
                datum['limit'] = float(datum_magnitude[1:])
                datum['magnitude'] = 0
            elif datum_magnitude == '--':
                datum_magnitude = ''
            else:
                datum['magnitude'] = float(datum_magnitude)

            ## Correct the timezone
            utc = TimezoneInfo(utc_offset=0*units.hour)
            time.format = 'datetime'
            value = {
                'timestamp': time.to_datetime(timezone=utc),
            }

            ## For each value,
            for column_name in datum.colnames:

                ## If the column is masked, skip. If the magnitude value is zero, skip magnitude and error.
                if not (np.ma.is_masked(datum[column_name])) \
                    and not (column_name == 'magnitude' and datum['magnitude'] == '0') \
                    and not (column_name == 'error' and datum['magnitude'] == '0'):

                    ## If the column is magnitude, record as a float.
                    if (column_name == 'magnitude' and datum['magnitude'] != '0'):
                        value[column_name] = float(datum[column_name])

                    ## Else, record as whatever it already is.
                    else:
                        value[column_name] = datum[column_name]

            photometry.append(value)


        # for row in photometry:
        #     print(row)

        print("Finished photometry processor.")

        return photometry
