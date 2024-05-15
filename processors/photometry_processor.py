import mimetypes

from astropy import units
from astropy.io import ascii as astropy_ascii
from astropy.time import Time, TimezoneInfo
import numpy as np

from tom_dataproducts.data_processor import DataProcessor
from exceptions import InvalidFileFormatException, OtherException

# print("Photometry Initialisation Bark!")

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
        print("Bark!")

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

        # print("Processing Bark 1!")
        photometry = []
        # print("Test2!")

        data = astropy_ascii.read(data_product.data.path)
        if len(data) < 1:
            raise InvalidFileFormatException('Empty table or invalid file type')

        ## Set all column names to lowercase
        for column_name in data.colnames:
            data[column_name].name = column_name.lower()

        ## --- Deal with column names ---
        ## Step 1: Time...
        if ('time' not in data.colnames) and ('mjd' not in data.colnames) and ('jd' not in data.colnames):
            raise OtherException("No time column found in file; Photometry requires a time column with the name 'time', 'mjd', or 'jd'.")
        ## Step 2: Magnitude...
        if 'magnitude' not in data.colnames: raise OtherException("No 'magnitude' column found in file; Photometry only supports magnitude.")
        ## Step 2: Error...
        if 'magnitude_error' in data.colnames and 'error' not in data.colnames:
            data['magnitude_error'].name ='error'

        ## Remove superfluous columns:
        for column_name in data.colnames:
            if column_name not in ['time', 'mjd', 'jd', 'telescope', 'magnitude', 'error', 'limit', 'source', 'filter']:
                data.remove_column(column_name)

        ## If Telescope, Filter, and Source columns aren't present, then create and fill in.
        if 'telescope' not in data.colnames:
            # print("Test!")
            s           = ['Unknown Telescope']
            s           *= len(data)
            data["telescope"] = s

        if 'filter' not in data.colnames:
            # print("Test!")
            s           = ['Unknown Filter']
            s           *= len(data)
            data["filter"] = s

        if 'source' not in data.colnames:
            # print("Test!")
            s           = ['Unknown Source']
            s           *= len(data)
            data["source"] = s

        for datum in data:
            if 'time' in datum.colnames:
                if float(datum['time']) > 2400000:
                    time = Time(float(datum['time']), format='jd')
                else:
                    time = Time(float(datum['time']), format='mjd')
            if 'mjd' in datum.colnames:
                time = Time(float(datum['mjd']), format='mjd')
            if 'jd' in datum.colnames:
                time = Time(float(datum['jd']), format='jd')
            if np.ma.is_masked(datum['magnitude']) and 'limit' not in datum.colnames:
                raise OtherException("One or more Magnitude values missing. Please check and re-upload.")
            utc = TimezoneInfo(utc_offset=0*units.hour)
            time.format = 'datetime'
            value = {
                'timestamp': time.to_datetime(timezone=utc),
            }
            for column_name in datum.colnames:
                if not np.ma.is_masked(datum[column_name]):
                    value[column_name] = datum[column_name]
            photometry.append(value)

        # print(len(data))


        # print(photometry[0])

        return photometry


        # photometry = []
        #
        # data = astropy_ascii.read(data_product.data.path)
        # if len(data) < 1:
        #     raise InvalidFileFormatException('Empty table or invalid file type')
        #
        # for datum in data:
        #     time = Time(float(datum['time']), format='mjd')
        #     utc = TimezoneInfo(utc_offset=0*units.hour)
        #     time.format = 'datetime'
        #     value = {
        #         'timestamp': time.to_datetime(timezone=utc),
        #     }
        #     for column_name in datum.colnames:
        #         if not np.ma.is_masked(datum[column_name]):
        #             value[column_name] = datum[column_name]
        #     photometry.append(value)

        return photometry
