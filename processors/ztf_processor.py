import mimetypes
#
from astropy import units
from astropy.io import ascii as astropy_ascii
from astropy.time import Time, TimezoneInfo
import numpy as np
#
# from tom_dataproducts.data_processor import DataProcessor
from django.contrib import messages
from exceptions import InvalidFileFormatException, OtherException
import pandas as pd
from ztfquery import lightcurve
import os
from tom_dataproducts.models import DataProduct, DataProductGroup, ReducedDatum
from data_processor import DataProcessor


# def ZTFProcessor(target, target_id, target_ra, target_dec):
#     print("ZTF Bark!")
#     print(target)
#     print(target_id)
#     print(target_ra)
#     print(target_dec)
#
#     print("-- ZTF: Looking for target...", end="\r")
#     lcq = lightcurve.LCQuery.from_position(target_ra, target_dec, 5)
#     ZTF_data_full = pd.DataFrame(lcq.data)
#     ZTF_data = pd.DataFrame({'JD' : lcq.data.mjd+2400000.5, 'Magnitude' : lcq.data.mag, 'Magnitude_Error' : lcq.data.magerr})
#
#     if len(ZTF_data) == 0:
#         raise Exception("-- ZTF: Target not found. Try AAVSO instead?")
#
#     print("-- ZTF: Looking for target... target found.")
#     print(lcq.__dict__)
#
#     df = ZTF_data
#     print(os.getcwd())
#
#     df.to_csv("./data/GEMTOM_ZTF_Test.csv")



class ZTFProcessor(DataProcessor):
    def process_data(self, data_product):
        """
        Routes a photometry processing call to a method specific to a file-format.

        :param data_product: Photometric DataProduct which will be processed into the specified format for database
        ingestion
        :type data_product: DataProduct

        :returns: python list of 2-tuples, each with a timestamp and corresponding data
        :rtype: list
        """
        # print(data_product.data.path)
        mimetype = mimetypes.guess_type(data_product.data.path)[0]
        # print(mimetype)
        if mimetype in self.PLAINTEXT_MIMETYPES:
            photometry = self._process_photometry_from_plaintext(data_product)
            return [(datum.pop('timestamp'), datum, datum.pop('source', '')) for datum in photometry]
        else:
            raise InvalidFileFormatException('Unsupported file type')

    def _process_photometry_from_plaintext(self, data_product):
        try:
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

            print("Processing ZTF Photometry...")
            photometry = []


            data = astropy_ascii.read(data_product.data.path)
            if len(data) < 1:
                raise InvalidFileFormatException('Empty table or invalid file type')

            if data.colnames[0] == "col1":
                data['col1'].name = 'index'
                data['col2'].name = 'jd'
                data['col3'].name = 'magnitude'
                data['col4'].name = 'error'
                data.remove_row(0)

            # print(data)
            # print("Test3!")

            ## Set all column names to lowercase
            for column_name in data.colnames:
                data[column_name].name = column_name.lower()

            ## --- Deal with column names ---
            ## Step 1: Time...
            if ('time' not in data.colnames) and ('mjd' not in data.colnames) and ('jd' not in data.colnames):
                # messages.error(None,
                #     'Error while fetching ZTF data; '
                # )
                raise Exception("No time column found in file; Photometry requires a time column with the name 'time', 'mjd', or 'jd'.")
                return redirect('/targets/104/?tab=manage-data', '/')

            ## Step 2: Magnitude...
            if 'magnitude' not in data.colnames: raise OtherException("No 'magnitude' column found in file; Photometry only supports magnitude.")
            ## Step 2: Error...
            if 'magnitude_error' in data.colnames and 'error' not in data.colnames:
                data['magnitude_error'].name ='error'

            ## Remove superfluous columns:
            for column_name in data.colnames:
                if column_name not in ['time', 'mjd', 'jd', 'magnitude', 'error']:
                    data.remove_column(column_name)

            print("Considering datapoints...")
            # print(data.colnames)
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
                utc = TimezoneInfo(utc_offset=0*units.hour)
                time.format = 'datetime'
                value = {
                    'timestamp': time.to_datetime(timezone=utc),
                }
                for column_name in datum.colnames:
                    if not np.ma.is_masked(datum[column_name]):
                        value[column_name] = datum[column_name]
                photometry.append(value)

            print("Photometry done!")

            return photometry

        except Exception as e:
            print(e)

        return photometry
