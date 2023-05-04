# ========================== #
# LIBRARIES
# ========================== #
import apecosm
import cartopy.crs as ccrs

# ========================== #
# PATHS
# ========================== #
data_path = '/home/adrien/MEGA/Code/Datarmor/FileZilla/run-spinup-fisheries/'
apecosm_config_path = data_path+'apecosm-config/'
fishing_config_path = data_path+'cnrm-fishing-config/'
output_path = data_path+'output/'
fishing_path = data_path+'fishing-outputs/'

# ========================== #
# LOAD DATA
# ========================== #
report_parameters = apecosm.read_report_params('report_params_conf1.csv')

# ========================== #
# REPORT
# ========================== #
apecosm.report(report_parameters, crs=ccrs.Mollweide())

#import pdfkit
#path = /home/adrien/MEGA/Code/python-apecosm/report/
#pdfkit.from_file([path+'html/config_meta.html', path+'html/config_report.html', path+'html/fisheries_report.html'], '/home/adrien/MEGA/Code/python-apecosm/tmp/report_test_total.pdf')