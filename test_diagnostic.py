# ========================== #
# LIBRARIES
# ========================== #
import apecosm
import cartopy.crs as ccrs

# ========================== #
# LOAD DATA
# ========================== #
#report_parameters = apecosm.read_report_params('report_params_conf1.csv')
report_parameters = apecosm.read_report_params('report_params_conf_simu_reference.csv')

# ========================== #
# REPORT
# ========================== #
apecosm.report(report_parameters, crs=ccrs.Mollweide())

#import pdfkit
#path = /home/adrien/MEGA/Code/python-apecosm/report/
#pdfkit.from_file([path+'html/config_meta.html', path+'html/config_report.html', path+'html/fisheries_report.html'], '/home/adrien/MEGA/Code/python-apecosm/tmp/report_test_total.pdf')