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