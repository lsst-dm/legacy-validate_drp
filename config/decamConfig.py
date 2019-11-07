# use "instcal" exposures from the community pipeline until DecamIsrTask is up to snuff
from lsst.obs.decam.decamNullIsr import DecamNullIsrTask
config.processCcd.isr.retarget(DecamNullIsrTask)
# Run meas_modelfit to compute CModel fluxes
import lsst.meas.modelfit
config.processCcd.calibrate.measurement.plugins.names |= [
    "modelfit_DoubleShapeletPsfApprox", "modelfit_CModel"]
config.processCcd.calibrate.measurement.slots.modelFlux = 'modelfit_CModel'
