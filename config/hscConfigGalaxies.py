import lsst.meas.modelfit
# We do not have transmission curves attached to our validation repos yet
config.processCcd.isr.doAttachTransmissionCurve = False
# these commissioning data do not have the correct header info to apply the stray light correction
config.processCcd.isr.doStrayLight = False
# Run CModel
config.processCcd.calibrate.measurement.plugins.names |= [
    "modelfit_DoubleShapeletPsfApprox", "modelfit_CModel"]
config.processCcd.calibrate.measurement.slots.modelFlux = 'modelfit_CModel'
config.processCcd.calibrate.catalogCalculation.plugins['base_ClassificationExtendedness'].fluxRatio = 0.985
