from .spatial import NRMSE, RMSE, MSE, NMSE, LInfinity, VMSE, VRMSE

from .spectral import binned_spectral_mse

# I hate that the linter is forcing an all function...
__all__ = ["NRMSE", "RMSE", "MSE", "NMSE",
           "LInfinity", "VMSE", "VRMSE", "binned_spectral_mse"] # I hate this

validation_metric_suite = [RMSE(), NRMSE(), LInfinity(), VRMSE(), binned_spectral_mse()]
