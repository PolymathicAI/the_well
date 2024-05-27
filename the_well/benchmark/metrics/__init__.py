from .spatial import MSE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity
from .spectral import binned_spectral_mse
from .wandb_plots import (
    field_histograms,
    plot_all_time_metrics,
    plot_power_spectrum_by_field,
)

# I hate that the linter is forcing an all function...
__all__ = [
    "NRMSE",
    "RMSE",
    "MSE",
    "NMSE",
    "LInfinity",
    "VMSE",
    "VRMSE",
    "binned_spectral_mse",
]  # I hate this

long_time_metrics = ["VRMSE", "RMSE", "binned_spectral_mse"]
validation_metric_suite = [RMSE(), NRMSE(), LInfinity(), VRMSE(), binned_spectral_mse()]
validation_plots = [plot_power_spectrum_by_field, field_histograms]
time_plots = [plot_all_time_metrics]
