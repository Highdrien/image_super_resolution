from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure

def compute_metrics(config, y_pred, y_true):
    metrics = []

    if config.metrics.PSNR:
        psnr = PeakSignalNoiseRatio()
        metrics.append(psnr(y_pred, y_true))
    # PSRN calculate difference pixel per pixel

    if config.metrics.MSSSIM:
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        metrics.append(ms_ssim(y_pred, y_true))
    # MS-SSIM is a multi-scale metrics based on SSIM
    # SSIM is more a visual metrics compared to PSNR: it looks at the luminance, contrast and structure to find visible differences
    # MS-SSIM is application of SSIM performed at multiple scales through a multi-step downsampling process to have better result.
    return metrics