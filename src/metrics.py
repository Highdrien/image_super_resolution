from torchmetrics import PeakSignalNoiseRatio


def compute_metrics(config, y_pred, y_true):
    metrics = []

    if config.metrics.PSNR:
        psnr = PeakSignalNoiseRatio()
        metrics.append(psnr(y_pred, y_true))
    
    return metrics