import torch


class DepthCompletionMetric:

    def __init__(self, max_depth=10.0):
        super(DepthCompletionMetric, self).__init__()

        self.t_valid = 0.0001

        self.count = 0

        self.max_depth = max_depth

        self.metrics = {
            'RMSE': 0.0,
            'MAE': 0.0,
            'iRMSE': 0.0,
            'iMAE': 0.0,
            'REL': 0.0,
            'D^1': 0.0,
            'D^2': 0.0,
            'D^3': 0.0
        }

        self.sum_metrics = {
            'RMSE': 0.0,
            'MAE': 0.0,
            'iRMSE': 0.0,
            'iMAE': 0.0,
            'REL': 0.0,
            'D^1': 0.0,
            'D^2': 0.0,
            'D^3': 0.0
        }

    def evaluate(self, pred, gt):
        n = gt.shape[0]
        pred = torch.clamp(pred, 0., self.max_depth)
        gt = torch.clamp(gt, 0., self.max_depth)
        with torch.no_grad():
            pred = pred.detach()
            gt = gt.detach()

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            mask = gt > self.t_valid
            num_valid = mask.sum()

            pred = pred[mask]
            gt = gt[mask]

            pred_inv = pred_inv[mask]
            gt_inv = gt_inv[mask]

            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25**2).type_as(ratio)
            del_3 = (ratio < 1.25**3).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            self.metrics['RMSE'] = rmse
            self.metrics['MAE'] = mae
            self.metrics['iRMSE'] = irmse
            self.metrics['iMAE'] = imae
            self.metrics['REL'] = rel
            self.metrics['D^1'] = del_1
            self.metrics['D^2'] = del_2
            self.metrics['D^3'] = del_3

            self.count += n
            for key in self.metrics.keys():
                self.sum_metrics[key] += self.metrics[key] * n

    def reset(self):
        for k in self.metrics.keys():
            self.metrics[k] = 0.0
            self.sum_metrics[k] = 0.0
        self.count = 0

    def average(self):
        avg = {}
        for k in self.metrics.keys():
            avg[k] = self.sum_metrics[k] / self.count
        return avg
