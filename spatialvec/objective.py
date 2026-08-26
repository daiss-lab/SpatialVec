import torch
import torch.nn.functional as F


def reconstruction_loss(model, batch, cfg):
    sdf_pred, occ_logits, boundary_logits, z = model(batch)

    sdf_gt = batch["sdf"]
    occ_gt = batch["occ"]
    boundary_gt = batch["boundary"]
    mask = batch["mask"]

    weight = (1.0 / (sdf_gt.abs() + cfg.sdf_loss_eps)).clamp(
        max=cfg.sdf_loss_clamp
    ) * mask

    sdf_error = F.smooth_l1_loss(sdf_pred, sdf_gt, reduction="none")
    l_sdf = (sdf_error * weight).sum() / weight.sum().clamp_min(1.0)

    l_occ = (
        F.binary_cross_entropy_with_logits(occ_logits, occ_gt, reduction="none") * mask
    ).sum() / mask.sum().clamp_min(1.0)

    l_boundary = (
        F.binary_cross_entropy_with_logits(
            boundary_logits, boundary_gt, reduction="none"
        )
        * mask
    ).sum() / mask.sum().clamp_min(1.0)

    l_code = z.pow(2).sum(dim=1).mean()

    loss = (
        cfg.sdf_weight * l_sdf
        + cfg.occ_weight * l_occ
        + cfg.boundary_weight * l_boundary
        + cfg.code_weight * l_code
    )

    with torch.no_grad():
        live = mask.sum().clamp_min(1.0)
        occ_acc = (
            ((torch.sigmoid(occ_logits) >= 0.5).float() == occ_gt).float() * mask
        ).sum() / live
        boundary_acc = (
            ((torch.sigmoid(boundary_logits) >= 0.5).float() == boundary_gt).float()
            * mask
        ).sum() / live

    metrics = {
        "loss": float(loss.detach()),
        "sdf": float(l_sdf.detach()),
        "occ": float(l_occ.detach()),
        "boundary": float(l_boundary.detach()),
        "occ_acc": float(occ_acc.detach()),
        "boundary_acc": float(boundary_acc.detach()),
    }
    return loss, metrics
