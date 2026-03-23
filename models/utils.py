import os
import torch
import matplotlib.pyplot as plt

def _plot_heatmap(M, title, path, vmin=None, vmax=None):
    fig = plt.figure()
    plt.imshow(M, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('key / source (j)')
    plt.ylabel('query / target (i)')
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    
# ---------- main visualizer ----------
@torch.no_grad()
def visualize_attn_maps(
    source_locs: torch.Tensor,   # (N, 3)
    edge_attn: torch.Tensor,     # (E, H) or (E, H, C) or (E,)
    alpha: torch.Tensor,         # (E, H)  # final attention from forward(..., return_attention_weights=True)
    out_dir: str = "./attn_viz",
    prefix: str = "layer0"
):
    """
    저장 경로: out_dir/prefix_*.png
    """
    os.makedirs(out_dir, exist_ok=True)
    device = source_locs.device
    N = source_locs.shape[0]

    # (1) 1 - normalized center distance
    d = torch.cdist(source_locs, source_locs, p=2)       # (N, N)
    dmax = d.max().clamp(min=1e-9)
    score = 1.0 - (d / dmax)                             # (N, N), 대각선은 1
    _plot_heatmap(
        score.detach().cpu().numpy(),
        f"{prefix} (1) 1 - normalized center distance",
        os.path.join(out_dir, f"{prefix}_1_centerdist.png"),
    )

    # (2) edge_attn (raw projection from edge_attr)
    if edge_attn is not None:
        H = edge_attn.shape[-1] if edge_attn.dim() >= 2 else 1
        edge_attn = edge_attn.reshape(N,N,H).permute(1, 0, 2)  # (N, N, H)
        A = edge_attn.detach().cpu().numpy()
        vmin, vmax = A.min(), A.max()

        # 헤드별
        for h in range(H):
            _plot_heatmap(
                A[:, :, h],
                f"{prefix} (2) edge_attn head {h}",
                os.path.join(out_dir, f"{prefix}_2_edgeattn_head{h}.png"),
                vmin=vmin, vmax=vmax
            )
        # 평균
        _plot_heatmap(
            A.mean(axis=2),
            f"{prefix} (2) edge_attn mean over heads",
            os.path.join(out_dir, f"{prefix}_2_edgeattn_mean.png"),
            vmin=vmin, vmax=vmax
        )

    # (3) alpha (final attention)
    if alpha is not None:
        H = alpha.shape[-1] if alpha.dim() >=2 else 1
        alpha = alpha.reshape(N,N,H).permute(1, 0, 2) # (N, N, H_alpha)
        B = alpha.detach().cpu().numpy()
        vmin, vmax = B.min(), B.max()

        for h in range(H):
            _plot_heatmap(
                B[:, :, h],
                f"{prefix} (3) alpha head {h}",
                os.path.join(out_dir, f"{prefix}_3_alpha_head{h}.png"),
                vmin=vmin, vmax=vmax
            )
        _plot_heatmap(
            B.mean(axis=2),
            f"{prefix} (3) alpha mean over heads",
            os.path.join(out_dir, f"{prefix}_3_alpha_mean.png"),
            vmin=vmin, vmax=vmax
        )

    print(f"[✓] Saved attention maps to: {os.path.abspath(out_dir)}")