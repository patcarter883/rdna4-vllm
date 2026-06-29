"""CAM step 1-2 (atlas) — build the d=4096 whitened spherical-code canonical-Z atlas from the
committee probe cards (CANONICAL_BUILD_PLAN §1.1-§1.2, CAM_DESIGN §2.1 dial #1).

INPUT  : the per-member probe cards in ckpt/probe/<slug>.pt, each carrying the FULL absolute tap
         embedding A[n_anchor, d_base] over the FIXED 102-anchor bank (sha 28a4acf8). Different
         members have different d_base and tokenizers; the relative-rep matrix R[n,n] is the
         base-neutral, dimension-agnostic alignment key.

RECIPE (the LOCKED build):
  1. UNIFORM z-scored relative-rep per member, recomputed from each card's stored A:
        Ac = center-across-anchors(A) ; Az = Ac / std-per-dim(Ac) ; R = cos(Az_i, Az_j).
     Z-SCORING (not plain centering) is mandatory — the Zaya CCA probe proved plain centered-cosine
     can COLLAPSE onto a single rogue dimension; per-dim standardization is the correct uniform
     relrep (Moschella 2209.15430 incl. per-feature standardization) and is a strict superset of
     centering, so it also tightens the well-behaved members.
  2. CONSENSUS mean-atlas (RLSA 2311.06547): Rbar = mean over members of R  -> [n,n]. Base-neutral
     by construction (every member contributes equally; no single model is the home turf). This is
     the consensus pairwise-anchor geometry the canonical hub embeds.
  3. EMBED the 102 canonical anchor-keys into d=4096. Rbar is a consensus cosine/correlation Gram;
     its eigendecomposition Rbar = U diag(lambda) U^T gives anchor coordinates Y = U sqrt(lambda+)
     (classical MDS / kernel-PCA on the consensus Gram). rank <= n_anchor-1, so the geometry lives
     in <=101 active dims; we place those in the leading coordinates of the d=4096 hub (the rest are
     a true null space the translators / store can use, kept zero in the canonical KEYS).
  4. WHITEN to isotropy: the raw MDS coordinates inherit the consensus eigenvalue spectrum (highly
     anisotropic — a few anchors-axes dominate). Whiten the ACTIVE block so its covariance across
     anchors is ~I (PCA/ZCA whitening) -> isotropic canonical coordinates. Capacity <=> isotropy
     (CAM_DESIGN §2.1): a whitened hub packs keys uniformly and lets translators stay near-affine.
  5. SPHERICAL-CODE shaping: project the 102 whitened keys to the unit hypersphere and run a
     repulsion (Tammes / U-Hop separation, 2410.23126 / 2404.03827) that MAXIMIZES the minimum
     pairwise angle (minimizes max pairwise cosine) while a fidelity anchor keeps the keys near
     their whitened consensus geometry (so the code still REFLECTS committee structure, not a
     generic sphere packing). Output keys = a near-optimal spherical code carrying the consensus.

OUTPUT : ckpt/atlas/canonical_z_v1_local6.pt with the canonical keys Z[n_anchor, d=4096], the
         consensus Gram, the whitening transform, build params + metadata, and the per-member
         alignment diagnostics.

Pure CPU linear algebra over small matrices ([102,102], [102,d], d<=4096) — NO GPU lease.
"""
import argparse
import hashlib
import os

import numpy as np
import torch


def _eigh(M):
    """Symmetric eigendecomposition via numpy (the image's CPU torch lacks LAPACK). Returns
    (evals ascending, evecs) as torch tensors — same convention as torch.linalg.eigh."""
    w, V = np.linalg.eigh(M.detach().cpu().double().numpy())
    return torch.from_numpy(w).float(), torch.from_numpy(V).float()

_HERE = os.path.dirname(os.path.abspath(__file__))

# The 6 in-hand LOCAL committee members (distinct slugs; the cyankiwi__/Zyphra__ files are byte
# duplicates of the slug-named MoE/Zaya cards and are intentionally excluded to avoid double-weight).
LOCAL6 = [
    "Qwen__Qwen3.5-4B",      # GDN hybrid (SSM)
    "Qwen__Qwen3-0.6B",      # GQA
    "unsloth__Llama-3.2-3B", # GQA
    "unsloth__gemma-3-4b-pt",# soft-cap
    "zaya",                  # CCA hybrid (the outlier)
    "qwen36_35b_a3b",        # GDN + MoE
]


def zscore_relrep(A):
    """UNIFORM z-scored relative-rep from a card's absolute tap embeds A[n, d_base] -> R[n, n].

    center across anchors -> per-dim standardize (z-score) -> L2-normalize rows -> cosine Gram.
    The per-dim std clamp avoids div-by-zero on dead dims; this is the relrep that fixes the Zaya
    single-rogue-dimension collapse and is a strict superset of plain centering."""
    A = A.float()
    Ac = A - A.mean(dim=0, keepdim=True)
    sd = Ac.std(dim=0, keepdim=True).clamp_min(1e-6)
    Az = Ac / sd
    Azn = torch.nn.functional.normalize(Az, dim=1)
    return Azn @ Azn.t()


def offdiag(M):
    n = M.shape[0]
    return M[~torch.eye(n, dtype=torch.bool)]


def load_members(probe_dir, slugs):
    members = []
    sha = None
    for slug in slugs:
        path = os.path.join(probe_dir, f"{slug}.pt")
        c = torch.load(path, map_location="cpu", weights_only=False)
        assert c["n_anchor"] == 102, f"{slug}: expected 102 anchors, got {c['n_anchor']}"
        if sha is None:
            sha = c["anchor_sha"]
        assert c["anchor_sha"] == sha, (
            f"{slug}: anchor_sha {c['anchor_sha'][:8]} != {sha[:8]} — members probed DIFFERENT banks; "
            "the atlas is only valid over a single fixed bank.")
        R = zscore_relrep(c["A"])
        members.append({
            "slug": slug,
            "model": c["model"],
            "geometry": c.get("geometry", "?"),
            "d_base": c["d_base"],
            "tap_layer": c.get("tap_layer"),
            "n_layers": c.get("n_layers"),
            "A": c["A"].float(),
            "R": R,
        })
    return members, sha


def consensus_atlas(members):
    """RLSA mean-atlas: equal-weight mean of the members' z-scored relreps -> Rbar[n,n]."""
    R = torch.stack([m["R"] for m in members])           # [M, n, n]
    Rbar = R.mean(dim=0)
    Rbar = 0.5 * (Rbar + Rbar.t())                        # symmetrize (guard fp drift)
    return Rbar


def embed_mds(Rbar, d_hub, rel_floor=1e-3):
    """Classical-MDS / kernel-PCA embedding of the consensus Gram into d_hub.

    Rbar is a consensus cosine Gram (diag ~1, symmetric PSD-ish). Eig-decompose, keep the spectrum
    above a RELATIVE floor (rel_floor * lambda_max) -> anchor coords Y_active[n, r] = U[:,+]*sqrt(lam).
    The relative floor drops the centering-induced rank-deficient null direction (eigenvalue ~0)
    whose whitening would otherwise be a meaningless 1/0 amplification. r is the GENUINE consensus
    rank — the dimensionality of the canonical active subspace the 102 keys are packed into."""
    evals, evecs = _eigh(Rbar)                            # ascending
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])
    floor = rel_floor * float(evals[0])
    keep = evals > floor
    r = min(int(keep.sum()), d_hub)
    lam = evals[:r].clamp_min(0)
    U = evecs[:, :r]
    Y_active = U * lam.sqrt().unsqueeze(0)                # [n, r]
    return Y_active, evals, r


def whiten(Y_active, alpha=0.6, eps_rel=1e-3):
    """SHRINKAGE PCA-whiten the active block toward isotropy.

    Pure whitening (alpha=1) maps the coordinate covariance to I — but for a point cloud that ALSO
    flattens the consensus pairwise geometry the keys must keep (retrieval ranking is a CAM
    non-negotiable). So we whiten by a CONTROLLABLE amount: scale each principal axis by
    lambda^(-alpha/2) instead of lambda^(-1/2). alpha=0 -> identity (full structure, anisotropic);
    alpha=1 -> full isotropy (structure destroyed). alpha in (0,1) trades isotropy for geometry
    preservation — the plan's "whiten to isotropy ... Z stays clean for retrieval" balance.

    eps_rel floors the per-axis eigenvalue at eps_rel*lambda_max so the rank-deficient null direction
    (eigenvalue ~0 from the n-point centering) is not amplified into noise. Returns the shrink-whitened
    block, the transform, the pre-whiten mean, and the covariance condition number before/after."""
    n = Y_active.shape[0]
    Yc = Y_active - Y_active.mean(dim=0, keepdim=True)
    cov = (Yc.t() @ Yc) / (n - 1)                         # [r, r]
    ev, V = _eigh(cov)                                    # ascending
    floor = eps_rel * float(ev.max())
    ev = ev.clamp_min(floor)
    scale = ev.pow(-alpha / 2.0)                          # lambda^(-alpha/2): shrinkage whitening
    W = V @ torch.diag(scale) @ V.t()
    Yw = Yc @ W
    covw = (Yw.t() @ Yw) / (n - 1)
    evw, _ = _eigh(covw)
    # condition number over the GENUINE active subspace: the n-point centering leaves one
    # rank-deficient null eigenvalue (floored), which is not part of the canonical geometry — drop
    # the single smallest eigenvalue from both spectra so the condition number measures real isotropy.
    ev_a = torch.sort(ev, descending=True).values[:-1].clamp_min(1e-12)
    evw_a = torch.sort(evw, descending=True).values[:-1].clamp_min(1e-12)
    cond_before = float(ev_a.max() / ev_a.min())
    cond_after = float(evw_a.max() / evw_a.min())
    return Yw, W, Yc.mean(dim=0), cond_before, cond_after, ev


def spherical_code(Yw, steps=4000, lr=0.05, fidelity=0.10, seed=0):
    """Shape the whitened keys toward a near-optimal spherical code (max-min pairwise angle).

    Project to the unit sphere, then gradient-repel: minimize a softmin-of-cosine energy (pushes the
    closest pairs apart, the Tammes objective) + a fidelity term anchoring each key near its whitened
    consensus direction (so the code still carries committee geometry, not a generic packing). Keys
    stay on the sphere by renormalizing each step. Returns the unit-sphere code Zc[n, r]."""
    torch.manual_seed(seed)
    Z0 = torch.nn.functional.normalize(Yw, dim=1)         # consensus directions on the sphere
    Z = Z0.clone().requires_grad_(True)
    n = Z.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    opt = torch.optim.Adam([Z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        Zn = torch.nn.functional.normalize(Z, dim=1)
        cos = Zn @ Zn.t()
        cos_off = cos[~eye].view(n, n - 1)
        # softmax-weighted repulsion: dominated by the CLOSEST pairs -> pushes the min-angle up.
        beta = 20.0
        w = torch.softmax(beta * cos_off, dim=1)
        repel = (w * cos_off).sum(dim=1).mean()
        fid = (1.0 - (Zn * Z0).sum(dim=1)).mean()         # 1 - cos to consensus direction
        loss = repel + fidelity * fid
        loss.backward()
        opt.step()
    with torch.no_grad():
        Zc = torch.nn.functional.normalize(Z, dim=1)
    return Zc.detach(), Z0


def code_stats(Z):
    """min/mean/max pairwise angle (deg) + cosine for a unit-sphere code Z[n, r]."""
    n = Z.shape[0]
    cos = (Z @ Z.t()).clamp(-1, 1)
    off = cos[~torch.eye(n, dtype=torch.bool)]
    ang = torch.rad2deg(torch.acos(off.clamp(-0.9999, 0.9999)))
    return {
        "min_angle_deg": float(ang.min()),
        "mean_angle_deg": float(ang.mean()),
        "max_cos": float(off.max()),
        "mean_cos": float(off.mean()),
    }


def per_member_alignment(members, Rbar, Zc):
    """How cleanly does each member map into the canonical hub?
    (a) member-R vs consensus Rbar  (rho over off-diagonals) — does this member agree with consensus?
    (b) member-R vs FINAL code Gram (cos of Zc) — does the spherical code preserve the member's
        pairwise geometry (the retrieval-ranking that must survive into the hub)?"""
    n = Rbar.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    code_gram = (Zc @ Zc.t())
    rb = Rbar[mask]
    cg = code_gram[mask]
    out = []
    for m in members:
        rm = m["R"][mask]
        rho_consensus = float(torch.corrcoef(torch.stack([rm, rb]))[0, 1])
        rho_code = float(torch.corrcoef(torch.stack([rm, cg]))[0, 1])
        out.append({
            "slug": m["slug"], "geometry": m["geometry"], "d_base": m["d_base"],
            "rho_to_consensus": rho_consensus, "rho_to_code": rho_code,
        })
    return out


def neutrality_check(members, Rbar):
    """Base-NEUTRALITY: leave-one-out. Drop each member, rebuild consensus, measure how much the
    consensus moves (rho of LOO-Rbar vs full Rbar). If one member DOMINATES, dropping it would move
    the consensus a lot (low rho). Near-1 LOO rho for ALL members == genuinely base-neutral."""
    n = Rbar.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    full = Rbar[mask]
    M = len(members)
    out = []
    for i in range(M):
        loo = torch.stack([members[j]["R"] for j in range(M) if j != i]).mean(0)
        loo = 0.5 * (loo + loo.t())
        rho = float(torch.corrcoef(torch.stack([loo[mask], full]))[0, 1])
        out.append({"dropped": members[i]["slug"], "loo_rho_to_full": rho})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default=os.path.join(_HERE, "ckpt", "probe"))
    ap.add_argument("--out", default=os.path.join(_HERE, "ckpt", "atlas", "canonical_z_v1_local6.pt"))
    ap.add_argument("--d-hub", type=int, default=4096)
    ap.add_argument("--members", nargs="+", default=LOCAL6)
    ap.add_argument("--whiten-alpha", type=float, default=0.5,
                    help="shrinkage whitening strength 0..1 (0=keep anisotropy/full structure, "
                         "1=full isotropy/structure-destroying); 0.6 balances isotropy vs geometry")
    ap.add_argument("--code-steps", type=int, default=4000)
    ap.add_argument("--code-lr", type=float, default=0.05)
    ap.add_argument("--fidelity", type=float, default=0.10)
    ap.add_argument("--smoke", nargs="*", default=None,
                    help="if given, build on just these slugs and print stats (no save)")
    args = ap.parse_args()

    torch.manual_seed(0)
    slugs = args.smoke if args.smoke else args.members
    print(f"[atlas] d_hub={args.d_hub} members({len(slugs)})={slugs}")

    members, sha = load_members(args.probe_dir, slugs)
    print(f"[atlas] anchor_sha={sha[:16]} (all members agree)")
    print("[atlas] per-member z-scored relrep:")
    for m in members:
        off = offdiag(m["R"])
        print(f"    {m['slug']:24s} geom={m['geometry']:5s} d_base={m['d_base']:5d} "
              f"L={m['tap_layer']}/{m['n_layers']} | Rz off-diag mean/std/max "
              f"{off.mean():+.3f}/{off.std():.3f}/{off.max():+.3f}")

    Rbar = consensus_atlas(members)
    off = offdiag(Rbar)
    print(f"[atlas] consensus Rbar: diag~{float(Rbar.diagonal().mean()):.4f} "
          f"off-diag mean/std/max {off.mean():+.3f}/{off.std():.3f}/{off.max():+.3f}")

    Y_active, evals, r = embed_mds(Rbar, args.d_hub)
    pos = evals[evals > 0]
    print(f"[atlas] MDS embed: rank r={r} (active dims of {args.d_hub}); "
          f"top-5 eig {[round(float(x),3) for x in evals[:5]]}; "
          f"eig spread max/min(+) {float(pos.max()):.3f}/{float(pos.min()):.4g}")

    Yw, W, ymean, cond_b, cond_a, cov_ev = whiten(Y_active, alpha=args.whiten_alpha)
    print(f"[atlas] shrinkage-whiten alpha={args.whiten_alpha}: cov condition number  "
          f"before={cond_b:.3e}  after={cond_a:.3e}  (isotropy = condition -> 1)")
    # how much consensus geometry survives the whitening (BEFORE spherical coding)?
    n = Rbar.shape[0]
    mask_n = ~torch.eye(n, dtype=torch.bool)
    Ywn = torch.nn.functional.normalize(Yw, dim=1)
    rho_w = float(torch.corrcoef(torch.stack(
        [Rbar[mask_n], (Ywn @ Ywn.t())[mask_n]]))[0, 1])
    print(f"[atlas]   whitened-key Gram vs consensus Rbar: rho={rho_w:+.3f} "
          f"(geometry retained through whitening)")

    Zc_active, Z0 = spherical_code(Yw, steps=args.code_steps, lr=args.code_lr,
                                   fidelity=args.fidelity)
    pre = code_stats(Z0)
    post = code_stats(Zc_active)
    print(f"[atlas] spherical-code (active r={r}): "
          f"min-angle {pre['min_angle_deg']:.2f} -> {post['min_angle_deg']:.2f} deg | "
          f"mean-angle {pre['mean_angle_deg']:.2f} -> {post['mean_angle_deg']:.2f} deg | "
          f"max-cos {pre['max_cos']:+.3f} -> {post['max_cos']:+.3f}")

    # place the active spherical code into the leading coordinates of the d=4096 hub; the trailing
    # (d_hub - r) coordinates are the canonical null space (kept zero in the KEYS — translators and
    # the store may use them). Keys stay unit-norm because the tail is zero.
    n = Zc_active.shape[0]
    Z = torch.zeros(n, args.d_hub)
    Z[:, :r] = Zc_active

    align = per_member_alignment(members, Rbar, Zc_active)
    print("[atlas] per-member alignment into Z:")
    for a in sorted(align, key=lambda x: x["rho_to_code"]):
        print(f"    {a['slug']:24s} geom={a['geometry']:5s} | rho->consensus {a['rho_to_consensus']:+.3f} "
              f"| rho->code(Z) {a['rho_to_code']:+.3f}")

    neut = neutrality_check(members, Rbar)
    print("[atlas] base-neutrality (leave-one-out consensus rho to full; near-1 = neutral):")
    for nv in sorted(neut, key=lambda x: x["loo_rho_to_full"]):
        print(f"    drop {nv['dropped']:24s} -> LOO rho {nv['loo_rho_to_full']:+.5f}")

    if args.smoke:
        print("[atlas] SMOKE — not saving.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    atlas = {
        "version": "v1_local6",
        "d_hub": args.d_hub,
        "rank_active": r,
        "n_anchor": n,
        "anchor_sha": sha,
        "members": [{"slug": m["slug"], "model": m["model"], "geometry": m["geometry"],
                     "d_base": m["d_base"], "tap_layer": m["tap_layer"], "n_layers": m["n_layers"]}
                    for m in members],
        "relrep": "zscore",                              # UNIFORM z-scored relrep (not plain centering)
        "build_params": {"code_steps": args.code_steps, "code_lr": args.code_lr,
                         "fidelity": args.fidelity, "whiten_alpha": args.whiten_alpha,
                         "embed_rel_floor": 1e-3, "whiten_eps_rel": 1e-3},
        # the canonical artifacts
        "Z": Z,                                          # [n_anchor, d_hub] canonical spherical-code keys
        "Z_active": Zc_active,                           # [n_anchor, r] active block (unit sphere)
        "Rbar": Rbar,                                    # [n,n] consensus mean-atlas Gram
        "mds_evals": evals,                              # consensus spectrum
        "whiten_W": W,                                   # [r,r] ZCA whitening transform
        "whiten_mean": ymean,                            # [r] active-block mean (pre-whiten)
        "cond_before": cond_b, "cond_after": cond_a,
        # validation report (so the checkpoint is self-describing)
        "alignment": align,
        "neutrality": neut,
        "code_stats_pre": pre, "code_stats_post": post,
    }
    torch.save(atlas, args.out)
    blob = torch.nn.functional.normalize(Z[:, :r], dim=1)
    zsha = hashlib.sha256(blob.contiguous().numpy().tobytes()).hexdigest()[:16]
    print(f"[atlas] saved -> {args.out}  (Z {tuple(Z.shape)}, active r={r}, Z-sha {zsha})")


if __name__ == "__main__":
    main()
