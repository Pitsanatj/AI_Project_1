"""
run_exp.py  —  ECSAnet → HybridECSAnet experiment launcher
============================================================
Wraps "main (2).py" with pre-defined configs for the B-series architecture
ablation (ECSANet → Hybrid Eff+ViT) and the combined contrastive chain.

Usage
-----
    python run_exp.py B0                     # ECSANet-S baseline
    python run_exp.py B4                     # paper's HybridEffNetV2M-ViT
    python run_exp.py B5                     # proposed hybrid with CBAM
    python run_exp.py B6                     # best: B5 + Macenko stain norm
    python run_exp.py B5 --epochs 30         # override epoch count
    python run_exp.py B5 --mag 100X 200X     # override magnification
    python run_exp.py B5 --dry_run           # print command, don't train
    python run_exp.py B5 --fine_tune_from ./outputs/B5/models/40X/best_model.pth

Architecture ablation chain (B-series)
───────────────────────────────────────
Each experiment adds exactly ONE component vs the previous row:

  B0   ECSANet-S          ← baseline (EfficientNetV2-S + CBAM, pure CNN)
                              Same architecture as SupCon E0

  B1   EffV2S ‖ DeiT-S    ← add smallest ViT branch (no CBAM on CNN)
                              measures: does ANY parallel ViT help?

  B2   EffV2S ‖ ViT-B/16  ← replace DeiT-S with full ViT-B/16 (no CBAM)
                              measures: bigger ViT vs smaller ViT?

  B3   EffV2S-CBAM ‖ ViT-B/16  ← add CBAM back to CNN branch
                                   measures: does CBAM help with ViT present?

  B4   EffV2M ‖ ViT-B/16  ← scale CNN: S → M (paper's HybridEffNetV2M-ViT)
                              measures: bigger CNN backbone?

  B5   EffV2M-CBAM ‖ ViT-B/16  ← add CBAM to M backbone  ★ proposed
                                   measures: CBAM on larger backbone?

  B6   B5 + Macenko stain norm  ← best stain normalisation  ★★ BEST
                                   measures: stain norm effect on hybrid

Contrastive experiment ordering (cross-folder plan)
─────────────────────────────────────────────────────
  After B-series determines the best architecture, the Modified SupCon
  pipeline (ESCANet/Sup_Con) is applied:

  Phase 1  (Architecture, this file, ESCANet/CNN+ViT)
    Step 1: B0  (2 GPU: run B0 + B1 in parallel, else sequential)
    Step 2: B2  (compare B1 vs B2 → pick ViT variant)
    Step 3: B3  (add CBAM to best-ViT setup)
    Step 4: B4  (scale CNN, independent)
    Step 5: B5  (combines CBAM + M backbone — read B3+B4 first)
    Step 6: B6  (Macenko on B5)

  Phase 2  (Contrastive, ESCANet/Sup_Con folder, already designed)
    Step 7: E0 baseline (if not already done)
    Step 8: E5 — best ECSANet-S + full SupCon pipeline  ★ proposed CNN-only
    Step 9: E6 — E5 + Macenko                            ★★ best CNN-only

  Phase 3  (Hybrid + Contrastive, future extension)
    Step 10: B5-SC — best hybrid (B5/B6 arch) + Modified SupCon 3-stage
             → apply SupCon pipeline from Sup_Con/main.py to hybrid model
             → only worthwhile if B5/B6 > E5/E6

Architecture comparison table (reference)
──────────────────────────────────────────
  Exp  | CNN branch                | ViT branch  | Params  | Emb dims
  ─────┼───────────────────────────┼─────────────┼─────────┼──────────
  B0   | EffNetV2-S + CBAM@stage6  | —           | ~21M    | 1280
  B1   | EffNetV2-S                | DeiT-S      | ~43M    | 1280+384
  B2   | EffNetV2-S                | ViT-B/16    | ~107M   | 1280+768
  B3   | EffNetV2-S + CBAM@1280    | ViT-B/16    | ~107M   | 1280+768
  B4   | EffNetV2-M                | ViT-B/16    | ~140M   | 1280+768
  B5   | EffNetV2-M + CBAM@1280    | ViT-B/16    | ~140M   | 1280+768
  B6   | EffNetV2-M + CBAM@1280    | ViT-B/16    | ~140M   | 1280+768
         (+ Macenko stain norm)

Key papers
──────────
  HybridEffNetV2M-ViT (2024): B4 is this paper's method.
  VTCNet, Li et al. (2024):   parallel CNN+ViT with feature-level fusion.
  ECSANet (this work):        CBAM injection + ablation of S/M + stain norm.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

# ── Shared base configs ─────────────────────────────────────────────────────────
_COMMON = dict(
    data_dir      = "./npy_data",
    num_workers   = 2,
    ckpt_interval = 5,
    magnification = ["40X"],
    weights_dir   = "./pretrained_weights",
    no_amp        = True,
    num_classes   = 8,
    optimizer     = "sgd",
    lr            = 1e-3,
    weight_decay  = 1e-2,
    momentum      = 0.9,
    scheduler_patience = 3,
    patience      = 25,
)

_REINHARD = dict(**_COMMON, preprocess="reinhard")
_MACENKO  = dict(**_COMMON, preprocess="macenko")

# Standard CE training hyperparams shared across all B exps
_STD = dict(
    epochs     = 50,
    batch_size = 16,
    augment    = "standard",
)

# ── Experiment definitions ──────────────────────────────────────────────────────
EXPERIMENTS: dict[str, dict] = {

    # ─── B0: ECSANet-S baseline ─────────────────────────────────────────────────
    # EfficientNetV2-S + CBAM (original ECSANet architecture)
    # Same model as ESCANet(Sup_Con) E0 — provides comparable anchor point.
    "B0": {
        **_REINHARD,
        **_STD,
        "model":      "ecsamet",
        "output_dir": "./outputs/B0",
        "experiment": "B0-ECSANetS-Reinhard",
    },

    # ─── B1: ECSANet-S + DeiT-S parallel (no CBAM on CNN) ─────────────────────
    # Adds the smallest/fastest ViT branch to measure: does any ViT help?
    # joint_dim = 1280 + 384 = 1664 → FC(1664→512→8)
    "B1": {
        **_REINHARD,
        **_STD,
        "model":        "hybrid_ecsanet",
        "cnn_backbone": "s",
        "vit_branch":   "deit_s",
        "use_cbam":     False,
        "fusion_dim":   512,
        "output_dir":   "./outputs/B1",
        "experiment":   "B1-EffV2S-DeiTS",
    },

    # ─── B2: ECSANet-S + ViT-B/16 parallel (no CBAM on CNN) ───────────────────
    # Replaces DeiT-S with full ViT-B/16: bigger ViT vs smaller ViT?
    # joint_dim = 1280 + 768 = 2048 → FC(2048→512→8)
    "B2": {
        **_REINHARD,
        **_STD,
        "model":        "hybrid_ecsanet",
        "cnn_backbone": "s",
        "vit_branch":   "vit_b16",
        "use_cbam":     False,
        "fusion_dim":   512,
        "output_dir":   "./outputs/B2",
        "experiment":   "B2-EffV2S-ViTB16",
    },

    # ─── B3: ECSANet-S + CBAM + ViT-B/16 parallel ─────────────────────────────
    # Adds CBAM(1280) to CNN branch: does CBAM still help with ViT present?
    # Note: CBAM operates at 1280ch (after all features, before avgpool).
    "B3": {
        **_REINHARD,
        **_STD,
        "model":        "hybrid_ecsanet",
        "cnn_backbone": "s",
        "vit_branch":   "vit_b16",
        "use_cbam":     True,
        "fusion_dim":   512,
        "output_dir":   "./outputs/B3",
        "experiment":   "B3-EffV2S-CBAM-ViTB16",
    },

    # ─── B4: ECSANet-M + ViT-B/16 parallel (no CBAM) ──────────────────────────
    # Scales CNN to EfficientNetV2-M: does the bigger backbone help?
    # This is the architecture from HybridEffNetV2M-ViT (2024) paper.
    "B4": {
        **_REINHARD,
        **_STD,
        "model":        "hybrid_ecsanet",
        "cnn_backbone": "m",
        "vit_branch":   "vit_b16",
        "use_cbam":     False,
        "fusion_dim":   512,
        "output_dir":   "./outputs/B4",
        "experiment":   "B4-EffV2M-ViTB16",
    },

    # ─── B5: ECSANet-M + CBAM + ViT-B/16 parallel  ★ PROPOSED ────────────────
    # Adds CBAM to M backbone — combines CBAM + bigger CNN + ViT.
    # Best expected model (Reinhard stain norm).
    "B5": {
        **_REINHARD,
        **_STD,
        "model":        "hybrid_ecsanet",
        "cnn_backbone": "m",
        "vit_branch":   "vit_b16",
        "use_cbam":     True,
        "fusion_dim":   512,
        "output_dir":   "./outputs/B5",
        "experiment":   "B5-EffV2M-CBAM-ViTB16-Reinhard",
    },

    # ─── B6: B5 + Macenko stain norm  ★★ BEST ──────────────────────────────────
    # Identical architecture to B5, but with Macenko stain normalisation.
    # Tests: does better stain norm improve the hybrid model?
    "B6": {
        **_MACENKO,
        **_STD,
        "model":        "hybrid_ecsanet",
        "cnn_backbone": "m",
        "vit_branch":   "vit_b16",
        "use_cbam":     True,
        "fusion_dim":   512,
        "output_dir":   "./outputs/B6",
        "experiment":   "B6-EffV2M-CBAM-ViTB16-Macenko",
    },
}

# ── Config → argv ───────────────────────────────────────────────────────────────
def config_to_argv(cfg: dict) -> list[str]:
    argv = []
    for k, v in cfg.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, list):
            argv.append(flag)
            argv.extend(str(x) for x in v)
        else:
            argv += [flag, str(v)]
    return argv


# ── Launcher ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="HybridECSAnet architecture ablation launcher (B0–B6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Experiments: " + "  ".join(EXPERIMENTS),
    )
    p.add_argument("exp", choices=list(EXPERIMENTS.keys()),
                   help="Experiment ID  (B0–B6)")
    p.add_argument("--epochs",         type=int,   default=None,
                   help="Override epoch count")
    p.add_argument("--mag",            nargs="+",
                   choices=["40X", "100X", "200X", "400X", "all"],
                   default=None, help="Override magnification(s)")
    p.add_argument("--fine_tune_from", default=None,
                   help="Override fine_tune_from weight path")
    p.add_argument("--seed",           type=int, default=None,
                   help="Random seed (default: use experiment default=42). "
                        "When set, appends /seed_{seed} to output_dir so "
                        "multiple seeds never overwrite each other.")
    p.add_argument("--dry_run",        action="store_true",
                   help="Print resolved command and exit without training")
    launch = p.parse_args()

    cfg = dict(EXPERIMENTS[launch.exp])    # copy — never mutate global

    if launch.epochs is not None:
        cfg["epochs"] = launch.epochs
    if launch.mag is not None:
        cfg["magnification"] = launch.mag
    if launch.fine_tune_from is not None:
        cfg["fine_tune_from"] = launch.fine_tune_from
    if launch.seed is not None:
        cfg["seed"]       = launch.seed
        cfg["output_dir"] = cfg["output_dir"] + f"/seed_{launch.seed}"

    cfg["auto_resume"] = False

    argv = config_to_argv(cfg)

    # ── Prerequisite guard ───────────────────────────────────────────────────
    if "fine_tune_from" in cfg:
        src = Path(cfg["fine_tune_from"])
        if not src.exists():
            print(f"\n  !!  WARNING: fine_tune_from not found: {src}")
            print(    "     Make sure the prerequisite experiment has completed.\n")

    # ── Pretty-print config ──────────────────────────────────────────────────
    is_hybrid  = cfg.get("model") == "hybrid_ecsanet"
    stain_tag  = "[Macenko]" if cfg.get("preprocess") == "macenko" else "[Reinhard]"
    model_tag  = cfg.get("model", "?")

    print("\n" + "=" * 70)
    print(f"  Experiment : {launch.exp}  ({stain_tag}  {model_tag})")
    print(f"  Mags       : {cfg['magnification']}")
    print(f"  Epochs     : {cfg['epochs']}")
    print(f"  Output dir : {cfg['output_dir']}")
    print(f"  Seed       : {cfg.get('seed', 42)}")

    if is_hybrid:
        cnn_s = cfg.get("cnn_backbone", "s").upper()
        vit   = cfg.get("vit_branch",   "vit_b16")
        cbam  = "[CBAM]" if cfg.get("use_cbam") else "[no CBAM]"
        fd    = cfg.get("fusion_dim", 512)
        emb   = 1280 + (768 if vit == "vit_b16" else 384 if vit == "deit_s" else 0)
        print(f"  CNN branch : EfficientNetV2-{cnn_s}  {cbam}")
        print(f"  ViT branch : {vit}")
        print(f"  Fusion     : {emb}-dim -> FC({fd}) -> 8-class")

    if "fine_tune_from" in cfg:
        print(f"  From       : {cfg['fine_tune_from']}")
    print("=" * 70)

    cmd_str = "python \"main (2).py\" \\\n    " + " \\\n    ".join(argv)
    print(cmd_str)
    print("=" * 70)

    # ── Ablation chain reminder ──────────────────────────────────────────────
    print("\n  Architecture ablation chain:")
    print("    B0  ECSANet-S (pure CNN baseline)")
    print("    B1  + DeiT-S branch (parallel ViT)")
    print("    B2  + ViT-B/16 branch (bigger ViT)")
    print("    B3  + CBAM on CNN branch")
    print("    B4  + EfficientNetV2-M (scale CNN)  ← paper's method")
    print("    B5  + CBAM on M backbone            ← proposed ★")
    print("    B6  + Macenko stain norm             ← best ★★")
    print()
    print("  Next: apply SupCon pipeline (ESCANet/Sup_Con/run_exp.py E5/E6)")
    print("  Then: best hybrid architecture + SupCon = ultimate model")
    print()

    if launch.dry_run:
        print("  [dry_run] Exiting without training.")
        return

    # ── Load and run main.py ─────────────────────────────────────────────────
    main_path = Path(__file__).parent / "main (2).py"
    if not main_path.exists():
        sys.exit(f"ERROR: '{main_path}' not found.")

    spec = importlib.util.spec_from_file_location("_main_module", main_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sys.argv = [str(main_path)] + argv
    mod.main()


if __name__ == "__main__":
    main()