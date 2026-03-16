"""
src/augment.py
━━━━━━━━━━━━━━
Augmentation pipelines applied AFTER preprocessing (on uint8 tensors).
New strategies can be added and selected via --augment CLI argument.
"""

from torchvision.transforms import v2


def build_augment(strategy: str) -> v2.Compose | None:
    """
    Returns an augmentation transform or None.

    Strategies
    ----------
    standard   : flip + affine  (paper default)
    augmix     : AugMix only
    combined   : flip + affine + AugMix
    none       : no augmentation
    """
    strategy = strategy.lower()

    flip_affine = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(
            degrees=(-45, 45),
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=(0, 10),
        ),
    ]

    augmix_step = [
        v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0),
    ]

    if strategy == "standard":
        steps = flip_affine
    elif strategy == "augmix":
        steps = augmix_step
    elif strategy == "combined":
        steps = flip_affine + augmix_step
    elif strategy == "none":
        return None
    else:
        raise ValueError(
            f"Unknown augmentation strategy '{strategy}'. "
            "Choose from: standard | augmix | combined | none"
        )

    return v2.Compose(steps)


def build_balance_augment() -> v2.Compose:
    """
    Applied only to minority-class samples during oversampling.
    Generates extra diversity for under-represented classes.
    """
    return v2.Compose([
        v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0),
    ])
