"""Anomaly detection methods compared in the study.

Every method implements:
    fit(train_images)            -> None      train_images: (k, H, W, 3) uint8
    score(images)                -> (scores (N,), maps (N, EVAL, EVAL) or None)
    loo_scores(train_images)     -> (k,) leave-one-out scores on training images, or None

PatchCore, PaDiM and PatchCore-DINOv2 are implemented here directly (rather than through
Anomalib's training engine) so that arbitrary k-shot subsets can be passed in; the
implementations follow the original papers. WinCLIP uses Anomalib's WinClipModel.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2 as T

from data import EVAL_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- helpers
def to_tensor(images: np.ndarray, size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.0)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


def normalise(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def batches(n: int, bs: int):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    radius = int(math.ceil(4 * sigma))
    xs = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    k = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = maps.unsqueeze(1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
    return x.squeeze(1)


def to_eval_maps(patch_maps: torch.Tensor) -> np.ndarray:
    m = F.interpolate(patch_maps.unsqueeze(1), size=(EVAL_SIZE, EVAL_SIZE),
                      mode="bilinear", align_corners=False).squeeze(1)
    return gaussian_blur(m).cpu().numpy().astype(np.float32)


def nn_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Euclidean distance from each query row to its nearest row in bank."""
    out = []
    for i in range(0, query.shape[0], chunk):
        d = torch.cdist(query[i:i + chunk], bank)
        out.append(d.min(dim=1).values)
    return torch.cat(out)


def greedy_coreset(features: torch.Tensor, ratio: float, seed: int, proj_dim: int = 128) -> torch.Tensor:
    """Greedy k-center coreset selection as in PatchCore (on a random projection)."""
    n = features.shape[0]
    m = max(1, int(n * ratio))
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(features.shape[1], proj_dim, generator=g).to(features.device)
    z = features @ proj
    selected = [int(torch.randint(n, (1,), generator=g))]
    min_d = torch.linalg.norm(z - z[selected[0]], dim=1)
    for _ in range(m - 1):
        idx = int(torch.argmax(min_d))
        selected.append(idx)
        min_d = torch.minimum(min_d, torch.linalg.norm(z - z[idx], dim=1))
    return features[torch.tensor(selected, device=features.device)]


class Method:
    name = "base"
    produces_maps = True
    supports_k0 = False

    def fit(self, train_images: np.ndarray) -> None:
        raise NotImplementedError

    def score(self, images: np.ndarray):
        raise NotImplementedError

    def loo_scores(self, train_images: np.ndarray):
        """Default: refit on k-1 images and score the held-out one."""
        k = len(train_images)
        if k < 2:
            return None
        out = []
        for i in range(k):
            keep = np.delete(np.arange(k), i)
            self.fit(train_images[keep])
            out.append(float(self.score(train_images[i:i + 1])[0][0]))
        self.fit(train_images)  # restore the full model
        return np.array(out)


# ---------------------------------------------------------------- PatchCore (CNN or DINOv2)
class PatchCore(Method):
    """PatchCore [Roth et al. 2022]: locally aggregated mid-level patch features,
    nearest-neighbour distance to a memory bank. Coreset subsampling (10%) is applied only
    when the bank exceeds `coreset_min` patches (i.e. the full training set)."""

    def __init__(self, backbone: str = "wrn50", coreset_ratio: float = 0.1,
                 coreset_min: int = 50_000, seed: int = 0):
        self.backbone_name = backbone
        self.name = "patchcore" if backbone == "wrn50" else "patchcore_dinov2"
        self.coreset_ratio, self.coreset_min, self.seed = coreset_ratio, coreset_min, seed
        if backbone == "wrn50":
            net = torchvision.models.wide_resnet50_2(
                weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            self.input_size = 256
            self.extractor = torchvision.models.feature_extraction.create_feature_extractor(
                net, return_nodes={"layer2": "l2", "layer3": "l3"})
        elif backbone == "dinov2":
            self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.input_size = 448  # 32 x 32 patch tokens
        else:
            raise ValueError(backbone)
        self.model = (self.extractor if backbone == "wrn50" else self.net).to(DEVICE).eval()
        self.bank = None
        self.grid = None

    @torch.no_grad()
    def _features(self, images: np.ndarray) -> torch.Tensor:
        """Returns (N, h*w, D) patch features."""
        feats = []
        for sl in batches(len(images), 16):
            x = to_tensor(images[sl], self.input_size).to(DEVICE)
            if self.backbone_name == "wrn50":
                f = self.extractor(x)
                l2 = F.avg_pool2d(f["l2"], 3, 1, 1)
                l3 = F.avg_pool2d(f["l3"], 3, 1, 1)
                l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
                z = torch.cat([l2, l3], dim=1)                  # (B, 1536, 32, 32)
                b, c, h, w = z.shape
                z = z.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
                z = F.adaptive_avg_pool1d(z, 1024).reshape(b, h * w, 1024)
            else:
                outs = self.net.get_intermediate_layers(x, n=[7, 11], reshape=True)
                z = torch.cat([F.avg_pool2d(o, 3, 1, 1) for o in outs], dim=1)  # (B, 1536, 32, 32)
                b, c, h, w = z.shape
                z = z.permute(0, 2, 3, 1).reshape(b, h * w, c)
            self.grid = (h, w)
            feats.append(z)
        return torch.cat(feats)

    def fit(self, train_images):
        f = self._features(train_images)
        bank = f.reshape(-1, f.shape[-1])
        if bank.shape[0] > self.coreset_min:
            bank = greedy_coreset(bank, self.coreset_ratio, self.seed)
        self.bank = bank

    @torch.no_grad()
    def score(self, images):
        f = self._features(images)
        n, p, d = f.shape
        dist = nn_distance(f.reshape(-1, d), self.bank).reshape(n, *self.grid)
        return dist.amax(dim=(1, 2)).cpu().numpy(), to_eval_maps(dist)

    def loo_scores(self, train_images):
        """Cheap exact LOO: drop the held-out image's own patches from the bank."""
        k = len(train_images)
        if k < 2:
            return None
        f = self._features(train_images)
        out = []
        for i in range(k):
            bank = torch.cat([f[j] for j in range(k) if j != i])
            out.append(float(nn_distance(f[i], bank).max()))
        return np.array(out)


# ---------------------------------------------------------------- PaDiM
class PaDiM(Method):
    """PaDiM [Defard et al. 2021] with ResNet-18, layers 1-3, 100 random dimensions, and
    a Gaussian per patch position. Covariance = sample covariance + eps*I. With k = 1 the
    sample covariance is zero and the model reduces to Euclidean distance to the mean,
    scaled by 1/eps; small-k results should be read with this in mind."""
    name = "padim"

    def __init__(self, n_dims: int = 100, eps: float = 0.01, seed: int = 0):
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.extractor = torchvision.models.feature_extraction.create_feature_extractor(
            net, return_nodes={"layer1": "l1", "layer2": "l2", "layer3": "l3"}).to(DEVICE).eval()
        g = torch.Generator().manual_seed(seed)
        self.idx = torch.randperm(64 + 128 + 256, generator=g)[:n_dims].to(DEVICE)
        self.eps = eps
        self.input_size = 256

    @torch.no_grad()
    def _features(self, images):
        out = []
        for sl in batches(len(images), 32):
            f = self.extractor(to_tensor(images[sl], self.input_size).to(DEVICE))
            size = f["l1"].shape[-2:]
            z = torch.cat([f["l1"]] + [F.interpolate(f[k], size=size, mode="nearest")
                                       for k in ("l2", "l3")], dim=1)
            out.append(z[:, self.idx])                          # (B, d, 64, 64)
        return torch.cat(out)

    @torch.no_grad()
    def fit(self, train_images):
        z = self._features(train_images)                        # (N, d, H, W)
        n, d, h, w = z.shape
        z = z.permute(2, 3, 0, 1).reshape(h * w, n, d)          # (P, N, d)
        mean = z.mean(dim=1)
        centred = z - mean[:, None]
        cov = centred.transpose(1, 2) @ centred / max(n - 1, 1)
        cov = cov + self.eps * torch.eye(d, device=DEVICE)
        self.mean, self.inv_cov, self.grid = mean, torch.linalg.inv(cov), (h, w)

    @torch.no_grad()
    def score(self, images):
        z = self._features(images)
        n, d, h, w = z.shape
        z = z.permute(2, 3, 0, 1).reshape(h * w, n, d) - self.mean[:, None]
        m = torch.einsum("pnd,pde,pne->pn", z, self.inv_cov, z).clamp_min(0).sqrt()
        maps = m.T.reshape(n, h, w)
        return maps.amax(dim=(1, 2)).cpu().numpy(), to_eval_maps(maps)


# ---------------------------------------------------------------- SimCLR
def simclr_augment(strength: str, size: int = 224):
    if strength == "standard":   # as in Chen et al. 2020
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
        ])
    if strength == "mild":       # geometry-light, no colour change: keeps colour/shape defects anomalous
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomHorizontalFlip(),
        ])
    raise ValueError(strength)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    n = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


class SimCLR(Method):
    """SimCLR trained on the k available normal images only.

    pretrained=False: ResNet-18 from random initialisation ("from scratch").
    pretrained=True:  ResNet-50 initialised from ImageNet weights ("fine-tuned").
    Score = mean cosine distance to the n_neighbors nearest memory embeddings, where the
    memory holds each training image plus `memory_views` mildly augmented views of it
    (so that k = 1 still gives a non-trivial memory). No anomaly maps are produced.
    """
    produces_maps = False

    def __init__(self, pretrained: bool, augment: str = "standard", steps: int = 500,
                 batch_size: int = 64, tau: float = 0.2, memory_views: int = 16,
                 n_neighbors: int = 5, seed: int = 0):
        self.pretrained, self.augment, self.steps = pretrained, augment, steps
        self.batch_size, self.tau, self.memory_views = batch_size, tau, memory_views
        self.n_neighbors, self.seed = n_neighbors, seed
        self.name = f"simclr_{'finetune' if pretrained else 'scratch'}_{augment}"
        self.input_size = 224

    def _build(self):
        torch.manual_seed(self.seed)
        if self.pretrained:
            enc = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            lr = 1e-4
        else:
            enc = torchvision.models.resnet18(weights=None)
            lr = 1e-3
        dim = enc.fc.in_features
        enc.fc = nn.Identity()
        head = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))
        self.encoder, self.head = enc.to(DEVICE), head.to(DEVICE)
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-6)

    def fit(self, train_images):
        self._build()
        base = to_tensor(train_images, 256, (0, 0, 0), (1, 1, 1)).to(DEVICE)  # raw [0, 1]
        aug = simclr_augment(self.augment)
        g = torch.Generator().manual_seed(self.seed)
        self.encoder.train(); self.head.train()
        for _ in range(self.steps):
            idx = torch.randint(len(base), (self.batch_size,), generator=g)
            x = base[idx.to(DEVICE)]
            v1 = normalise(torch.stack([aug(img) for img in x]))
            v2 = normalise(torch.stack([aug(img) for img in x]))
            loss = nt_xent(self.head(self.encoder(v1)), self.head(self.encoder(v2)), self.tau)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        self.encoder.eval(); self.head.eval()
        self._build_memory(train_images)

    @torch.no_grad()
    def _embed(self, images: np.ndarray, views: int = 0) -> torch.Tensor:
        out = []
        mild = simclr_augment("mild")
        for sl in batches(len(images), 32):
            x = to_tensor(images[sl], 256, (0, 0, 0), (1, 1, 1)).to(DEVICE)
            centre = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            out.append(F.normalize(self.encoder(normalise(centre)), dim=1))
            for _ in range(views):
                out.append(F.normalize(self.encoder(normalise(torch.stack([mild(i) for i in x]))), dim=1))
        return torch.cat(out)

    @torch.no_grad()
    def _build_memory(self, train_images):
        views = self.memory_views if len(train_images) <= 16 else 0
        self.memory_owner = []
        mem = []
        for i in range(len(train_images)):
            mem.append(self._embed(train_images[i:i + 1], views))
            self.memory_owner += [i] * (1 + views)
        self.memory = torch.cat(mem)
        self.memory_owner = torch.tensor(self.memory_owner, device=DEVICE)

    @torch.no_grad()
    def _knn_distance(self, emb, memory):
        sim = emb @ memory.T
        kk = min(self.n_neighbors, memory.shape[0])
        return (1 - sim.topk(kk, dim=1).values).mean(dim=1)

    @torch.no_grad()
    def score(self, images):
        return self._knn_distance(self._embed(images), self.memory).cpu().numpy(), None

    @torch.no_grad()
    def loo_scores(self, train_images):
        """Approximate LOO: the encoder is not retrained (too costly), but the held-out
        image's own memory entries are excluded. Because the encoder has seen the image,
        this threshold is biased low; the realised test FPR is reported alongside it."""
        k = len(train_images)
        if k < 2:
            return None
        emb = self._embed(train_images)
        return np.array([float(self._knn_distance(emb[i:i + 1],
                                                  self.memory[self.memory_owner != i])[0])
                         for i in range(k)])


# ---------------------------------------------------------------- WinCLIP
class WinCLIP(Method):
    """WinCLIP / WinCLIP+ [Jeong et al. 2023] via Anomalib's WinClipModel
    (ViT-B-16-plus-240, LAION-400M weights, window scales 2 and 3)."""
    name = "winclip"
    supports_k0 = True

    def __init__(self, class_name: str):
        from anomalib.models.image.winclip.torch_model import WinClipModel
        self.model = WinClipModel().to(DEVICE).eval()
        self.class_name = class_name

    def _x(self, images):
        return to_tensor(images, 240, CLIP_MEAN, CLIP_STD).to(DEVICE)

    @torch.no_grad()
    def fit(self, train_images):
        refs = None if train_images is None or len(train_images) == 0 else self._x(train_images)
        self.model.setup(self.class_name, refs)
        if refs is None:
            self.model.k_shot = 0

    @torch.no_grad()
    def score(self, images):
        scores, maps = [], []
        for sl in batches(len(images), 16):
            out = self.model(self._x(images[sl]))
            s, m = (out.pred_score, out.anomaly_map) if hasattr(out, "pred_score") else out
            scores.append(s.float().cpu())
            maps.append(m.float())
        maps = torch.cat(maps)
        maps = F.interpolate(maps.unsqueeze(1), size=(EVAL_SIZE, EVAL_SIZE), mode="bilinear",
                             align_corners=False).squeeze(1)
        return torch.cat(scores).numpy(), maps.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- registry
def build(method: str, class_name: str, seed: int):
    if method == "patchcore":
        return PatchCore("wrn50", seed=seed)
    if method == "patchcore_dinov2":
        return PatchCore("dinov2", seed=seed)
    if method == "padim":
        return PaDiM(seed=0)          # fixed dimension subset, as in the original method
    if method.startswith("simclr_"):
        _, init, aug = method.split("_")
        return SimCLR(pretrained=(init == "finetune"), augment=aug, seed=seed)
    if method == "winclip":
        return WinCLIP(class_name)
    raise ValueError(method)


ALL_METHODS = [
    "simclr_scratch_standard", "simclr_scratch_mild",
    "simclr_finetune_standard", "simclr_finetune_mild",
    "padim", "patchcore", "patchcore_dinov2", "winclip",
]


def timed(fn, *args):
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0
