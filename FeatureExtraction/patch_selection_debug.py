import sys
import os
import io
import json
import shutil
import logging
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import h5py
from PIL import Image, ImageDraw

from azureml.core import Datastore, Run
from azure.storage.blob import ContainerClient, ContentSettings

from huggingface_hub import login, HfApi
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import openslide
import cv2

# ---- Genel ayarlar ----
Image.MAX_IMAGE_PIXELS = None

# ---- Konfig (ENV destekli) ----
OUTPUT_FEATURES_DIR = "test2_features"   # blob içi hedef klasör
PNG_UPLOAD_DIR = "test_pngs"                             # blob içi görsel hedef klasör
LOCAL_JSON = "patches.json"                         # tek JSON dosyası (liste)

# Patch selection hiperparametreleri
PATCH_SIZE = 256
STRIDE = 256
THUMB_MAX = 2048
SAT_THRESH = 20
VAL_THRESH = 30
MIN_TISSUE_FRAC = 0.10
MAX_TILES_SCAN = 6000
VIS_SAMPLE_EVERY = 1

# Patch kalite kontrolleri
BLUR_VAR_MIN = 40
V_MIN, V_MAX = 40, 245
S_MIN = 12
DARK_FRAC_MAX = 0.20
OPEN_K = 3
MIN_CC_AREA_PX = 500

BATCH_SIZE = 32                    # kaç dosyayı bir seferde işleyelim
MAX_PATCHES_PER_SLIDE = 2500       ###
INFER_BATCH = int(os.environ.get("INFER_BATCH", "512"))  # mini-batch inferans boyutu
DEBUG = os.environ.get("DEBUG", "0") == "1"
USE_AMP = os.environ.get("USE_AMP", "1") == "1"  # CUDA varsa yarı hassasiyet
MAX_SLIDES = int(os.environ.get("MAX_SLIDES", "0"))      # sadece debug için limit (0=hepsi)

logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("patch-uni2h-job")

def gpu_mem():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / (1024**2)
        r = torch.cuda.memory_reserved() / (1024**2)
        return f"{a:.0f}MB alloc / {r:.0f}MB reserv"
    return "CPU"

HF_TOKEN = (os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HF_TOKEN")
            or "").strip()

if not HF_TOKEN:
    raise RuntimeError("HF token bulunamadı (HF_TOKEN / HUGGINGFACE_HUB_TOKEN yok).")

try:
    who = HfApi().whoami(token=HF_TOKEN)
    logging.info(f"HF whoami OK, user={who.get('name') or who.get('email')}")
except Exception as e:
    raise RuntimeError(f"HF token geçersiz/expire: {e}")

try:
    login(token=HF_TOKEN)
    HfApi().model_info("MahmoodLab/UNI", token=HF_TOKEN)
    logging.info("HF erişim OK: MahmoodLab/UNI")
except Exception as e:
    raise RuntimeError(
        "HF auth/erişim hatası. Büyük olasılıkla modele erişim onayı yok.\n"
        f"Detay: {e}\n"
        "HF model sayfasına gidip (MahmoodLab/UNI) hesabınla 'Request/Accept access' yapmalısın."
    )

def get_thumb_with_scale(slide, max_side=THUMB_MAX):
    W0, H0 = slide.dimensions
    scale = max_side / float(max(W0, H0))
    tw, th = int(round(W0 * scale)), int(round(H0 * scale))
    img = slide.get_thumbnail((tw, th)).convert("RGB")
    return img, scale

def tissue_mask_from_thumb(thumb_rgb, sat_thresh=SAT_THRESH, val_thresh=VAL_THRESH):
    hsv = np.array(thumb_rgb.convert("HSV"))
    sat = hsv[..., 1]
    val = hsv[..., 2]
    return (sat > sat_thresh) & (val > val_thresh)

def refine_mask(mask_bool):
    m = (mask_bool.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_CC_AREA_PX:
            keep[labels == i] = 255
    return (keep > 0)

def grid_coords_from_mask_nonuniform(mask, scale_x, scale_y, patch_size_l, stride_l,
                                     min_frac=MIN_TISSUE_FRAC, max_tiles=MAX_TILES_SCAN):
    th, tw = mask.shape
    coords = []
    max_tx = max(0, tw - patch_size_l)
    max_ty = max(0, th - patch_size_l)
    for ty in range(0, max_ty + 1, max(1, int(round(stride_l)))):
        for tx in range(0, max_tx + 1, max(1, int(round(stride_l)))):
            frac = mask[ty:ty + patch_size_l, tx:tx + patch_size_l].mean()
            if frac >= min_frac:
                x0 = int(round(tx * scale_x))
                y0 = int(round(ty * scale_y))
                coords.append((x0, y0))
    if len(coords) > max_tiles:
        idx = np.linspace(0, len(coords) - 1, max_tiles).astype(int)
        coords = [coords[i] for i in idx]
    return coords

def patch_ok(pil_rgb):
    arr = np.array(pil_rgb.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    s_mean = float(hsv[..., 1].mean())
    v_mean = float(hsv[..., 2].mean())
    dark_frac = float((gray < 30).mean())
    if var < BLUR_VAR_MIN: return False, "blur"
    if v_mean < V_MIN or v_mean > V_MAX: return False, "v_out"
    if s_mean < S_MIN: return False, "s_low"
    if dark_frac > DARK_FRAC_MAX: return False, "dark_blob"
    return True, ""

def overlay_patches(thumb_rgb, coords_l0, scale, patch_size, out_path, sample_every=VIS_SAMPLE_EVERY):
    img = thumb_rgb.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    s = scale
    for i, (x0, y0) in enumerate(coords_l0):
        if (i % sample_every) != 0:
            continue
        x1 = int(round(x0 * s)); y1 = int(round(y0 * s))
        x2 = int(round((x0 + patch_size) * s)); y2 = int(round((y0 + patch_size) * s))
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
    img.save(out_path)

def select_patches_multilevel(slide, max_patches=MAX_PATCHES_PER_SLIDE):
    """
    Basit çok seviyeli: 6 -> 5 -> 4 -> 3 seviyeleri.
    """
    W0, H0 = slide.dimensions
    base_thumb, thumb_scale = get_thumb_with_scale(slide, THUMB_MAX)

    target_levels = [6, 5, 4, 3]
    kept_all = []
    reasons = {"blur": 0, "v_out": 0, "s_low": 0, "dark_blob": 0, "other": 0}
    per_level_counts = {}
    grid_total = 0

    for lvl in target_levels:
        if lvl >= slide.level_count:
            continue

        w_l, h_l = slide.level_dimensions[lvl]
        thumb_l = slide.get_thumbnail((w_l, h_l)).convert("RGB")
        mask0 = tissue_mask_from_thumb(thumb_l, SAT_THRESH, VAL_THRESH)
        mask = refine_mask(mask0)

        down = slide.level_downsamples[lvl]
        patch_size_l = max(1, int(round(PATCH_SIZE / down)))
        stride_l = max(1, int(round(STRIDE / down)))

        scale_x = W0 / float(thumb_l.size[0])
        scale_y = H0 / float(thumb_l.size[1])

        coords_lvl = grid_coords_from_mask_nonuniform(
            mask=mask,
            scale_x=scale_x,
            scale_y=scale_y,
            patch_size_l=patch_size_l,
            stride_l=stride_l,
            min_frac=MIN_TISSUE_FRAC,
            max_tiles=MAX_TILES_SCAN
        )
        grid_total += len(coords_lvl)

        kept_lvl = []
        for (x0, y0) in coords_lvl:
            try:
                patch = slide.read_region((x0, y0), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            except Exception:
                reasons["other"] += 1
                continue
            ok, why = patch_ok(patch)
            if ok:
                kept_lvl.append((x0, y0))
            else:
                reasons[why if why in reasons else "other"] += 1

            if len(kept_all) + len(kept_lvl) >= max_patches:
                kept_all.extend(kept_lvl)
                per_level_counts[f"level_{lvl}"] = len(kept_lvl)
                return kept_all[:max_patches], base_thumb, thumb_scale, reasons, per_level_counts, grid_total

        per_level_counts[f"level_{lvl}"] = len(kept_lvl)
        kept_all.extend(kept_lvl)
        if len(kept_all) >= max_patches:
            break

    return kept_all[:max_patches], base_thumb, thumb_scale, reasons, per_level_counts, grid_total

def extract_features_for_slide(slide_path, model, transform, device):
    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        logger.error(f"Slayt açılamadı: {slide_path}. Hata: {e}")
        return None

    W0, H0 = slide.dimensions
    coords, base_thumb, thumb_scale, reasons, lvl_counts, grid_total = select_patches_multilevel(slide)
    if not coords:
        slide.close()
        return None

    # Patchleri hazırla (CPU'da)
    patch_tensors = []
    for (x0, y0) in coords:
        patch_img = slide.read_region((x0, y0), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        patch_tensors.append(transform(patch_img).unsqueeze(0))  # (1,C,H,W)

    slide.close()

    n_patches = len(patch_tensors)
    if DEBUG:
        logger.info(f"[DBG] slide={os.path.basename(slide_path)} "
                    f"kept_patches={n_patches} batch={INFER_BATCH} "
                    f"device={'cuda' if torch.cuda.is_available() else 'cpu'} mem={gpu_mem()}")

    all_feats = []
    idx = 0
    bs = max(1, INFER_BATCH)
    t0_all = time.time()

    # CUDA performansı için (sabit input boyutlarında iyi sonuç verir)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        while idx < n_patches:
            j = min(idx + bs, n_patches)
            # batch'i birleştir ve cihaza taşı
            batch = torch.cat(patch_tensors[idx:j], dim=0)
            t0 = time.time()
            try:
                batch = batch.to(device, non_blocking=True)
                if torch.cuda.is_available() and USE_AMP:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = model(batch)
                else:
                    out = model(batch)
                all_feats.append(out.detach().cpu())

                dt = time.time() - t0
                if DEBUG:
                    logger.info(f"[DBG] batch [{idx}:{j}) -> {j-idx} patch | "
                                f"{dt*1000:.0f} ms | ~{(j-idx)/max(dt,1e-3):.1f} patch/s | mem={gpu_mem()}")
                idx = j  # başarılıysa ilerle

            except RuntimeError as e:
                # OOM yakala ve batch boyutunu düşürerek tekrar dene
                if "out of memory" in str(e).lower() and torch.cuda.is_available() and bs > 32:
                    if DEBUG:
                        logger.warning(f"[DBG] OOM @[{idx}:{j}], bs={bs}. Cache temizleniyor, bs yarıya iniyor.")
                    del batch
                    torch.cuda.empty_cache()
                    bs = max(32, bs // 2)
                    continue
                else:
                    raise
            finally:
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    total_dt = time.time() - t0_all
    final_feats = torch.cat(all_feats, dim=0).numpy()

    if final_feats.shape[0] != n_patches:
        raise RuntimeError(f"Feature sayısı uyuşmuyor: feats={final_feats.shape[0]} vs patches={n_patches}")

    if DEBUG:
        logger.info(f"[DBG] DONE slide={os.path.basename(slide_path)} "
                    f"batches=~{(n_patches + INFER_BATCH - 1)//INFER_BATCH} "
                    f"total={total_dt:.2f}s | avg={(n_patches/max(total_dt,1e-3)):.1f} patch/s")

    try:
        run = Run.get_context()
        run.log("kept_patches", n_patches)
        run.log("infer_batch", INFER_BATCH)
        run.log("use_amp", int(USE_AMP and torch.cuda.is_available()))
        run.log("infer_total_s", total_dt)
        run.log("patches_per_s", n_patches / max(total_dt, 1e-3))
    except Exception:
        pass

    meta = {
        "width_height_level0": [int(W0), int(H0)],
        "thumb_size": [int(base_thumb.size[0]), int(base_thumb.size[1])],
        "thumb_scale": float(thumb_scale),
        "rejected_counts": reasons,
        "n_tiles_grid": int(grid_total),
        "n_tiles_kept": int(len(coords)),
        "infer_batch": int(INFER_BATCH),
        "use_amp": int(USE_AMP and torch.cuda.is_available())
    }
    return final_feats, coords, base_thumb, meta

def main():
    run = Run.get_context()
    if run.id.startswith("Offline"):
        from azureml.core import Workspace
        ws = Workspace.from_config(path="config.json")
    else:
        ws = run.experiment.workspace

    datastore = Datastore.get(ws, "workspaceblobstore")
    container_client = ContainerClient(
        account_url=f"https://{datastore.account_name}.blob.core.windows.net/",
        container_name=datastore.container_name,
        credential=datastore.account_key
    )

    pyramids_blobs = [
        b.name for b in container_client.list_blobs(name_starts_with="test2_pyramid/")
        if b.name.lower().endswith(".tiff")
    ]
    if MAX_SLIDES > 0:
        pyramids_blobs = pyramids_blobs[:MAX_SLIDES]

    logger.info(f"Pyramids: {len(pyramids_blobs)} TIFF dosyası bulundu.")
    logger.info(f"Settings -> INFER_BATCH={INFER_BATCH}, USE_AMP={int(USE_AMP)}, "
                f"MAX_PATCHES_PER_SLIDE={MAX_PATCHES_PER_SLIDE}, DEBUG={int(DEBUG)}, MAX_SLIDES={MAX_SLIDES}")

    # Model / transform
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Cihaz: {device} | mem={gpu_mem()}")
    model.to(device).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    total = len(pyramids_blobs)
    failed_files = []
    bad_data_count = 0
    processed_count = 0
    batch = []

    Path("temp").mkdir(exist_ok=True)
    t_start = time.time()

    for i, blob_name in enumerate(pyramids_blobs, 1):
        file_name = os.path.basename(blob_name)
        temp_dir = os.path.join("temp", f"{i}")
        os.makedirs(temp_dir, exist_ok=True)
        local_tiff = os.path.join(temp_dir, file_name)
        local_h5 = local_tiff.replace(".tiff", ".h5")

        # Blob'dan indir
        try:
            with open(local_tiff, "wb") as f:
                f.write(container_client.download_blob(blob_name).readall())
        except Exception as e:
            failed_files.append({"filename": file_name, "status": f"pyramids download error: {e}"})
            bad_data_count += 1
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue

        batch.append((file_name, local_tiff, local_h5, temp_dir))

        if len(batch) == BATCH_SIZE or i == total:
            for file_name, local_tiff, local_h5, temp_dir in batch:
                t0 = time.time()
                try:
                    res = extract_features_for_slide(local_tiff, model, transform, device)
                    if res is None:
                        failed_files.append({"filename": file_name, "status": "no_patches_or_open_failed"})
                        bad_data_count += 1
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        continue

                    feats, coords, base_thumb, meta = res

                    # H5'e yaz ve blob'a yükle
                    with h5py.File(local_h5, "w") as hf:
                        hf.create_dataset("feats", data=feats)
                    datastore.upload_files(files=[local_h5],
                                           target_path=OUTPUT_FEATURES_DIR,
                                           overwrite=True)

                    stem = Path(local_tiff).stem
                    local_png = os.path.join(temp_dir, f"{stem}_patches.png")
                    overlay_patches(base_thumb, coords, meta["thumb_scale"], PATCH_SIZE, local_png, VIS_SAMPLE_EVERY)
                    with open(local_png, "rb") as fp:
                        container_client.upload_blob(
                            name=f"{PNG_UPLOAD_DIR}/{Path(local_png).name}",
                            data=fp.read(),
                            overwrite=True,
                            content_settings=ContentSettings(content_type="image/png")
                        )

                    record = {
                        "id": file_name,
                        "width_height_level0": meta["width_height_level0"],
                        "thumb_size": meta["thumb_size"],
                        "thumb_scale": meta["thumb_scale"],
                        "patch_size": PATCH_SIZE,
                        "stride": STRIDE,
                        "min_tissue_frac": MIN_TISSUE_FRAC,
                        "n_tiles_grid": meta["n_tiles_grid"],
                        "n_tiles_kept": meta["n_tiles_kept"],
                        "rejected_counts": meta["rejected_counts"],
                        "coords_level0_xy": [[int(x), int(y)] for (x, y) in coords[:5000]],
                        "infer_batch": meta.get("infer_batch"),
                        "use_amp": meta.get("use_amp")
                    }

                    # Tek liste JSON'u güncelle
                    data_list = []
                    if os.path.exists(LOCAL_JSON):
                        try:
                            with open(LOCAL_JSON, "r", encoding="utf-8") as fp:
                                existing = json.load(fp)
                                data_list = existing if isinstance(existing, list) else [existing]
                        except Exception:
                            logger.warning("patches.json okunamadı, yeni dosya oluşturulacak.")
                            data_list = []
                    data_list.append(record)
                    with open(LOCAL_JSON, "w", encoding="utf-8") as fp:
                        json.dump(data_list, fp, ensure_ascii=False, indent=2)

                    shutil.rmtree(temp_dir, ignore_errors=True)
                    processed_count += 1
                    logger.info(f" {processed_count}/{total} bitti: {file_name}  ({time.time() - t0:.1f} sn)")
                except Exception as e:
                    failed_files.append({"filename": file_name, "status": f"processing error: {e}"})
                    bad_data_count += 1
                    shutil.rmtree(temp_dir, ignore_errors=True)

            batch.clear()

    if failed_files:
        pd.DataFrame(failed_files).to_csv("failed_uni2h_files.csv", index=False)

    logger.info(f" İşlem tamamlandı. Başarılı: {processed_count} / {total} | "
                f"Bozuk data sayısı: {bad_data_count} | Süre: {(time.time()-t_start)/60:.1f} dk")

if __name__ == "__main__":
    main()
