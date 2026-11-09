# Magik Color Correct — v1.1.0

Reference-based color correction for ComfyUI. The source keeps its resolution, batch size and alpha. Reference resolution can differ. Statistical methods do not require alignment; Upscale color restore requires matching framing.

## Install

1. Extract `Magik_Color_Correct` into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI. Add **Magik → Image → Magik Color Correct**.

Uses NumPy and PyTorch already included in ComfyUI. No extra packages, custom frontend, network calls, or model downloads.

## Connect

- `image`: image(s) to change.
- `reference`: image(s) whose colors you want to copy.
- Output `image`: corrected image(s).

Start with **Gaussian optimal transport**, strength **0.7–1.0**. Increase `preserve_luminance` toward **1** to keep the source's brightness structure.

## Upscale color restore (new in v1.1)

Select **Upscale color restore** in `method`. Connect the upscaled image to `image`, and the original before upscaling to `reference`.

Start with `strength=1`, `color_strength=1`, `brightness_strength=1`, `correction_radius=2`, `preserve_luminance=0`.

The reference is resized to output resolution. A smooth Lab difference map restores local color/lightness; only the low-frequency difference is added to the upscale, preserving its high-frequency detail before final gamut clipping. Smoothing uses three separable box passes approximating a Gaussian. No models or extra dependencies are needed.

- `color_strength`: scales correction of Lab a/b color channels.
- `brightness_strength`: scales correction of Lab L lightness.
- `correction_radius`: approximate blur sigma as a percentage of the output short edge. 2% means about 41 pixels at a 2048px short edge. Lower values follow smaller features but can imprint reference details or amplify misalignment. Higher values correct broader casts and tolerate small changes.
- Existing `strength`, `mask`, batching and alpha preservation still apply.
- `preserve_luminance` further reduces brightness correction: effective brightness strength is `brightness_strength × (1 − preserve_luminance)`.
- `sample_pixels`, `iterations`, and `seed` apply to statistical methods only.
- Disconnect `source_statistics_mask` and `reference_mask` for this mode; it uses spatial correspondences instead. Use `mask` to restrict the corrected area.

Requires the same framing and aspect ratio (within 1% for rounding). No registration/crop detection is performed. A matching aspect ratio alone does not guarantee alignment. Strong redraws, crops, or moved objects can produce halos; increase the radius or use a statistical method. Full-frame Lab smoothing uses additional CPU memory and interruption checks between passes. This mode also works with aligned images of equal resolution.

## Methods

| Method | Purpose |
| --- | --- |
| Gaussian optimal transport | Default. Matches Lab means and full covariance using a regularized symmetric Gaussian transport map. |
| Reinhard (Lab) | Fast channel mean/std matching. Conservative starting point for similar scenes. |
| Histogram matching | Matches per-channel Lab quantiles; can be aggressive on unrelated images. |
| Histogram + Gaussian | Histogram transfer followed by covariance correction. This is this node's own hybrid, not color-matcher's HM-MVGD-HM. |
| Sliced optimal transport | Iterative matching along deterministic random orthogonal directions in Lab. Models more complex distributions; slower. |

These are classical and advanced statistical color-distribution methods, not neural semantic relighting or a claim of latest benchmark SOTA. They cannot tell skin from sky: use masks to exclude irrelevant colors. No existing node implementation is copied or imported.

## Controls

- `strength`: 0 bypasses; 1 full result; up to 2 extrapolates.
- `preserve_luminance`: restores the source CIELAB L channel before RGB conversion. Out-of-gamut RGB clipping can change final lightness.
- `reference_mode`: First reference reuses reference 1. Pair by index accepts a single shared reference or an exact batch-size match; mismatches raise a clear error.
- `sample_pixels`: maximum samples used per image for estimating statistics, default 65,536. Does not resize the output. Sampling is deterministic.
- `iterations`: applies only to sliced optimal transport, default 12. Higher values cost more time.
- `seed`: controls sampling and transport directions.

## Optional masks

White selects; black excludes. Masks resize bilinearly to the corresponding image. Supply one shared mask or one mask per corresponding image batch.

- `mask`: soft mask for where correction is applied. Does not affect learned statistics.
- `source_statistics_mask`: source pixels used to learn the transform.
- `reference_mask`: reference pixels used to learn the transform.

Statistics masks use values >0.5 and must select at least two pixels. When reference mode is First reference, the first reference mask is used. Image alpha is preserved but is not automatically a statistics mask; connect an explicit mask if needed. ComfyUI Load Image alpha masks may use the opposite convention, so invert where appropriate.

## Batches, memory, and video

Processes images sequentially on CPU with ComfyUI interruption checks and batch progress. Output retains input dtype and device. Uses bounded statistics samples and 131,072-pixel transform chunks. Sliced OT repeats estimation per chunk and can be slow on large images. Full input/output image tensors and per-frame Lab buffers still require memory.

Each frame estimates its own source distribution. This is useful for independent images but may cause flicker in video; this version does not implement temporal smoothing or a shared fixed video transform.

Input RGB is interpreted as display-referred sRGB in [0,1], not scene-linear HDR. Nonfinite input is rejected; processed RGB is clamped. Strength 0 returns the original tensor directly.

## Validation

Python compilation and numerical tests cover Lab round-trip accuracy, all five statistical methods, identity transfer, deterministic results, constant-color inputs, finite results and Gaussian covariance matching. Eight tests also cover spatial identity/bypass, known color casts, separate color/lightness strengths, opposite local casts, detail retention, and tiny dimensions. Run `python Magik_Color_Correct/test_color_core.py` without ComfyUI. PyTorch and a running ComfyUI were unavailable in the build environment, so ComfyUI execution, UI and GPU-device behavior still need testing on the target installation.

## Background reading

- Reinhard et al., *Color Transfer between Images* (2001).
- Pitié et al., iterative probability distribution transfer / color grading.
- https://github.com/hahnec/color-matcher — algorithms used by several existing ComfyUI color matching nodes.
- https://github.com/kijai/ComfyUI-KJNodes — existing ColorMatch node interface.
- https://github.com/pengbo-learn/python-color-transfer — classical color transfer approaches.
