# Super-Resolution and Video Frame Interpolation: Comparing Loss Functions

This repository contains the code and recorded super-resolution results for my M.S. thesis in Applied Statistics at California State University, Long Beach. The project investigates how the choice of loss function during super-resolution training affects image reconstruction and subsequent video frame interpolation.

The workflow combines **super-resolution (SR)**, which reconstructs higher-resolution images from lower-resolution inputs, with **video frame interpolation (VFI)**, which predicts an intermediate frame from two surrounding frames. The goal is to compare how different training objectives affect the quality of both the restored images and the interpolated middle frame.

**[Read the thesis on ProQuest](https://www.proquest.com/docview/3391144167/77E463CD2D394634PQ/1?sourcetype=Dissertations%20&%20Theses)**

## How the project works

### 1. Prepare low-resolution and reference images

The experiments use frames from Vimeo-90K. For SR training, cached image sequences are loaded into NumPy arrays and flattened into individual frames. The dataset code creates training, validation, and test splits, then generates paired low-resolution and high-resolution images using bicubic downsampling.

The main SR configuration uses **4x— scaling in each spatial dimension**. Training uses random crops; validation and testing use images cropped to dimensions divisible by the scaling factor.

### 2. Train SRResNet with different loss functions

Separate SRResNet models learn to reconstruct the high-resolution reference using seven loss functions:

| Loss function | Main emphasis |
| --- | --- |
| MSE | Squared pixel differences |
| L1 | Absolute pixel differences |
| G-Loss | Pixel differences and image gradients at multiple scales |
| SSIM Loss | Local structural similarity |
| MS-SSIM | Structural similarity across multiple scales |
| MS-SSIM + L1 | Structural similarity combined with pixel accuracy |
| Huber | A combination of squared and absolute error behavior |

Training records loss, PSNR, and SSIM, and saves model checkpoints. The best SRResNet checkpoint is selected using validation PSNR.

### 3. Fine-tune each model as an SRGAN

Each SRResNet variant initializes an SRGAN generator. The generator is then fine-tuned with a VGG19 feature-based content loss and an adversarial loss, while a discriminator learns to distinguish reconstructed images from reference images.

**The loss names attached to the SRGAN variants identify their SRResNet pretraining loss. All variants use the same perceptual and adversarial objectives during SRGAN fine-tuning.** The best SRGAN checkpoint is selected using validation content loss.

### 4. Interpolate the missing middle frame

The VFI notebook uses Vimeo-90K triplets containing `im1.png`, `im2.png`, and `im3.png`. In the standard benchmark, each frame is downsampled by 4x—, and each available SRGAN restores the frames to their original dimensions.

The restored first and third frames are passed to a pretrained **TLB-VFI** model to predict the middle frame. The original high-resolution middle frame serves as the reference for evaluation; it is not supplied to TLB-VFI during prediction.

The benchmark also evaluates bicubic upsampling followed by TLB-VFI and records direct SR reconstruction of the middle frame for comparison. TLB-VFI is used for inference rather than retrained in this workflow.

## Expected outputs

After training and evaluation, the workflow produces:

- **Model checkpoints** for each trained SRResNet and SRGAN variant.
- **Training histories** containing loss and reconstruction metrics across epochs.
- **Restored endpoint frames and a predicted middle frame** for each evaluated triplet and SRGAN variant.
- **CSV metric tables** with per-triplet results and aggregate means, standard deviations, and sample counts.
- **Visual comparisons**, including reference images, reconstructed images, training curves, metric charts, and absolute-error heatmaps.

The expected research outcome is a comparison of loss functions: which objectives produce accurate SR reconstructions, and how those differences carry through to interpolation. Better SR scores do not automatically guarantee better interpolated frames, so the two stages are evaluated separately.

Quality is measured with **PSNR** and **SSIM**, where higher values indicate closer agreement with the reference under each metric. The SR module computes these metrics on the luminance (Y) channel, while the VFI wrapper computes them on RGB images. Scores from the two evaluation paths should be interpreted within their respective settings.

An optional experiment upscales native-resolution frames by 4×—or 16×— before interpolation, then resizes the outputs back to native resolution for evaluation. The 16× case applies the 4× SR model twice and may require substantial GPU memory.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| `sr_modular_runner_notebook.ipynb` | Loads cached frames, runs SRResNet and SRGAN training, and generates SR comparisons |
| `srgan_tlbvfi__wrapper.ipynb` | Loads trained SRGANs and pretrained TLB-VFI, runs triplet benchmarks, and generates VFI comparisons |
| `sr_modular/` | Model architectures, loss functions, data preparation, training, evaluation, and utilities |
| `sr_project_outputs/metrics/` | Recorded training histories and the SR validation/test summary |

Generated checkpoints are saved under `sr_project_outputs/checkpoints/`. The VFI notebook writes results under its configured `SR_OUTPUTS_ROOT/vfi_comparisons/`, with separate folders for the standard 4x— benchmark and optional native-upscale experiments.

## Recorded SR results

The committed `model_metrics_summary.csv` contains results for seven SRResNet and seven SRGAN variants. Among the recorded SRGAN variants, SSIM-loss pretraining has the highest test PSNR and SSIM: approximately **29.946 dB** and **0.8482**, respectively.

These values describe the saved **SR evaluation**, not the combined SR + VFI benchmark. Consult the thesis for the full experimental analysis and conclusions.

## Running the workflow

The notebooks were developed for Google Colab and include Drive mounting cells and paths that must be configured for your environment.

1. Prepare the Vimeo-90K data and cached NumPy frames used by the SR runner.
2. Configure the paths, loss selection, training parameters, and data splits in the SR notebook and `sr_modular/config.py`.
3. Run the SR notebook to train SRResNet variants, fine-tune their SRGAN counterparts, and save checkpoints and metrics.
4. Obtain the [TLBVFI repository](https://github.com/ZonglinL/TLBVFI), its dependencies, and pretrained weights. Configure the wrapper's repository, dataset, output, and checkpoint paths.
5. Run the VFI wrapper to compare the available SRGAN variants on selected Vimeo test triplets.

## Acknowledgment

The interpolation stage uses **TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation**, by Zonglin Lyu and Chen Chen. Credit for the TLB-VFI architecture, pretrained weights, and original implementation belongs to its authors.

- [Original TLBVFI repository and citation information](https://github.com/ZonglinL/TLBVFI)
- [M.S. thesis on ProQuest](https://www.proquest.com/docview/3391144167/77E463CD2D394634PQ/1?sourcetype=Dissertations%20&%20Theses)
- [M.S. defense presentation--mock version](https://www.youtube.com/watch?v=TGduKT1s6a4)
