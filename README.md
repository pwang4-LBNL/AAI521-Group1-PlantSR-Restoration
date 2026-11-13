# Plant Image Restoration — AAI 521 Final Team Project

A three-person project using the PlantSR dataset to fine-tune pretrained generative models for **denoising, 4× super-resolution, colorization, and inpainting**, then integrate the four tasks into a web application.

**Planning window:** November 9–December 7, 2026 (America/Los_Angeles).

**Final submission deadline:** Monday, December 7, 2026.

## 1. Project scope

We will build a plant image restoration system covering four tasks, fine-tune pretrained models on PlantSR, and connect the models to a web application. Fine-tuning will be performed.

| Component | Work plan | Outputs |
| --- | --- | --- |
| Image denoising | Train on noisy/clean plant image pairs | Checkpoint, restored images and test metrics |
| Image super-resolution | Train on 64×64 inputs and 256×256 reference crops for 4× SR | Bicubic, pretrained and fine-tuned comparisons |
| Image colorization | Train on grayscale/color pairs | Colorized images, color error and perceptual assessment |
| Image inpainting | Train on masked/complete pairs | Restored regions, boundary analysis and test metrics |
| Integration and deployment | Connect all four models to a Flask web application | Upload, task selection, restoration, comparison and download |
| Evaluation and user testing | Compare models on held-out data and collect user feedback | Results, failure analysis and application improvements |
| Final deliverables | Prepare the report, GitHub code base and team presentation | Report PDF, repository link, video MP4 and submission inventory DOCX |

**Research question:** Does plant-domain fine-tuning improve restoration relative to the original pretrained models and simple baselines while preserving plant structure?

Sharper or more colorful outputs do not necessarily contain accurate botanical details. We will report failures and negative results as well as improvements.

## 2. Team responsibilities

| Role | Member | Restoration tasks | Project responsibilities | Reviewer |
| --- | --- | --- | --- | --- |
| A | Peng Wang — GitHub: pwang4-LBNL | Super-resolution | Data preparation and splits, shared evaluation, botanical detail analysis, report coordination | Ali |
| B | Jinyuan He | Denoising and colorization | GitHub code organization, updates, code review, automated checks and reproducibility | Peng |
| C | Ali Abdul-Haneed | Inpainting | Application development, model integration, deployment and user testing | Jinyuan |

Each member can choose suitable pretrained models from Hugging Face and use the training framework that best fits their tasks and available hardware. Jinyuan can use different models and training frameworks for denoising and colorization. Sharing code is optional.

The team will use the same dataset splits, sample IDs, evaluation rules and application interface so that independently developed models can be compared and integrated. Model choices and training configurations will be documented for each task.

**Every member's responsibilities:**

- Select, fine-tune and document the model(s) for their assigned task(s).
- Evaluate results and explain improvements, limitations and representative failures.
- Review another member's work and support integration.
- Write their report section and present their contribution in the team video.
- Participate in reproducibility checks and final submission review.

## 3. Four-week schedule

### Week 1 — Data, model selection and setup
**November 9–15, 2026**

Colab GPUs are available, and Ali also has a more powerful personal GPU. The team will use these resources for initial experiments and larger training runs, recording GPU memory and training speed to match each model to suitable hardware.

| Owner | Main tasks |
| --- | --- |
| Peng | Audit PlantSR, freeze data splits, prepare paired inputs and run initial SR baselines. |
| Jinyuan | Select denoising/colorization models; organize the repository, dependencies and code checks. |
| Ali | Select an inpainting model, prepare masks and build the application layout. |
| All | Run short fine-tuning trials, verify checkpoint reload and agree on the common inference interface. |

**Milestone:** Shared data splits, selected models and runnable training paths for all four tasks.

### Week 2 — Fine-tuning and integration
**November 16–22, 2026**

| Owner | Main tasks |
| --- | --- |
| Peng | Fine-tune SR and build the shared evaluation pipeline. |
| Jinyuan | Fine-tune denoising/colorization; review code and monitor repository checks. |
| Ali | Fine-tune inpainting; integrate model wrappers and deploy a development version. |
| All | Compare validation results, review failures and select model settings. |

**Milestone:** Fine-tuned checkpoints and an application that runs all four restoration tasks.

### Week 3 — Evaluation and user testing
**November 23–30, 2026**

| Owner | Main tasks |
| --- | --- |
| Peng | Run final held-out evaluation, consolidate figures/tables and analyze botanical details. |
| Jinyuan | Analyze denoising/colorization results; review GitHub code and verify reproducible execution. |
| Ali | Lead user testing, fix application issues and verify deployment from another device. |
| All | Draft report sections and presentation slides; freeze the experiment results and demo workflow. |

**Milestone:** Results, user feedback, report drafts and a stable demo ready for the final week.

Select checkpoints and settings using validation data before final testing. Keep the test set separate from model tuning, and document any later changes that affect reported results.

### Week 4 — Report, code and presentation
**December 1–7, 2026**

| Owner | Main tasks |
| --- | --- |
| Peng | Consolidate the final report and appendix, figures and results; coordinate the submission inventory. |
| Jinyuan | Finalize GitHub code, dependencies, README and run instructions; check the repository from a clean environment. |
| Ali | Finalize the deployment and user-testing summary; prepare and record the application demonstration. |
| All | Record and edit the team presentation, review the PDF and code links, and submit the complete package. |

- **December 1–2:** Consolidate the report PDF and appendix; finish GitHub cleanup and presentation slides.
- **December 3–5:** Record each member's presentation and the application demo; edit the final MP4.
- **December 6:** Review all files and links, complete the inventory DOCX and verify instructor access.
- **Monday, December 7:** Submit the final team project package.

### Final submission package

| Deliverable | File or link |
| --- | --- |
| Written report and appendix | `PlantSR_Final_Report.pdf`, including the appendix; list any separate appendix files if used |
| Code base | GitHub repository link for `AAI521-PlantSR-Restoration` |
| Team presentation video | `PlantSR_Presentation.mp4` and a video link if applicable |
| Final deliverable inventory | `Final_Deliverable_Inventory.docx`, submitted with the final team assignment |

The inventory Word document will list the team name and **Peng Wang, Jinyuan He and Ali Abdul-Haneed**, followed by the report/appendix filenames, code repository link, and presentation video filename/link under the three deliverable headings.

## 4. Teammate task checklists

### A — Peng Wang

- [ ] **A1:** Audit image readability, dimensions, orientation and duplicate/near-duplicate content.
- [ ] **A2:** Freeze split manifests and stable source-image IDs, including the source folder when filenames overlap.
- [ ] **A3:** Provide aligned HR/LR pairs and shared degradation/metadata conventions.
- [ ] **A4:** Select an SR model; run Bicubic and original pretrained-model baselines.
- [ ] **A5:** Fine-tune SR, verify checkpoint reload and select the checkpoint using validation data.
- [ ] **A6:** Implement shared per-image metrics, comparison tables and botanical detail analysis.
- [ ] **A7:** Review Jinyuan's training/evaluation and coordinate the final report and inventory.
- [ ] **A8:** Write the problem, data, SR and evaluation sections; prepare slides and record the associated presentation.

### B — Jinyuan He

- [ ] **B1:** Select suitable denoising and colorization models, using separate frameworks where helpful.
- [ ] **B2:** Prepare each model's paired-data loader, training configuration and checkpoint workflow.
- [ ] **B3:** Fine-tune denoising and evaluate across the specified noise strengths.
- [ ] **B4:** Fine-tune colorization and assess reference color error and visual quality.
- [ ] **B5:** Supply task-specific checkpoints, inference wrappers, configurations and training logs.
- [ ] **B6:** Lead GitHub code organization and maintenance; review pull requests and monitor formatting, lint and functional checks.
- [ ] **B7:** Review Ali's inpainting code and verify installation, checkpoint loading and inference from a clean environment.
- [ ] **B8:** Finalize dependencies, run instructions and repository documentation with the team.
- [ ] **B9:** Write the denoising, colorization and code implementation sections; prepare slides and record the associated presentation.

### C — Ali Abdul-Haneed

- [ ] **C1:** Select an inpainting model; generate masks and verify covered-area ratios and alignment.
- [ ] **C2:** Fine-tune inpainting and evaluate missing regions and boundary failures.
- [ ] **C3:** Build upload, task selection, mask handling, preview, processing status and download features.
- [ ] **C4:** Integrate all four model handlers through the common interface and manage GPU memory and request concurrency.
- [ ] **C5:** Lead application deployment; document startup steps and verify access and inference from a second device.
- [ ] **C6:** Lead user testing, summarize feedback, implement improvements and retest the application.
- [ ] **C7:** Review Peng's data/SR work and support training on the available GPU resources.
- [ ] **C8:** Write the inpainting, application, deployment and user-testing sections; prepare slides and record the application demo.

## 5. Dataset and paired inputs

The [PlantSR dataset on Figshare](https://figshare.com/articles/dataset/PlantSR_Dataset/24648150) contains 1,030 plant images: 830 official training images and 200 official test images. The dataset is licensed under CC BY 4.0.

We divided the official training images into training and validation sets by original image. The official test set was retained subject to the documented audit and duplicate checks. The final split counts are recorded in `prepared/summary.json`, and any exclusions are listed in `prepared/manifests/excluded.csv`.

### Download and use the prepared dataset

1. Download [PlantSR_prepared_for_team.zip](https://drive.google.com/file/d/1j_vx9Yuow1pu8MzYmbGY4udXBpB4dBx1/view?usp=sharing) from Google Drive.
2. Extract the ZIP and set your dataset root to the extracted `prepared/` folder. Resolve image and crop paths in the CSV manifests relative to that folder.
3. Use the same `train.csv`, `val.csv`, and `test.csv` manifests under `prepared/manifests/` for all four restoration tasks.

The [preprocessing notebook](notebooks/PlantSR_Preprocessing_Center256.ipynb) documents the audit, split assignment, and crop export.

### Training and evaluation inputs

- Assign splits before cropping, augmentation, or degradation. Keep each original image and all its derivatives in the same split.
- During training, sample random 256 × 256 crops from the full-size training images, then apply the degradation for each task.
- For validation and testing, use the saved 256 × 256 center crop for each image. The manifests record the crop coordinates.
- Use fixed, reproducible validation and test degradations, including noise and masks where applicable. Use identical inputs and targets when comparing pretrained and fine-tuned models.
- Use native-resolution crops as reference targets. Report results as a crop-based benchmark. Full-image performance requires a separate evaluation.

| Task | Model input | Reference target | Initial recipe |
| --- | --- | --- | --- |
| Denoising | Noisy 256×256 RGB crop | Clean crop | Gaussian noise; training sigma 5–25 in 0–255 units; primary test sigma 15, optional 5/25 robustness sets |
| Super-resolution | 64×64 RGB image | 256×256 RGB crop | Fixed Bicubic 4× downsampling |
| Colorization | Grayscale crop replicated to three channels | Original color crop | Fixed grayscale conversion |
| Inpainting | Crop with missing regions | Complete crop | Rectangular/free-form masks; roughly 5–20% covered during training; fixed test masks |

For inpainting, **mask 1 = missing/reconstruct; mask 0 = preserve**.

Manifest fields: `sample_id, source_image_id, split, task, input_path, target_path, mask_path, crop_coordinates, degradation_parameters, seed`.

## 6. Model and training plan

Model selection is flexible. Each task owner will choose a pretrained generative model with downloadable weights, a usable fine-tuning workflow and hardware requirements that fit the available GPUs. The four tasks can use different architectures, Hugging Face checkpoints and training frameworks.

| Task | Owner | Model selection and adaptation |
| --- | --- | --- |
| Super-resolution | Peng | A pretrained 4× SR generator, such as [Real-ESRGAN](https://huggingface.co/kornia/real_esrgan); fine-tune on aligned LR/HR pairs |
| Denoising | Jinyuan | A suitable pretrained generative restoration model; fine-tune on noisy/clean pairs |
| Colorization | Jinyuan | A suitable pretrained colorization or conditional image-generation model; fine-tune on grayscale/color pairs |
| Inpainting | Ali | A suitable pretrained inpainting or conditional image-generation model; fine-tune on masked/complete pairs |

[InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) is one option for investigating conditional image editing. It is optional; each owner can select a more suitable model for their task. Model selection will consider restoration quality, preservation of plant structure, training support and compute cost.

For each task, keep a copy of its original pretrained checkpoint for the before/after comparison. Save fine-tuned weights separately with task and version identifiers. Model-specific environments can be maintained separately, while inference wrappers follow the shared application interface.

**Training workflow**

1. Prepare aligned input/target pairs and apply the same geometric transforms to inputs, targets and masks.
2. Run a short 50–100-step trial, verify finite losses and parameter updates, and reload the saved checkpoint in a fresh process.
3. Adjust batch size, learning rate and training duration using validation results and available GPU memory.
4. Fine-tune the selected model and save checkpoints, configurations and training logs.
5. Select the final checkpoint on validation data and evaluate it against the original pretrained model on identical test inputs.

**Starting configurations**

- SR: HR 256×256 / LR 64×64, batch size 1–4 and learning rate around `1e-5` for paired reconstruction fine-tuning.
- Conditional diffusion, if selected: 256×256 benchmark images, batch size 1 and gradient accumulation; configure trainable modules, preprocessing and losses for the chosen architecture.
- Use mixed precision, gradient checkpointing or compatible parameter-efficient fine-tuning where useful.
- Record model revision, code commit, environment, data split, trainable parameters, loss, training steps and seed.
- Fix inference settings, including prompts and sampling parameters when applicable, using validation data before final testing.

**Model-specific implementation notes**

- Load weights into the matching architecture and checkpoint format.
- For Real-ESRGAN, describe L1-only generator adaptation as reconstruction fine-tuning; full GAN retraining also requires the adversarial training components.
- For InstructPix2Pix, preserve an existing 8-channel image-conditioned input layer when adapting a training script.
- For adapter-based fine-tuning, verify that saved adapters reload with the intended base model and image-conditioning workflow.

## 7. Evaluation rules

For every task, compare **a simple baseline, the original pretrained model, and the fine-tuned model** on identical held-out inputs.

| Task | Simple baseline | Evaluation focus |
| --- | --- | --- |
| Denoising | Noisy input and a traditional denoiser such as non-local means | PSNR/SSIM/LPIPS, noise reduction, leaf-vein preservation |
| Super-resolution | Bicubic interpolation | PSNR/SSIM/LPIPS, edge artifacts, inference time |
| Colorization | Grayscale repeated as RGB, explicitly labeled a weak baseline | Color error, perceptual similarity, naturalness, failure analysis |
| Inpainting | Traditional fill such as OpenCV Telea | Missing-region error, boundary continuity, plausible structure; full-image scores only as supplementary results |

Use a consistent RGB convention, value range, crop/border policy and output size. LPIPS requires its own expected normalization. Aggregate per original image, retain matched comparisons, and report failures rather than silently dropping them.

For diffusion, fix sampling settings and seeds; do not select the best sample after looking at the test reference. If multiple seeds are evaluated, report all of them and avoid counting them as independent source images.

For inpainting, final compositing may preserve visible pixels exactly:

`final = (1 - mask) * input + mask * generated`

Report the compositing step and missing-region results. Excellent scores on copied visible pixels are not evidence of model restoration quality.

Colorization and inpainting can have multiple plausible answers. Reference error is useful but does not establish unique physical truth. Inspect structure, color, texture and invented details.

Results fields: `sample_id, task, method, checkpoint_revision, seed, metric_name, metric_value, inference_seconds, status, error_note`.

## 8. Application and user testing

Shared inference interface:

`restore(image, task, mask=None, seed=42)`

Each task wrapper will return the output image, task/model version, processed dimensions, elapsed time and success/error information.

Ali will build the application to:

- Accept supported JPG/PNG inputs and display the actual crop/processing dimensions.
- Offer all four tasks with meaningful defaults.
- Support mask upload or drawing for inpainting, with a preview.
- Show processing state, input/output comparison and PNG download.
- Queue GPU work and avoid loading all large checkpoints simultaneously when memory is limited.
- Run real inference; cached example images may supplement the demo but cannot replace the workflow.
- Show PSNR/SSIM only in a benchmark context with a known reference, not for arbitrary user uploads.

Ali will deploy the Flask application with an appropriate serving process and GPU support where needed. The deployment documentation will include the application location, environment and startup instructions. Docker can be used for packaging.

Ali will coordinate user testing with **3–5 volunteers**. Each tester will upload an image, choose a task, perform inpainting with a mask and download a result. The team will record task completion, confusing controls, perceived quality and waiting time, then document issues, fixes and retest outcomes.

## 9. Repository organization

| Path | Contents |
| --- | --- |
| `README.md` | Project scope, ownership, milestones and run instructions |
| `configs/` | Per-task training/inference configurations |
| `src/data/` | Data audit, splits and paired-data generation |
| `src/train/` | Task-specific training scripts and model adapters |
| `src/inference/` | Four task wrappers and model loading |
| `src/evaluation/` | Metrics, aggregation and figure generation |
| `app/` | Flask application and UI |
| `tests/` | Data, inference, integration and application checks |
| `.github/workflows/` | Automated repository checks maintained by Jinyuan |
| `notebooks/` | Colab entry points and exploratory analysis |
| `data/manifests/` | Source-image splits and sample metadata |
| `reports/` | Metrics, figures, report PDF, appendix and user-testing notes |
| `presentation/` | Slides, narration outline and video access information |
| `docs/` | Environment, model provenance, deployment and final deliverable inventory |

Keep full datasets, large checkpoints, video files, credentials and temporary uploads out of ordinary Git commits. Document retrieval locations and checksums using suitable artifact/model storage. Cite reused data, model weights and code, and retain their applicable licenses.

## 10. Collaboration

- Use short task branches such as `feat/sr`, `feat/denoise`, `feat/colorize`, `feat/inpaint` and `feat/app`.
- Open focused pull requests with a description, validation results and reproduction steps.
- Have another teammate review before merging; Jinyuan will coordinate code review and monitor repository checks.

