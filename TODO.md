# TODO

## Done
- [x] Set up project
- [x] Set up Conda environment
- [x] Verify CUDA/GPU
- [x] Load pretrained ViT
- [x] Predict single image
- [x] Add Top-5 prediction
- [x] Verify ImageNet dataset
- [x] Read ImageNet labels
- [x] Read meta.mat
- [x] Build ImageNetDataset
- [x] Build DataLoader
- [x] Add Gaussian Blur level 4
- [x] Evaluate base ViT on clean and blurred images

## Next
- [ ] Create training folder
- [ ] Write fine-tuning script
- [ ] Fine-tune ViT on Gaussian Blur level 4
- [ ] Save checkpoint
- [ ] Evaluate fine-tuned ViT
- [ ] Compare base vs fine-tuned model

## Later
- [x] Logit Lens correct-class probability for Blur-4 and Noise-4
- [x] First correct-prediction layer for Blur-4 and Noise-4
- [x] Expected Calibration Error (ECE)
- [x] Attention entropy analysis for Blur-4 and Noise-4
- [x] Attention squared-difference and cosine-similarity analysis
- [ ] Representation similarity analysis
- [x] Vanilla ReLU and BatchTopK SAE training/evaluation pipeline
- [ ] Complete full 15-epoch training for all eight level-4 SAEs
- [ ] Generate final SAE similarity distributions
