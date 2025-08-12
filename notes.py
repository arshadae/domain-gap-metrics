NOTES = {
    "FID": "Fréchet Inception Distance; lower is better. Measures distance between Gaussian fits to Inception features of the two domains. Sensitive to both mean and covariance shifts. Good for global distribution similarity.",
    "KID": "Kernel Inception Distance; mean and std shown. Polynomial kernel MMD on Inception features; unbiased and more stable on small datasets than FID. Lower is better.",
    "PAD": "Proxy A-distance; lower is better (0 = indistinguishable domains). Computed via domain classifier accuracy on Inception features. High PAD indicates large separability between domains.",
    "DomainClassifierAccuracy": "Accuracy of logistic regression domain classifier in PAD calculation. Higher accuracy indicates a larger domain gap. 0.5 means domains are indistinguishable.",
    "MMD2_RBF": "Squared Maximum Mean Discrepancy with multi-scale RBF kernels on Inception features; lower is better. Measures distance between domain feature distributions without assuming Gaussianity.",
    "CKA_linear_paired": "Linear Centered Kernel Alignment between paired domain features (paired by index after truncation to min(N_A,N_B)). Higher means more similar representations (1 = identical). Sensitive to feature-space geometry.",
    "LPIPS_set_to_set": "Learned Perceptual Image Patch Similarity averaged over random cross-domain image pairs. Lower is better; requires `lpips` package. Correlates with human perceptual similarity.",
    "ChiSquare_RGB_Hist": "Chi-square distance between average RGB histograms of the domains; lower indicates more similar global color distribution. Ignores spatial structure.",
    "TSNE": "2D t-SNE embedding of Inception features, saved as a PNG file (suffix includes eval_name if provided). Useful for visualizing local feature-space overlap between domains.",
    "UMAP": "2D UMAP embedding of Inception features (if available), saved as a PNG file (suffix includes eval_name if provided). Preserves more global structure than t-SNE.",
    "Features": "Unless task metrics are requested, all metrics are computed using InceptionV3 pool3 (2048-D) features extracted from resized and center-cropped RGB images.",
    "Limitations": "Feature-level metrics (FID, KID, PAD, MMD, CKA, LPIPS, histogram distances, t-SNE, UMAP) capture distribution and representation similarity, not task-specific performance. A small domain gap in these metrics does not guarantee equal downstream accuracy. For a reliable assessment, combine these with task-specific metrics (classification accuracy/F1, segmentation IoU, detection mAP) where possible, as they directly measure impact on the intended end task."
}
