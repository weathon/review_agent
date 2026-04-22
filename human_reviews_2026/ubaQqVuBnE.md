# Nonparametric Data Attribution for Diffusion Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Data attribution for generative models seeks to quantify the influence of individual training examples on model outputs. Existing methods for diffusion models typically require access to model gradients or retraining, limiting their applicability in proprietary or large-scale settings. We propose a *nonparametric* attribution method that operates entirely on data, measuring influence via patch-level similarity between generated and training images. Our approach is grounded in the analytical form of the optimal score function and naturally extends to multiscale representations, while remaining computationally efficient through convolution-based acceleration. In addition to producing spatially interpretable attributions, our framework uncovers patterns that reflect intrinsic relationships between training data and outputs, independent of any specific model. Experiments demonstrate that our method achieves strong attribution performance, closely matching gradient-based approaches and substantially outperforming existing nonparametric baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose Nonparametric Diffusion Attribution (NDA), a gradient-free data-attribution method for diffusion models that scores patch-level influence using the analytic optimal score, and is optimized with convolution-based operations. They target black-box/proprietary settings with no access to a model's parameters, and show attributions that outperform prior baselines and approach gradient-based methods on CIFAR-10 and CelebA.

### Strengths
The paper seems practical and easy enough to follow in situations where no model access is available. The authors also report spatially interpretable results, and the method is shown to outperform significantly the tested prior baselines.

### Weaknesses
1. A few core assumptions and scope limits should be made explicit. Most derivations rely on naive score-matching or score identities for standard diffusion losses, so general statements about “model-agnostic” influence seem like an overreach.  For example, if we inject an exact-reconstruction penalty in the loss, the results may no longer hold.

2. Compute requirements seem concerning. The convolutional trick proposed by the authors helps with peak memory blow-up, but the per-image convolution over all training patches and multiple timesteps/scales won't scale very well. 

3. The authors test their work in relatively low-resolution settings (up to 64×64).

4. The visual distribution tested a relatively simple, with generally smooth contours and simple patches.

### Questions
(number references to the Weekness section)

On (2). It would be valuable to report wall-clock vs. dataset size, have some theoretical scaling analysis on compute for data size/resolution, and give scaling guidances. 

On (3). Are the multiscale tricks and hyperparameters stable at higher resolutions or for more scales?—an experiment at ≥256×256 would be very valuable. 

On (4). How does the algorithm behave with more complex distributions (e.g. Imagenet)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a nonparametric approach to data attribution for diffusion models motivated by scalability. It computes patch-level influence scores derived from the optimal score-function formulation. Effectiveness is evaluated via Linear Datamodeling Score (LDS) and counterfactual removal-and-retrain experiments.

### Strengths
The paper is well motivated and well presented. The proposed approach is solid, although more explanations will be helpful (see weakness). In the evaluated scenarios, the methods seem to perform well.

### Weaknesses
1. While the method is motivated by a score-function view, the estimator still just behaves like a complicated version of a similarity measure against training patches. Additional ablations/theory to isolate what truly helps will be very appreciated. For example, do naive patch-level similarities across timesteps already recover most of the effect? 
2. The paper is motivated by scalability. However, it seems that the paper still only evaluates on smaller models that generate low-res images.

### Questions
As the authors also point out, patch-level attribution seems to be interpretable. Do you think that for diffusion model kind image generation models, patch-level attribution is just fundamentally a better approach compared to image-level attribution, as it provides more fine-grained signals?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a method for quantifying the relevancy of individual training samples to the outputs generated by diffusion models without requiring model access or gradients. The method performs patch-level similarity comparisons between generated and training images to quantify the attribution. Evaluation results using CIFAR-2, CIFAR-10, and CelebA datasets, show that NDA performance is close to gradient-based approaches and outperforming existing nonparametric baselines.

### Strengths
1. Proposing an efficient query-only/nonparametric, data attribution approach for diffusion models.
2. The method is based on analytical properties of the diffusion score function.
3. The patch-level mapping can be used for understanding the relationship between training and generated images.
4. Experiments show consistent improvements over the baselines.

### Weaknesses
1. Figure 1 – The proposed method is not clear from the figure. I suggest improving the figure.
2. No clear description of what are the requirements for performing the attribution
3. Not clear how the ground-truth was generated?
4. Experiments should be extended to larger and more complex datasets (e.g., ImageNet) to prove scalability and generalization.
5. Comparisons with other strong nonparametric or model-free baselines is missing, for example: 

[a] CustomMark: Customization of Diffusion Models for Proactive Attribution
[b] Montrage: Monitoring training for attribution of generative diffusion models

6. Method's runtime analysis is missing.

### Questions
1. Please explain what are the requirements for performing the attribution?
2. How the ground-truth was generated?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Nonparametric Diffusion Attribution (NDA), a novel, model-agnostic method for data attribution in diffusion models. The key insight is to reinterpret the weighting term from the analytical, optimal score function of a diffusion process as an influence score. By grounding this in a local score function, the method derives a patch-wise similarity metric that is computationally efficient and requires no access to model parameters. The method demonstrates strong empirical performance on the Linear Datamodeling Score (LDS) when compared to several baseline methods.

### Strengths
1. Novel Non-Parametric Framework: The paper makes a valuable contribution by framing the attribution problem from a non-parametric, data-driven perspective.
2. Theoretical Motivation: The method is motivated by reinterpreting the weighting term (W_t) from the analytical form of the optimal score function (Eq. 5-6) as a measure of influence. This provides a strong theoretical justification for why this specific form of similarity should be effective.
3. Strong Empirical Performance (vs. Chosen Baselines): On the LDS and counterfactual retraining metrics, the proposed NDA method is shown to be highly effective. It substantially outperforms other full-image non-parametric baselines (Raw Pixel, CLIP).

### Weaknesses
1. Unsupported Claim of Model-Independence: The paper makes a strong, highly valuable claim that "our empirical results show that the attribution scores remain consistent across a variety of architectures and training regimes" (Page 6, Line 281). However, there are no empirical results presented in the manuscript (including the appendix) to substantiate this.

2. Missing "Apples-to-Apples" Baselines: The paper's core method (NDA) is patch-based. However, the non-parametric baselines it is compared against ("Raw Pixel" and "CLIP Similarity") are full-image-based. This is an "apples-to-oranges" comparison that inflates the perceived contribution of NDA. The strong performance seen might not come from the specific "score-function" formulation but simply from the act of using patches, which is a known technique. Some "naive patch-based" baselines include:
    * Patch-wise Raw Pixel: A method that computes L2 similarity on top-k raw patches and aggregates them.
    * Patch-wise Feature Similarity: A method that uses a standard feature extractor (e.g., CLIP, a pretrained CNN, or DINOv2) to embed patches and then computes a top-k aggregated similarity.

3. Failure to Disentangle Causal Factors: The paper's novelty rests on two assumptions: theoretical motivation via optimal score function and localization. The current experiments do not disentangle these. It is unclear whether the method's good performance comes from the "locality" assumption (Eq. 9) or the "optimal score" formula itself (Eq. 6). The authors should provide an ablation study for an "Image-wise NDA" baseline. This method would be derived directly from the full-image optimal score function (Eq. 5) and would use the full-image weighting term $W_t$ (Eq. 6) as the influence score, with no patch decomposition.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
