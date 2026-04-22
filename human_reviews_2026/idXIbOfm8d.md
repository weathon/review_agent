# JET-Diff: Joint-Encoding Tensor Diffusion Model for Accurate DTI Reconstruction from Sparse DWIs

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Diffusion Tensor Imaging (DTI) is an advanced magnetic resonance imaging (MRI) technique for characterizing white matter microstructure. Conventional DTI protocols require multiple diffusion-weighted imaging (DWI) acquisitions across numerous directions, resulting in long scan times, motion artifacts, patient discomfort, and reduced clinical utility. Current deep learning approaches frequently yield diffusion tensors that are anatomically inconsistent or physically implausible. We introduce Joint-Encoding Tensor Diffusion (JET-Diff), a framework that synthesizes the full six-component diffusion tensor in 3D from a highly undersampled DWI acquisition. Specifically, we propose a latent diffusion model operating on a set of coupled latent tensors derived from sparse DWIs and diffusion tensor components, which improves anatomical fidelity and encourages physically consistent tensors. JET-Diff leverages a novel anatomical autoencoder to disentangle structural information from tensor properties, yielding a compact and expressive latent space optimized for generative performance. Experiments on the Human Connectome Project (HCP) Young Adult dataset demonstrate that JET-Diff improves reconstruction accuracy and produces geometrically consistent diffusion tensors, as evidenced by SPD-aware validity metrics such as Log-Euclidean and tractography-based distances.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a diffusion-model-based approach for DTI reconstruction from undersampled DWI data. The method trains a latent diffusion model to synthesize DTI conditioned on DWI measurements. The main novelties are: 1) a new autoencoder architecture for the latent space, which uses multi-scale features from DWI to improve the decoding of DTI images from the latent space, as well as a "Joint MLP block" to capture correlations among tensor components; 2) an additional pretraining stage to learn the unconditional joint distribution of DTI and DWI, before training the conditional diffusion model for DTI synthesis. Experiments were conducted on the public human connectome dataset and includes comparisons to existing methods as well as ablation studies on each proposed component.

### Strengths
- The paper identifies important problems in existing approaches for DTI reconstruction, e.g., Diff-DTI -- the anatomical inconsistency due to using 2D models and the physical implausibility caused by directly estimating DTI-derived parameter maps without reconstructing the tensor image itself.

### Weaknesses
- The rationales for the proposed approaches can be better explained -- some components seem redundant and could be replaced with simpler alternatives. Empirical evidence that supports their benefits can also be improved. 
- Most technical improvements appear specific for the DTI problem, with limited generalizability.
- Naming of some method components can be misleading. Some standard techniques, e.g., cross-attention, are given a new name, e.g., "Joint-Encoding Tensor Attention". The contributions and novelties can be better clarified.
- Several notations, abbreviations, and technical terms are not clearly defined, making the paper relatively hard to follow.

### Questions
- On the proposed pretraining step
  - What is motivation for learning the joint distribution of DTI and DWI? Why will it benefit the reconstruction of DTI from DWI? Given that the problem is not to synthesize entirely new pairs of DTI-DWI data, but only to estimate DTI given DWI inputs, learning this joint distribution seems an overkill and unnecessary.
  - Empirical evidence supporting the benefit of this pretraining step is lacking. Currently, only visual comparison on one test example is provided between the models trained with and without this pretraining step (Section 4.3.3). No numerical comparisons (e.g., PSNR over the entire test set) were presented.
  - In the existing visual comparison, were the two models trained for the same number of epochs? If not, this may lead to an unfair comparison.

- On the proposed latent diffusion model and autoencoder
  - The autoencoder is first trained on each of the six tensor components independently. Then, to model inter-correlation among tensor components, the authors proposed to add a "Joint MLP block" at the end of the decoder. Why is such a two-stage approach employed, in favor of a more straightforward approach that applies the autoencoder to all tensor components simultaneously, by viewing them as a six-channel image? Have the authors compared these two approaches?
  - Similarly, why is the diffusion model applied independently to each tensor component (line 218), rather than to the entire tensor image directly, by treating it as a six-channel image? Have the authors compared these two approaches?
  - Can the authors clarify the difference between "Joint-Encoding Tensor Attention" and the standard Cross-Attention mechanism used in diffusion models for conditional image synthesis?
  - Can the authors clarify the meaning of their name "Multi-Tensor diffusion model"? Based on my understanding, this name suggests that the diffusion model is implemented for both DWI and DTI, hence "Multi-Tensor". However, DWI is not a tensor image -- it's a sequence of measurements obtained at different gradient directions.

- No comparison is conducted with existing DTI reconstruction methods, e.g., Diff-DTI, whose limitations motivated this work.

- Questions on notations:
  - Equation in line 82: The symbols are undefined.
  - Line 220: $\epsilon_c$ is not defined.
  - What does $z_{c,t}$ represent? Although $z_{c,t}^X$ and $z_{c,t}^Y$ are defined, the notation without superscript is not. Based on Section 3.3.3, it seems that $z_{c,t}$ is an abbreviation of the DTI latents $z_{c,t}^Y$. If so, consider unifying the notation.
  - In the defition of pretraining loss (line 252), why is there {} around $z_{c,t}$? If it denotes the set of $z_{c,t}^Y$ for all $c$, why is there still a sum over $c$ in the equation?
  - Consider adding definitions for MD, RD, FA and Color FA for readers not familiar with DTI.
  - Consider defining mean tract core distance.

Other comments:
- The comparison with CycleGAN may not be meaningful, since CycleGAN is developed for unpaired data. This work assumes that paired data is available for training, and, I assume, all other methods are trained on paired data, which makes this an unfair comparison.
- In the ablation experiment on autoencoder, what is the design of the "standard autoencoder"? Was it trained only on DTI? If so, how is the subsequent diffusion model adapted, which takes both the latent representations of DWI and DTI as input?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a diffusion model for recovering "Diffusion Tensor Images" (DTI), a relatively expensive brain imaging modality that captures directional flow in the brain, using a much smaller number of Diffusion Weighted Images (DWI) than is typically used (going from 90 to 4). DTI captures biologically rich signals, but long scan times have limited its clinical and research adoption. The proposed approach uses diffusion models to solve an inverse problem - trying to reconstruct the full diffusion tensor (6 components) from only 3 directional scans - with the hope that strong biological priors make accurate reconstruction possible with less data. The paper proposes several methods to improve the reconstruction, including an improved latent space autoencoder, multi-stage training, and a refinement stage for tensor consistency. Small but consistent improvements are reported.

### Strengths
- This is an interesting and valuable problem. My understanding from collaborators is that DTI is more useful and informative for many reasons, but that adoption is limited due to the time/expense of obtaining the scans. Speeding this up and/or making it possible to augment DTI datasets with semi-synthetic DTI using fewer DWI scans would be beneficial for the field. 

- I thought the introduction to the problem for the ICLR audience seemed good. 

- The metrics, baseline, and comparison methods all seemed reasonable and well chosen to me. I was happy to see the ADE as a logical baseline. Showing results on scalar measures, tensor component reconstruction, and tractography were exactly what I was hoping to see. In particular, one might hope that tractography benefits most from accurate directional flow reconstruction. 

- Training a good autoencoder is important and tricky. The decoder with DWI cross attention seems like it may be the most useful component in this paper.

### Weaknesses
Generally, the training procedure (4 stages, App.C) seems overly complex and not well justified. 
The use of joint diffusion with joint unconditional pretraining and the refinement stage complicate the training pipeline, but their incremental benefits are small and not clearly demonstrated to generalize beyond this dataset. It appears that the main performance gains come from the DWI-aided decoder and improved autoencoder representation rather than from the diffusion modeling itself. (see detailed comments below)

Based on the presented results, I don't think this paper substantially moves the needle toward solving the DTI problem, and the broader methodological takeaways for the ICLR community are unclear. 

## Joint diffusion versus conditional

I'm skeptical about one of the core claims in the paper, that a joint diffusion is useful / necessary compared to standard conditional diffusion (like the LDM baseline would use). 
The single example figure in Fig. 6, 4.3.3, doesn't really convince me that this makes sense. This 2-stage training (joint diffusion, then conditional) complicates the model and goes contrary to typical practices. More convincing evidence would be needed (for me) to consider adopting this idea. 

I believe that the only substantial difference in performance comes from the DWI-aided decoder. The difference between standard AE and anatomical AE in the ablation study (Table 3) is about the same as the difference between your method and baseline LDM in Table 2 (PSNR/SSIM gains on the order of 0.03–0.05).  If I understood correctly, the baseline LDM uses the standard autoencoder rather than the anatomical version, which likely explains most of the performance gap.

## Training stages
In Table 3, you show the influence of one of the four training stages, joint refinement, and again here the differences seem small to justify the extra training. 
While the ablation studies do show some small improvements from multiple training stages, an equally plausible hypothesis is that the improvements stem from more overall optimization steps rather than the specific form of joint diffusion. A single-stage conditional model trained with comparable compute might achieve similar results.


Overall, I appreciate the paper’s ambition and relevance, but the empirical evidence does not yet justify the complexity of the proposed pipeline. Clarifying the contribution of each stage — especially the necessity of joint diffusion — and demonstrating consistent benefits in downstream tasks would strengthen the case considerably.

### Questions
See questions in weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
**Background for those unfamiliar**: Diffusion MRI maps water diffusivity within the brain along a set of angles (diffusion gradients) at each voxel, thereby creating multidimensional images (3D space + 2D angles). Diffusion primarily occurs along the axons within the brain, so mapping the directions of the flow can create surrogates for neural tracts. The Diffusion Tensor Imaging (DTI) model assumes that each voxel can be represented by a 3x3 covariance matrix (the diffusion tensor, an ellipsoid at each voxel), and the major-axis direction of the ellipsoid is the direction of the flow.

DTI reconstruction requires a minimum of six angles and often much more (30+ at least) in practice to actually estimate the tensor and direction from noisy clinical data. This paper proposes a generative model that aims to reconstruct the diffusion tensor from a very undersampled angular acquisition. It performs experiments on a retrospectively undersampled version of the HCP-YA dataset and achieves better results than some non-diffusion MRI-specific generative models (e.g., CycleGAN).

### Strengths
- The front matter of the paper is reasonably clear in describing its goals and is likely accessible to an audience that is not deeply familiar with dMRI. It is a good introduction to non-deep learning work in diffusion tensor imaging.
- It is appreciated that the model operates volumetrically, instead of slice-by-slice (although I will note that historically this slicewise processing was commonly caused by GPU memory issues when dealing with large 6D spatial+angular images. This paper downsamples the images spatially, so I do not know if their method can handle full resolution images volumetrically)

### Weaknesses
## Experimental
### Baselines
While the field has made tremendous progress in the last 15 years towards better DTI (or other models like ODFs) reconstruction using both better iterative prior-based methods and deep learning, *none* of it is benchmarked against in this paper. The paper instead presents benchmarks against nearly decade-old image-to-image generative models that are completely unrelated to the task at hand (e.g., CycleGAN, Pix2Pix). The sole DTI-related method (ADE) is never used in practice, as it simply sets off-diagonal terms to zero.

This is perplexing, as the paper does cite *some* recent methods (L099 and L125) but does not benchmark against them because they're not volumetric or skip some intermediate steps that this paper proposes. If these are indeed limitations, then they should be demonstrated experimentally.

Further, DTI is only one possible representation of the underlying diffusion signal used en route to estimating fiber tracts. Many other representations, such as fODFs and dODFs, exist, and there are methods that directly predict raw diffusion gradients. For example:
- [SparseWars](https://www.sciencedirect.com/science/article/pii/S1053811918307699) covers and benchmarks major iterative prior based methods in this field and includes code to implement many of its methods. 
- As deep learning methods, some methods regress scalar maps from undersampled DWIs ([1](https://pubmed.ncbi.nlm.nih.gov/27071165/), [2](https://pubmed.ncbi.nlm.nih.gov/30426558/)), employ CycleGAN-like methods to estimate DTI ([1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406190/)), use anatomical conditioning similarly to the proposed method to predict raw DWIs or tractograms ([1](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00259/123726), [2](https://arxiv.org/abs/2106.13188)), super-resolve undersampled DWIs ([1](https://m-lyon.github.io/publication/lyon2023spatio/lyon2023spatio.pdf)), estimate model fits using undersampled data ([1](https://arxiv.org/abs/2411.11819)), and many more. Please see [Karimi24](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00353/124918/Diffusion-MRI-with-machine-learning) for an up-to-date review of machine learning in Diffusion MRI analysis. 

While many of the above do not tackle DTI reconstruction specifically, they can all be used to estimate scalar maps and tractograms used for benchmarking in this paper and some of them should be included as baselines.

### Datasets and evaluation

- There are challenge and benchmark datasets with simulated ground truth for fiber tractography and general dMRI analysis that can be used for evaluation, including an [ISMRM challenge](https://tractometer.org/) and a [MICCAI challenge](https://www.sciencedirect.com/science/article/pii/S1053811923003828). They can be used to supplement the analysis presented in this paper.
- It is unclear why the HCP-YA dataset is resampled to 2 mm isotropic. Doing so introduces significant local mixing in the diffusion tensor, which is already a coarse representation.
- Table 2: diffusion tensor components cannot be compared using PSNR and SSIM, which are defined only for images. See the references above for better evaluation strategies based on fiber-related downstream tasks, or if comparing diffusion tensors is required, one can compute the Riemannian distance between SPD matrices to do so.

## Methodological 

- The proposed method is highly convoluted. It involves four training stages (3 pretraining and 1 finetuning) before it can make any predictions, which makes it very hard to integrate into new labs on new datasets. 
- Moreover, while the methods section spells out *what* it does, it does not attempt to motivate *why* the method is designed the way that it is and reads like a sequence of disconnected components. This unfortunately makes the methods section hard to read and follow. For example, it is never spelt out why this problem specifically needs a latent diffusion model. 
- Sec 3.2.2.: it is entirely unclear why just adding more layers would make it a valid tensor. 
- The paper claims that a joint latent space between diffusion gradients (DWIs) and DTI is needed. This is strange, as DTI is analytically derived from the DWIs; it is unclear what benefit such a joint latent space provides.

### Questions
I'm happy to be corrected on any of the above and look forward to the discussion period. 

My primary questions pertain to:
- Please elaborate on the motivation behind the selection of the current baselines and why none of the related machine learning works in diffusion MRI analysis were included in the experiments.
- Please elaborate on the justification for the methodological design.

The weaknesses above include more secondary questions as well.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents JET-Diff (Joint-Encoding Tensor Diffusion), a novel framework designed to reconstruct high-quality Diffusion Tensor Imaging (DTI) from a limited number of diffusion-weighted images (DWIs). Conventional DTI acquisition typically requires over 30 DWIs across multiple gradient directions with long scan times and motion artifacts. In contrast, JET-Diff reconstructs complete six-component diffusion tensors using only four DWI volumes. The framework relies on two main components, (i) an anatomical autoencoder which utilizes information decoupling to separate anatomical context from tensor-specific properties and (ii) a Multi-Tensor Latent Diffusion (MTLD) which captures the joint distribution of DWI and DTI latent representations through a unified diffusion process. Comprehensive evaluation on the Human Connectome Project (HCP) dataset shows that JET-Diff significantly outperforms baseline methods such as CycleGAN, Pix2Pix, ResViT, and standard latent diffusion models.

### Strengths
1. Relevant problem and clear problem setting: the paper addresses a critical problem of high acquisition time of scans. The joint modeling of DWI-DTI distribution is well defined and technically sound.

2. Novel framework: the paper presents several contributions with an information decoupling to separate tensor information and anatomical context. Moreover, a lightweight MLP block is used for the joint refinement phase and proved efficient. The use of multi-axis cross-attention is also interesting as it reduces computational burden of treating 3D medical images and reinforces spatial understanding.

3. Thorough validation: the paper uses various metrics to evaluate performance (NMSE, PSNR, SSIM) demonstrating the versatility and robustness of the current method. Comparison with 3D baselines is provided on a large-scale dataset of 973 HCP subjects.

4. Extended ablation: the paper validates the relevance of the main components: DWI-aided decoder, joint refinement, and unconditional pre-training.

### Weaknesses
1. Unclear Methodology: Some implementation details are missing regarding the positive semi-definiteness. The dimensions and activations used for the Joint MLP block are not specified. 

2. Pre-processing: resampling to 2mm isotropic is performed but no analysis is shown of the effect of resolution on performance. There is no sufficient information to understand how the gradient vectors in the DWI volumes were chosen.

3. Baseline Comparison: the paper introduces SuperDTI or FlexDTI as baselines but they are not shown in the result section. The conditioning mechanism employed in the vanilla LDM is not clear and may not be comparable to other baselines.

4. Statistical rigour: the paper is lacking signifance testing for the different methods. Moreover, confidence intervals and standard deviations are not shown.

5. Generalization: there are concerns that some empirical choices might prevent generalisation. Typically, multiple b-values are used in real-world protocols but only b=1000 s/mm² is used in these experiments. The model uses the HCP dataset of healthy subjects suggesting the method may fail on data with pathology or data from external sources.

### Questions
1. There is no runtime comparisons at inference. How is the "lightweight" joint refinement module quantifiable? What are the memory requirements ? 

2. How does performance scale with the number of inputs ? 

3. The model is trained on b=1000 s/mm². Did you study the generalization to other b-values? 

4. The input resolution may have an impact on performance. What is the minimum and maximum resolution required?

### Soundness
3

### Presentation
3

### Contribution
3
