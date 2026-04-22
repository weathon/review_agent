# No Alignment Needed for Generation: Learning Linearly Separable Representations in Diffusion Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Efficient training strategies for large-scale diffusion models have recently emphasized the importance of improving discriminative feature representations in these models. A central line of work in this direction is representation alignment with features obtained from powerful external encoders, which improves the representation quality as assessed through *linear probing*. Alignment-based approaches show promise but depend on large pretrained encoders, which are computationally expensive to obtain. In this work, we propose an alternative regularization for training, based on promoting the **L**inear **SEP**arability (LSEP) of intermediate layer representations. LSEP eliminates the need for an auxiliary encoder and representation alignment, while incorporating linear probing directly into the network’s learning dynamics rather than treating it as a simple post-hoc evaluation tool. Our results demonstrate substantial improvements in both training efficiency and generation quality on flow-based transformer architectures such as SiTs, achieving an FID of 1.44 on $256 \times 256$ ImageNet and FID of 1.66 on $512 \times 512$ ImageNet dataset.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Linear Separability (LSEP), a regularization strategy for diffusion transformers which explicitly encourages linearly separable representations without external alignment. The high-level idea is to jointly train a linear probe with the denoising objective thereby enhancing feature separability in the early layer. The authors introduce three design components (1) classification specific conditioning, (2) random cropping for patch-level separability, and (3) time-dependent loss weighting. Experiments on SiT models show that LSEP improves both training speed and generation quality achieving comparable results to alignment-based methods.

### Strengths
- The proposed training framework is simple yet efficient, achieving both faster convergence and improved generation quality without relying on external modules.

- The paper is well-written and easy to follow. The motivation and ideas are clearly articulated, figures and visualizations make the concepts easy to understand.

- The ablation studies on each design components are well-designed and detailed which effectively validates each design choice.

### Weaknesses
- The study mainly focus on linear probing as the main indicator of representational quality. It is unclear whether this approach generalizes beyond class-labeled settings (e.g., text-to-image or video generation). It would be helpful to discuss how linear separability could be defined or leveraged in such contexts.

- Although linear separability is a useful cue, good linear separation does not necessarily imply better representation. The setup therefore feels somewhat constrained and overly tied to the classification context. When combined with REPA (Table 2 and Figure 7), it achieves much higher performance and clearer separability, showing that linear probing alone does not fully explain the improvements in generation quality.

- The rationale for selecting the weighting parameter $\omega_{\text{class}}$ is not well explained. Its value (e.g., 0.03) appears empirically chosen; an ablation study or sensitivity analysis would strengthen this aspect.  

- Higher resolution (e.g., 512x512) experiments would help demonstrate scalability.

### Questions
- The linear probe module is already conditioned on timestep embeddings. Could the authors elaborate on why additional time-dependent weight is still necessary? While Table 1 shows empirical improvements, a deeper explanation of its role would be helpful.  

- The authors mention that LSEP continues to improve and approaches REPA. Does this trend persist with longer training (e.g., beyond 4M steps)?

-  All experiments are conducted only on SiTs. Is this also applicable to DiTs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a regularization method, Linear SEParability (LSEP), to improve the training efficiency and generation quality of diffusion models, without relying on external pre-trained encoders.

The idea is to incorporate a linear classifier, inspired by linear probing in post-hoc evaluation, directly into diffusion training as an auxiliary objective, encouraging the model to learn more linearly separable intermediate representations. The approach is straightforward but also includes several specific training strategies and hyper-parameter choices.

Experiments on the ImageNet 256x256 dataset show that LSEP accelerates convergence and improves final performance across various model sizes of SiT using the original SD-VAE. It also demonstrates a synergistic effect when combined with existing alignment-based methods like REPA.

### Strengths
- The paper's primary strength lies in its novel and straightforward idea: a linear classifier can directly be an effective training regularizer. This provides a new, simple alternative to the dominant paradigm of external representation alignment.
- Although the idea of supervised training is simple, several tricks like patch-level random cropping are clever and effective.
- The empirical results are convincing. The method demonstrates consistent improvements over baselines. The finding that LSEP and REPA are complementary is also valuable.

### Weaknesses
- Training cost: The paper claims improvements in training efficiency, demonstrated by training iterations vs. FID scores.
  - However, the proposed method appears to require two separate forward passes through the first few layers, as the two branches require different class-conditioning (Figure 1, Section 3.2).
  - This could introduce significant computational overhead, potentially doubling the cost for a portion of the model, and this overhead is not discussed. A wall-clock time (or compute) vs. FID plot would be helpful.
- On linear separability: The paper is motivated by a claim that enhancing linear separability is the mechanism driving the improvements in generation. However, evidence for this link is sparse.
  - The main results in Table 1 do not report the linear probing accuracy for each configuration, making it impossible to directly correlate the degree of separability with the final FID. If we plot the (acc, FID) points for all 21 configurations in Table 1, will it demonstrate a clear, strong correlation?
  - In Table 2, it is also unclear how REPA and LSEP independently (and jointly) improve linear separability.
  - While Figure 4(a) shows an increase in accuracy, the absolute value remains relatively low (below 60%, much weaker than supervised learning), which warrants further discussion.
- Generalizability to other settings: The method, in its current form, explicitly relies on class labels for the auxiliary loss. This dependency limits its applicability to other tasks such as text-to-image or unconditional generation.
  - It would strengthen the paper to demonstrate how LSEP could be adapted for such scenarios, for instance, by training on class or attribute labels as a regularizer while still performing text-conditional or unconditional generation for the primary task.
  - Can the input conditions and labels for auxiliary classification be decoupled?
- Loss weighting design: Eq. (7) appears somewhat hand-crafted and might be sensitive to changes in the training-time timestep sampling (e.g., LogNorm distribution in SD3 and VA-VAE).
  - The ablation study in Table 1 shows that this parameter is carefully tailored to a very narrow range of values ([0.0275, 0.0325]).
  - The paper also does not show how the general magnitude of this value (around 0.03) is determined, as the cross entropy loss itself is already low enough (<0.1, according to Figure 6).

### Questions
Please refer to the weaknesses regarding the correlation between linear separability and performance, and the generalizability to other conditioning signals.

The current manuscript also reads like a strong empirical study with good performance but leaves many questions unanswered. For example, will other well-known techniques in supervised learning (e.g., label smoothing) be helpful? Will other discriminative tasks (other than a simple classification) also be beneficial? Why does a classifier trained on diffusion backbone lag behind supervised ViTs?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to accelerate the training of diffusion transformers without additional computational costs by using a linear classifier. Specifically, instead of aligning the intermediate features of the diffusion transformers with a pretrained vision encoder, it uses intermediate features to be linearly separable to classify the category of the given image. The authors demonstrate that the proposed method (LSEP) shows competitive results with REPA.

### Strengths
1. The idea is simple yet effective: It can even be used with REPA to further accelerate the training of the diffusion transformer.

2. Extensive analysis of the design choices facilitates the practitioners to use this approach.

### Weaknesses
1. LEAP is limited only to the dataset that has human annotations (i.e., labels). How can we use this method to accelerate training diffusion models that do not have ground-truth labels, e.g., T2I generation (MS-COCO)?

2. Even though REPA uses an external encoder as conditioning, the reported REPA results do not use any external labels (i.e., unsupervised), but LSEP needs labels (i.e., supervised). In some sense, LSEP uses more information to generate images, but shows worse performance and slower training than REPA (in terms of training time).

### Questions
1. Please answer the Weaknesses.

2. Table 2 shows SiT + REPA + LSEP. However, in Table 3, the authors report only SiT + LSEP. Can SiT + REPA + LSEP achieve SOTA performance?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- The paper proposes LSEP, a simple regularization that inserts a trainable linear probe at an intermediate layer of a diffusion transformer and co-optimizes it with the standard denoising loss, with the goal of directly promoting linear separability of intermediate features during training rather than measuring it post-hoc. 
- On ImageNet 256×256 with SiT backbones, LSEP reduces FID versus the SiT baseline, improves linear-probe accuracy at early layers, and shows complementarity with alignment methods like REPA
- The paper positions LSEP as a no-alignment alternative that can also be combined with alignment (e.g., REPA) to further speed training and improve generation quality, and it discusses limitations such as the lack of experiments on text-to-image, video, or higher resolutions.

### Strengths
- The paper tackles a concrete and important problem of improving the quality of features learnt by generative models. 
- The intervention is architecturally minimal and easy to adopt, since it adds a single linear head and combines it with the standard denoising loss via a simple weight.
- The experiments show consistent FID gains over baseline SiT and demonstrate complementarity with REPA, with clear tables and figures that place the numbers in context and avoid relying solely on a single configuration. 
- The writing is generally clear about the two-branch setup and provides a useful hyperparameter grid for reproduction.

### Weaknesses
- The goal of pretraining is to learn general-purpose, task-agnostic features that transfer across datasets, label spaces, and objectives (e.g., detection, segmentation, retrieval, open-set recognition), but this method explicitly supervises a linear probe on ImageNet class labels during pretraining and pressures intermediate features to be separable with respect to that specific taxonomy. As a result, the learned geometry is optimized for class-conditional ImageNet generation and linearly decoding those same labels, which does not guarantee that the features capture fine-grained structure needed by unrelated downstream tasks and may bias representations toward dataset-specific shortcuts.
- The probe and denoiser use different class-conditioning schedules, which creates gradient-level tension; the paper notes that $ρ_L=1$ causes a mismatch but does not analyze interference between the two schedules or test mitigations such as stop-gradient into shared embeddings or separate class embeddings.
- The paper trains its linear probe on pooled features (and sometimes pooled crops), which can make the average representation linearly separable while leaving individual patch tokens messy. Without token-level evaluations, the claim that LSEP improves patch-space geometry remains unclear and could reflect pooling shortcuts 

- The paper sets the probe’s unconditioning probability very high but below one, yet it does not test whether the method remains stable under label noise or whether those occasional non-null conditioning steps make the probe exploit conditioning artifacts rather than visual content. It will be interesting to see if the reported gains survive mislabeled classes and still hold when the class-conditioning pathway is perturbed or ablated.

### Questions
See weaknesses above

### Soundness
3

### Presentation
3

### Contribution
3
