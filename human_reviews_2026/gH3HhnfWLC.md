# Block Recurrent Dynamics in Vision Transformers

- Avg Score: 6.80
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 8, 4

## Abstract
As Vision Transformers (ViTs) become standard backbones across vision, a mechanistic account of their computational phenomenology is now essential. Despite architectural cues that hint at dynamical structure, there is no settled framework that interprets Transformer depth as a well-characterized flow. In this work, we introduce the $\textbf{Block-Recurrent Hypothesis (BRH)}$, arguing that trained ViTs admit a block-recurrent depth structure such that the computation of the original $L$ blocks can be accurately rewritten using only $k \ll L$ distinct blocks applied recurrently. Across diverse ViTs, between-layer representational similarity matrices suggest few contiguous phases. To determine whether this reflects reusable computation, we operationalize our hypothesis in the form of block recurrent surrogates of pretrained ViTs, which we call Recurrent Approximations to Phase-structured TransfORmers ($\texttt{Raptor}$).  Using small-scale ViTs, we demonstrate that phase-structure metrics correlate with our ability to accurately fit $\texttt{Raptor}$ and identify the role of stochastic depth in promoting the recurrent block structure. We then provide an empirical existence proof for BRH in foundation models by showing that we can train a $\texttt{Raptor}$ model to recover $94$\% of DINOv2 ImageNet-1k linear probe accuracy in only 2 blocks. To provide a mechanistic account of these observations, we leverage our hypothesis to develop a program of $\textbf{Dynamical Interpretability}$. We find $\textit{\textbf{(i)}}$ directional convergence into class-dependent angular basins with self-correcting trajectories under small perturbations $\textit{\textbf{(ii)}}$ token-specific dynamics, where $\texttt{cls}$ executes sharp late reorientations while $\texttt{patch}$ tokens exhibit strong late-stage coherence reminiscent of a mean-field effect and converge rapidly toward their mean direction and $\textit{\textbf{(iii)}}$ a collapse of the update field to low rank in late depth, consistent with convergence to low-dimensional attractors. Altogether, we find that a compact recurrent program emerges along the depth of ViTs, pointing to a low-complexity normative solution that enables these models to be studied through principled dynamical systems analysis.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper interprets vision transformers under a Block-Recurrent Hypothesis, investigating if the original computation that such models perform using L blocks can be achieved using k << L recurrent blocks while reproducing the internal activations of the full model. The paper then views these new models under a program of dynamical systems analysis.

### Strengths
I appreciate the very thorough account of the motivation, construction, and analysis of the Block-Recurrent Hypothesis. The mathematical definitions and propositions look sound, and the experiments around the identification of functional reuse and dynamical interpretability are well characterized. Understanding the dynamical structures within vision transformers that have so ubiquitously entered as de-facto models for visual tasks is interesting and necessary, and the paper does a good job at teasing some of that out. Specifically, the paper systematically explores the emergence of blocks using a controlled setting with a tiny vision transformer, and then scales the analysis to a larger DINOv2 ViT-base model. Additionally, it performs a series of experiments to understand dynamical properties of ViT tokens, offering a unique perspective on the appropriateness of directional (angular) geometry as a paradigm for dynamical analysis.

### Weaknesses
For table 1 and figure 5, it seems that the performance experiments were not repeated multiple times with different partitions or different random model seeds. I would have preferred to see some form of error bars quantifying how robust or varied the metric results are, and what to take from that. Also, in line 317, it says that accuracy saturates at k=4 without evidence to support that (what happens at k=5?).

In section 3, the manuscript states that the Raptor program will be tested on "Foundation models". What qualifies as a foundation model is highly debated between researchers, but regardless of the fact, simply testing on a ViT-base does not provide a convincing argument for the claim. For example, what happens when you scale to, say, a ViT-Large? Can Raptor be successfully applied to other objective functions outside of DINOv2, say MoCo-v3?

More generally, the authors show block dynamics to emerge across diverse ViTs. However, the ViTs that are tested seem to be trained on either a supervised loss or a contrastive loss. Do block dynamics emerge in a ViT that is trained on a local reconstruction loss, such as an MAE? How generalizable is the BRH?

### Questions
Most of the representational structure that is explored in this paper involves looking at the level of the output of transformer blocks. I would be curious to know how interactions within the attention matrices change across the model hierarchy, or whether you similarly observe functional reuse at the output of the attention layer in each block. Transformers can be thought of as possessing two computational pathways within each block, a fast one that is the result of the residual connection, and a slow one that is the result of a self-attention module. How do these two separate pathways interact with the Block-Recurrent Hypothesis, if at all? This is not a limitation of this current manuscript, but I would love to hear thoughts from the authors if they have any.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors hypothesise (via their so-called Block-Recurrent Hypothesis) that ViTs operate in blocks, and that the computation of individual blocks may be compressible while allowing the same outputs.

They use the BRH's validation to train Raptor, which has weight-tied layers, almost acting like an RNN in the direction of transformers.

Finally, the authors offered a novel approach of 'dynamical interpretability' to analyse Raptor as a recurrent systems in the depth dimension

### Strengths
Really well motivated hypothesis, analysis and subsequent model design. Analysis was simple and straightforward - starting with representational similarity analysis then validating the connection to computational similarity by constructing Raptor. In turn, the construction/training objectives of Raptor were simple and easy to follow. The paper was a joy to read over all.

### Weaknesses
Introduction to different embeddings/tokens for those with a background in recurrent systems rather than ViTs would be very helpful.

### Questions
How many 'timesteps' was each recurrent block in Raptor run for? Table 3 provides layer splits, but could you clarify this more explicitly in the main text? For instance, does Block 1 in the k=3 configuration iterate 7 times to predict layers 1-7?

"Unlike classical distillation, which typically supervises logits (and occasionally a few intermediate “hints”), we enforce one-to-one alignment of all layers representations across the entire depth for the same inputs." - were distillation methods of this form compared against? Are there baseline compression methods that Raptor can be compared against?

The ε in the definition of BRH was never explicitly measured IIUC - does this vary across datasets/architectures?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the Block-Recurrent Hypothesis (BRH) in Vision Transformers (ViTs), which posits that the deep stacks of layers in a trained ViT can be decomposed into a small number of computational phases that repeatedly perform similar transformations. To operationalize this idea, the authors propose Raptor (Recurrent Approximations to Phase-structured Transformers), a framework that reconstructs a full ViT using only a few weight-tied recurrent blocks.

Instead of directly proving recurrence analytically, Raptor demonstrates the existence of block recurrence by example— training recurrent modules to reproduce not only the final outputs but the entire layer-wise activation trajectories of the teacher ViT. The resulting models achieve remarkable fidelity: with only k = 3 recurrent blocks, Raptor can recover ≈97% of DINOv2 ViT-B’s ImageNet-1k linear probe accuracy and a layer-wise R² of 0.73, indicating strong internal consistency between the reconstructed and original representations.

The work thus offers a constructive, empirical validation of the BRH, showing that the depth of ViTs can be “compressed” into a small number of iterative computational phases without major loss of functionality. It further opens the door to analyzing ViTs as dynamical systems, revealing phase-wise convergence behavior, token-specific dynamics, and low-rank collective motion in the representational flow across depth.

### Strengths
* Rather than relying on correlation or representational similarity alone, the authors operationalize the block-recurrence hypothesis by functionally reconstructing the ViT’s internal computations.

* Raptor replicates the performance and hidden representations of large ViTs (e.g., DINOv2 ViT-B) with only a few recurrent modules, across classification, segmentation, and depth estimation tasks.

* The two-stage Teacher Forcing + Autoregressive procedure, with gradual annealing and end-to-end refinement, shows careful engineering and clear causal analysis (e.g., TF-only collapse).

* The dynamical characterization—phase-wise convergence, angular attractors, weak contraction, and token-specific stability—offers valuable interpretability for transformer dynamics.

* The work suggests a compact, iterative view of transformer computation, resonating with ideas in dynamical systems, implicit networks, and recurrent neural architectures.

### Weaknesses
* The experiments mainly focus on image classification and linear probing; while segmentation and depth tasks are included, the demonstrations remain limited compared to the full spectrum of real-world applications.

* The use of max-cut–based phase segmentation is heuristic; the stability or uniqueness of the phase decomposition is not fully explored.

* The need for depth scaling indicates that the effective dynamics of ViT layers are non-stationary, so full recurrence may not hold uniformly across depth.

* While the recurrence hypothesis is supported empirically, it is not yet clear whether such recurrence can be directly exploited for efficiency or deployment in large-scale multimodal or generative models.

### Questions
The analysis and empirical validation of the block-recurrent hypothesis are quite compelling. However, has the team explored whether the recurrent compression (k ≪ L) can be leveraged for deployment efficiency—for instance, reducing parameters and FLOPs in large foundation models (e.g., MLLMs or multimodal ViTs) without sacrificing performance?

In principle, if a ViT can be represented with a few recurrent blocks, one might expect significant efficiency gains in inference-time or memory-limited environments. Has any investigation been conducted to measure these potential deployment benefits in practice?

Beyond classification, could Raptor-style recurrence be applied to dense or structured vision tasks (e.g., detection, captioning, video understanding)? It would be interesting to see whether the same “computational phase” structure holds when the model needs spatially adaptive or temporally extended reasoning.

Given the recent integration of ViT backbones into multimodal large language models, do the authors see any prospects for shared recurrent computation across modalities, or for cross-modal recurrence alignment?

Finally, could the dynamical interpretation of ViTs—especially the phase-wise attractor structure—help in regularizing or stabilizing the training of much deeper transformer-based architectures?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the Block-Recurrent Hypothesis (BRH) for Vision Transformers (ViT): a pretrained ViT’s depth can be organized into contiguous “phases” such that the full computation can be reproduced by reusing a small set of shared blocks applied recurrently. The paper then instantiates the BRH by proposing Raptor, a weight-tied surrogate that uses max-cut and matches the activation of pretrained ViTs. Experiments on DINOv2 show that Raptor recovers most of the linear-probing accuracy of the original model with only a fraction of layers. Building on BRH, they further propose a dynamical interpretability view of depth, reporting directional (angular) attractors, token-specific dynamics, and low-rank update collapse in late layers.

### Strengths
1. From a hypothesis to an actionable algorithm along with interpretability insights, the paper is very insightful and can be very valuable to any follow-up research in ViTs.
2. Extensive experiments, both small and large scale, are presented to test the proposed hypothesis.
3. The paper is clearly written and well presented.
4. The stochastic depth section is very insightful and interesting.

### Weaknesses
1. The large-scale experiments only used DINOv2. While it is a very widely-used backbone, experiments on other models such as SigLIP could greatly strengthen the work and show its broader applicability.
2. ViTs are now extensively used in the context with vision-language models (VLMs) now. When tuned alongside the LLM, it is natural to think whether the proposed hypothesis still holds.
3. Compute tradeoffs are not fully quantified. While FLOPs per iteration is considered, wall-clock, memory behavior, and parallelism effects of recurrent blocks vs conventional ViTs aren’t thoroughly analyzed.

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a fresh perspective on vision transformers, showing that their computational flow can be explained through a block-recurrent computation structure. To support their claims, the authors present several visualizations and training experiments. After establishing the connection between vision models and block-recurrent structure, the authors leverage this perspective to propose a new interpretability approach inspired by principles from dynamical systems, which can be used to better understand vision models.

### Strengths
- The paper presents a fresh view on the computational flow of vision transformers, providing a better understanding of these important models. 

- As far as I know, the idea is novel and creative. 

- The authors include extensive analyses, including ablation studies and interpretability experiments.

### Weaknesses
**W.1. Limited empirical analysis for the main claim:**

I believe the main idea in the paper (the recurrent block view) should be justified through additional experiments. In its current form, the argument relies heavily on the power of distillation, which is a very strong model compression technique. To verify that this is not the primary cause of the success of Raptor-like models, I suggest the authors conduct the following two experiments:

**(i) Block structure variation:** Train Raptor-like models with a similar number of blocks and layers per block, but where each block includes a non-contiguous (fragmented) set of layers (for example, instead of [1,2,3], [4,5], [6,7], use [1,4,6], [2,3,7], [5]). Please note that previous work [a] which use smart grouping found that sharing parameters between non-adjacent layers can be very effective, somewhat contradicting the results reported in this paper.

**(ii) Causal intervention analysis:** At test time (without fine-tuning or retraining), replace the weights of K layers with (a) weights from different layers within the same block, and (b) weights from random layers elsewhere in the model (not from the same block). If, for several K values, the accuracy under option (a) is consistently higher, this would provide much stronger support for the authors’ claim.


**W.2. Strengthen the connection to dynamical systems:**

From my perspective, the current motivation is somewhat vague. For example, a devil’s advocate might ask what differentiates the proposed approach from the well-known layer-by-layer analysis. I suggest that the authors include a dedicated section to motivate their decision to use dynamical systems. Incorporating technical terms and principles from the field of dynamical systems could make the argument more compelling and insightful.

**W.3. Minor:** 

There are several visualization issues. For example, the colors in Fig. 6 (right) are unclear, with three similar shades of green. Additionally, the excessive use of vspace or other compression commands makes the text too dense.

[a] DYNAMIC LAYER TYING FOR PARAMETER-EFFICIENT TRANSFORMERS. David et al. ICLR 24

### Questions
Q.1. Do the authors think that their insights can shed more light on the connections between vision diffusion models and Raptor-like vision transformers, given that they used significantly fewer parameters and recurrent computation? Such insights could provide valuable information on the optimal trade-offs between FLOPs and parameters in vision models.

Q.2. What is the compute gap required to train Raptor models compared to the baseline? Please provide relative numbers to help illustrate how challenging it is to train such models.

### Soundness
2

### Presentation
2

### Contribution
3
