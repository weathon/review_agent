# Patronus: Interpretable Diffusion Models with Prototypes

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Uncovering the opacity of diffusion-based generative models is urgently needed, as their applications continue to expand while their underlying procedures largely remain a black box. With a critical question -- how can the diffusion generation process be interpreted and understood? -- we proposed *Patronus*, an interpretable diffusion model that incorporates a prototypical network to encode semantics in visual patches, revealing *what* visual patterns are learned and *where* and *when* they emerge throughout denoising. This interpretability of Patronus provides deeper insights into the generative mechanism, enabling the detection of shortcut learning via unwanted correlations and the tracing of semantic emergence across timesteps. We evaluate *Patronus* on four natural image datasets and one medical imaging dataset, demonstrating both faithful interpretability and strong generative performance. With this work, we open new avenues for understanding and steering diffusion models through prototype-based interpretability. Our code is available at [nina-weng.github.io/patronus.github.io](https://nina-weng.github.io/patronus.github.io/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Patronus, an interpretable diffusion model that integrates a prototypical network to encode semantic concepts within visual patches. The model learns prototypes that represent localized semantic patterns and uses their activation vectors to guide the diffusion process. Unlike post-hoc interpretability methods, Patronus embeds interpretability directly into the architecture, allowing visualization of what prototypes represent, where and when they emerge during generation, and how they interact with semantic attributes. Experiments on multiple datasets show that Patronus achieves competitive or superior generation quality and meaningful prototype disentanglement.

### Strengths
1. The paper introduces an architectural approach to intrinsic interpretability in diffusion models through prototype learning, differing from previous post-hoc analyses.

2. The integration between prototype activations and diffusion conditioning is mathematically consistent, with theoretical reasoning showing that adding the condition does not degrade likelihood.

3. Evaluation on many datasets (including a medical imaging dataset) demonstrates both semantic interpretability and competitive FID performance.

### Weaknesses
1. Some theoretical explanations are verbose and could be streamlined; a visual overview of the training pipeline would improve accessibility.

2. Joint optimization of prototype encoder, conditional DDPM, and latent diffusion may pose scalability issues for higher-resolution or text-conditioned tasks.

3. The paper acknowledges that global attributes (e.g., age or gender) are not well captured due to the patch-based encoder, but offers no structural remedy. However, I think this is also acceptable for a short-term work.

### Questions
1. How stable is prototype learning when scaling the model size or dataset diversity? Are prototypes consistent across training runs?

2. Can Patronus be extended to text-to-image diffusion models like Stable Diffusion, where semantics are conditioned on language rather than visual features?

3. How sensitive are prototype visualizations to noise in the prototype activation vector? Do small perturbations yield consistent semantic control?

Actually I am not an expert on diffusion theory research but a researcher on diffusion modal application, so I will check the review from other reviewers and the responses from authors to adjust my final rating.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces "Patronus," a new type of diffusion model that tries to open up the "black box" of how these models generate images. The core idea is to build a "prototypical network" directly into the U-Net. This network learns a set of visual patterns (prototypes) from the data. As the model denoises an image, it maps its internal features to these prototypes, letting us see what patterns are being used, where they're being placed, and when (at which timestep) they appear. The authors show this can be used to understand the model's logic, catch it "cheating" by learning bad correlations (shortcut learning), and track how concepts form from noise.

### Strengths
Overall, the work is interesting. 

1.	The core idea is highly original. We’ve seen prototypes used for interpretability in classification, but applying this idea to understand the internal dynamics of a diffusion model as it generates is a very creative and insightful leap.

2.	The ability to detect "shortcut learning" is a particularly high-impact claim, especially for the medical imaging applications they explore, where you absolutely cannot have a model focusing on the wrong artifacts.

### Weaknesses
I have a few key reservations that kept me from being completely convinced.

1.	My main concern is about the classic "interpretability vs. performance" trade-off. The authors claim "strong generative performance," but adding a whole prototype network inside the U-Net can't be computationally free. I really need to see a clear analysis of the cost. What’s the hit to the FID score compared to a standard baseline with the same parameter count? What's the extra latency during inference? This needs to be quantified.

2.	I'm also worried about scalability. The experiments mentioned seem to be on smaller-scale datasets (like CIFAR or low-res medical images). I'm skeptical that this method, as-is, would scale to high-resolution (e.g., 256x256 or 512x512) and high-complexity (e.g., ImageNet) datasets. The computational cost of computing patch-prototype similarity at every single step for large batches and high-res feature maps seems like it would be a major bottleneck.

3.	The entire utility of this method hinges on the quality of the learned prototypes. The paper needs to do more to convince me that these prototypes are (a) genuinely meaningful to a human, (b) diverse and not just "collapsed" to a few simple textures (even though the authors directly use a loss function to make it diverse), and (c) actually represent what the model is really thinking. Qualitative examples can be cherry-picked, so I’m looking for a more quantitative analysis of prototype quality.

4.	This method is only tested on unconditional diffusion models (here we do not treat the prototype activation vector as ‘real’ conditions). Could this method be applied to text/image conditioned diffusion models as well?

### Questions
To help clarify these points, I have a few questions for the authors:

1.	**The Performance Trade-off**: Could you provide a direct, head-to-head comparison of generative quality (FID, etc.) and computational cost (GFLOPs, wall-clock time) between Patronus and a standard diffusion baseline of similar size?

2.	**The "Shortcut Learning" Claim**: This is a big selling point. Could you walk me through more concrete examples from your experiments (maybe from the medical dataset) where Patronus clearly identified an unwanted correlation that a baseline model was exploiting?

3.	**Scaling Up**: What are the real bottlenecks in scaling Patronus to something like ImageNet? Have you thought about or explored any approximations (e.g., to the prototype matching step) to make it more efficient at high resolutions?

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
3

### Summary
This paper introduces Patronus, a diffusion model that integrates a prototypical network within the denoising architecture to provide intrinsic interpretability.
Instead of relying on post-hoc feature visualization or external semantic encoders, Patronus learns patch-based prototypes that encode localized semantic patterns (''what"), their spatial location ("where"), and temporal emergence during denoising ("when").
Prototype activation vectors condition the diffusion process, allowing direct visualization, manipulation, and bias diagnosis (i.e., shortcut learning).
Empirical results across five datasets:
- Achieves competitive or superior FID and latent quality (AUROC/TAD)
- Produces interpretable prototype semantics
- Enables diagnosis of unwanted correlations (e.g., hair color - smile);
- Reveals temporal emergence of visual features.

### Strengths
- Combines a prototypical encoder and a conditional diffusion process, embedding interpretability directly rather than via post-hoc probing.
- "What, where, when" decomposition is compelling and concretely supported by prototype visualizations and activation dynamics.
- Prototypes correspond to semantically meaningful attributes (e.g., smile, hair color, collar) and can be manipulated for controllable image editing.
- Demonstration of shortcut learning (hair color ↔ smile) is convincing and highlights societal relevance.
- Ablations explore prototype distinctness, disentanglement, and conditional generation.
- Figures and examples are intuitive and make the technical ideas clear.

### Weaknesses
- The prototype extraction is largely adapted from ProtoPNet, with limited theoretical or algorithmic innovation in how prototypes are integrated beyond being used as conditional guidance.
- The interpretability claims rely mostly on qualitative visualization; no quantitative evaluation (e.g., localization accuracy, faithfulness, or human interpretability studies) is provided.
- While visualizations are appealing, there is no test showing that manipulating a prototype truly corresponds to causal semantic change rather than correlated artifacts.
- It’s unclear how many prototypes are needed, whether semantics are stable across seeds, or how sensitive the results are to prototype dimensionality.
- Comparison is limited to DiffAE and InfoDiff. More modern interpretable or controllable diffusion baselines (e.g., DDPM inversion [1]) could be considered.
- The paper could better position itself relative to prior encoder-conditioned approaches. What specifically differentiates the prototype activations from lower-dimensional semantic vectors used in DiffAE-like methods.
- Theoretical claims in Section 3.5 (showing conditioning "cannot degrade" the ELBO) are informal and would benefit from a more rigorous derivation or empirical validation.

[1] Huberman-Spiegelglas et al., CVPR’24.

### Questions
1. How stable are the prototypes across random seeds? Do the same semantic concepts consistently emerge?
2. Can the authors quantify interpretability (e.g., localization or faithfulness metrics)?
3. How does Patronus compare to direction-based interpretability methods (e.g., PCA directions, linear semantic editing)?
4. Beyond hair - smile, does it detect other spurious links (e.g., makeup - gender cues)?
5. How do performance and disentanglement vary with number of prototypes and prototype dim?
6. Could the prototype activation vector be used for cross-domain transfer or editing (e.g., applying "smile" from one dataset to another)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Patronus, a diffusion model that incorporates prototypes to provide more interpretable generation. The model learns patch-level prototypes and uses prototype activation vectors to condition the diffusion process. The authors claim that this enables understanding of what semantic patterns emerge, and where and when they appear during denoising. The paper also proposes a sampling-based prototype visualization strategy. Experiments on several datasets are presented, focusing on semantics, reconstruction, guidance, and bias analysis.
The goal of embedding interpretability into diffusion models is timely and important. The direction of prototype-based interpretability for diffusion is interesting.

### Strengths
* Tackles an important problem: interpretability in diffusion models.
* Clear motivation for moving toward intrinsic interpretability rather than post-hoc analysis.
* Prototype idea has conceptual appeal and builds on a known line of interpretable ML research.
* Good results in certain controlled settings with simple datasets.

### Weaknesses
* **Clarity issues:** The paper is written in a cryptic way and often hides a simple idea behind heavy verbal descriptions. The prototype module is not sufficiently described at the engineering and procedural level. For example, it remains unclear whether the prototype encoder is a standard CNN, how it is trained relative to the denoiser, and whether prototypes are learned or chosen a priori. Section 3.1 in particular does not provide essential clarity on the prototype encoder architecture and training scheme.
* **Name inconsistency:** This is actually a minor comment, but the acronym “Prototype-Assisted TRANsparent diffuSion model” does not actually map to “Patronus”.
* **Proposition lacks rigor:** The stated proposition is informal and the proof is neither formal nor convincing. There is no precise mathematical statement or measurable criterion. The argument reads as a loose justification rather than a proof. It should be rewritten to be mathematically precise or removed.
* **Reverse DDIM reference is vague:** The paper refers to a “reverse DDIM process” (line 313) but never clearly defines how it is implemented.
* **Weak empirical depth:** Experiments rely mainly on small and simple datasets (FMNIST, CIFAR-10, CelebA subsets). This limits confidence in claims about semantic interpretability and scalability. Higher-resolution and more complex datasets (ImageNet, large natural scenes) would significantly strengthen the case.
* **Interpretability not fully grounded:** The prototype activations are visually inspected and interpreted manually. There is no systematic evaluation of interpretability quality, faithfulness, or prototype consistency beyond selected qualitative examples.
* **Simple baseline missing:** A naive “mask-based or patch-embedding supervision” or a direct attention-map visualization baseline would help demonstrate that prototypes add more than a different form of patch scoring.

### Questions
1. What exactly is the architecture of the prototype encoder? Is it a ResNet? How many layers? Is it trained jointly or frozen after pretraining?
2. Are prototypes randomly initialized and learned, or do you initialize from real patches?
3. How is the reverse DDIM performed in practice? Please provide pseudocode or a precise description.
4. Try to be more clear, direct, and less cryptic in the explanation.

### Soundness
3

### Presentation
2

### Contribution
3
