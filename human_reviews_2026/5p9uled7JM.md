# Flow Autoencoders are Effective Protein Tokenizers

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 10

## Abstract
Protein structure tokenizers enable the creation of multimodal models of protein structure, sequence, and function. Current approaches to protein structure tokenization rely on bespoke components that are invariant to spatial symmetries, but that are challenging to optimize and scale. We present Kanzi, a flow-based tokenizer for tokenization and generation of protein structures. Kanzi consists of a diffusion autoencoder trained with a flow matching loss. We show that this approach simplifies several aspects of protein structure tokenizers: frame-based representations can be replaced with global coordinates, complex losses are replaced with a single flow matching loss, and SE(3)-invariant attention operations can be replaced with standard attention. We find that these changes stabilize the training of parameter-efficient models that outperform existing tokenizers on reconstruction metrics at a fraction of the model size and training cost. An autoregressive model trained with Kanzi outperforms similar generative models that operate over tokens, although it does not yet match the performance of state-of-the-art continuous diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
They build a discrete VQVAE based tokenizer for protein structures, where the encoder maps 3D coordinates to discrete codes and the decoder is a DiT-based flow matching model. They additionally train an autoregressive prior over the latent codes for generation. The approach, Kanzi, simplifies aspects of prior approaches, including the use of a diffusion loss (as opposed to prior SVD-based losses), replacing frame representations with global coordinates, and using standard attention instead of SE(3) invariant architectures. Kanzi outperforms prior token-based structure generative models and is more parameter efficient.

### Strengths
- The authors perform extensive benchmarking against prior approaches on both reconstruction and generative tasks. The results demonstrate that the model is competitive against the prior SOTAs on both tasks.
- They perform a series of ablations on encoder variants (invariant vs not invariant), attention window sizes, and model size.
- Simplifying the auto-encoder loss is pretty significant, as it significantly reduces the cost of training from O(L^3) or O(L^2) to O(L).
- Using a diffusion decoder also allows you to use classifier-free guidance and the diffusion noise scales to balance the tradeoff between diversity and sample quality.

### Weaknesses
- The idea isn't incredibly novel. The paper is a combination of many design choices (diffusion decoder, DiT rather than SE(3) invariant attention, FSQ discretization) rather than a single great idea. 
- You mentioned the cost of loss functions used in prior works. Maybe you can do an experiment measuring the iteration speeds and memory scaling of each approach? Note also that diffusion model autoencoders have a multiplicity (number of diffusion timesteps per step), which can increase the memory use / runtime of this approach. Also, despite these supposed gains, you only trained on proteins of length <256. An experiment scaling to larger systems may be worthwhile.

### Questions
- Do you think the sliding window attention would still suffice for longer sequences?
- While the model achieves superior reconstruction and competitive generation for token‐based methods, it still lags continuous diffusion models. What are the main causes of this gap?

### Soundness
4

### Presentation
4

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
This paper introduces Kanzi, a non–SE(3)–equivariant diffusion autoencoder–based protein structure tokenizer, trained using a single flow-matching objective on global 3D coordinates. The authors argue this architecture simplifies training relative to SE(3)-invariant tokenizers and demonstrate strong reconstruction and unconditional structure generation performance compared to recent tokenizers such as ESM3, DPLM2, FoldToken, and IST.

### Strengths
- Clear motivation to simplify protein structure tokenization by removing architectural and loss engineering complexity.

- Technically solid implementation using flow-matching and FSQ-based codebooks.

- Strong reconstruction quality across multiple protein benchmarks despite relatively small model size and training data.

- Interesting empirical findings: non-equivariant encoders outperform equivariant ones under flow objectives, codebook utilization emerges late in training.

- Introduction of rFPSD as a distributional reconstruction metric is potentially valuable.

### Weaknesses
1. No evaluation of SE(3) stability / invariance — critically, the tokenizer may output different tokens for rotated versions of the same protein, which invalidates many downstream use cases (retrieval, clustering, homology, interpretability). This is not even measured.

2. No retrieval / similarity / StructTokenBench[1]-style evaluation, even though retrieval is a core purpose of structured tokenization (cf. FoldSeek[2], ).

3. No conditional generation experiments, despite stating generative capability as a core contribution; all reported results are unconditional only.

4. Claims of “smaller model rivaling ESM3/DPLM2” are not apples-to-apples — those are protein sequence / multimodal models, not structure-only.

5. Scope of “tokenizer quality” is too narrow — heavily focused on reconstruction, insufficient multi-dimensional evaluation (e.g. sensitivity, explainability, stability, controllability).

[1] Protein Structure Tokenization: Benchmarking and New Recipe

[2] Fast and accurate protein structure search with Foldseek

### Questions
1. Does rotating a protein structure change the tokenization output? Have you evaluated rotational consistency quantitatively?

2. Why is retrieval / homology search omitted? Do Kanzi tokens perform poorly under similarity search?

3. Can Kanzi support conditional generation (e.g., topology, motif, scaffold constraints)? If yes, why is it not reported?

4. You do not perform residue-level local centering or relative-frame coordinate normalization (e.g., per-residue local frame / backbone-centric coordinates) — without such normalization, how are the learned tokens supposed to be interpretable or physically meaningful?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Kanzi, a flow-based protein structure tokenizer that replaces SE(3)-invariant components with a simpler diffusion autoencoder trained using a flow matching loss. The approach removes the need for complex invariant losses and geometric attention while maintaining or even improving reconstruction performance compared to prior tokenizers like ESM3, DPLM2, and FoldToken. 

Kanzi is then used to train an autoregressive model (Kanzi-AR) for structure generation, demonstrating competitive designability and diversity.

### Strengths
- **The experimental setup is solid and shows careful design choices** — e.g., diverse test datasets, detailed RMSD/TM metrics, and fair consideration of computational trade-offs. 

- **The writing is clear, and figures are well-designed.** The methodology section carefully explains the training and inference setup. 

- **Simplified yet effective formulation.** The use of a single flow matching loss instead of multiple SE(3)-invariant losses reduces training complexity and improves stability. This work is more like an altogether engineering refinement than a conceptual breakthrough. The proposed simplification aligns with recent diffusion autoencoder trends in vision. Though the novelty in biological context is limited, the "making it simple and scalable" is indeed a very important next step for this field.

### Weaknesses
- **Performance is not that out-standing**: The reconstruction performance and generation performance do not look that out-standing. ESM3 seems to still be the best of all. Inference-time sampling tricks from Proteina seems to be able to boost Kanzi-AR to a next level, though unfortunately neither ESM3 or DPLM2 used this trick. This might be an unfair comparison.

- **Only evaluating on reconstruction and generation, missing representation quality**: From image domain, there are some papers discussed about one point: not necessarily the best reconstruction quality leads to a better tokenizer. The representation quality also matters. e.g., see the table 1 in RAE paper (https://arxiv.org/pdf/2510.11690v1) for MAE-B and DINOv2-B. And there is a benchmark designed for structure tokenizer representation quality: from AminoAseed [1] paper. Highly suggest to also benchmark representation quality rather than reconstruction quality alone.

- **Missing comparisons to other structure tokenizer benchmarks**: see questions.

- **Minor typos**: There are also minor typographical issues (e.g., lines 794–797 contain “¿” and “¡” characters).


[1] Protein Structure Tokenization: Benchmarking and New Recipe

### Questions
1. Compare with two more baselines: 
- Cheap [1]
- AminoAseed [2]

2. Add representation quality evaluation. For example, the benchmark from AminiAseed [2].

3. Questionable use of diffusion sampling in AR models. Table 3 claims Kanzi-AR uses inference-time sampling tricks (Eqn. 5), which conceptually apply to diffusion-based samplers, not discrete autoregressive decoders. As I can understand, AR simply uses the discrete structure tokens for autoregressive training and sampling. There should not be any diffusion sampling process involved? Clarification is needed.

[1] Tokenized and continuous embedding compressions of protein sequence and structure 

[2] Protein Structure Tokenization: Benchmarking and New Recipe

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The authors present a flow-based protein structure auto-encoder ("Kanzi") useful as a protein tokenizer. The primary contribution is that, as far as I am aware, it is the only example of flow-based structure auto-encoder. Secondary to that, but still very important, is that the encoder and decoder are not invariant or equivariant, continuing the recent trend to show that such complex models are not required to adequately model proteins. The authors additionally use the model to generate novel structures.

### Strengths
The authors presentation is mostly very clear, and it does a good job of discussing the previous literature. They present a novel approach to the problem of structure tokenization. I particularly encouraged to see that the output is simply the backbone atomic coordinates, which should greatly simplify the tokenizer's use. It is intriguing that the model uses real space coordinates _and_ a sequence id embedding; see questions for more on this.

I like that the authors show that the resulting tokenizer can be used to generate designable structures. But see weaknesses for additional comments on this.

Clearly Kanzi is successful: the 30M parameter version trained with "under optimized" hyperparameters is best or second best across most metrics, covering the core set of structure prediction data, while being _much_ smaller. 

The discussion of ablations is especially useful, although I do wish it were more detailed.

### Weaknesses
This is a small weakness but the authors seem to flip between using "diffusion" and "flow matching" to describe their approach. It seems that flow matching better matches what they're doing--either that or there is something that needs to be clarified substantially in their writing. In any case diffusion and flow matching are not exactly the same thing, so the authors should be precise.

There are two significant weaknesses in the paper that I'd like to see addressed:
1. The authors should examine how their tokenizer performs at other downstream tasks that are relevant for protein language models. Many are detailed in a recent paper from ICML 2025: Xinyu Yuan et al., Protein structure tokenization: Benchmarking and new recipe.

2. While the authors show that their auto-regressive structure generator is capable of generating designable structures using the conventional definition of generating a sequence which folds back to the same structure, this is highly dependent on the particular folding model used. The authors chose ESMFold, which I found curious given they train on a sample of AFDB. The authors should comment on this choice given that better models are available.

### Questions
The authors might want to cite Ellmen, ... Deane "Transformers trained on proteins can learn to attend to Euclidean distance", which includes an interesting discussion of how attention mechanisms process real space coordinates.

### Soundness
4

### Presentation
3

### Contribution
4
