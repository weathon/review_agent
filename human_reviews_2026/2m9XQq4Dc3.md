# BAR: Refactor the Basis of Autoregressive Visual Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Autoregressive (AR) models, despite their remarkable successes, encounter limitations in image generation due to sequential prediction of tokens, e.g. local image patches, in a predetermined row-major raster-scan order. Prior works improve AR with various designs of prediction units and orders, however, rely on human inductive biases. This work proposes Basis Autoregressive (BAR), a novel paradigm that conceptualizes tokens as basis vectors within the image space and employs an end-to-end learnable approach to transform basis. By viewing tokens $x_k$ as the projection of image $\mathbf{x}$ onto basis vectors $e_k$, BAR's unified framework refactors fixed token sequences through the linear transform $\mathbf{y}=\mathbf{Ax}$, and encompasses previous methods as specific instances of matrix $\mathbf{A}$. Furthermore, BAR adaptively optimizes the transform matrix with an end-to-end AR objective, thereby discovering effective strategies beyond hand-crafted assumptions. Comprehensive experiments, notably achieving a state-of-the-art FID of 1.15 on the ImageNet-256 benchmark, demonstrate the ability of BAR to overcome human biases and significantly advance image generation, including text-to-image synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Basis Autoregressive, a unified and learnable framework for autoregressive visual generation that treats tokens as projections of an image onto a learned basis via a linear transform y = Ax. By optimizing an orthogonal, end-to-end trainable transform matrix A jointly with the AR model, BAR refactors token sequences into more informative orders and units, theoretically preserving equivalence to MAR and xAR objectives and practically improving generation. A residual objective encourages early bases to capture maximal information. BAR subsumes prior designs (e.g., VAR, PAR, RAR, FAR, xAR) as special cases and achieves state-of-the-art results on ImageNet-256, with additional results of ImageNet-512, text-to-image generation, and extensive ablations.

### Strengths
* The proposed BAR paradigm is fresh and compelling, offering a unified framework that theoretically derives and subsumes a range of recent AR variants.
* The method is simple and end-to-end trainable without complicated designs. 
* It achieves strong results on ImageNet-256, comparable to current state-of-the-art diffusion models, and also includes extensive ablations that validate the effectiveness of each component. The paper also validates its scalability in terms of model size, resolution, and data scale (t2i generation).
* The paper is clearly written, well-structured, and easy to follow.

### Weaknesses
I do not see any major flaws in this paper. I have a few discussion points, and if addressed, I would be happy to raise my score:

* The residual loss feels somewhat heuristic. The paper’s motivation critiques recent AR work for relying on hand-crafted inductive biases, yet the residual loss reintroduces a prefix-ordering bias. More diagnostics on how the learned ordering correlates with information content, and comparisons to alternative ordering regularizers, would clarify why this particular choice is preferable.

* Orthogonality constraint on A: While enforcing orthogonality yields clean equivalence to MAR/xAR and stabilizes training, it restricts the search space to energy-preserving transforms. It remains unclear whether relaxing to general invertible A could provide further gains. An exploration of non-orthogonal parameterizations would be valuable.

### Questions
* Why can A be trained end-to-end jointly with the AR model? What outcomes would we expect from a two-stage training scheme instead?

* In Appendix Figure 18, the quality of text-to-image results does not look good, with many high-frequency artifacts. Please explain the likely causes.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Basis Autoregressive (BAR), a novel paradigm that reformulates autoregressive (AR) image generation by viewing token sequences as basis vectors in a linear space. Traditional AR models generate images in raster-scan order, which disregards spatial structure and heavily depends on human-designed heuristics. BAR achieves state-of-the-art FID 1.15 on ImageNet-256, demonstrates scalability to 512×512 resolution, and generalizes to text-to-image tasks with consistent improvements across metric

### Strengths
BAR reformulates AR models as linear-space transformations, offering a mathematically elegant and generalizable view that subsumes prior ad hoc approaches.

The introduction of an orthogonal, end-to-end learnable transform matrix A replaces human inductive biases with data-driven optimization.

BAR achieves new SOTA on multiple benchmarks (e.g., FID 1.15 on ImageNet-256) and demonstrates speed–quality advantages with fewer parameters.

### Weaknesses
1. While mathematically rigorous, the paper could better explain the intuition behind how the learned basis improves spatial dependency modeling.
2. The exploration of non-orthogonal or adaptive-rank transformations could broaden the framework’s generality.
3. Although efficiency is claimed, a detailed analysis of training overhead (from learning (A)) versus vanilla AR would be valuable.
4.  The connection between BAR’s linear transform and classical linear subspace learning (e.g., PCA, ICA) is briefly implied but not discussed explicitly.
5.  Equation (9)–(10) and residual objective explanation could use more intuitive narrative; some notations (e.g., (ỹ_k)) are underdefined at first appearance.

### Questions
1. Does the orthogonality constraint limit the expressivity of (A)? Would relaxing it (e.g., via low-rank factorization) yield additional improvements?
2. How does BAR perform in **long-sequence autoregression** (e.g., 1k+ tokens), where cumulative numerical error in (A^{-1}) may become significant?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes BAR (Basis Autoregressive), a new paradigm for AR visual generation. Instead of following a fixed raster-scan token prediction order, BAR introduces a learnable linear transform matrix A that redefines the basis of token prediction space. The authors claim that existing AR variants (e.g., VAR, RAR, PAR, FAR, xAR) can be expressed as special cases of A. They further propose a residual training objective to encourage informative early tokens and show extensive experiments achieving FID 1.15 on ImageNet-256, surpassing diffusion and AR baselines. The method is compatible with both MAR and xAR architectures and improves efficiency while reducing inductive bias.

### Strengths
1. Unified theoretical framework: Reformulates a broad range of AR improvements under a single matrix transformation paradigm, offering conceptual clarity.
2. Learnable basis with end-to-end training reduces reliance on handcrafted heuristics and inductive biases.
3. Strong empirical results: Achieves new SOTA on ImageNet-256 and competitive performance on ImageNet-512 and text-to-image tasks.
4. Compatible with multiple AR architectures (MAR, xAR), indicating generality rather than architecture-specific tweaking.
5. Residual objective design mimics coarse-to-fine prediction in a principled way rather than manually defining scales.
6. Visualization of learned basis provides interpretability insights into the emergence of hierarchical token prediction.

### Weaknesses
1. **Theoretical depth vs. practical benefit**: While the “unified” perspective is appealing, the core operation (learnable linear transform on tokens) is conceptually simple. The novelty may be perceived as incremental unless the generality claim is further formalized or proven beyond examples.
2. **Orthogonality constraint and optimization stability**: The reliance on orthogonal projection raises the question of whether the learned A collapses to simple permutations or low-rank patterns. More analysis is needed.
3. **Lack of ablation on scalability of A**: The matrix is N×N, which scales quadratically with sequence length. It's unclear how this approach behaves at 1024+ tokens or 1024² resolution.
4. **Training cost overhead**: The paper reports inference-time speedup but does not clearly quantify the extra training overhead introduced by optimizing A.

### Questions
1. Does A converge to a stable configuration, or does it keep drifting during training? Any visualization of A evolution?
2. Can BAR be applied to autoregressive text LLMs? If not, what limits the transfer?

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
3

### Summary
This paper introduces Basis Autoregressive (BAR), a framework for visual autoregressive (AR) generative modeling that rethinks how image data is factorized for sequential prediction. Rather than adhering to the fixed, human-designed raster-scan order, BAR mathematically formalizes token sequences as projections onto learned basis vectors, proposing an end-to-end approach to optimize this basis using a parameterized linear transformation matrix $\mathbf{A}$. The framework unifies various previous AR methods as special cases of matrix $\mathbf{A}$ and claims to transcend the limitations of hand-crafted inductive biases, showing strong empirical results (notably FID of 1.15 on ImageNet-256). The paper provides theoretical justification, thorough ablations of the learned basis, and visualizations of both the basis and progressive generation.

### Strengths
1. Unified and General Framework: BAR provides a mathematically coherent and unifying lens for understanding and extending autoregressive visual models. By interpreting previous methods as instances of linear basis transformations, the paper brings much-needed formalism to a field heavily reliant on heuristics.
2. Mathematical Depth and Rigor: The paper gives detailed derivations and proofs (see Section 3.3, and Appendix A) demonstrating the theoretical validity and equivalence of BAR to well-known AR objectives (e.g., MAR, xAR), subject to the choice of loss and basis.
3. Visualization: Visualization of generation (Figures 4, 8, 9) and qualitative samples (Figures 5, 10, 11) effectively demonstrate the dynamics and diversity of the BAR approach.

### Weaknesses
1. Experimental Reproducibility Gaps: While the main algorithm and hyperparameters are explained (see Section C and Table 8), some necessary details for replication are spread across appendices and main text, and others (e.g., code availability, certain training curves beyond ImageNet, data preprocessing for alternative datasets) are left somewhat vague. For a contribution at this level, full transparency would further strengthen the impact.
2. Dependence on VAE/Tokenizer Quality: Since the BAR method operates atop latent tokens produced via VAE or similar encoders (e.g., KL-16 tokenizer), its success is inextricably tied to the information bottleneck of the encoder. While this is noted in the limitations, reliance on a potentially lossy front-end may restrict the impact in domains where end-to-end optimization is required.
3. Clarity and Structure Issues: The mathematical notation is dense and sometimes overly compressed. For example, the definitions of prefix sums and the role of the reverse transform $\mathbf{A}^{-1}$ in generation/decoding steps are not always explicitly illustrated or described in algorithmic terms (see Figure 2 and the pseudo-code).
Some implementation choices, such as hard vs. soft orthogonal projection, are only detailed in ablation (Table 6) and could use a deeper explanation in the main text.

### Questions
Please provide responses to the issues raised in my Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3
