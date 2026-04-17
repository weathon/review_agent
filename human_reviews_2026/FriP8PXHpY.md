# Circuit Complexity Bounds for Visual Autoregressive Model

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Understanding the expressive ability of a specific model is essential for grasping its capacity limitations. A recent breakthrough in image generation is the introduction of Visual Autoregressive ($\mathsf{VAR}$) Models, which employ a scalable coarse-to-fine "next-scale prediction" framework. We investigate the circuit complexity of the VAR model and establish a bound in this study. Our primary result demonstrates that the VAR model is equivalent to a simulation by a uniform $\mathsf{TC}^0$ threshold circuit with hidden dimension $d$ and $\mathrm{poly}(d)$ precision. This is the first study to rigorously highlight the limitations in the expressive power of VAR models despite their impressive performance. We believe our findings will offer valuable insights into the inherent constraints of these models and guide the development of more efficient and expressive architectures in the future.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides the first circuit-complexity analysis of Visual Autoregressive models, formally showing that a fixed-depth (O(1)), poly(d)-precision VAR—comprising bicubic up-interpolation, attention, MLP/LN, convolutions, and a constant-depth VQ-VAE decoder—can be simulated by a DLOGTIME-uniform TC0 threshold-circuit family of polynomial size and constant depth. By mathematically formulating VAR’s coarse-to-fine “next-scale” architecture and proving TC0 realizations for each component (including exp approximation and matrix ops), the work establishes a strong uniform TC0 upper bound on VAR’s expressive power, thereby clarifying inherent computational limitations and aligning VAR with recent TC0 bounds for Transformers and SSMs.

### Strengths
* First circuit-complexity treatment of VAR models; clean formalization of the “next-scale” autoregressive architecture by applying the known TC0 techniques (floating-point ops, exp approximation, matrix/convolution ops, attention) to a new domain.
* I'm not an expert in this domain, but the stage-wise proof structure makes the argument easy to follow; explicit statements of depth/size bounds and uniformity aid reproducibility.

### Weaknesses
* The main TC0 upper bound critically relies on O(1) depth and poly(d) precision/width; the paper does not analyze how bounds change when depth scales with resolution or when precision changes, limiting relevance to large practical VARs. Provide depth-parameterized bounds or discuss thresholds where the simulation may leave TC0.

### Questions
* Many bounds assume hm, wm ≤ poly(d). In practical VARs, d may be modest while image size grows large—how would your analysis scale if hm, wm are the primary growth parameters?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper analyzes the expressiveness of visual autoregressive models using circuit complexity, formalizing VAR modules and proving they can be simulated by shallow threshold circuits under certain assumptions. The main result shows that the entire VAR pipeline can be simulated in O(1) depth and polynomial size, connecting VAR architectures to established circuit complexity classes.

### Strengths
- The paper attempts to formalize the components of VAR models and provides some logical structure.

- The authors reference existing circuit complexity results and applies them to the VAR setting.

### Weaknesses
- There is substantial overlap (e.g., Figure 1) with another ICLR submission (submission 2833), both focusing on theoretical complexity of VARs and even using exactly the same figures. This strong similarity raise concerns about originality and possible being written by LLMs.

- The contribution is mainly upper bounds. no lower bounds or separation results are provided, limiting theoretical novelty.

- Some key assumptions (e.g., constant depth/layers) in this paper do not match real-world VAR configurations, reducing practical relevance.

- Figure 1 contains unreadable symbols and overlaps. Moreover,

### Questions
How would the reported results change if the number of layers grows with input size, rather than being constant?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper establishes the first circuit complexity bound for Visual AutoRegressive (VAR) models, demonstrating that they can be simulated by a DLOGTIME-uniform TC0 threshold circuit with constant depth, poly(n) size, and poly(n) precision, despite their strong empirical performance in image generation. The authors systematically analyze each component including up-interpolation blocks, attention layers, convolution blocks, and the VQ-VAE decoder to prove that all can be computed within TC0 under realistic precision assumptions.

### Strengths
The paper provides a technical result that extends known TC0 bounds to VAR models.

### Weaknesses
1. The paper largely follows existing circuit complexity analysis techniques developed for Transformers and Mamba, offering limited novel methodological or theoretical advancements.
2. While the paper claims theoretical limitations for VAR, it fails to reconcile this with its strong empirical performance, leaving the tension between theory and practice unaddressed.
3. What practical implications does the paper’s conclusion that VAR models lie within TC0 have for real-world modeling or algorithm design?
4. Figure 1 is identical to that in https://openreview.net/forum?id=S3Fq8E9jb7, raising serious concerns about originality and proper attribution.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper conducts a comprehensive theoretical investigation into the circuit complexity bounds of Visual Autoregressive (VAR) models, which adopt a coarse-to-fine "next-scale prediction" framework for image generation. The core result demonstrates that VAR models (with hidden dimension d and poly(d) precision) can be simulated by DLOGTIME-uniform $TC^0$ threshold circuits with polynomial size and constant depth. These findings reveal inherent expressive limitations of VAR models despite their empirical performance advantages.

### Strengths
Due to significant differences between my research domain and the circuit complexity/visual autoregressive modeling field of this paper, I am unable to provide a substantive assessment of its originality, quality, clarity, and significance from a professional perspective. The paper appears to address an underexplored gap (circuit complexity analysis of VAR models) and presents a structured theoretical framework with detailed definitions and proofs, which suggests careful academic rigor. However, a precise evaluation of whether its contributions (e.g., novel formulations, complexity bounds) are impactful or original within the field requires expertise in computational complexity that I do not possess.

### Weaknesses
See Strengths.

### Questions
No question

### Soundness
3

### Presentation
3

### Contribution
3
