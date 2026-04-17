# Compressibility measures Complexity: Minimum Description Length meets Singular Learning Theory

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
We study neural network compressibility by using singular learning theory to extend the minimum description length (MDL) principle to singular models like neural networks. Through extensive experiments on the Pythia suite with quantization, factorization, and other compression techniques, we find that complexity estimates based on the local learning coefficient (LLC) are closely and, in some cases, linearly correlated with compressibility. Our results provide a path toward rigorously evaluating the limits of model compression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper extends the Minimum Description Length (MDL) principle to singular models via Singular Learning Theory (SLT), showing that the local learning coefficient (LLC) governs description-length redundancy for neural networks. It proposes an operational compressibility metric (the critical quantization level / factorization fraction under a loss-tolerance ϵ) and empirically studies its relationship to LLC on Pythia models up to 6.9B parameters. The main finding is a strikingly linear relation between LLC and critical bits for uniform symmetric quantization over long training ranges ($R^2$ ≈ 0.98), while factorization and other perturbations show weaker but consistent trends.

### Strengths
- A singular-MDL formulation that ties compressibility to degeneracy (not curvature) and places LLC as the leading term in code redundancy. This is conceptually neat and well-motivated.
- Operational metric for compressibility with a practical measurement protocol (critical $n_q$ ​ at tolerance $\epsilon$)
- Consistent, near-linear LLC–critical-bits relation across model sizes and checkpoints; robustness across  $\epsilon$ values is examined.
- Transparent discussion of limitations of SGLD-based LLC estimation.

### Weaknesses
- My main concern is about the narrow compression baselines, especially for pruning/factorization; non-uniform quantization missing. **Quantization**: main experiments use symmetric uniform grids with range m optimized to minimize loss; no comparisons to stronger PTQ baselines (AWQ[4], GPTQ[5], group/mixed-precision). This limits the generality of the reported LLC–bits linearity. **Factorization**: appendix uses plain SVD truncation with heuristic layer choices; no SVD-LLM[1] style evaluations; the LLC–compressibility relation is generally monotonic but clearly weaker than quantization and can flatten on larger models. **Pruning**: appendix reports random attention-head pruning with brief retraining; there is no magnitude/Wanda[2]/FLAP[3]; authors do not report a clean LLC–critical-pruning fraction relation due to noisy curves. Hence the pruning story is inconclusive.
- All experiments are on dense Pythia transformers. It remains unknown whether LLC predicts compressibility for other architectures like, Mixture-of-Experts.
- SGLD-based estimation lacks firm theory at LLM scales and is sensitive to hyperparameters; this is acknowledged, but it means the measured $\lambda$ may differ from true LLC in unknown ways, especially outside the quantization setting.


[1] Wang, Xin, et al. "Svd-llm: Truncation-aware singular value decomposition for large language model compression." arXiv preprint arXiv:2403.07378 (2024).

[2] Sun, Mingjie, et al. "A simple and effective pruning approach for large language models." arXiv preprint arXiv:2306.11695 (2023).

[3] An, Yongqi, et al. "Fluctuation-based adaptive structured pruning for large language models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 10. 2024.

[4] Lin, Ji, et al. "Awq: Activation-aware weight quantization for on-device llm compression and acceleration." Proceedings of machine learning and systems 6 (2024): 87-100.

[5] Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).

### Questions
- What about the estimation performance of LLC on any possible combinations of compression methods: pruning + quantization (one can first apply pruning and then do quantization)?
- Would LLC be used to guide the hyperparameters/initialization of MoE pre-training?

### Soundness
2

### Presentation
3

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
The paper aims at bridging the gap between information theory and practical compression of Large Language Models by exploring the model space with performance-induced metrics and the Local Learning Coefficients theory. The introduction and background are solid, the proven theorems are sound. The experimental design is sound but lacking detailed explanations. The experimental results generally agree with theoretical findings but lack finer interpretations and broader discussion.

### Strengths
- The background is well-written and explains the problem well even to unaware readers. The references are well-placed and, in my view, fairly exhaustive.
- The paper discusses focuses on the rarely-discussed information-theoretical background of LLM compression, which is novel and interesting.

### Weaknesses
- The results indicate that the proposed analysis is mostly accurate for intermediate stages of training and seemingly non-transferable between model sizes. This limits the applicability to the analysis of Post-Training Quantization of heavily-overtrained LLMs and the predictive power.
- The paper (including Appendix) is missing crucial information about the experimental setup and conducted experiments, such as LLM training hyper-parameters (e.g., optimizer, learning rate, batch size) and the observed training loss curves that could potentially help to interpret the results and gauge the method's applicability.
- Theorem proofs, including some that aren't novel, occupy the majority of the paper's main body, leaving little space for experimental results and discussion.
- Interpretation of the observed trained dynamics (from the Compressibility perspective) are lacking (e.g., how models get seemingly harder to compress the longer the training). Broader discussion (e.g., how the trends depends on model size) are missing.

Overall, the paper seems incomplete at this stage.

### Questions
Questions: 

1. It is known from the area of dynamic-bitwidth compression, that certain model layers have disproportionally large effect on model degradation [https://proceedings.mlr.press/v139/hubara21a/hubara21a.pdf] and KL-Divergence specifically [https://aclanthology.org/2025.naacl-long.543/]. From this, one would expect that the estimated LLCs would be substantially different for various parts of the same model (e.g., various layers/blocks). Can your analysis be applied on this finer level?

2. How do train steps (i.e., from 0 to 90K) translate into the number of consumed tokens? How do those token-to-parameter ratios correspond to the compute-optimal estimations of Hoffmann et al. and the corresponding ratios of modern models (e.g., Llama-3, Gemma3, Qwen3).

3. Can the proposed method (LLC/compressibility estimation) be applied to existing pre-trained models (e.g., Llama-3, Gemma3, Qwen3)?

Editorial notes:
- 196 and 201: m is both multiplicity and lower bound? Confusing notation.
- 290: typo: loglog m -> loglog n
- 306: inconsistent \frac vs / in O(...)

### Soundness
2

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
2

### Summary
The paper's objective is its attempt to establish a theoretical measure of neural network complexity by combining two statistical frameworks, namely minimum description length and singularity theory. 

The authors propose that the Local Learning Coefficient (LLC), a recently introduced metric metric derived from SLT, can serve as an architecture-independent measure that accurately predicts a model's intrinsic compressibility limit. To validate this theory, the authors perform experiments on the Pythia model suite (a series of large language models). They found that the LLC was correlated with the observed compression rate across various techniques (quantization, factorization, etc.).

### Strengths
The topic is generally of interest, since model compression is important for model deployment: the size of the effective throughput are improved with it. This leads to push the resource/accuracy Pareto frontier of models. From this perspective, a theory that offer some foundation for understanding neural network compression is of interest. 

A good measure of compressibility is something that is missing. 

The paper is quite extensive in the treatment of different cases (with the shortcoming of considering the loss only).

### Weaknesses
The validation is limilted to the Pythia model suite and measured against training loss on the pre-training dataset. This limited scope may fail to generalize the findings to other models. While useful for scaling laws, restricting the analysis to a single family of models trained on a specific corpus limits the claim of "architecture-independent" applicability.

The observed linear correlation between LLC and compression rate is likely due to the inherent structure of the pre-trained weights, which is already a known phenomenon in model compression heuristics. Without a corresponding analysis that shows the LLC predicts the drop in capability on MMLU or other downstream tasks after compression, the measure remains an academic curiosity rather than a practical tool.

Also, the authors admit (p.2 in F.2.) that the framework ``has yet to explain any capabilities gains via post-training methods like various forms of fine-tuning (e.g., LoRA) and reinforcement learning (e.g., RLHF).'' Given that the most significant and rapidly changing complexity in modern LLMs comes from instruction tuning and alignment (which fundamentally restructure the model's energy landscape), a complexity measure that ignores this utility does not seem very adequate. The framework applies only to the initial, unaligned state, offering limited insight into the model deployed in practice.

### Questions
Have you experiments that would measure whether the conclusions remain valid when the model performance is validated on downstream tasks?

### Soundness
3

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
4

### Summary
The paper reinterprets the main findings of singular learning theory (SLT), originally formulated within Bayesian asymptotics analysis, in the language the MDL framework. It then establishes a conceptual and mathematical link between the SLT-derived local learning coefficient (LLC) and neural network compressibility (via quantization). Based on this connection, the authors present empirical evidence that the geometric quantity represented by the LLC correlates with how compressible a model is.

I recommend the rejection of the paper in its current state because of the problems with its loose contextualization within the field's conceptual landscape, insufficient justification for some of the claims made, and poor presentation. However, I provide a weak rejection since I believe the paper makes an important and promising connection between loss landscape geometry and compressibility.

### Strengths
- The paper addresses an important problem in the field: while compression is rampant in research and practice, theory of compressibility is lagging behind. Improving the understanding of what makes models compressible is an important endevaour.
- The connection between LLC and compressibility (i.e. quantizability) is very interesting and very promising as a future direction for following research (albeit relatively underdeveloped in this paper).
- The techniques used in combining MDL and SLT frameworks can be useful for following research.
- The paper is written with a clear and accessible language (although not structured so, see below).

### Weaknesses
The paper has three major weaknesses that should be diligently addressed. I will briefly list these weaknesses here and provide more detail in the next section. The main three weaknesses of the paper are:

1. Conceptual imprecision and lack of contextualization within the literature
2. Insufficient substantiation for some of the claims
3. Poor presentation of the preceding and existing work

### Questions
### Weakness 1

The paper is over-indexed to a very specific conceptual landscape and the assumptions that it implies early on, and mostly ignores alternative frameworks, making it hard to situate the paper and evaluate its contributions in the context of the larger learning theoretic literature. More specifically, the notions such as "complexity" and "compressibility" have been used in the literature with diverse definitions to produce useful theoretical and empirical results e.g. [1, 2, 3, 4]. The paper ignores alternative possible definitions of these phenomena and early on commits to the notion of "complexity $\approx$ compressibility". Although it is perfectly fine for the paper to commit to a theoretical framework and particular definitions, this is helpful only if this is done *explicitly*, and *after* the acknowledgment of alternative definitions (or at least existence thereof). Instead, the paper prematurely concludes this topic with statements like "It is clear that compressibility in the above sense must be related to ideas like minimum description length (MDL)". 

Note that By collapsing complexity and compressibility, the paper obscures an important conceptual distinction: unlike the paper's intimation of complexity $\approx$ compressibility, its results imply a different, and perhaps more meaningful conclusion, i.e. asymptotically complexity $\propto \lambda$ (from MDL - SLT connection), whereas compressibility $\propto \lambda/d$ (L368). This situates complexity as an absolute, and compressibility as a relative notion. Note that this is not a trivial nor cosmetic distinction: this implies that LLC can be used as a notion of model complexity *across* and *within* architectures, but as a notion of model compressibility only *within* architectures (also implied by Fig. 3, right). This is a fixable problem, but is a tangible negative result of the conceptual imprecision that should be decisively addressed.
 
### Weakness 2

The paper makes the claim that "We study neural network compressibility by using singular learning theory to extend the minimum description length (MDL) principle to singular models like neural networks." I believe that this claim requires strong qualification. Although the authors bridge Watanabe’s Bayesian asymptotics and information-theoretic MDL framework, they end up translating the existing Bayesian–singular theory (Watanabe’s asymptotic framework) into MDL language, rather than introducing new inferential machinery that the Bayesian version didn’t already imply. Indeed, the authors' main result is identical in form to Watanabe's original result, expressed for redundancy rather than free energy. In contrast to this, in my opinion the more novel contribution of the paper is the connection between LLC and quantization (Section 3.3) - and unfortunately this section is theoretically underdeveloped. And, importantly, to support the former point, note that Section 3.3 could have been written with the machinery of original SLT theory.

### Weakness 3

Lastly, the paper does a poor job at onboarding the reader to the existing work, which it relies heavily on - this challenges the feasibility of the paper as a standalone work. For example, the paper does not attempt to formally introduce SLT are anywhere in the main paper. More dramatically, the paper directs the reader to a separate *book* for the definition of two central variables in the main theorem of the paper (and for SLT). This and similar omissions make it difficult for the paper to stand alone. A concise but formal introduction to SLT and MDL - including definitions of key quantities such as the learning coefficient and multiplicity - is essential for accessibility to a general ML audience. The paper allocates two pages for the proof of Theorem 1: some of this space can and should be allocated to onboarding the reader to the previous work systematically, including central variables they use in their own theory such as RLCT (similarly, if needed, fine-tuning low-dimensionality review can be relegated to the appendix).

### Minor comments
- L159: Please specify what $\mathcal{H}(q)$ denotes.
- L325: Please explain "minimally singular" vs "singular"
- L357: "our in the"
- L463: The paper makes repeated emphasis on the model families hosted on Pythia framework. Is there any particular reason for this? If the models do not have a theoretically important mutual property, this implementation detail should be deemphasized.

[1] Jiang*, Y., Neyshabur*, B., Mobahi, H., Krishnan, D., & Bengio, S. Fantastic Generalization Measures and Where to Find Them. ICLR. 2019.

[2] Suzuki, T., Abe, H., & Nishimura, T. Compression based bound for non-compressed network: Unified generalization error analysis of large compressible deep neural network. ICLR. 2021.

[3] Sefidgaran, M., & Zaidi, A. Data-Dependent Generalization Bounds via Variable-Size Compressibility. IEEE Transactions on Information Theory, _70_(9), 6572–6595. 2024.

[4] Lu, H., Zhou, Y., Liu, S., Wang, Z., Mahoney, M. W., & Yang, Y. AlphaPruning: Using heavy-tailed self regularization theory for improved layer-wise pruning of large language models. NeurIPS. 2024.

### Soundness
3

### Presentation
2

### Contribution
2
