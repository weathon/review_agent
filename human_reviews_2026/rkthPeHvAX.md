# From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 8

## Abstract
Humans organize knowledge into compact conceptual categories that balance compression with semantic richness. Large Language Models (LLMs) exhibit impressive linguistic abilities, but whether they navigate this same compression-meaning trade-off remains unclear. We apply an Information Bottleneck framework to compare human conceptual structure with embeddings from 40+ LLMs using classic categorization benchmarks (Rosch, 1973a; 1975; McCloskey & Glucksberg, 1978). 
We find that LLMs broadly agree with human category boundaries, yet fall short on fine-grained semantic distinctions. Unlike humans, who maintain "inefficient" representations that preserve contextual nuance, LLMs aggressively compress, achieving more optimal information-theoretic compression at the cost of semantic richness. Surprisingly, encoder models outperform much larger decoder models in agreement with human categories, suggesting that understanding and generation rely on distinct representational mechanisms.
Training-dynamics analysis reveals a two-phase trajectory: rapid initial concept formation followed by architectural reorganization, during which semantic processing migrates from deep to mid-network layers as the model discovers increasingly efficient, sparser encodings.
These divergent strategies, where LLMs optimize for compression and humans for adaptive utility, reveal fundamental differences between artificial and natural intelligence. This highlights the need for models that preserve the conceptual "inefficiencies" essential for human-like understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies how LLMs and humans trade-off compression and meaning in conceptual representations. Building on Rate–Distortion Theory and the Information Bottleneck (IB), the authors propose an objective function to combines information-theoretic compression with geometric coherence. Using digitized, classic human categorization benchmarks and embeddings from many models (encoders and decoders), they report three core findings: (i) LLMs broadly align with human category boundaries, (ii) LLMs capture weak typicality gradients relative to humans, and (iii) LLM-derived clusters achieve lower L (greater compression efficiency) than human categories, suggesting humans preserve more semantic richness at the expense of statistical efficiency. The paper releases the digitized benchmarks and positions the framework as a diagnostic for monitoring compression–meaning balance.

### Strengths
1. Originality. A clear, unified information-theoretic lens that ties together clustering complexity (compression) and semantic coherence (meaning), then uses it to compare humans and LLMs at scale. The encoder-decoder contrast is especially thought-provoking.
2. Quality. Careful benchmark curation (classic cognitive datasets), comprehensive model coverage, and multi-angle evaluation (category alignment, typicality, and L-curves); The training-dynamics analysis with 57 OLMo-7B checkpoints is helpful.
3. Clarity. The paper’s structure is easy to follow; figures that separate boundary alignment vs. typicality vs. L-frontiers help the reader parse what “efficiency” means in practice.
4. Significance. The finding that humans are less efficient but more semantically rich challenges a common “optimality” narrative and invites new objectives/architectures for more human-aligned understanding.

### Weaknesses
1. Many correlations are modest. Please provide bootstrap confidence intervals, multiple-comparison controls, and seed variability for clustering and correlation estimates.
2. The conclusion that humans are statistically suboptimal depends on the chosen geometry and L weighting. Most results fix $\beta=1$; the paper should report sensitivity curves over $\beta$ and discuss how human/LLM rankings vary under plausible trade-off weights. Also clarify how human distortion is computed from typicality/membership data to avoid mismatched metrics.
3. Encoder-decoder gap interpretation. The architecture result is intriguing, but confounded by training objective differences (MLM vs. autoregressive), tokenizers, and pretraining corpora. A controlled study with matched data/tokenizers and frozen-head probes would strengthen the claim.

### Questions
1. How exactly is the Distortion term computed for human categories, from typicality distances, membership uncertainty, or an inferred geometry? Please add a short methodological box that maps human ratings to the variance term and discuss limitations.
2. If typicality is computed against category names, can you replicate with prototype centroids (learned per category) or descriptive definitions, and do encoder-decoder gaps shrink?
3. With a common tokenizer/corpus and matched parameter counts, do masked-LM encoders still outperform autoregressive decoders on AMI and typicality?
4. During the mid-layer migration phase, what happens to the complexity-distortion frontier layer-wise? Does the “efficiency” shift track changes in heads/FFNs (e.g., sparsity, attention concentration)?
5. Can you test the framework on non-noun categories (events, relations) or multilingual replicas?

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
4

### Summary
This paper conducts a rigorous analytical re-examination of the boundaries of “understanding” in large language models (LLMs). The authors demonstrate that current LLMs still struggle to master fine-grained semantic distinctions, which is the critical element shaping human-level comprehension. Besides, Information-theoretic optimality does not equate to human-level understanding. Maximizing compression efficiency diverges from achieving semantic alignment fundamentally. In addition, focusing solely on decoder-based language models is unlikely to improve alignment with human comprehension capabilities. These findings reveal a fundamental divergence in the strategies employed by LLMs and humans for understanding natural language: LLMs rely on statistical compression, while humans depend on semantic richness.

### Strengths
(S1) The paper offers a theoretically motivated and quantitatively explicit framework that fuses Rate–Distortion Theory and the Information Bottleneck to provide a new perspective on the compression–meaning trade-off in LLMs. The use of semantic compactness as an internal proxy for meaning fidelity is original.

(S2) The empirical analysis is well-executed and leverages high-quality, historically grounded human categorization datasets. Their public release greatly facilitates reproducibility and establishes a lasting benchmark for semantic understanding in language models.

(S3) The evaluation spans widely of model architectures (encoder, decoder, and static word embeddings) and scales (from 300M to 70B parameters). The inclusion of both static and contextual embeddings, and the analyses across training checkpoints like OLMo-7B, provides a multi-angle validation of the proposed framework’s generality and interpretive depth.

(S4) The paper is written with exceptional clarity and precision, presenting complex theoretical ideas in an accessible and logically coherent manner, making the idea easy to follow.

### Weaknesses
(W1) Although the analysis is thorough, one might consider including token-level efficiency statistics in future works, though this does not affect the validity of the current findings.

(W2) The study could be further enriched by considering computational efficiency (e.g., token-level cost) as an additional axis in the compression–meaning landscape. Doing so may illuminate how efficiency interacts with semantic representation in practice.

(W3) This study only scoped in English, it would be interesting as future work to examine whether similar compression–meaning trade-offs hold across multilingual models, given that linguistic granularity may modulate semantic richness.

(W4) The study currently focuses on conceptual categorization, exploring how it extends to relational or compositional understanding could broaden its applicability and lasting its significance.

### Questions
1. Could the authors comment on how the main results might change if the distortion term in the $\mathcal{L}$ objective were replaced with a more cognitively grounded metric, such as human similarity judgments or human-rated typicality scores?
2. It would be useful to include a brief sensitivity analysis or supplementary report across several $\beta$ values to illustrate whether the LLM–human divergence remains consistent under varying compression–meaning trade-offs.
3. Could the authors clarify whether the proposed $\mathcal{L}$ is intended to generalize beyond categorical tasks, or whether it should be viewed as specific to concept-formation settings rather than to other forms of understanding such as relational reasoning, compositional generalization, or contextual disambiguation?
4. Would the authors consider adding a brief discussion on what kinds of inductive biases or representational mechanisms might help future models better align with human conceptual structure?
5. Although the compression is measured at a bit level in the current formulation, could the authors discuss what theory or empirical understanding about the compression at token-level in language models that this framework could provide?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates the relationship between compression and meaning in human and machine representations, proposing that large language models (LLMs) achieve broad categorical alignment with human judgments but lack fine-grained semantic distinctions. The authors frame this as a compression–meaning tradeoff: LLMs optimize for efficient representation (compression) at the expense of semantic richness, while humans maintain inefficient but more structured representations that support flexible reasoning.
The study evaluates a diverse set of LLMs across multiple benchmarks of human conceptual categories, comparing model cluster structures (via token embeddings and k-means) with human category structures. Analyses include rate–distortion metrics and mutual information between model-derived clusters and human-labeled categories. Key findings include that LLMs achieve near-optimal compression–distortion tradeoffs, whereas human representations appear suboptimal by information-theoretic measures. Additionally, encoder-only models better align with human judgments than decoder-only models, suggesting differences between recognition and generation mechanisms.

### Strengths
* The study compares a wide range of LLM architectures and sizes.
* Compression and concept formation is a deep question at the intersection of cognitive science and machine learning.
* This study links displines between information theory and the prototype theories of human concept learning.

### Weaknesses
* The paper takes prototype theory as a given framework for human concept representation, but this theory has been very controversial compared to Exemplar theory (Medin & Schaffer, 1978) as an alternative explanation. Modern cognitive models are usually hybrid and integrate prototype and exemplar components. The manuscript should acknowledge this longstanding debate and should not take for granted that the cognitive system for humans are prototype-like. 

* The paper lacks an introduction or figures about the background and the set up for the cognitive experiments that they compare LLMs with, and how they relate to compression-meaning tradeoff.  A schematic figure illustrating the overall task (compression–meaning tradeoff, human vs. model representation pipeline) is missing, especially considering the importantance to introduce the cognitive experiment clearly for this crowd of audience. 

*  The paper is difficult to read and repetitive, I would suggest to remove some of the colorful RQ01 blocks. 
- Fig. 1 (left): The figure has only a single square, so the rest of them are decoder architectures? 
- Fig. 2 (right): I would not term this as categorical success

* Unsupported claims:
    * Line 421: The statement that “human conceptual systems, though appearing suboptimal, serve distinct cognitive needs such as flexible generalization and causal reasoning” is unsubstantiated within the current analysis.
    * Line 72: “Challenges the popular assumption that statistical optimality equals understanding” — what is the “popular assumption” here? Please clarify.

* line 90: "cognitive studies applying information theory to human concept learning without connecting to modern llms", that is not true, see Wu et al. 2025. about prior cognitive modeling work applying information theory to human and LLM, in the context of rate–distortion tradeoffs.

* fig 1 left: there is just one square
* line 323: I would not term this as gradients

### Questions
* What precisely are the human categories, and how are they measured?
* How is mutual information calculated between human and model representations, and between human categories and other human categories?
* What prompts were used for contextual embedding extraction?
* What exactly is meant by “compression–meaning tradeoff” in cognitive terms?
* What are the exact prompts used to elicit contextual embeddings from the LLMs? And how much is it deviating from the classical behavioral experiments. 

Reference:

_Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification learning. Psychological Review, 85(3), 207–238_

_Wu, S., Thalmann, M., Dayan, P., Akata, Z., & Schulz, E. (2025). 
Building, Reusing, and Generalizing Abstract Representations from Concrete Sequences.
In The Thirteenth International Conference on Learning Representations (ICLR 2025)_

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper compares how large language models and humans organize conceptual categories, exploring whether models trade compression for meaning in a similar way to people. It digitizes classic psychology datasets on human categorization and analyzes embeddings from over forty models, both encoders and decoders. Using an information-theoretic framework inspired by rate–distortion theory, it quantifies the balance between information compression and semantic fidelity.

### Strengths
- very interesting and important premise 
- Systematic, broad comparison covering 40+ models and also layer-wise comparisons.
- Formulation/digitization of the datasets from cognitive science is a good contribution
- Transparent discussion on the limitations

### Weaknesses
- The main question lies in how strongly to rely on the proposed metrics to infer “human-like” learning. The information–compression trade-off captures geometric efficiency in embedding space, but it is not clear whether this translates to human-style conceptual abstraction or reasoning.
- Dependency on parameters and metrics - The authors themselves acknowledge that "architectural design and pre-training objectives significantly influence a model's ability to abstract human-like conceptual information."
Is cosine similarity to category names, too narrow to capture the richness of human judgments?
- robustness to different similarity measures, clustering methods, and parameter settings is not tested.
- How does current large scale frontier models perform under similar analysis ?

### Questions
- Metrics - Need more information about the Adjusted Mutual Information (AMI), Normalized Mutual Information (NMI), and
Adjusted Rand Index (ARI) metrics.

### Soundness
2

### Presentation
4

### Contribution
3
