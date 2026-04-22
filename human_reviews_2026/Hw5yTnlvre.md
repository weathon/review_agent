# The Language of Time: a Language Model Perspective on Time Series Foundation Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Large language models have established a successful paradigm of training foundation models on massive datasets, extending this approach to multiple domains. Time series foundation models extend this paradigm, demonstrating exceptional cross-domain transfer and prediction capabilities in both industrial and academic scenarios. This creates a paradox: while time series from different domains reflect distinct dynamical systems that should limit transferability, empirical results demonstrate remarkable cross-domain performance.
To resolve this paradox, this paper investigates the representation learning mechanisms and generalization capabilities of patch-based time series foundation models from both theoretical and experimental perspectives. We demonstrate that these models fundamentally extend language model representations from deterministic vectors to probabilistic distributions, enabling effective cross-domain transfer. Our analysis shows that time series patches can be quantized into discrete vocabularies with statistical properties similar to natural language.
This theoretical framework explains how time series models inherit the robust representation and transfer abilities of large language models, accounting for their superior performance in temporal tasks. Our work provides a rigorous theoretical foundation for understanding, evaluating, and improving the safety and reliability of large-scale time series foundation models for time series analysis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper explores a theory-driven explanation for why time series admit transferability by casting them into a language-like paradigm: segment sequences into patches, quantize into tokens, and form a temporal vocabulary so the data can be processed by general-purpose sequence models. Methodologically, it presents (i) a channel-agnostic encoding/aggregation idea intended to produce stable token streams under multi-channel inputs; (ii) a vocabulary construction pipeline using standard clustering (e.g., K-Means); and (iii) a theory-informed narrative linking empirical observations to principles: faithful discretization (ε-covering/quantization bounds), Zipf-like natural statistics, capacity & generalization (hypothesis-space inclusion, stability bounds), and an information-theoretic view (patching as IB-style compression). Experiments report vocabulary statistics, visual analyses, and hyperparameter sensitivity (P, K) across datasets, pointing to the scalability and cross-domain applicability of this unified preprocessing-and-discretization pipeline (patch-token-vocabulary).

### Strengths
- Unifying perspective: converting time series into token sequences (patch→token→vocabulary) enables reuse of general sequence-model tooling across tasks/domains.  
- Practical simplicity: the windowing + clustering pipeline is easy to implement and scale, and integrates with existing systems.  
- Interpretability with evidence: Zipf/long-tail statistics, separability, and sensitivity analyses help explain how tokenized time behaves in distributional and geometric terms.  
- Coherent theoretical narrative: the paper connects discretization, statistical structure, capacity/generalization, and IB intuitions into a single storyline.

### Weaknesses
1. (Section 2.2) “Channel-agnostic” should be citation-backed or down-scoped  
   - Issue: The aim is to explain why transferability exists; channel-agnosticity is one facet of that explanation, not a claim to be proven here. Current wording may read as a strong conclusion.  
   - Suggestion (pick one):  
     (A) Citation-backed — add representative prior work that demonstrates transfer under channel changes/heterogeneity (multi-sensor/multimodal/cross-device/cross-subject), so the statement is literature-backed; or  
     (B) Down-scope — rephrase it as an explanatory/inductive attribute (“a plausible factor enabling transferability”), rather than a strong empirical claim within this paper.

2. (Section 2.2.1) Quantization choice and assumptions untested  
   - Issue: Only K-Means is used (Euclidean convexity, isotropic clusters); no comparisons with GMM, VQ-VAE, Product Quantization; no analysis of distance metrics (DTW, cosine, Mahalanobis).  
   - Suggestion: Add ablations with alternative quantizers/metrics; report effects on Zipf fit, within-cluster variance, and downstream metrics with significance tests.

3. (Section 3.1) Intra-cluster semantics not analyzed under faithful discretization  
   - Issue: The theorem bounds global ε but does not examine whether within-cluster deviations harm semantic consistency, cause boundary confusion, or break temporal smoothness.  
   - Suggestion: Report correlations between within-cluster variance and downstream performance (even lightweight).

4. (Section 3.2) Misalignment between the *core question* and the *explanation*, plus lack of validation  
   - Issue: §3.2 asks *why* time series follow Zipf and what it reveals about structure, yet it assumes a two-parameter GEM prior and derives Zipf—turning a causal “why” into “reproducing Zipf under a chosen prior.” There is no statistical validation that time series are GEM-like (parameter estimation, GOF/posterior checks).  
   - Suggestion: Clearly distinguish explanatory hypothesis from causal proof; treat §3.2 as a generative interpretation; add minimal quantitative checks or qualify the claims accordingly.

### Questions
- Will you add citations that directly evidence channel-agnostic transfer (e.g., multivariate TS across sensors/devices/subjects/modalities) so that §2.2 is literature-backed?  
- If not adding such citations, would you down-scope channel-agnosticity to an explanatory phrasing consistent with the paper’s “why-explanation” focus?  
- How would Zipf/long-tail stats and downstream metrics change with GMM/VQ-VAE or DTW/cosine/Mahalanobis distances?  
- Do intra-cluster errors correlate with downstream performance (a lightweight analysis would suffice)?  
- Can you provide a minimal statistical check for a GEM-like process to better address the causal “why” in §3.2?

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
The core argument of this paper is that patch-based foundation models for time series can be regarded as a generalized form of large language models (LLMs). Its key insight lies in showing that time series patches can be quantized into “distributional tokens” exhibiting quasi-linguistic statistical properties (e.g., Zipf’s law), which explains the model’s strong cross-domain generalization ability.

From an experimental perspective, the paper demonstrates the superiority of patches over point-based representations and presents a “vocabulary” of time series. From a theoretical standpoint, it further establishes a rigorous mathematical framework—supported by a series of theorems and lemmas—to justify and validate the reasonableness and advantages of treating time series as a form of language.

### Strengths
1. The paper presents a novel perspective by interpreting time series through the lens of language modeling, effectively bridging the gap between time series sequences and natural language sentences. The insight presented in this paper is interesting and addresses a valuable and important problem in the design of time series foundation model encoders.
2. The argumentation is comprehensive: beyond the empirical evidence provided by experiments, the paper develops an end-to-end, logically coherent mathematical framework for the “time series as language” paradigm, drawing from representation theory, model capacity, stochastic process properties, and learning theory.

### Weaknesses
1. (major) The experimental and theoretical analyses in this paper are well executed; however, the connection to foundation models themselves is insufficient. This limitation is mainly reflected in two aspects:
    1. Lack of causal linkage between “distribution token” and model inference behavior: 
    In the experimental section, the paper clusters time series after segmenting them into patches and observes the phenomenon of a “probability cloud,” suggesting that time series tokens are distributional tokens, with a corresponding theoretical explanation provided (Theorem A.4, Centroid–offset decomposition). However, the logical connection from this observation to the conclusion that “the model essentially operates on probability distributions” is incomplete. The paper does not provide evidence showing that the model (e.g., Transformers) can truly process such “probability clouds” as unified computational entities. At present, the “Centroid–offset” framework appears more as an effective modeling strategy for describing the fact of clustering results, rather than evidence that the model itself has evolved to manipulate probabilistic objects.
    2. Disconnected Analysis between "Quasi-Linguistic Properties" and Downstream Performance: The experimental section powerfully demonstrates the quasi-linguistic properties of time series but fails to establish a causal link between the "quality" of these properties and a model's performance on downstream tasks, such as forecasting. For instance, the paper notes a trade-off where larger patches (P=128) yield better linguistic plausibility (Zipf conformity), while smaller patches (P=32) offer superior structural fidelity (clustering quality). It remains unclear which of these characteristics is more critical for achieving high performance in downstream tasks (e.g. forecasting) and what the underlying reasons are.
2. (minor) Some of the theoretical proofs deviate considerably from practical conditions. In arguing that patch-based models have stronger representational capacity (Theorem 3.3), the proof (specifically Lemma A.3 in Appendix A.2) relies on a key assumption that the dictionary contains all possible patches of length 1. In a continuous space, this implies an uncountably infinite vocabulary. Theorem 3.2 attempts to theoretically explain the emergence of Zipf’s law by assuming that the token generation process follows a GEM distribution, from which Zipf’s law is derived. However, the paper does not provide any prior justification for why the generation process of time series “tokens” should naturally conform to a GEM distribution.

### Questions
1. Please refer to Weakness 1. The authors could further strengthen the paper by applying the key observations and hypotheses derived from their analyses to time series foundation models, and reporting corresponding experimental results. Such evidence would help verify whether these insights are valid in practice and whether they indeed contribute to improving the predictive performance of time series foundation models.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper reframes time series foundation models as language models over patch-level “distributional tokens.” Concretely, each short segment is assigned to a motif cluster and the model predicts a continuous offset around that centroid, yielding a cluster-and-offset pipeline that explains transfer across domains. Empirically, the authors compile a large “time-series vocabulary” across 37 heterogeneous datasets and show language-like statistics. Theoretically, they provide quantization/coverage error bounds and a capacity hierarchy showing that the hypothesis space of pointwise models is strictly contained in that of patch models.

### Strengths
1. The paper tackles an engaging problem and advances a theoretical account of why TSFMs can transfer and generalize across domains.
2. The study substantiates its claims with empirical evidence on diverse real-world time-series corpora, using these datasets to test the central assumptions of the proposed framework.

### Weaknesses
1. The empirical validity appears sensitive to tokenization design, including codebook size, patch length, normalization, and clustering, leaving questions about the robustness and cross-domain stability of these hyperparameters.
2. The theoretical guarantees rely on smoothness and boundedness assumptions; while strong assumptions are acceptable in a theoretical treatment, their realism under non-stationarity and abrupt regime shifts remains insufficiently examined.

### Questions
1. When deploying to a new domain with shifted distributions, how should the codebook evolve?
2. How specifically do patch length and codebook size trade bias and variance in the cluster-and-offset scheme.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Current Time Series Foundation Models have demonstrated strong practical performance and maintain generalization and robustness across domains; however, their interpretability remains underexplored. This paper takes the perspective of treating time series as a “language”, borrowing linguistic insights—particularly distributional patterns from natural language—to explain the effectiveness of Time Series Foundation Models. The authors conduct extensive comparative experiments to analyze how factors such as patch size and cluster size influence the linguistic properties of time series and further propose a theoretical framework to support their hypothesis.

### Strengths
Time series in foundation models are segmented into patches based on stride and patch size, analogous to tokens in LLMs. The paper identifies this resemblance and attempts to transfer linguistic distribution laws—such as Zipf’s law—from natural language modeling to time-series modeling, providing a novel and insightful analogy that helps explain model effectiveness.

### Weaknesses
While the analogy-based narrative and theoretical construction are relatively coherent, several aspects are counterintuitive, and some experimental evidence appears insufficient to fully support the claims.

### Questions
1. Vocabulary Size Limitation:  
   Modern LLMs typically have vocabularies in the hundreds of thousands (e.g., Qwen3: ~151,936 tokens), where Zipf’s law manifests as a power-law distribution explaining language expressiveness. In contrast, this paper adopts a cluster size of 32, meaning only 32 “time-series tokens.” Such a small vocabulary weakens the explanatory force of Zipf-like reasoning.


$ P(X = k) = \frac{1/k^s}{H_{N,s}}, \quad k = 1, 2, \dots, N $
$ H_{N,s} = \sum_{n=1}^{N} \frac{1}{n^s} $

When N goes to Inf:
$ P(X = k) = \frac{1/k^s}{\zeta(s)}, \qquad s > 1 $


2. Blurry Cluster Boundaries:  
   The paper notes that latent time-series representations follow probability density distributions, leading to fuzzy boundaries between clustered tokens—visible in visualization results. Additional experiments are needed to explore how this probabilistic nature impacts model performance. For instance, if all time series within a cluster are treated as one patch token, what are the trade-offs in downstream tasks?

3. Fixed Patch Size Issue:  
   The study uses a fixed patch size of 32 across time series with different frequencies and sampling rates. This design strongly affects the “time-series as language” interpretation, but the paper does not analyze its implications—especially concerning frequency scalability in foundation models.

4. Generalization Across Settings:  
   The authors fix both patch size and cluster number to 32 and derive conclusions from this setup. It is unclear whether these conclusions hold under different patch or cluster configurations. More analysis or experiments are needed to validate generalization.

5. Insufficient Explanation of Cluster Centers:
   In Appendix (page 27), the authors visualize 32 cluster centers, implying these represent fundamental time-series patterns that can cover a broad range of temporal dynamics. However, the explanation is insufficient—particularly regarding how these 32 patterns capture the diversity of time-series structures, similar to how DCT basis components reconstruct images in JPEG compression.

6. Concerns on Realism of Theoretical Assumptions
Theorems assume β-mixing, GEM distribution, stability conditions, etc. Do real time-series datasets satisfy these assumptions? Have the authors validated β-mixing or GEM-like process assumptions empirically? Some TS domains (finance, climate, sensors) may not have β-mixing or stationary structure.

### Soundness
3

### Presentation
4

### Contribution
4
