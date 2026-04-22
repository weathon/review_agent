# Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6, 6

## Abstract
Personalized large language models (LLMs) rely on memory retrieval to incorporate user-specific histories, preferences, and contexts. Existing approaches either overload the LLM by feeding all the user's past memory into the prompt, which is costly and unscalable, or simplify retrieval into a one-shot similarity search, which captures only surface matches. Cognitive science, however, shows that human memory operates through a dual process: Familiarity, offering fast but coarse recognition, and Recollection, enabling deliberate, chain-like reconstruction for deeply recovering episodic content. 
Current systems lack both the ability to perform recollection retrieval and mechanisms to adaptively switch between the dual retrieval paths, leading to either insufficient recall or the inclusion of noise.
To address this, we propose RF-Mem (Recollection–Familiarity Memory Retrieval), a familiarity uncertainty-guided dual-path memory retriever. 
RF-Mem measures the familiarity signal through the mean score and entropy. High familiarity leads to the direct top-$K$ Familiarity retrieval path, while low familiarity activates the Recollection path. In the Recollection path, the system clusters candidate memories and applies $\alpha$-mix with the query to iteratively expand evidence in embedding space, simulating deliberate contextual reconstruction.
This design embeds human-like dual-process recognition into the retriever, avoiding full-context overhead and enabling scalable, adaptive personalization. Experiments across three benchmarks and corpus scales demonstrate that RF-Mem consistently outperforms both one-shot retrieval and full-context reasoning under fixed budget and latency constraints. 
Our code can be found in the Reproducibility Statement.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes RF-Mem, a retrieval framework for LLM personalization. The core idea is to adaptively choose between two memory-retrieval modes, inspired by dual-process human memory:
Familiarity: fast dense top-K retrieval when the query matches past user memory with high confidence.
Recollection: iterative, multi-hop expansion when confidence is low — retrieve candidates, cluster them, mix cluster centroids with the original query embedding, and retrieve again to surface contextual / episodic details.
A simple gate based on mean similarity and similarity-score entropy decides which path to use. The goal is to surface only the most relevant personal memories for context, instead of injecting the full user history.
The method is evaluated on multi-session personalization benchmarks with up to ~1M tokens of per-user memory. RF-Mem outperforms (i) no memory, (ii) giving the whole history, (iii) plain dense retrieval, and (iv) recollection-only, while keeping inference context small and latency close to dense retrieval. The authors also show RF-Mem working on top of summarized memory stores, not only raw logs.

### Strengths
### 1. Strong motivations and grounding in cognitive science. 

The familiarity vs recollection split is explicitly motivated by dual-process models of human memory. The authors lay out a concrete computational mapping (confidence gating → familiarity, uncertainty → recollection). This gives the method conceptual clarity and makes the design feel principled.

### 2. Clear formalism / presentation.

The framework is described cleanly, with the retrieval controller, the gating rule, and the recollection procedure all specified in enough detail to reimplement. 

### 3. Scales to large memories. 
RF-Mem remains effective under very large user histories (hundreds of thousands to a million tokens) where “put everything in context” is impossible and vanilla dense retrieval starts missing important long-tail details.

### 4. Method is practical with low overhead. 
The approach adds only modest latency over standard dense retrieval and explicitly tracks context budget. No LLM fine-tuning is required. It is interesting to see how much latency the dual process formulation saves over pure recollection, without losing performance.

### 5. Modularity. 
RF-Mem can sit on top of summarized memories rather than raw transcripts, suggesting it can plug into existing assistant memory stacks.

### Weaknesses
### 1. Lack of real-user evaluations.

All experiments use synthetic or benchmarked personalization data. There’s no human evaluation of helpfulness or preference alignment, and no analysis of noisy, contradictory, or sensitive histories.

### 2. Heuristic gating mechanism.
The decision to enter recollection mode is based on manually chosen similarity / entropy thresholds. There’s no learned controller and no robustness study across domains. Furthermore, the authors do not detail why certain thresholds were selected, or how one should optimise these.

### 3. Baselines.
Comparisons are mostly against: full context, dense top-K, and recollection-only. Stronger long-term memory agents (e.g. reflective / hierarchical memory systems) aren’t fully represented.

### 4. Safety/privacy.
Since RF-Mem is explicitly better at surfacing detailed personal history, it can also resurface sensitive information the user didn’t mean to bring into the current turn. The authors acknowledge this risk but do not quantify or mitigate it.

### Questions
1. How are the gating thresholds selected? Could the gate be learned?
2. Can you show qualitative failure cases where the gate picked the wrong mode?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RF-Mem (Recollection-Familiarity Memory Retrieval), a new framework for personalizing Large Language Models (LLMs) by improving how they retrieve user-specific memory. The authors argue that current methods for memory retrieval are inefficient, either by overloading the LLM with the user's entire history (which is unscalable) or by using a simple, one-shot similarity search that only captures surface-level matches. Inspired by cognitive science, RF-Mem is based on the Recollection-Familiarity Dual-Process Theory of human memory. Experiments on three benchmarks show that RF-Mem consistently outperforms both one-shot retrieval and full-context methods, achieving higher accuracy and recall while maintaining low latency.

### Strengths
1. The "Recollection" path is a novel iterative mechanism that uses KMeans clustering and $\alpha$-mixing to reconstruct evidence chains purely in the embedding space, without requiring a costly LLM-in-the-loop for query reformulation.
2. The paper includes a full suite of supporting analyses in the appendix. This includes detailed sensitivity studies for all key hyperparameters ($\alpha$, $\tau$, B, F, K), a compelling qualitative case study that visually demonstrates the method's superiority,  and a formal theoretical analysis that mathematically grounds the risk-minimizing properties of the switching mechanism.

### Weaknesses
1. The primary experimental baseline ("Dense Retrieval") is a non-iterative, one-shot method. This is a weak baseline, as many advanced RAG systems already use multi-step or iterative retrieval.
2. This adaptive switch mechanism is heuristic and lacks robustness. The optimal thresholds are highly dependent on the specific embedding model used and the score distribution of the dataset.

### Questions
1. To improve robustness and reduce tuning overhead, have the authors considered a learned switching policy? A small, lightweight classifier trained on the probe retrieval's features (mean score, entropy, etc.) might provide a more generalizable switch than fixed thresholds.

### Soundness
3

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
5

### Summary
The paper proposes RF-Mem, a dual path that gates between fast “Familiarity” retrieval and slower “Recollection” based on mean similarity and an entropy signal. 
When the gate flags uncertainty, it runs an in-embedding recollection loop that retrieves, clusters with KMeans, α-mixes cluster centroids with the query, and iterates under tight beam/fanout/round caps. 
Experiments on personalized generation and retrieval tasks show a better accuracy–latency balance than one-shot dense retrieval or full-context prompting.

### Strengths
1. The cognitive grounding is crisp and the gate is concrete (mean score + entropy with a sharpness λ), so the controller is easy to reason about. 

2. Recollection is lightweight and model-agnostic since it lives entirely in vector space using simple clustering and linear mixing. 


3. Compute stays bounded because the loop exposes explicit knobs for beam width, fanout, and depth instead of open-ended expansion.

### Weaknesses
1. The many thresholds and weights feel hand-tuned, with little guidance for auto-tuning across users and domains. 

2. KMeans centroids can be unstable on anisotropic or overlapping memory clusters, so α-mixing may drift the query off-intention. 

3. Entropy over the top-K score simplex depends on λ and retriever scale, which could make the gate brittle when swapping encoders or normalizations. 

4. The controller assumes unit-normalized cosine and a sub-Gaussian mean-similarity story, which may not hold in heavy-tailed similarity landscapes. 

5. The latency write-up emphasizes averages, while real deployments care about tail spikes when recollection fans out.

### Questions
1. Would a density-based or spectral clustering help when memory clusters are non-spherical or imbalanced. 



2. Does the gate remain calibrated when swapping embedding models or scaling memory to much larger corpora. 



3. Can you add early-stop checks that detect “recollection loops” and cut branches before they waste budget.

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
3

### Summary
The paper “Evoking User Memory: Personalizing LLM via Recollection–Familiarity Adaptive Retrieval” proposes RF-Mem, a dual-process memory retrieval framework for personalized large language models. Inspired by cognitive science, the model integrates two complementary retrieval modes—familiarity, for fast and confident recognition, and recollection, for slower and more deliberate evidence reconstruction. RF-Mem measures familiarity using the mean similarity score and entropy, dynamically switching between the two paths based on uncertainty. When familiarity is high, it performs efficient one-shot retrieval; when uncertainty increases, it activates a clustering-based recollection process that iteratively refines queries through an α-mixing strategy in embedding space. This design allows the system to balance efficiency with depth, maintaining the low latency of one-shot retrieval for straightforward queries while engaging in more thorough evidence gathering for ambiguous ones. Experiments on several benchmarks demonstrate that RF-Mem outperforms traditional one-shot and full-context methods in both accuracy and efficiency, achieving scalable, adaptive, and human-like personalization for LLMs.

### Strengths
This paper presents a highly original and conceptually grounded contribution by introducing RF-Mem, a dual-process memory retrieval framework inspired by the Recollection–Familiarity theory in cognitive science. The idea of aligning LLM personalization with human memory processes is both novel and intellectually appealing, extending beyond conventional retrieval-augmented generation paradigms. Methodologically, the paper is well executed: the formulation of familiarity uncertainty through mean similarity and entropy is simple yet effective, and the recollection mechanism—implemented via clustering and α-mixing—offers a practical and computationally efficient way to approximate deliberate evidence reconstruction in embedding space. The experiments are comprehensive, covering multiple personalized benchmarks and retriever architectures, and the empirical results convincingly support the method’s claimed benefits in both accuracy and efficiency. The paper is also clearly written and well-structured, maintaining strong conceptual coherence between cognitive theory and technical implementation.

### Weaknesses
Despite its conceptual elegance, the paper’s empirical validation should be further strengthened. 
First, the experimental comparisons are restricted to retrieval-only baselines, omitting stronger LLM-based retrieval or reasoning systems such as query rewriting, iterative retrieval (e.g., Search-Mem), or graph-based methods. Including such more advanced baselines would better position RF-Mem within current state-of-the-art approaches and clarify whether its advantages hold beyond standard dense retrieval. 
Second, the paper lacks an ablation study of the gating mechanism. It is unclear how much the mean similarity score and entropy respectively contribute to routing decisions, and how often each retrieval path is activated. 
Third, the definition of uncertainty is limited to similarity score distributions, which may fail to capture semantic ambiguity or misleading high-confidence matches; this limitation should be acknowledged and discussed.

### Questions
1. Could adaptive or learned gating mechanisms outperform the hand-tuned parameters used here?
2.The recollection process relies on clustering in embedding space. Have the authors tested alternative approaches, such as graph-based retrieval or semantic expansion via LLM-generated paraphrases, to validate the robustness of the proposed method?
3.As the field increasingly uses LLMs for complex retrieval tasks, can you demonstrate RF-Mem's advantage over a baseline that uses a powerful LLM  for a single step of query expansion or decomposition before a dense retrieval step?
4.The entire framework, including the uncertainty estimation, relies heavily on the quality of the underlying embedding model.  Did you observe significant performance variance across the three retrievers (MiniLM, MPNet, BGE) in terms of the routing accuracy?  Is there a risk that a poorly calibrated embedding model could lead to a cascading failure in the gating mechanism?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RF-Mem, a dual-process memory retrieval framework inspired by cognitive science, combining a fast Familiarity path and a deliberate Recollection path for personalized LLM memory retrieval. By using a familiarity-uncertainty gating mechanism and a retrieve–cluster–mix procedure, RF-Mem adaptively retrieves evidence under varied user-memory relevance conditions.

### Strengths
1. The paper effectively motivates its design using the Recollection–Familiarity Dual-Process Theory (Lines 61–83) and successfully maps the cognitive analogy into a computational retriever.

2. The familiarity-uncertainty gating (Lines 162–201) provides principled decision logic for choosing between retrieval modes, reducing unnecessary expansion while preserving robustness.

3. The adaptive study (Lines 432–454) shows that RF-Mem complements rather than replaces indexing approaches like MemoryBank, strengthening its practicality.

### Weaknesses
1. The α-mix formula (Lines 246–257) is introduced but lacks theoretical clarity on why linear mixing is optimal for query expansion.

2. $\tau$ (Lines 174–201) is fixed globally but may vary significantly across users, domains, or embedding models.

3. Although the paper compares fairly against retrieval-only baselines (Lines 289–303), retrieval today heavily relies on query rewriting, which is absent from the comparisons.

4. The ethical considerations are brief and do not address challenges like preference over-amplification, exposure to sensitive data, or user-profiling bias.

### Questions
1. Could the authors consider ablation comparing linear vs. nonlinear mixing (e.g., gating networks, learned interpolation), or reference theoretical grounding such as manifold-preserving mixing used in retrieval augmentation?

2. Could the authors consider learned gating or meta-calibrated thresholds conditioned on query distribution or similarity statistics?

3. Include comparisons to LLM-based query reformulation modules such as LD-Agent or MemoCue, since they also address deeper reasoning retrieval.

### Soundness
3

### Presentation
4

### Contribution
3
