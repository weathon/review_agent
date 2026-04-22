# LLM DNA: Tracing Model Evolution via Functional Representations

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 6, 4, 6, 6

## Abstract
The explosive growth of large language models (LLMs) has created a vast but opaque landscape: millions of models exist, yet their evolutionary relationships through fine-tuning, distillation, or adaptation are often undocumented or unclear, complicating LLM management. Existing methods are limited by task specificity, fixed model sets, or strict assumptions about tokenizers or architectures. Inspired by biological DNA, we address these limitations by mathematically defining *LLM DNA* as a low-dimensional, bi-Lipschitz representation of functional behavior. We prove that LLM DNA satisfies *inheritance* and *genetic determinism* and establish its existence. Building on this theory, we derive a general, scalable, training-free pipeline for DNA extraction. In experiments across 305 LLMs, DNA aligns with prior studies on limited subsets and achieves superior or competitive performance on various tasks. Beyond these tasks, DNA comparisons uncover previously undocumented relationships among LLMs. We further construct the evolutionary tree of LLMs using phylogenetic algorithms, which align with shifts from encoder-decoder to decoder-only architectures, reflect temporal progression, and reveal distinct evolutionary speeds across LLM families.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper defines a DNA to LLMs in order to build a distance metric between LLMs to produce very interesting mapping at a fairly low cost (only need a forward pass and no need to retrain a model when adding a new LLM to the map).

### Strengths
- importing concepts from genetics to apply them to LLMs is inspiring
- the method is well documented and math supports the methods
- the subject addressed is important as more and more models are published everyday with lower and lower transparency about training details especially for private models

### Weaknesses
- **Important related work missing** : This paper adresses a nearly identical topic with similar methods than another paper published last year in the same conference https://iclr.cc/virtual/2025/poster/28195 without discussing and comparing with it. An in depth comparison of both methods would be very important as it would provide a good baseline to compare both the theoretical framework as well as results in practice increasing the impact of the paper.
- **Definition of LLM evolution unclear** :The paper defined clearly what is genetic determinism and inheritance but didn't clearly define what is evolution in LLMs while this is a crucial concept in the paper. Is it a model A finetuned from model B ? Is it the general evolution of pretraining sets ? Is it something else ? This concept is central to the analogy and could deserve a much more in depth explanation strengthening even further the already strong basis of the paper.
- **No limitations discussion** : are there cases where the algorithm doesn't work as expected ? It is important to discuss the upsides of the methods but also the parts where it doesn't perform particularly well while explaining why. This can only make the paper more interesting for the reader as well as fostering follow up work.

### Questions
- The method presented used classical benchmarks to test the completion of models on these benchmarks. Some models are suspected to be contaminated on classical benchmarks like MMLU. This could bias the metric making models with different origins appear with a similar DNA. Wouldn't it be better to use text not included in classical benchmarks to avoid this ?
- The algorithm seems to capture very well some relationships between models in Figure 4. However some relationships are not captured such as Meta-Llama-3-8B that is not close to Meta-Llama-3-8B-Instruct while the latter is likely finetuned from the former. What could cause this difference ? (there are many other examples of models that are not close to each other in the figure)
- In Figure 4, clusters are based on organizations that trained the models. However similarity between LLMs could either come from a similar base model (but trained on different finetuning sets) or because they come from different pretraining sets (but were finetuned on the same finetuning set). How does this plays out in the results ? Does one scenario lead to much higher similarity than the other ?

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
3

### Summary
This paper proposes LLM DNA, a low-dimensional representation of large language models obtained through random projection on model outputs. The authors formally define DNA as a bi-Lipschitz embedding, prove its existence via the Johnson-Lindenstrauss lemma, and demonstrate its utility for detecting model relationships across 305 models. Experiments show 95.7% AUC in relationship detection and reveal previously undocumented model relationships through phylogenetic tree analysis.

### Strengths
1. The paper addresses an interesting problem of tracing LLM evolutionary relationships through the DNA concept, demonstrating potential as a practical tool. 

2. The experimental scale of 300+ models is substantial, and the discovery of false negative samples shows promising value.

### Weaknesses
1. The paper's core issue is presenting standard dimensionality reduction through two layers of excessive framing that obscure rather than clarify the contribution:
   - Mathematical formalization: Sections 3-4 occupy too much space establishing Hilbert space formulation, bi-Lipschitz conditions, and applying the Johnson-Lindenstrauss lemma. However, this entire framework is generic, applying identically to image classifiers, recommender systems, or any function class. Theorems 3.4-3.5 contribute no LLM-specific insights, merely restating that this is random projection with standard techniques.
   - Biological framing: The DNA terminology ("inheritance," "genetic determinism" in lines 108-115) lacks citation to biological literature supporting these characterizations. For the ML community, framing this as "low-dimensional manifold learning" would be far more transparent and align with established understanding of dimensionality reduction methods. Why is phylogenetic tree construction necessary for analyzing LLM relationships? Since the method only requires a distance matrix, are there simpler established methods that could achieve similar relationship visualization without requiring biological background knowledge? 

2. Related work inadequately contextualizes the approach. The "Representation of LLMs" in Section 2 mentions model routing and ensemble learning without explaining these tasks or their connection to LLM DNA extraction, leaving unclear how this work differs from or builds upon existing representation learning methods.

3. Baseline comparisons are critically insufficient. This is fundamentally model similarity measurement, yet experiments only compare against "Random" and "Greedy" strawmen plus EmbedLLM.

Overall, I recommend the authors reduce the extensive mathematical and biological framing that appears sophisticated but contributes no novel insights, and instead devote more effort to contextualizing the work within existing literature and conducting more rigorous experiments.

### Questions
1. Phylogenetic trees in biology represent species-level divergence, yet your tree shows iterative model versions like Qwen2.5 and Qwen3 as separate branches. Do you think the biological evolution analogy genuinely fits the nature of LLM development, or does it create conceptual mismatches?

### Soundness
3

### Presentation
2

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
This paper introduces LLM DNA, a formal framework to identify and trace the evolutionary relationships of large language models (LLMs). The authors address the challenge that with millions of LLMs existing, their origins through fine-tuning, distillation, or adaptation are often unclear.

### Strengths
-one of the paper's  strengths is its rmathematical grounding. It doesn't just propose a heuristic; it formally defines LLM DNA using the Bi-Lipschitz condition and uses the Johnson-Lindenstrauss (JL) Lemma to prove its existence.

-Unlike methods that require retraining to compare models, the LLM DNA extraction pipeline is fixed. Once the random projection matrix $A$ and prompt set $\mathcal{S}_t$ are defined, any new model can have its DNA extracted independently4. This makes the system highly scalable and durable.

-The method does not rely on model internals like architecture, weights, or tokenizers. Since it only evaluates text outputs , it works on black-box API models (like GPT-4) just as easily as open-source models, and it can compare models of completely different types

### Weaknesses
-While the Mantel test shows stability for general datasets, the DNA is still a snapshot of the model's behavior on only $t=600$ prompts. If this prompt set were narrowly-scoped (e.g., all prompts were Python coding problems), the resulting DNA would likely represent a "coding DNA" rather than the model's general functional behavior.

-.The DNA measures functional similarity (i.e., "do these models behave alike?"). If two models from different companies were trained independently but happened to achieve a very similar functional state (convergent evolution), their DNAs would be nearly identical.

### Questions
-the paper sets the final DNA dimension $L=128$. How was this number chosen? How sensitive is the DNA's quality to this choice? If $L=32$, would the system lose the resolution to distinguish highly similar models (e.g., Llama-3-8B vs. Llama-3-8B-Instruct)? 

-How sensitive is the method to minor fine-tuning? If I fine-tune Llama 3 on just 1,000 new samples, how much will its DNA $\tau_{f_2}$ change from the original $\tau_{f_1}$? Is the DNA distance $d_\tau$ predictably correlated with the amount or domain of fine-tuning data?

-The study includes models from 100M to 70B parameters. Is the method equally effective across this entire range? Does using a powerful embedding model $\phi$ on the (often lower-quality) outputs of a 100M model "over-beautify" its functional representation, making its DNA seem more capable than it is?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes LLM DNA, a mathematically grounded, low-dimensional representation intended to capture an LLM’s functional behavior. The approach: (i) define DNA via a bi-Lipschitz map over a space of LLM functions, (ii) justify existence with a Johnson–Lindenstrauss (JL) argument, and (iii) provide a training-free pipeline. Empirically, DNAs cluster related models and support simple routing.

### Strengths
- The bi-Lipschitz view + JL existence yields a clean, theory-backed rationale rather than an ad-hoc embedding and is well principled.
- The training-free pipeline is straightforward, practical, and appears to work across heterogeneous and API-only models.
- The high-level theory (bi-Lipschitz definition, JL-style existence, and concentration arguments) appears reasonable for finite settings.
- The method is agnostic to architecture and tokenizers, and doesn't require rebuilding an embedding space when adding LLMs to the evaluation set. This means it can be extended to the closed-source LLMs and can be used to uncover undocumented relationships between models (p7-8, Appendix C.2), which is an advantage over EmbedLLM.

### Weaknesses
- "DNA" quality likely hinges on the chosen sentence-embedding backbone and its dimensionality. Robustness to encoder swaps (and multilingual/domain shifts) is not yet demonstrated.
- There is an insignificant difference in performance for the model routing test between the standard baselines.
- The motivation for the benefits that this type of representation unlocks (namely, phylogenetic analysis on models) is not sufficient. The authors show and discuss a phylogenetic tree, but discuss the genetic evolutionary rate without drawing the connection to its significance in this context and seem to be borrowing from authority of biological genetic evolutionary analysis.
- The mathematical guarantees are for finite sets and bounded inputs; it’s unclear how distortion and stability behave as input lengths or the set of models grow. Please discuss the unbounded-input case and evolving model pools.

### Questions
- How can the trees be useful in practice?
- Since there is only a small performance difference, are there other benefits to using this method over others?
- How do results change with different sentence-embedding models and embedding dimensionalities?
- What is the empirical trade-off between projection dimension d, distortion of pairwise distances, and downstream tasks (lineage/routing)?

### Soundness
3

### Presentation
3

### Contribution
3
