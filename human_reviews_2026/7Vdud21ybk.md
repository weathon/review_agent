# FineScope: SAE-guided Data Selection Enables Domain-Specific LLM Pruning & Fine-Tuning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Large Language Models (LLMs) are typically trained on diverse, general-purpose datasets, enabling broad generalization but incurring substantial computational costs. However, real-world applications often require efficient models tailored to narrow, domain-specific tasks. In such settings, large model capacity and generality are unnecessary, and traditional fine-tuning pipelines struggle under resource constraints. We introduce FineScope, a framework that addresses this challenge by tightly coupling domain-aware data selection with model pruning and fine-tuning. Starting from a small set of user-provided seed examples, FineScope trains sparse autoencoders (SAEs) on intermediate model activations to automatically extract semantically aligned examples from large unlabeled corpora. The curated dataset then guides structured pruning to preserve domain-relevant substructures and supports self-distillation fine-tuning to recover task-specific performance. Experiments across STEM, humanities, social sciences, math, and coding domains show that FineScope consistently outperforms baseline fine-tuning approaches while enabling up to $35\%$ parameter pruning. On math reasoning tasks, it achieves an average improvement of ~11.50 points across pruned models. Code will be available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FineScope, a sparse autoencoder (SAE)-guided framework for fine-grained data selection in LLM fine-tuning. The main idea is to decompose intermediate representations of a pretrained model into sparse latent features via an SAE, and then use these features to estimate the importance and diversity of training samples. FineScope selects or reweights data based on these latent scores, aiming to retain samples that are semantically rich and representative. Experiments on multiple benchmarks covering reasoning, instruction following, and coding show that FineScope outperforms random sampling and several heuristic data selection methods.

### Strengths
1. The paper addresses a practical and relevant problem, i.e., data selection for efficient LLM fine-tuning.
2. The approach is conceptually simple and leverages sparse representation learning to provide an interpretable basis for sample selection.

### Weaknesses
I have the following concerns. *If the authors could properly address them during the rebuttal phase, I am willing to raise my score.*

1. The methodological novelty is limited. The framework mainly applies an existing sparse autoencoder formulation to compute feature-based data scores, without introducing new architectural or algorithmic components. The contribution feels more like an application of known techniques than a principled innovation.
2. The authors assume that SAE latent dimensions correspond to meaningful semantic directions, yet provides no quantitative or theoretical justification. Without analysis of what these latent factors represent or how they align with domain-level semantics, the interpretability claim remains speculative.
3. The empirical gains are modest and occasionally within variance ranges. The paper lacks statistical significance testing or multiple random seeds, making it hard to assess robustness.
4. Some important recent baselines [1,2,3] are missing. The paper would benefit from either a comparison with these methods or a discussion of their relevance.

[1] Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models. EMNLP 2024.

[2] How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition. ACL 2024.

[3] Boosting Multi-Domain Fine-Tuning of Large Language Models through Evolving Interactions between Samples. ICML 2025.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FineScope, a framework designed for efficiently adapting large, general-purpose language models (LLMs) to specific domains, particularly under resource constraints. Recognizing that standard fine-tuning requires extensive domain data and model compression often degrades performance, FineScope proposes a tightly coupled process. It starts with a few seed examples and uses Sparse Autoencoders (SAEs), trained on intermediate LLM activations, to automatically curate a domain-specific dataset ($D_s$) from a large unlabeled corpus by identifying semantically similar samples. This curated dataset ($D_s$) then guides a structured pruning process to preserve relevant model components and is used in a modified self-distillation fine-tuning (SDFT) step to recover task performance on the pruned model. The authors present experiments across several domains showing that FineScope allows significant pruning (up to 35%) while outperforming baseline fine-tuning methods.

### Strengths
1. The framework addresses the practical and important challenge of creating efficient, domain-specialized LLMs from large generalist models, which is crucial for deployment in resource-limited environments .




2. FineScope introduces a novel integration of data selection, pruning, and fine-tuning, leveraging internal model representations (via SAEs on activations) to guide the entire adaptation process, rather than treating these as separate steps .




3. The approach is data-efficient, requiring only a small set of seed examples to bootstrap the curation of a larger, domain-aligned dataset from unlabeled text, reducing reliance on costly annotated data.

### Weaknesses
1. The overall pipeline is highly complex, involving multiple sophisticated stages. The process includes training numerous SAEs (potentially one per layer ), selecting Top-K activations , processing a large unlabeled corpus for embeddings and similarity search , performing guided structured pruning , generating a distilled dataset via SDFT, and finally fine-tuning. This complexity could hinder reproducibility and practical implementation.

2. The computational cost of the data curation phase may be substantial and is not analyzed. While the final fine-tuning uses a small dataset ($D_s$), the upfront cost of training SAEs and especially processing a large unlabeled corpus (U) to extract activations/embeddings and compute similarities could be very high. Without a cost breakdown, the framework's overall efficiency compared to alternatives is unclear.

3. The framework's performance likely depends significantly on the quality of the SAEs and associated hyperparameters. Effective data curation hinges on the SAEs successfully capturing domain-relevant features from activations. This process might be sensitive to choices like SAE architecture, sparsity penalty ($\lambda$), the specific LLM layers used, and the Top-K activation selection criteria, requiring careful tuning (though Top-K effects are studied).

### Questions
1. The method trains separate SAEs, potentially for each layer. How are the embeddings/features from these multiple SAEs utilized during the data curation step (Section 3.1.3)? Are features from a specific layer chosen, or are they aggregated across layers before similarity computation?

2. The modified SDFT stage involves a teacher model. How critical is the choice and capability of this teacher model for successfully recovering the performance of the pruned student model?

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
4

### Summary
This paper introduces FineScope, a unified framework that combines domain-aware data selection with model pruning and fine-tuning for efficient adaptation of large language models. The approach leverages Sparse Autoencoders (SAEs) trained on intermediate layer activations to extract semantically relevant examples from large unlabeled corpora, starting from a small seed set of user-provided examples. The curated dataset guides structured pruning to preserve domain-relevant substructures and enables modified self-distillation fine-tuning to recover performance. Experimental validation across STEM, humanities, social sciences, mathematics, and coding domains demonstrates that FineScope maintains competitive performance while enabling up to 35% parameter reduction, with notable improvements averaging 11.50 points on mathematical reasoning tasks in pruned models.

### Strengths
1. The paper presents a novel integration of SAE-based interpretable representations for simultaneous data curation and model compression. While SAEs and structured pruning exist independently, their joint application for domain-specific adaptation represents a meaningful contribution. The use of Top-K activation filtering to reduce SAE training overhead while maintaining quality is a practical innovation that addresses scalability concerns.
2. The experimental design is reasonably comprehensive, spanning multiple model architectures (Vicuna-7B, MathCoder-CL-7B, LLaMa 3.1-8B) and diverse evaluation domains. The inclusion of ablation studies examining modified SDFT and pruning dataset effects (Table 4) strengthens the empirical validation. Comparisons against GPT-3 variants and OLMO-7B provide useful context, though the performance gaps suggest room for improvement in certain settings.
3. The paper is generally well-structured with clear motivation and methodology sections. Figure 2 effectively illustrates the two-stage pipeline, making the approach accessible. The mathematical formulations are presented with adequate notation, though some technical details could benefit from expansion.
4. The work addresses a genuine practical challenge - deploying specialized LLMs under resource constraints without access to curated domain datasets. The consistent performance improvements across domains demonstrate practical value, particularly for settings where computational resources are limited.

### Weaknesses
1. The paper lacks theoretical analysis of why SAE embeddings should outperform raw embeddings for data selection beyond empirical results in Table 5. While the claim about "interpretable features" is made repeatedly, there's insufficient explanation of what makes these features more suitable for guiding pruning decisions. The connection between sparsity in SAE representations and domain relevance needs deeper investigation.
2. Despite emphasizing efficiency, the paper provides no runtime or memory analysis comparing FineScope to baselines. Training L separate SAEs (one per layer), computing embeddings for large corpora, and performing cosine similarity calculations introduce non-trivial overhead. The claim of "computational cost" reduction is undermined without concrete measurements. Figure 3 shows training time for different K values but doesn't compare against end-to-end baseline costs.
3. The method introduces several hyperparameters: Top-K value, number of seed examples, SAE sparsity penalty λ, pruning ratio r, yet their sensitivity is inadequately addressed. While K=128 is justified in Section 4.3, the choice of ~10 seed samples and K=100 for final selection (Section 3.1.3) appears arbitrary. How robust is performance to these choices? What happens with 5 vs. 20 seeds?
4. Despite claims of broad applicability, evaluations are restricted to STEM, humanities, social sciences, math, and coding. Critical domains like medicine, law, and finance, where domain-specific LLMs have gained significant traction (referenced in Related Work), remain unexplored. The gap between claiming "domain-specific" adaptation and testing on relatively broad academic categories is notable.
5. Section 3.1.3 states seeds are "representative of the target domain" but provides minimal detail on selection criteria beyond Figure 6's visualization. The reliance on GPT-4 for seed generation (mentioned in implementation details) raises questions about reproducibility and applicability when such resources are unavailable. How would performance degrade with manually selected seeds?
6. The "Random" baseline uses datasets of the same size as FineScope, but there's no comparison against other data selection methods (e.g., embedding-based retrieval without SAEs, gradient-based selection, or uncertainty sampling). The claim of superiority over "traditional fine-tuning pipelines" would be stronger with more comprehensive baselines.
7. Tables 1-3 report point estimates without confidence intervals or significance tests. Given the relatively small performance margins in some cases (e.g., 0.83% difference for LLaMa3.1 STEM between Full-OI and FineScope in Table 1), statistical validation is necessary to confirm the improvements are not due to random variation.

### Questions
1. Why train separate SAEs per layer rather than a unified model across layers? Have you investigated attention-based aggregation of multi-layer representations, which might capture hierarchical domain features more effectively?
2. Can you provide concrete wall-clock time and memory comparisons for the complete FineScope pipeline versus standard fine-tuning? How does the method scale to models beyond 8B parameters, particularly given the need to train L SAEs?
3. What is the performance variance across different seed sets? Have you tested with seeds selected by domain experts versus those generated by GPT-4? Does the method degrade gracefully with fewer or lower-quality seeds?
4. The gradient-based Top-K selection (Equation 6) requires computing gradients during forward passes. Could you clarify the computational implications and compare against simpler magnitude-based selection?
5. How does FineScope handle ambiguous or multi-domain samples? For instance, mathematical biology spans STEM domains- would it be selected for both, and how might this affect specialization?
6. Have you explored reversing the pipeline- fine-tuning first with SAE-selected data, then pruning? Could this retain more domain-relevant parameters?

### Soundness
3

### Presentation
3

### Contribution
3
