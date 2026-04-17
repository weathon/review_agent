# Supervised Graph Contrastive Learning for Gene Regulatory Networks

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Graph Contrastive Learning (GCL) is a powerful self-supervised learning framework that performs data augmentation through graph perturbations, with growing applications in the analysis of biological networks such as Gene Regulatory Networks (GRNs). The artificial perturbations commonly used in GCL, such as node dropping, induce structural changes that can diverge from biological reality. This concern has contributed to a broader trend in graph representation learning toward augmentation-free methods, which view such structural changes as problematic and to be avoided. However, this trend overlooks the fundamental insight that structural changes from biologically meaningful perturbations are not a problem to be avoided but a rich source of information, thereby ignoring the valuable opportunity to leverage data from real biological experiments.
Motivated by this insight, we propose SupGCL (Supervised Graph Contrastive Learning), a new GCL method for GRNs that directly incorporates biological perturbations from gene knockdown experiments as supervision. SupGCL is a probabilistic formulation that continuously generalizes conventional GCL, linking artificial augmentations with real perturbations measured in knockdown experiments and using the latter as explicit supervisory signals.
To assess effectiveness, we train GRN representations with SupGCL and evaluate their performance on downstream tasks. The evaluation includes both node-level tasks, such as gene function classification, and graph-level tasks on patient-specific GRNs, such as patient survival hazard prediction. Across 13 tasks built from GRN datasets derived from patients with three cancer types, SupGCL consistently outperforms state‑of‑the‑art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a new method called SupGCL for learning representations of gene regulatory networks (GRNs).

- SupGCL directly incorporates real biological perturbation data (gene knockdown experiments) as supervision in the contrastive learning setup.

- Instead of only using random augmentations, they use measured changes in network structure (after the perturbation) as positive pairs, aligning the representation learning more closely with biologically meaningful changes.

- The paper evaluates SupGCL on multiple different tasks (both node-level and graph-level) derived from patient GRN data across multiple cancer types and shows that SupGCL consistently outperforms state-of-the‐art baselines.

### Strengths
### 1. Technical Soundness

- The paper presents a probabilistic generalization of traditional Graph Contrastive Learning (GCL), introducing biologically supervised augmentations via gene knockdown data. The authors mathematically link supervised and unsupervised contrastive objectives through the augmentation temperature parameter ($\tau_a$) and establishes that standard GCL is a limiting case when $\tau_a \to \inf$.

- Algorithm 1 provides a clear training loop using importance sampling, softmax-based probability modeling, and AdamW optimization, improving computational tractability. 

- The method transforms artificial augmentations (random node/edge dropout) into biologically grounded perturbations (gene knockdowns) that serve as supervisory signals.

- Loss functions are well-defined for both node- and augmentation-level contrastive objectives, with consistent normalization and temperature scaling across levels. The inclusion of a theoretical corollary (linking SupGCL to GRACE) adds credibility.

### 2. Experimental Soundness

- SupGCL is evaluated on multiple downstream tasks across multiple cancer types, covering node-level and graph-level settings. This demonstrates robust empirical coverage.

- The authors compare SupGCL against multiple baselines, ranging from traditional GCL methods, augmentation-free approaches, to supervised GRN inference models.

- The $\tau_a$ ablation study validates the theoretical claim: as $\tau_a$ increases (less supervision), performance converges to that of GRACE, supporting the model’s theoretical soundness.

- Embedding visualization and clustering metrics (NMI, ARI) show that SupGCL yields clearer subtype separation than other baselines, confirming representational quality.

### Weaknesses
### 1. Technical Limitations

- The framework relies on sampling-based estimation of normalization constants and importance sampling of probabilities. While efficient, it introduces stochastic noise that might affect convergence stability. No convergence analysis or complexity bounds are provided.

- Although knockdown data is used as biological supervision, the mapping from experimental perturbations (teacher GRNs) to graph-level augmentations is only heuristically justified, not theoretically proven to represent equivalent "positive pairs."

- The model assumes directed graphs with homogeneous node types and dense connectivity. However, GRNs are often sparse, hierarchical, and multi-modal. This could challenge generalizability beyond the datasets used.

### 2. Experimental Limitations

- Improvements are generally small (1–3%), and statistical significance is not tested across all metrics. The paper acknowledges that SupGCL "did not achieve statistically significant superiority in every single task."

- Cross-domain generalization is not good. Pre-training on one cancer type does not generalize to others — the model "fails to improve performance on downstream tasks for other cancer types"

- Although the algorithm uses sampling for efficiency, the paper lacks runtime comparisons or scalability analysis relative to baseline GCL methods.

### Questions
See the weakness section. In addition:

- SupGCL’s dual-graph supervision (G–H pairs for each gene knockdown) may scale poorly to full-genome datasets with thousands of genes.

- The approach requires availability of knockdown experiments; many genes lack this data, limiting applicability.

- Importance sampling introduces stochasticity that may bias gradients during optimization.

- Although biologically motivated, the paper doesn't rigorously formalize why knockdown graphs are "valid augmentations" in the contrastive sense.

- The paper does not clarify whether directionality is preserved in the encoder's message passing, which could distort regulatory logic.

- Evaluations are restricted to three cancer types and a single knockdown dataset. Results may not generalize to other tissues, diseases, or GRN reconstruction pipelines.

- Although several baselines are compared, there’s no inclusion of modern biological foundation models or multi-omics graph frameworks (e.g., MUSE-GNN).

- Variance in learned embeddings or predictive uncertainty (important for biological tasks) is not explored.

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
2

### Summary
This paper proposes a supervised graph contrastive learning framework named SupGCL for cancer drug response prediction based on gene regulatory networks (GRNs). The method attempts to leverage gene knockdown experimental data as supervision signals to guide the graph contrastive learning process, aiming to improve performance on downstream tasks.

### Strengths
1. The authors' integration of gene regulatory networks with drug response prediction represents a promising interdisciplinary research direction.
2. The paper covers both graph-level and node-level tasks, demonstrating extensive experimentation.
3. The idea of using real gene knockdown data as supervision signals is forward-looking and innovative.

### Weaknesses
1. The paper claims to have "theoretically proven that existing GCL methods are special cases of the proposed SupGCL." This statement is logically untenable. SupGCL is a highly domain-specific method whose core relies on GRNs as edge features and biological perturbation data as supervision signals. In contrast, general graph contrastive learning methods (e.g., GraphCL, GRACE) are designed to be universal and do not depend on such specific prior knowledge or external experimental data.

2. The paper lacks comparisons with some recent state-of-the-art unsupervised baselines. Furthermore, is it fair to compare a method using supervision signals against unsupervised GCL? Supervised contrastive learning [1] itself is not a novel concept. The authors are advised to discuss the similarities and differences between their approach and existing supervised contrastive learning frameworks.

3. If the authors believe that artificial data augmentations (e.g., random edge dropping) disrupt the biological realism of GRNs, while knockdown experiments represent genuine biological perturbations, then forcing the model to bring together a "distorted view" and a "true perturbed view" is logically contradictory. This is equivalent to requiring the model to treat a "noisy, unrealistic" state as "similar" to a "real, meaningful" state. Such a design may lead the model to learn merely a "denoising" capability rather than authentic perturbation-response patterns, potentially even corrupting the learned representations. The authors need to provide further clarification on this point.

4. The current contrastive framework of SupGCL only exploits the "correlation" in knockdown data, learning the association between "knocking down Gene X" and "gene expression change Y", and fails to fully leverage the potential of causal data, remaining at the level of "correlational learning."

5. Although the paper analyzes robustness to noise in estimated GRNs, it completely overlooks a discussion on the quality of the knockdown data itself. The performance of SupGCL heavily depends on the knockdown efficiency, off-target effects, and reproducibility of the knockdown experiments. If the knockdown data is of poor quality (e.g., low efficiency, severe off-target effects), the supervision signal itself becomes biased, a problem more fundamental than GRN estimation noise. The authors are recommended to include a discussion on the quality of knockdown data.

[1] Khosla P, Teterwak P, Wang C, et al. Supervised contrastive learning[J]. Advances in neural information processing systems, 2020, 33: 18661-18673.

### Questions
See the weaknesses above.

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
4

### Summary
This paper introduces SupGCL, a supervised graph contrastive learning framework tailored for Gene Regulatory Networks (GRNs). Unlike traditional augmentation-based or augmentation-free GCL methods, SupGCL leverages real biological perturbations—specifically, gene knockdown experiments—as supervisory signals to guide both node-level and augmentation-level contrastive learning. The authors present a unified probabilistic framework that generalizes traditional GCL as a special case and validate their approach through extensive experiments across 13 downstream tasks spanning three cancer types. Across all evaluations, SupGCL consistently outperforms existing baselines.

### Strengths
Novel Supervision Source: The use of gene knockdown data as a real-world supervisory signal for contrastive learning is both innovative and biologically meaningful.

Theoretical Generalization: The framework extends GCL into a probabilistic supervised model, showing that prior unsupervised GCL methods are special cases. Theoretical proofs and ablation studies reinforce this claim.

Experimental Rigor: The study includes comprehensive experiments across multiple tasks, cancer types, and baselines, providing strong empirical support.

Biological Relevance: By grounding augmentations in actual biological perturbations, SupGCL maintains biological fidelity—crucial for GRN interpretability and biomedical relevance.

### Weaknesses
Limited Generalizability: As noted by the authors, SupGCL trained on one cancer type does not transfer effectively to others, limiting its broader biomedical applicability.

Dependence on External Data: The framework’s reliance on knockdown data from LINCS constrains its use to settings where such experimental data exist, reducing scalability for rare or novel conditions.

Modest Gains in Some Tasks: Although performance improvements are consistent, the magnitude of gains is modest in certain node-level tasks, prompting consideration of the tradeoff between added supervision and practical benefit.

### Questions
Cross-Cancer Generalization: Incorporating domain adaptation techniques or pan-cancer pretraining could enhance the model’s transferability across different cancer types.

Perturbation-Free Extension: Developing a fallback mechanism—such as simulated perturbations informed by biological pathways—would make SupGCL applicable when knockdown data are unavailable.

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
4

### Summary
This paper aims to improve upon standard graph contrastive learning (GCL) paradigm particularly for gene regulatory networks (GRNs).
The main motivation for this paper is that, standard GCL which utilize artificial random augmentations (node dropping, edge dropping, feature perturbation) can produce structures that diverge from "biological reality" and the representations learned lack any kind of biological supervision /signal during pre-training that might be beneficial to be encoded for down stream tasks.

Based on this insight, the authors propose supervised GCL (SupGCL), particularly for GRNs where they utilize real experimental gene knockdown data as supervision (via "teacher GRNs") during the contrastive learning phase. SupGCL minimizes the KL divergence between the similarity distributions of real and simulated gene knockdowns, at both node and graph levels, so the authors claim that the learned graph embeddings respect true biological perturbation structure.

The authors argue that with their method is superior to augmentation-free GCL approaches as they completely ignore structural augmentations. The authors also show that when the temperature param tends to infinity in SupGCL we recover back the standard node level GCL contrastive objective, allowing one to control the amount of "biological" supervision.

Empirically, the authors utilize LINCS and TCGA data, which each contain ~1K real gene knockdown experiments for pretraining supervision GCL.
They consider both node level and graph level downstream classification tasks. Both these task follow the standard protocol of finetuning with MLP heads before 10-fold CV evaluation.

### Strengths
1) Clear motivation for the approach of using teacher GRNs via real biological experimental data. The loss formulation is sound, and paper provides good intuition for it. The formulation is novel and an extension to GCL when we have access to such supervision data.

2) Paper is easy to follow and detailed.

3) Empirically shows SupGCL with low temperatures values surpasses GRACE (standard node level GCL) and with high temp values approaches its performance. This experiment / ablation partially supports the authors claims.

### Weaknesses
1) While the main motivation is valid, the loss formulation and use of supervision goes against the goal of self-supervised pre-training which standard GCL follows. The proposed formulation assumes availability of real world biological data (e.g., gene knockdown data). The authors don't discuss the ease / cost of obtaining them. The main benefits of standard GCL which is large scale self-supervision is lost with SupGCL.


2) Experiments suggest the improvement SupGCL obtains is very marginal across the different node and graph level tasks. The difference particularly with the best baseline is within 1 std deviation for a majority of the tasks. This does point that the supGCL loss formulation is not very optimal and weakens the authors claim that SupGCL consistently outperforms SoTA baselines.

3) The authors do not discuss the results in Table 2 and 3 in detail. This is the main experiment to back the claims the authors have in the paper. What is the intuition for w/o - pretrain for e.g., is primarily (5/9) being the best baseline. Do the chosen downstream tasks not require expensive GCL pre-training ?.

4) There is a lot of existing literature on learning structural augmentations in a self-supervised fashion that have shown superior performance compared to standard GCL. This work does not consider them in experiments nor discuss them. Learned augmentations have the benefit of pre training without costly supervision which is the case for this work. [e.g., 1,2,3]

5) One of the goals of standard GCL formulation is also that it can help in transfer learning. And many of the baseline methods evaluate pre-training representations in this setting. The authors don't discuss if this is even applicable for their formulation.

Refs
[1] Suresh, S., Li, P., Hao, C., & Neville, J. (2021). Adversarial graph augmentation to improve graph contrastive learning. Advances in Neural Information Processing Systems, 34, 15920-15933.

[2] Yin, Y., Wang, Q., Huang, S., Xiong, H. and Zhang, X., 2022, June. Autogcl: Automated graph contrastive learning via learnable view generators. In Proceedings of the AAAI conference on artificial intelligence (Vol. 36, No. 8, pp. 8892-8900).

[3] Feng, S., Jing, B., Zhu, Y. and Tong, H., 2024. Ariel: Adversarial graph contrastive learning. ACM Transactions on Knowledge Discovery from Data, 18(4), pp.1-22.

### Questions
1) GRNs need to be estimated for the "teacher" views as well which encode the results of the knockdown experiment. It would good for the authors to comment on how expensive SuPGCL is in end-to-end computational complexity compared to self-supervised approaches such as GRACE / SGRL and even simpler and baselines like w/o - pretrain.

2) Other than zeroing out features for nodes and surrounding edges, were other artificial augmentations tried to simulate knockdown genes. Like for example 1) randomizing the knockdown node features 2) removing a portion of the edges surrounding the knockdown node etc. The question is more about the motivation of the chosen approach for simulation.

### Soundness
3

### Presentation
4

### Contribution
3
