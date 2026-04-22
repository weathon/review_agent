# Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Inductive Knowledge Graph Reasoning (KGR) aims to discover facts in open-domain KGs containing unknown entities and relations, which poses a challenge for KGR models in comprehending uncertain KG components. Existing studies have proposed Knowledge Graph Foundation Models (KGFMs) that learn structural invariances across KGs to handle this uncertainty. Recently, Large Language Models (LLMs) have demonstrated strong capabilities for open-domain knowledge reasoning. As a result, the latest research has focused on LLM-based KGFMs that integrate LLM knowledge with KG context for inductive KGR. However, the intrinsic knowledge of LLMs may be overshadowed by sparse KG context, leading to LLM knowledge distortion, which can cause irreversible damage to model reasoning. Moreover, existing LLM-based KGR methods still struggle to fully constrain generative hallucinations in LLMs, severely limiting the credibility of reasoning results. To address these limitations, we propose a Knowledge Reasoning Language Model (KRLM) that achieves unified coordination between LLM knowledge and KG context throughout the KGR process. Specifically, we design a Knowledge Reasoning Language (KRL) instruction format and a KRL tokenizer to align LLM knowledge with KG representations. Then, we propose a KRL attention layer that coordinates intrinsic LLM knowledge with additional KG context through a dynamic knowledge memory mechanism. Finally, a structure-aware next-entity predictor is proposed, which strictly constrains the reasoning results within a trustworthy knowledge domain. Extensive experimental results on 25 real-world inductive KGR datasets demonstrate the significant superiority of the proposed KRLM in both zero-shot reasoning and fine-tuning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents the Knowledge Reasoning Language Model (KRLM), a novel framework for inductive knowledge graph reasoning (KGR). The primary goal is to address the "knowledge distortion" problem in existing LLM-based approaches, where sparse contextual information from a knowledge graph (KG) can interfere with or override the dense intrinsic knowledge of the LLM. KRLM aims to create a more unified coordination between the LLM's internal knowledge and the external KG's structural knowledge.

### Strengths
The paper clearly identifies and articulates the "knowledge distortion" problem, which is a significant and practical challenge in combining LLMs with KGs. The goal of unifying the two knowledge sources is highly relevant.

The proposed KRLM is a sophisticated system where each component is designed with a clear purpose that ties back to the central goal of knowledge coordination. The KRL instruction format, the memory-augmented attention, the structured predictor, and the mutual distillation loss all work in concert. This is a significant engineering and research effort.

### Weaknesses
The proposed KRLM architecture is very complex. It involves multiple GNNs (for relations, entities, and the projection head), a modified attention mechanism, and a custom tokenizer, all integrated with a base LLM. While the results are strong, this complexity raises questions about its practical scalability and efficiency. The computational complexity analysis in Appendix E confirms that this is a heavy model. Could the authors comment on the trade-offs between this complexity and the performance gains? Is it possible to achieve similar benefits with a simpler architecture?

The experiments are based on Llama2-7b. While this is a reasonable choice, the field of LLMs is moving incredibly fast. How dependent are the architectural choices and performance gains on this specific base model? Would the same knowledge coordination mechanisms be as effective or even necessary with more advanced, capable models (e.g., Llama-3, Mistral, etc.) that might have better inherent reasoning and context-handling abilities?

### Questions
see weaknesses

### Soundness
2

### Presentation
2

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
This paper presents KRLM, an LLM-based foundation model for inductive knowledge graph (KG) reasoning. It aims at mitigating the issue of knowledge distortion. KRLM introduces several key components: a custom knowledge reasoning language (KRL) instruction format and tokenizer, a KRL attention layer that dynamically integrates intrinsic LLM knowledge with KG context through a memory mechanism, and a structure-aware next-entity predictor. Extensive experiments across 28 datasets demonstrate that KRLM consistently outperforms state-of-the-art models in both zero-shot and fine-tuning settings.

### Strengths
- The paper proposes a well-integrated architecture that effectively aligns LLM representations with KG structure.

- The proposed method shows consistent performance gains across a wide range of datasets, particularly in inductive settings, with thorough comparisons to both structural and LLM-based baselines.

### Weaknesses
- The contribution is primarily empirical; a stronger theoretical justification for how KRLM’s architectural choices address knowledge distortion would enhance the paper’s depth.

- While KRLM employs a memory-efficient tokenizer, the paper lacks discussion on computational efficiency, including training time and parameter overhead compared to prior models.

- The model heavily relies on KRL-format instructions, yet the design choices—such as instruction length, style, and vocabulary—are not systematically analyzed.

### Questions
- The concept of “knowledge distortion”, while intuitive, is loosely defined. Can it be quantified or diagnosed more rigorously? A deeper analysis would help ground this notion.

- Table 1 emphasizes accuracy-based metrics (e.g., MRR, Hits@10), but omits key efficiency metrics such as FLOPs, memory footprint, and wall-clock time for fine-tuning vs. pretraining. These are crucial for assessing practical viability.

- The paper would benefit from qualitative examples illustrating how KRLM produces more faithful or interpretable reasoning compared to baseline LLMs.

### Soundness
3

### Presentation
2

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
The paper proposes KRLM, a Knowledge Reasoning Language Model for inductive knowledge graph reasoning where entities/relations may be unseen. KRLM aims to coordinate intrinsic LLM knowledge with sparse KG context to mitigate knowledge distortion and reduce hallucinations. The method introduces (i) a KRL instruction format and a KRL tokenizer that map entities and relations into unified tokens aligned with LLM text representations; (ii) a KRL attention layer with a dynamic knowledge memory to balance LLM priors and KG evidence during reasoning; and (iii) a structure-aware next-entity predictor that constrains decoding to a trusted knowledge domain. Across 25 real-world inductive KGR datasets, KRLM is reported to outperform prior methods in both zero-shot and fine-tuned settings.

### Strengths
1.The paper focuses on an important pain point: leveraging LLM prior knowledge without letting sparse KG signals distort or override it, while also constraining generative hallucinations.

2.The KRL instruction + tokenizer provide a concrete alignment mechanism between symbolic KG elements and LLM token space, which is practical and reusable beyond a single dataset.

3.The KRL attention layer with dynamic memory is a sensible architectural step toward balancing textual priors and structured context, and the design is easy to ablate.

4.Results on 25 inductive benchmarks (zero-shot and fine-tuning) suggest robustness across datasets and settings, which strengthens external validity.

### Weaknesses
1.While the KRL interface and attention layer are coherent, the overall contribution currently reads as a well-engineered integration of known ingredients. The paper needs to sharpen what is theoretically or algorithmically new.

2.There is no efficiency study, yet the method introduces extra tokens (KRL), a memory mechanism, and constrained decoding. It is needed to provide per-query token counts, latency, and memory usage broken down by components, and compare to strong LLM-based KGR and KGFM baselines under matched budgets. 

3.Ambiguity around the “dynamic knowledge memory.” It’s not clear how the memory is constructed, updated, and queried. How are conflicts between LLM priors and KG entries resolved? 

4.The paper should specify how KRL embeddings, attention layers, and predictor parameters are initialized (random, vocabulary-tied, or from pretrained adapters), and whether the backbone LLM is fully fine-tuned, LoRA-adapted, or frozen. Include sensitivity to model scale and initialization choices.

5.The central claim, coordinated KRL reduces LLM knowledge distortion, needs direct measurement. It is needed to add controlled ablation studies that vary KG sparsity, conflicting triples, and noisy relations, reporting distortion/hallucination metrics.

### Questions
How exactly does the structure-aware predictor enforce structural constraints?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the KRLMl, a novel framework designed to address the problem of knowledge distortion in LLM-based KGR models. Specifically, KRLM aims to integrate the inherent knowledge of LLMs with the structural context of KGs to improve the accuracy and reliability of fact reasoning in open-domain KGs. The proposed model shows strong performance in both zero-shot and fine-tuned inductive KGR tasks, outperforming existing models in several benchmark datasets.

### Strengths
1. The paper proposes a sophisticated mechanism for integrating the structural knowledge of KGs with the intrinsic knowledge of LLMs.
2. The experiment results indicate KRLM's superiority in both zero-shot reasoning and fine-tuning scenarios, outperforming several state-of-the-art models.
3. The model is well-grounded in both theory and practice, with clear explanations of the new modules

### Weaknesses
1. The reasoning complexity of KRLM, particularly during fine-tuning and inference, is a significant concern. While the model shows excellent performance, the computational overhead may limit its practical deployment in large-scale real-time applications. The authors briefly mention the computational complexity but do not provide enough detail.
2. The paper does not sufficiently address how the model would perform or be adapted to environments with limited computational resources or sparse KGs.
3. Tranditional knowledge graph reasoning methods need to be dicussed.

### Questions
The paper assumes that the knowledge memory will contain relevant information for reasoning. However, in cases where the relevant entities or relations are not part of the memory, will the model's performance dramatically drops?

### Soundness
4

### Presentation
3

### Contribution
3
