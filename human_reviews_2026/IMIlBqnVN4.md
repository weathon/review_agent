# HeliCon: Dual-Level Contrastive Alignment for Robust Medical VQA under Long-Tailed Distribution

- Avg Score: 4.00
- Decision: Reject
- Scores: 8, 2, 2, 4

## Abstract
Learning robust multimodal representations for medical visual question answering (Med-VQA) is challenging due to the imbalanced distribution of semantic concepts. Frequent clinical patterns form robust embedding structures, while rare yet clinically important concepts often yield fragile representations, hindering reliable reasoning. To address this issue, we propose HeliCon, a dual-level contrastive alignment framework that follows a conceptual “double helix” structure. It intertwines two complementary mechanisms: (1) memory banks at the instance and prototype levels, which preserve sample diversity while enforcing semantically meaningful clustering; and (2) contrastive learning objectives at the hard and soft levels, which refine head embeddings and transfer relational knowledge to tail concepts. Collectively, these mechanisms enhance the learning of robust and semantically consistent multimodal representations across both frequent and rare concepts. At inference, a retrieval-augmented mechanism further enriches contextual reasoning by leveraging relevant answer embeddings from the training set. Experiments on Med-VQA benchmarks demonstrate that HeliCon achieves improved performance, with a particularly 3.51$\%$ absolute gain over the state-of-the-art on the PathVQA dataset, highlighting its effectiveness in producing robust representations under long-tailed data distributions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes HeliCon, a framework designed to improve medical visual question answering (Med-VQA) under long-tailed answer distributions, where rare clinical concepts are underrepresented and poorly learned.HeliCon introduces a dual-level contrastive alignment mechanism that enhances the robustness and semantic structure of multimodal embeddings:

Two memory banks 
• an instance-level memory preserving sample diversity, and
• a prototype-level memory maintaining category-wise semantic structure.

Dual contrastive objectives 
• Hard contrastive learning aligns head (frequent) concepts at both instance and prototype levels, and
• Soft contrastive learning transfers relational knowledge from head to tail (rare) concepts via KL-divergence alignment.

Retrieval-augmented inference :enriches test-time reasoning by retrieving semantically related answer embeddings from the training set and integrating them through cross-attention.

Extensive experiments on VQA-RAD, SLAKE, and PathVQA show that HeliCon consistently outperforms prior methods, achieving up to 3.51% absolute gain over the state of the art. The approach yields more structured, semantically coherent, and tail-robust embeddings, improving both answer accuracy and generalisation in medical multimodal reasoning.

### Strengths
Originality:
The paper introduces a dual level contrastive alignment framework that explicitly targets answer distribution bias in medical visual question answering (MedVQA), a dimension largely overlooked by prior work which has mostly focused on question bias or general VQA settings. The idea of using head (frequent) concepts as structural anchors to guide tail (rare) concept alignment through hard and soft contrastive objectives is both conceptually novel and practically valuable. The integration of instance level and prototype level memory banks provides a new, coherent perspective for balancing diversity and semantic clustering in multimodal learning.

Quality:
The methodology is technically sound and well motivated, combining principles from contrastive learning, long tailed recognition, and retrieval augmented reasoning. Each component, memory bank design, dual level contrastive losses, and retrieval based inference, is clearly defined and empirically justified. The experimental evaluation is comprehensive, spanning multiple datasets (VQA RAD, SLAKE, PathVQA), several LLM backbones (GPT2, StableLM, Mistral, LLaMA2), and detailed ablation studies that isolate the contribution of each component. The improvements are consistent, and qualitative analyses (tSNE, correlation statistics) convincingly support the claims.

Clarity:
The paper is exceptionally clear and well structured, adhering closely to ICLR standards. The motivation flows logically from the introduction to the formulation, with helpful diagrams (Figures 1 and 2) illustrating both intuition and architecture. Mathematical notation is consistent, and implementation details are sufficient for reproducibility. The inclusion of explicit training objectives, parameter settings, and model variants strengthens transparency.

Significance:
The contribution is highly relevant to both the medical AI and representation learning communities. It addresses a real world challenge, learning from imbalanced clinically meaningful data, while offering methodological insights that generalise beyond MedVQA to other multimodal or long tailed domains (for example retrieval based reasoning and grounded language models). The demonstrated 3.5 percent absolute improvement on PathVQA and consistent gains elsewhere highlight the framework’s practical utility.

Overall:
A technically rigorous, clearly written, and conceptually well integrated paper. It meaningfully extends contrastive learning to long tailed multimodal reasoning and provides both empirical and structural insights. Its combination of originality, execution quality, and broader applicability makes it a strong and significant contribution for the ICLR community.

### Weaknesses
1. Limited theoretical grounding of the soft alignment mechanism
The proposed prototype level soft contrastive objective, formulated as a Kullback–Leibler divergence between head and tail distributions, is empirically effective but theoretically underdeveloped. The paper does not provide a clear justification for why this formulation is the most appropriate means of transferring relational structure from head to tail concepts. A stronger theoretical analysis or connection to existing frameworks such as knowledge distillation (Hinton et al., 2015) or manifold alignment (Wang and Mahadevan, 2009) would strengthen the conceptual depth of the work. The authors could also provide an empirical comparison with alternative formulations such as Earth Mover’s Distance or cosine similarity based transfer.

2. Incomplete exploration of retrieval augmented inference
Although the retrieval component contributes to performance gains, its design and evaluation remain superficial. The retrieval mechanism is limited to a simple top K search followed by parameter free cross attention, without examining scalability or sensitivity to retrieval noise. There is no comparison with stronger retrieval based baselines such as LLaVA Med or RAG LLMs. A deeper analysis showing how retrieval quality affects answer accuracy and interpretability would make this part more convincing.

3. Lack of computational efficiency and scalability analysis
The framework maintains two memory banks updated throughout training, which could introduce nontrivial memory and time overhead, especially for large medical datasets. However, the paper does not report training time, GPU usage, or the impact of bank size on convergence. Including these results would clarify the practicality of the method in real world clinical settings.

4. Moderate novelty relative to prior contrastive works
While the integration of instance and prototype level alignment is elegant, similar multi level or hierarchical contrastive frameworks have appeared in prior work such as MUMC (Li et al., 2023b) and PMC CLIP (Lin et al., 2023). The main novelty lies in addressing answer distribution bias rather than introducing a fundamentally new learning principle. The paper could strengthen its originality by positioning HeliCon as a more general long tailed contrastive framework and empirically validating it beyond the medical VQA domain, for example on standard VQA v2 or GQA datasets.

5. Missing error and failure case analysis
While qualitative visualizations are presented, the paper does not discuss failure cases where the model retrieves misleading examples or confuses semantically similar but clinically distinct answers. Including a small number of failure analyses with attention maps or embedding plots could provide valuable insight into remaining biases and guide future improvements.

### Questions
Justification of the soft alignment mechanism:

Could you clarify why Kullback–Leibler divergence was chosen as the formulation for transferring head–tail relational knowledge?

Would other formulations (for example, cosine similarity or Wasserstein distance) lead to different behaviour?

A brief theoretical or empirical justification would strengthen confidence that this choice is principled rather than heuristic.

Efficiency and scalability analysis:

The dual memory banks may introduce computational overhead. Could you provide quantitative results on GPU memory usage, training time, and scalability with dataset size?

Clarifying this would help assess whether the approach is practical for large scale clinical deployment.

Retrieval augmented inference evaluation:

How is retrieval implemented and how sensitive is performance to retrieval noise or the number of retrieved samples (K)?

Could you compare your retrieval fusion strategy with standard RAG or knowledge retrieval baselines to demonstrate robustness and generality?

Generalisation beyond medical datasets:

The framework seems general for any long tailed multimodal problem. Have you considered evaluating it on a non medical dataset (for example, GQA or VQA v2) to show broader applicability?

Even a brief experiment or discussion could highlight that the method is not limited to the clinical domain.

Analysis of head–tail balance and parameter sensitivity:

The model’s performance depends on the frequency threshold separating head and tail classes and on weighting parameters λ₁ and λ₂.

Could you include or summarise a sensitivity analysis showing how performance changes with these values?

### Soundness
3

### Presentation
3

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
HeliCon introduces a dual-level contrastive alignment framework for medical VQA that explicitly handles long-tailed answer distributions. It aligns frequent answers via instance- and prototype-level hard losses, while transferring relational knowledge to rare answers through soft KL alignment against head prototypes. A retrieval-augmented inference module enriches decoding with token embeddings from similar training cases. Extensive experiments show consistent gains, especially on rare diseases, and reveal a near-linear embedding space that makes rare samples retrievable and diagnostically precise.

### Strengths
1. The paper is well-structured and clearly written.
2. The experiments are comprehensive and the arguments are complete.

### Weaknesses
1. Regarding the RAI module, the paper appears to omit any dedicated ablation experiments. Moreover, in the existing ablation studies, it is never explicitly stated whether the RAI module remains active after adding SSC or PHC losses. Consequently, readers cannot clearly discern the actual contribution of the RAI module to the model’s performance.

2. The F-test in Table 3 does not provide a corresponding p-value, and significance should be indicated in bold.

### Questions
1. The paper does not clearly define n_thresh. My understanding is that n_thresh denotes the frequency of an answer in the dataset and is used to distinguish between head and tail samples—am I correct?

2. No ablation on the threshold itself. The three QA datasets used in the paper differ drastically in size. Fixing the threshold at 20 could plausibly cause almost all samples in PathVQA to be labeled as head and nearly all samples in VQA-RAD as tail. The authors should therefore report the exact counts of head and tail samples in each dataset. If a threshold other than 20 were used, would the multimodal embedding similarities and answer embedding similarities still fall into the same linear space?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper tackles a key weakness in medical visual question answering (Med-VQA): models perform well on common (“head”) clinical concepts but struggle with rare (“tail”) ones that are often more medically important. This happens because existing vision–language models learn mostly from frequent examples, leading to biased or fragile representations.

The authors propose HeliCon, a new framework that makes these representations more balanced and semantically meaningful. Think of it as a “double-helix” alignment system that connects two strands: Dual memory banks, where one is for individual samples (diversity) and one for concept prototypes (structure). Dual contrastive objectives, where “hard” contrastive learning is used for frequent cases and “soft” alignment for rare ones.

Together, these mechanisms allow the model to learn effectively from both abundant and scarce examples while maintaining coherent relationships between medical concepts across the distribution.

### Strengths
The strength is three-fold across problem statement, experimentation, and inclusion of an ablation. 
- The paper correctly identifies a real and underexplored issue in Med-VQA — long-tailed answer distributions (most prior work focused on question bias). This reframing is fresh and important.
- In experiments, HeliCon shows that this setup works well in practice. It beats strong baselines (LLaVA-Med, MUMC, Med-MoE) and shows consistent open-ended gains, particularly on PathVQA (+3.5%), which is a tough benchmark.
- The authors include ablations.

### Weaknesses
There are four main weakness areas here:
- HeliCon conceptually focuses on the “tails” and provides solid representational evidence that rare samples become more semantically coherent and better integrated in the embedding space. However, in practice, the main numerical gains in recall and accuracy still come from the “head” classes. The tail improvements are visible but relatively modest, suggesting that the method stabilizes learning across the distribution rather than dramatically shifting performance toward the tails.

- The paper claims improved “reasoning,” but relies only on recall and BLEU metrics for open-ended questions. There’s no reasoning-specific or factual consistency test, so the reasoning claim is overstated.

- Dual contrastive levels, memory banks, and retrieval augmentation have each been used in prior VQA or multimodal representation works (e.g., MUMC 2023b; Parashar 2024). HeliCon integrates them nicely but doesn’t introduce a fundamentally new algorithmic insight.

- Performance partly depends on the large biomedical pretraining (600 K pairs). It’s unclear how much of the gain comes from the proposed dual-contrastive method versus the pretrained initialization.

### Questions
You mention pretraining on 600 K image–text pairs from PMC-15M. Which components of HeliCon are included in this pretraining (e.g., visual encoder, projection layers, language model)? How much of the downstream improvement can be attributed to this initialization rather than to the proposed dual-level contrastive alignment?

The paper frequently refers to “reasoning” and “contextual reasoning,” but no reasoning-specific benchmarks or metrics are reported. Could you clarify what definition of reasoning you adopt and how the reported metrics (recall, BLEU-1, F1) reflect reasoning capability rather than lexical overlap?

Your paper emphasizes addressing the tail-side challenge in medical VQA by transferring knowledge from frequent (head) to rare (tail) concepts. However, while the embedding correlation analysis suggests stronger semantic alignment for tail samples, the quantitative gains (e.g., recall and F1 improvements) appear larger for head samples. Could you clarify whether this indicates that HeliCon primarily improves overall embedding structure rather than directly enhancing reasoning accuracy on rare concepts? In other words, how should we interpret the gap between representational improvement and task-level performance for tail samples?

### Soundness
3

### Presentation
3

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
The paper introduces HeliCon, a novel dual-level contrastive learning framework for long-tailed medical visual question answering. It builds instance- and prototype-level memory banks to structure multimodal embeddings via contrastive alignment and employs a Retrieve and Fusion module during inference to recall semantically related samples, thereby improving reasoning and generalization on tail concepts.

### Strengths
1. They achieve high performance compared with the methods used for comparison.

2. The paper presents substantial results showing that the proposed components are effective.

3. The paper is well written, and the proposed method can be reproduced.

### Weaknesses
1. The paper does not include commonly used benchmark methods [A, B] for comparison, particularly those that show high zero-shot performance on the same VQA-RAD, SLAKE, and PathVQA datasets. (It seems that different metrics were used for the comparison methods, so it is difficult for the reviewer to make a fair judgment about the performance.)

[A] Chen, Junying, et al. "Towards injecting medical visual knowledge into multimodal llms at scale." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.

[B] Lin, Tianwei, et al. "HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation." Forty-second International Conference on Machine Learning.

2. The improvements in tail performance shown in Table 3 are marginal. Additionally, the performance on tail samples is particularly important for understanding the effectiveness of the proposed components. Therefore, results for all datasets on the tail samples should be provided.

3. It seems that the model is only applicable to short-answer VQA scenarios. If the answers are longer than a few words (as is common in recent datasets), can the proposed method still perform well in such cases? Otherwise, the contribution of the proposed method is too limited.

4. I acknowledge that head and tail samples are treated separately, but what is the motivation for using contrastive learning to address the long-tail problem instead of other loss functions? Further elaboration regarding this motivation is required. 

5. I wonder if the model would maintain high performance in Table 2 (the ablation study on learning strategies) when RAG is excluded. I am also curious whether contrastive learning truly helps improve model learning or the similarity retrieval process in RAG.

### Questions
1. Provide clear information about the baselines described in the results section.

2. Does varying λ1 and λ1 impact performance, and why were they set to 0.5?

### Soundness
3

### Presentation
3

### Contribution
2
