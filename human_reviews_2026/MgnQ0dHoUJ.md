# UPrompt: Bidirectional Multi-granularity Learning for Vision-Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The prompt learning paradigm for vision-language models is effective yet faces the dilemma of balancing granularity: global prompts lack fine-grained semantic awareness, while local prompts ignore overall contextual associations, leading to limited cross task generalization. This dilemma exists in dense prediction tasks.
Inspired by the U-Net framework that unifying multi-level representations across different granularities, we propose UPrompt, a novel bidirectional multi-granularity prompt learning framework for vision-language models.
Similar to how U-Net integrates fine and coarse features through symmetric encoder-decoder pathways with cross-level connections, UPrompt constructs parallel multi-granularity representations in both visual and textual modalities, where coarse-to-fine cascaded enhancement propagates global contextual information to refine local details, while fine-to-coarse hierarchical supervision ensures semantic consistency across scales. 
Extensive experiments on 17 benchmarks validate our effectiveness. Our method outperforms MAMET and VPKE by +4.1 and +7.3 rSum on MSCOCO, surpasses CoCoA-Mix by +5.09\% in base-to-novel generalization, while maintaining competitive performance with minimal overhead (coarse-grained) and matching PSRC with 1/3 cost (medium-grained).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a coarse-to-fine prompt learning approach that considers different levels of granularity. A set of learnable embeddings are progressively downsampled and upsampled through a UNet, with per-level exits for each granularity. Using two similar UNets for each modality, a per-level supervision can be applied to each granularity. The outputs, along with the visual and text features at each granularity, are then forwarded to the corresponding encoders, giving rise to the standard similarity loss. The results show marginal improvement w.r.t. existing methods, and ablation studies showcase the contribution of each of the elements introduced in the paper.

### Strengths
The paper proposes an interesting paradigm to prompt learning whereby different levels of granularity are combined and can be used according to the task under consideration. There is some merit in the method, as well as novelty, that are worth considering.

### Weaknesses
The paper is poorly written and notation is loosely used. Figure 2, which is essential to the understanding of the method, lacks a proper legend and flow explanation. It is not clear to me for example if the visual representations from the input image are computed at different granularities or these are directly taken from the CLIP’s last layer output. My understanding is that only E^K is learnable and the rest of the layers are computed from the previous ones, but I am not completely sure from examining Figure 2. It would be also good to mention in Section 3 how the prompts are gradually made coarser or finer (it is just mentioned in l. 295 that Llava is used). 

In l. 182 it is mentioned that “Visual features integrate granularity-specific prompts”. How? How are the granularity-specific prompts defined? 
In l. 192 it is mentioned how the alignment is measured in a granularity-specific layer, but it is still not clear at this point exactly what is being learned and what is computed from pooling the learned parameters. Please clarify.
In l. 199-202 it is mentioned that “Simple granularity stacking in (…). To address these challenges (…) (Fig. 2)”. How exactly this is depicted in Fig. 2 is not clear to me, given the loosen informative nature of Fig. 2.
Is there any form of attention between the input image and the image embeddings during the forward to the UNet? It is not clear from Fig. 2 if this is actually the case. I understand that different level of visual features are used but it is not clear how. For the text case, this is more clear as it is specifically mentioned in Fig. 2. 
My understanding considering the above is that the learnable prompts are a 14x14 tensor with M (undefined?) channels, that are forwarded to a 4-layer UNet producing in the decoding part a set of 4 level-specific prompts. The UNet, along with the embeddings, is learned using per-level supervision and the full prompt learning Lguide loss. Please clarify. 

The results in Table 2 are not really promising compared to state of the art works. 
How sensitive is the method to using a different caption rewriter (i.e. instead of Llava)? Similarly, how sensitive is the method to using a different prompt resolution (i.e. beyond 14x14)? 

In summary, the writing and presentation give rise to many concerns and doubts regarding the reproducibility of the proposed approach, which need further clarification. The method is technically sound and seems to produce reasonable results, and therefore I am borderline, leaning towards accept, with this paper, hoping for further clarification from the authors.

### Questions
Please refer to the weaknesses section

### Soundness
3

### Presentation
2

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
This paper proposes UPrompt, a U-Net-inspired framework to fix the granularity trade-off in VLM prompt learning—where global prompts miss fine details and local prompts lack global context. It uses two key components: Coarse-to-Fine Cascaded Enhancement (CE, injects global context into fine features via cross-attention) and Fine-to-Coarse Hierarchical Supervision (HS, uses finest-grained alignment to regularize coarser levels). Tested on 17 benchmarks, UPrompt outperforms baselines in cross-modal retrieval (e.g., 571.1 rSum on Flickr30K), few-shot classification (85.13% 16-shot accuracy), and OOD generalization, while offering performance-efficiency flexibility.

### Strengths
1. Targets a clear, unaddressed gap: single-granularity limits in VLM prompting, with direct links to performance flaws in existing methods.

2. Innovative U-Net adaptation: modality-specific granularity (spatial pooling for vision, semantic enrichment for text) plus bidirectional flow—backed by theoretical proofs (Propositions 1-2) rare in multi-granularity prompt work.

3. Rigorous experiments: ablations isolate CE/HS value, efficiency analyses show UPrompt-M matches PSRC’s accuracy with 1/3 cost, and visualizations confirm CE/HS work as intended.

4. Practical: adaptive granularity lets users pick coarse (low cost) or fine (high performance) setups.

### Weaknesses
1. Manual granularity design: 4 levels for classification/3 for retrieval are chosen without guiding heuristics, reducing usability for non-experts.

2. Llama 3-8B dependence: no tests on smaller LLMs (e.g., Llama 3-1B) to see if text hierarchy quality holds, or how LLM overhead offsets UPrompt’s efficiency gains.

3. Limited HS failure analysis: reversed “Coarse-to-Fine Supervision” performs poorly, but no examples (e.g., bad attention maps) show why coarse signals mislead fine modeling.

### Questions
1. Do you have preliminary data on how granularity count (3,5,6) affects tasks like FGVCAircraft vs. MSCOCO? Can you give a simple heuristic for choosing levels?

2. How does swapping Llama 3-8B for smaller models (e.g., T5-small) hurt/help text granularity and downstream performance? What’s the LLM’s share of UPrompt’s total compute?

3. If the finest-grained alignment has errors (e.g., mislabeled pairs), does HS spread those errors to coarser levels? Any tests on HS robustness?

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
5

### Summary
The paper presents UPrompt, a U-Net–inspired multi-granularity prompt-learning framework for adapting Vision–Language Models. It addresses the well-known granularity dilemma in prompt learning—global prompts capture overall semantics but miss fine detail, while local prompts capture details but lose global context. Extensive experiments on 17 benchmarks show consistent improvements.

### Strengths
- This paper is well organized and easy to follow.
- Introduces U-Net's philosophy into multi-granular prompt learning, exploring bidirectional information flow across modal granularities with demonstrated effectiveness;
- Multi-granularity attention maps qualitatively support the claims of semantic consistency and contextual coherence.

### Weaknesses
- Limited Novelty. Several recent works like TAP and HiCroPL, already explore multi-level or hierarchical prompts. UPrompt mainly formalizes these ideas within a U-shaped structure rather than introducing a fundamentally new mechanism. Besides, in the design, the number of granularities and textual levels are manually predefined, which lacks adaptive granularity selection, which limits scalability and automation.
- The textual side depends on Llama-3 generation heuristics. This may inject bias and complicate reproducibility. The paper does not analyze sensitivity to prompt-generation quality.
- The methodologies for constructing multi-level image and text granularities require further enrichment. Current approaches primarily rely on pooling and text attribute addition, lacking comprehensive comparisons between different granularity construction techniques
- In the granularity ablation study (Figure 4, left), performance continues to rise. It remains uncertain whether the peak performance has been reached, and how different granularity interval strategies might affect results

### Questions
Please highlight the significant novel designs.

### Soundness
2

### Presentation
2

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
This paper proposes UPrompt, a U-Net-inspired bidirectional multi-granularity prompt learning framework for vision–language model (VLM) adaptation. The core ideas are:
1. Coarse-to-Fine Cascaded Enhancement (CE) – injects global context from coarse layers into fine layers;
2. Fine-to-Coarse Hierarchical Supervision (HS) – distills knowledge from the finest layer to coarse ones to mitigate semantic drift.

The method achieves consistent gains across cross-modal retrieval, few-shot classification, base-to-novel generalization, and out-of-distribution (OOD) tasks, while enabling a controllable trade-off between accuracy and computational cost.

Overall, this paper proposed a reasonable idea, and the results looks reasonable. The multi-granularity + bidirectional flow idea is intuitive and broadly applicable. If the authors can supplement comparisons with stronger backbones, provide the resolution × layer ablation, and clarify reproducibility for LLM-generated text hierarchies, I would lean toward accept. However, the authors should also clearly address my concern in order to at least keep my original score.

### Strengths
The idea is conceptually intuitive yet systematic, the transfer of U-Net’s “multi-scale + skip connection” idea into “multi-granularity prompts + bidirectional information flow” looks intuitive to me.

The coarse-to-fine CE and fine-to-coarse HS sounds reasonable. The proof seems to be correct and easy to understand.

Comprehensive experiments covering retrieval, classification, base-novel, and OOD, with clear reporting of cost-vs-performance trade-offs.

Implementation details (e.g., temperature, prompt length) are good enough to understand conceptually.

### Weaknesses
1. Text hierarchy generation via Llama 3-8B introduces possible prior leakage and reproduction variability; comparisons with purely templated or rule-based text construction would clarify fairness.

2. Lack of comparison with stronger or newer VLM backbones such as SigLIP/SigLIP-2, EVA-CLIP, or E5-V; the reported results are all on CLIP ViT-B/16.

3. Self-distillation bias: HS uses the model’s own fine-grained layer as the teacher; if that layer mis-aligns, the error may propagate.

4. Complexity analysis could be deeper: CE’s cross-granularity attention likely adds overhead, but FLOPs/memory/throughput statistics are not that detailed. Fig.5 is very complex and a little bit hard to understand.

5. Statistical significance: averages are reported across datasets, but per-dataset variance or confidence intervals are missing. Also, Table 6 in supp has wrong bold for UCF task. Is this a typo? This increases my skepticism towards the overall soundness of the reported results.

### Questions
1. Experiments fix the input at 224×224 (14×14 tokens). If higher-resolution inputs (336/384) or denser backbones (ViT-L/H) are used, does the optimal number of granularities K change? Could you provide a resolution × layer-count ablation to disambiguate “benefit from more layers” vs. “compensation for limited resolution” or at least ablation study results with another resolution?

2. Have the authors tested UPrompt on or against SigLIP / SigLIP-2 and other recent contrastive VLMs? Since UPrompt acts at the prompt level, it should in principle transfer; evidence of such portability would strengthen the claim of generality.

3. If a different LLM (e.g., Qwen, Mixtral) or a rule-based templating scheme is used to form the text hierarchies, how sensitive are the results? A cross-LLM study or a purely templated baseline would clarify robustness, though I think this won't be a big problem. The authors can discuss this if time is not allowed to add experiments.

4. When the finest layer provides noisy supervision, does HS amplify the error? Have the authors explored soft teacher mixing (e.g., weighted fine + medium layers) or consistency regularization to mitigate self-distillation bias? This would provide more insightful understanding to the effectiveness of UPrompt.

5. Currently the inference layer is manually chosen. Could a lightweight gating or uncertainty-based controller dynamically select the appropriate granularity at test time for adaptive efficiency?

### Soundness
3

### Presentation
3

### Contribution
3
