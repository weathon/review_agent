# MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5, 5

## Abstract
We introduce MoMa, a novel modality-aware mixture-of-experts (MoE) architecture designed for pre-training mixed-modal, early-fusion language models. MoMa processes images and text in arbitrary sequences by dividing expert modules into modality-specific groups. These groups exclusively process designated tokens while employing learned routing within each group to maintain semantics-based adaptivity. Our empirical results reveal substantial pre-training efficiency gains through this modality-specific parameter allocation. Under a 1-trillion-token training budget, the MoMa 1.4B model, featuring 4 text experts and 4 image experts, achieves impressive FLOPs savings: 3.7x overall, with 2.6x for text and 5.2x for image compared to a compute-equivalent dense baseline, measured by pre-training loss. This outperforms the standard expert-choice MoE with 8 mixed-modal experts, which achieves 3x overall FLOPs savings (3x for text, 2.8x for image). These results demonstrate MoMa's potential to significantly advance the efficiency of mixed-modal, early-fusion language model pre-training, paving the way for more resource-efficient and capable multimodal AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces MoMa, a novel architecture for early-fusion pre-training of mixed-modal language models that utilizes modality-specific mixture-of-experts to enhance computational efficiency and model scalability across various multimodal tasks.

### Strengths
* Considering the current trends in multimodal pre-training, which increasingly rely on large-scale language models and extensive data, pursuing research in efficient architectures, exemplified by MoE, to optimize computational efficiency is a promising research direction.
* Generally, this paper is well-written and very easy to understand.
* The performance experiments and ablation studies are comprehensive and rich in detail.

### Weaknesses
* I have some concerns regarding the novelty of the paper. To my knowledge, there have been numerous attempts to apply MoE architecture in MLLMs to achieve a balanced performance and efficiency. These attempts have explored two paradigms: **shared modal experts** (*e.g.*, MoE-LLaVA [1]) and **decoupled modal experts** (*e.g.*, CogVLM [2]).  They correspond exactly to the MoMa 8x and MoMa 1t1i baselines. Given that these studies share a similar research motivation with MoMa, a comprehensive comparison with related works would better highlight the novelty of this paper.

* Regarding the experimental results, it can be observed from Fig 3 that the modality decoupling strategy (*i.e.*, MoMa 4t4i) appears to exhibit better convergence during pre-training compared to the modality sharing baseline (*i.e.*, MoMa 8x). However, the results presented in Tab 3 and Tab 4 suggest that MoMa 8x actually performs better across various benchmarks than MoMa 4t4i. It does not sufficiently demonstrate the superiority of expert decoupling.

* I wonder whether the authors have considered simultaneously modeling both modality sharing and decoupling.  Could it lead to improved outcomes?  Specifically,  I am curious if there has been any attempt to assign certain experts as modality-shared while designating others as modality-specific. It may potentially balance the commonalities and specificities between modalities more effectively. 

[1] Lin B, Tang Z, Ye Y, et al. Moe-llava: Mixture of experts for large vision-language models[J]. arXiv preprint arXiv:2401.15947, 2024.

[2] Wang W, Lv Q, Yu W, et al. Cogvlm: Visual expert for pretrained language models[J]. arXiv preprint arXiv:2311.03079, 2023.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces MoMa (modality-aware mixture-of-experts), a novel architecture designed for pre-training mixed-modal, early-fusion language models that process images and text in arbitrary sequences. MoMa divides expert modules into modality-specific groups, which exclusively process designated tokens while maintaining semantics-based adaptivity through learned routing within each group. 

The key contributions and findings of the paper are:
- MoMa achieves significant pre-training efficiency gains by allocating parameters modality-specifically.
- MoMa outperforms standard expert-choice MoE with 8 mixed-modal experts, which achieves 3× overall FLOPs savings.
- MoMa maintains a relatively modest throughput reduction while significantly improving efficiency and performance. The paper also evaluates MoMa's inference-time performance on language modeling data and downstream tasks, showing consistent improvements over the dense baseline.

### Strengths
The paper introduces MoMa, a modality-aware mixture-of-experts (MoE) architecture tailored for early-fusion language models. This approach represents a creative evolution in multimodal language processing by combining modality-specific experts that process either text or image tokens separately. 

The methodology is rigorous, with a strong experimental design that includes detailed comparisons between MoMa and dense baseline models. 

The paper is well-structured, and the key concepts are presented clearly, such as modality-specific expert grouping, hierarchical routing, and load balancing strategies.

### Weaknesses
- The innovation in the methodology presented in this paper is limited. The proposed decoupled modality approach lacks originality, as it has already been introduced in [1]. Additionally, the use of mixture of experts (MoE) is not a novel contribution, as multimodal MoE has been previously established[2].

- The paper lacks extensive validation and comparison across a broad range of benchmarks. While the authors provide comparisons for purely language downstream tasks in Table 4, there is a notable absence of direct performance comparisons on visual-language tasks. I appreciate the approach taken in Table 3 to assess the multimodal capabilities of the pre-trained models through perplexity, but I would encourage the authors to evaluate the model's performance on general visual-language tasks and compare it with some state-of-the-art understanding models, such as Flamingo[3], MM1[4], etc. Given that the primary innovation of this paper focuses on addressing multimodal scenarios, such comparisons are essential.

[1] Wang, Weihan, et al. "Cogvlm: Visual expert for pretrained language models." arXiv preprint arXiv:2311.03079 (2023).

[2] Lin, Bin, et al. "Moe-llava: Mixture of experts for large vision-language models." arXiv preprint arXiv:2401.15947 (2024).

[3] Alayrac, Jean-Baptiste, et al. "Flamingo: a visual language model for few-shot learning." Advances in neural information processing systems 35 (2022): 23716-23736.

[4] McKinzie, Brandon, et al. "Mm1: Methods, analysis & insights from multimodal llm pre-training." arXiv preprint arXiv:2403.09611 (2024).

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper mainly focuses on the early-fusion of multi-modal model pretraining. The authors proposed a Mixture of Modality-aware Experts (MoMa) Module to replace the standard dense connection layer. To ensure a balanced loading for each expert, they predefined a bucket size to select topk tokens among the batch data. An auxiliary router is trained to secure a causality for model inference. Experimental results show the effectiveness of the proposed method.

### Strengths
The introduction of modal-specific MoE into the early-fusion arch foundation models is interesting. Many scaling pre-training experiments are conducted to verify the effectiveness of the proposed method.

### Weaknesses
1. In the Expert-choice routing, loading balance is mainly considered where each expert will select topk tokens from the current batch data. However, how to ensure that all input tokens can be selected by at least one expert? How can the diversity of the experts be maintained?
2. In the abstract, it is claimed that MoMa 4t4i achieves impressive FLOPs savings: 3.7× overall, with 2.6× for text and 5.2× for image compared to a compute-equivalent dense baseline. How these numbers are calculated? In Table1, MoMa has much more parameters than the dense one for the same amount of active FLOPs per token. Additionally, in Table2, MoMa shows lower training throughput than the dense one. Hoth of these two tables may not clearly verify the efficiency of the proposed method.
3. For the modality-aware expert, the motivation of this work argues that different modalities have distinct characteristics and information densities (L131) and proposes the MoE for the dense layer. How about the modal-specific attention layers? Or the interleaved mix-modal and modal-specific attention layers?
4. In Table 3, MoE 8x outperforms MoMa 4t4i in 4 out of 11 settings, and in Table 4, it shows better performance in 5 out of 8 benchmarks. Given that MoE 8x uses a mixture-of-modality design, does this suggest that a mixed-expert approach itself is more beneficial, without necessarily requiring a unique expert for each modality?
5. In L168-171, “However, EC routing compromises causality in auto-regressive language modeling, as each expert selects the top tokens to process in a batch by comparing token scores across all tokens.” I am a little bit confused about why causality is an issue here. Although selecting the top-k tokens involves looking at future tokens, this process only affects the selection and doesn't incorporate future tokens' embeddings into the calculation of each token itself.
6. Some minor questions:
- Lines 254-255 state, “In MoD architectures, we implement MoD in alternating layers, beginning with layer 0, using a layer capacity factor cdc_dcd​ of 25%.” The term MoD is unclear as it lacks a prior introduction in the paper. Although supplementary section B presents MoD-related experiments, none are included in the main paper, making this mention unnecessary.
- L178, details of auxiliary routers should be in Supplementary section A.2.

### Questions
The questions listed above are essential for readers to gain a clearer understanding of the proposed method. A detailed response to these points is expected to help clarify potential confusion.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a modality-aware mixture-of-experts architecture for pre-training mixed-modal, early-fusion language models. This structure processes image and text tokens with two groups of expert modules in the FFN layer. Two types of routers are trained to control expert selection: the main routers generate the weight values, and tokens are selectively routed to an expert based on thresholding the auxiliary router score. Under a 1-trillion-token training budget, the performance of a model with 4 text experts and 4 image experts achieves significant FLOPs savings.

### Strengths
- This paper shows the effectiveness of a modality-aware processing structure on a large image-language model.
- The evaluation performance shows that this method achieves a clear reduction in FLOPs.
- The ablation and analysis are comprehensive, considering the significant training cost of the pre-training experiment.

### Weaknesses
- The technical novelty of the proposed method is limited, since grouping has been used in FFN layers, and previous works have separately processed image and text tokens, e.g., using image encoders to process the image tokens. It requires further justification of the novelty of the proposed method.

- Is there an ablation study to see if using both image/text and image+text groups of MoE would be better than using image/text groups of MoE?

- The FLOPs saving for training is given. While the FLOPs during training do not strictly align with training time, and MoE structure would make the training more complex, the full training time comparison should be given.

- Another concern is that using two groups of experts will lead to more MoE layers. The inference cost, e.g., inference time, should also be compared with the standard MoE model.

- From Tables 3 and 4, the performance improvement is inferior to the MoE model in some cases. Is it possible that the FLOPs saving comes at the expense of performance drop?

### Questions
- The clarification of the technical novelty.
- The training/test time comparison.
- The performance comparison with the MoE model.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The manuscript presents MoMa, a novel architecture for pre-training mixed-modal early-fusion language models utilizing a mixture-of-modality-aware experts strategy. By dividing experts into modality-specific groups and using learned routing, MoMa achieves significant efficiency gains in pre-training, highlighted by a comprehensive suite of experiments. This approach addresses the challenges associated with computational costs in scaling mixed-modal models and offers a modular framework that could be extended to various multimodal tasks.

### Strengths
The paper addresses the challenge in mixed-modal learning with an architecture that significantly reduces computational overhead through modality-specific expert allocation. 
It presents a thorough experimental setup, including detailed ablation studies that underline the architecture's strengths in different settings.

### Weaknesses
1. The paper could be improved by a more detailed comparison with existing methods that utilize modality-specific experts. The current discussion does not fully delineate how MoMa advances beyond prior art in terms of both design and performance metrics.

2. For performance evaluation, the results in Table 3 to assess the multimodal capabilities of the pre-trained models through perplexity are not enough. Common benchmarks (e.g., SEED, MMBench, VQAv2) are encouraged to introduced to compare it with other state-of-the-art models under the same setting.

### Questions
see above weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
