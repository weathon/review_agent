# Exploring the Potential of Encoder-free Architectures in 3D LMMs

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Encoder-free architectures have been preliminarily explored in the 2D Large Multimodal Models (LMMs), yet it remains an open question whether they can be effectively applied to 3D understanding scenarios. In this paper, we present the first comprehensive investigation into the potential of encoder-free architectures to alleviate the challenges of encoder-based 3D LMMs. These long-standing challenges include the failure to adapt to varying point cloud resolutions during inference and the point features from the encoder not meeting the semantic needs of Large Language Models (LLMs). We identify key aspects for 3D LMMs to remove the pre-trained encoder and enable the LLM to assume the role of the 3D encoder: 1) We propose the LLM-embedded Semantic Encoding strategy in the pre-training stage, exploring the effects of various point cloud self-supervised losses. And we present the Hybrid Semantic Loss to extract high-level semantics. 2) We introduce the Hierarchical Geometry Aggregation strategy in the instruction tuning stage. This incorporates inductive bias into the LLM layers to focus on the local details of the point clouds. To the end, we present the first Encoder-free 3D LMM, **ENEL**. Our 7B model rivals the state-of-the-art model, PointLLM-PiSA-13B, achieving 57.91%, 61.0%, and 55.20% on the classification, captioning, and VQA tasks, respectively. Our results show that the encoder-free architecture is highly promising for replacing encoder-based architectures in the field of 3D understanding. The code is released at https://github.com/Ivan-Tang-3D/ENEL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces ENEL, an "encoder-free" 3D LMM. It features a pretraining scheme with a Hybrid Semantic Loss for learnable early LLM layers and a Hierarchical Geometry Aggregation strategy (gridding, attention, pooling/unpooling) for instruction tuning. The authors report their 7B model matches or exceeds encoder-based baselines on Objaverse captioning, classification, and VQA, supported by detailed ablations.

### Strengths
This paper poses the ambitious question of whether 3D LMMs can be "encoder-free." It clearly articulates two key challenges motivating this approach—resolution mismatch and semantic misalignment—using qualitative visualisations. To address these, the authors present a concrete system (ENEL) featuring two main contributions:

1.  A thorough empirical exploration of self-supervised objectives (including masked modelling, reconstruction, KD, and contrastive), which culminates in a final "Hybrid" loss for pretraining.
2.  The authors report competitive results against strong baselines like PointLLM and PointLLM-PiSA, while also noting potential compute and memory savings compared to PointLLM.

### Weaknesses
1.	“Encoder-free” claim is overstated.
The “token embedding module” is explicitly a lightweight Point-PN–style hierarchy: FPS downsampling, k-NN grouping, and learnable layers repeated 2–4 times, followed by a projection. This is a parameter-efficient encoder, not the near-identity/VQ/linear tokenizer style commonly implied by “encoder-free” in 2D works. The paper itself calls it “a lightweight variant of Point-PN.” Please either narrow the claim or compare against true encoder-free tokenizers (e.g., single linear/conv projection without FPS/k-NN).  

	2.	Hierarchy moved into the LLM rather than removed.
The Hierarchical Geometry Aggregation repeatedly performs grid partitioning, intra-cell self-attention with a gate, mean-pooling, then later propagation (unpooling). Functionally, this recreates an encoder-like local-to-global hierarchy inside the decoder stack; it’s more of an architectural relocation of parameters than removal of hierarchical encoding. A head-to-head compute/param attribution (how many FLOPs/params sit in token-embedding + aggregation blocks vs a standard 3D encoder) is needed to justify “encoder-free” beyond naming. 
 
	3.	Loss weighting fixed to 1 without principled tuning.
The Hybrid Semantic Loss sums masked-token prediction and patch reconstruction with coefficients all set to 1, then adds the language CE loss—again with unit weight. This choice is arbitrary; one would expect a meaningful sensitivity analysis or a schedule (e.g., ramping reconstruction, temperature/ratio sweeps). Currently, the paper states “coefficients all equal to 1” with no further study.  

	4.	Fairness of comparisons / settings.
Several results swap backbones or data (e.g., ENEL-7B* uses Qwen2.5-7B and ShapeLLM data), while SOTA comparisons use different sizes. Stronger controls are needed: same LLM, same data, same token budget, and matched FLOPs/epoch, to isolate the architectural effect.  

	5.	What is the minimal “tokenizer” needed?
Table 1 suggests deeper token-embedding helps (3 layers best), but the “-Encoder” baseline collapses. This invites a systematic study from pure linear projection → 1-layer MLP → Point-PN-lite (2–4 layers) at matched token counts, to identify the true necessity of FPS/k-NN (and their cost).  

	6.	Compute story remains ambiguous.
Table 8 claims wall-clock/VRAM/FLOP gains vs PointLLM, but the new costs introduced by repeated grid builds, gated self-attention within many cells, and propagation are not separately profiled; neither is the token-embedding block. A per-stage FLOP/param breakdown would clarify where savings come from.  

	7.	Evaluation choices.
Heavy reliance on GPT-4 scores for captioning is common, but adding human preference or task-specific structured metrics (beyond Sentence-BERT/SimCSE) would strengthen claims—especially since the method tunes the text generator itself. (No citation needed; suggestion.)

### Questions
1. “Encoder-free” claim is not substantiated. The proposed pipeline still performs hierarchical local aggregation (FPS/k-NN grouping, grid partitioning, intra-cell/gated self-attention, pooling/unpooling). Functionally, this is an encoder—even if its parameters are “moved” into early LLM blocks. Unless “encoder-free” is defined merely as “no separate backbone module,” the claim is misleading.

2. Sweep the Hybrid loss coefficients (e.g., λmask, λrecon ∈ {0.1, 0.3, 1, 3}) and report stability/performance; try temperature/normalization changes for contrastive/KD, or curriculum schedules. 

3. Minimal tokenizer & hierarchy:
(a) Replace FPS+k-NN with a single linear projection (no sampling) to bound the true need for geometric preprocessing.
(b) Compare your aggregation to simple param-efficient adaptations closer to 2D “encoder-free” (e.g., LoRA on early LLM blocks; decoupled FFN) at equal parameter/FLOP budgets.

4. You motivate with resolution mismatch (Fig. 1). Please add a systematic resolution sweep for all tasks with matched token budgets, and report failure modes. 

5. Provide a per-module FLOP/param/time breakdown for PointLLM vs ENEL: token embedding, aggregation/propagation (including grid building), and LLM compute.

### Soundness
2

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
4

### Summary
This paper presents the systematic investigation into encoder-free architectures for 3D LMMs. The authors propose two methods: (1) LLM-embedded Semantic Encoding with a novel Hybrid Semantic Loss during pre-training to capture high-level 3D semantics, and (2) Hierarchical Geometry Aggregation during instruction tuning to enable better perception of local geometric structures. The resulting model, ENEL-7B, achieves competitive performance with state-of-the-art encoder-based models on classification, captioning, and VQA tasks, demonstrating the viability of encoder-free architectures for 3D understanding.

### Strengths
1. The paper conducts extensive ablation studies comparing various encoder-free strategies, including different self-supervised losses and architectural components. This thorough investigation provides valuable insights and practical guidance for the community when designing 3D LMMs.

2. The experiments cover multiple benchmarks across different tasks, with both GPT-4 evaluation and traditional metrics reported.

### Weaknesses
1. All core investigations are built on PointLLM (Aug 2023), a 2-year-old model, while conveniently citing ShapeLLM (2024) only in Table 5. This is fundamentally problematic—the rapid evolution of LLMs makes findings from ancient baselines nearly irrelevant. The authors fail to justify why encoder-free strategies weren't explored on ShapeLLM, raising serious questions about whether these "insights" generalize at all or merely exploit weaknesses of an obsolete architecture.

2. Figure 3 is confusing and poorly designed. The distinctions between loss functions are unclear, undermining a key contribution of the paper.

3. When finally tested with modern components (Qwen2.5-7B + ShapeLLM data), ENEL-7B* achieves only 2% gain over ShapeLLM-13B on QA. This trivial improvement seriously undermines the paper's central claim that encoder-free is a promising direction—it suggests the approach offers minimal benefit as base models improve, questioning the entire motivation.

4. Including BLEU/ROUGE/METEOR in Table 5 directly contradicts ShapeLLM's findings (and the authors' own admission in A.3.2) that these metrics are meaningless for 3D LMMs. 

5. The paper claims to be "the first comprehensive investigation" yet provides no related work discussion in the main paper (only in Appendix A.1). Reviewers are not obligated to read Appendix, and without proper context in the main text, it is impossible to assess the actual novelty and how this work differs from prior encoder-free efforts in 2D LMMs or recent 3D LMMs.

6. Table 5 omits many recent 3D LMMs from 2024-2025 (e.g., MiniGPT-3D, GreenPLM, and others). Without related work in the main paper, it's unclear whether this is due to incomplete survey or intentional cherry-picking. This makes it impossible to fairly assess ENEL's standing relative to the current state-of-the-art.

### Questions
1. Why not leverage the true advantage of encoder-free during training? The paper claims encoder-free architectures can "adapt to varying point cloud resolutions" (a key motivation in Abstract and Figure 1a). Can you train ENEL with mixed resolutions (e.g., randomly sample 2K-16K points per batch)? This would be infeasible for encoder-based models but natural for encoder-free models.

2. Can ENEL scale to scene-level understanding? All experiments are conducted on object-level point clouds (Objaverse, ModelNet40). Does ENEL maintain its advantages when handling much larger scene-level point clouds (e.g., ScanNet, 3D-GRAND)? This is crucial to assess whether encoder-free architectures can serve as a general solution for 3D understanding beyond isolated objects.

I appreciate the systematic investigation approach, but the evaluation does not sufficiently demonstrate the practical advantages of encoder-free design nor compare against modern baselines. If the authors can address these concerns, I would be willing to reconsider my score.

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
3

### Summary
Towards 3D understanding using LLM architecture, this work proposed an encoder-free method that works well on classification, captioning and QA tasks. First, during pre-training stage, multiple self-supervised losses are tried and compared. Second, at the instruction-tuning stage, geometry aggregation and propagation operations are baked into the LLM layers to better use the local details of 3D point clouds.

### Strengths
1. It is interesting to consider encode-free alternative for 3D LLM solutions. It has the potential to overcome the restricted resolution issue, and to reduce the semantic gaps between encoder and LLM. 
2. The proposed approach achieves good performance on 3D benchmarks and sometimes is state of the art.

### Weaknesses
1. The approach is quite incremental in terms of contributions. 1) For hybrid loss choices, combining masked and reconstruction loss seems less exciting to me. 2) The geometry aggregation is designed specifically for 3D point clouds. In some way, it moves the functionality of 3D point encoder inside the LLM. Although it might work, this leads to the loss of generality of using an general-purpose LLM. 

2.  Back to the claims of the paper. Two limitations motivate this work. 1) The variable resolution. However, there's no results to show that the method works for the case, and how does it work. Maybe benchmarks for this is missing. 2) The semantic gap between encoder and LLM. I would like to see some evidence of this first. Then, I would like to see some results of this work alleviating the issue.

### Questions
1. What are other models that you could use as baseline except `PointLLM-7B`?
2. What is the performance of the proposed method using following ablations? The reasoning behind this is to see how the vanilla version without the `claimed contributions` works. 
- a. without geometry aggregation, eg. $l=0$ 
- b. the variant with $H=0$

### Soundness
2

### Presentation
2

### Contribution
2
