# DKP: Semantic Consistency Distillation via Key-layer Pre-alignment with Large Vision-language Models for Cross-Modal Retrieval

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Recent retrieval solutions based on large vision-language models (VLMs) have shown promising performance by aligning vision and language representations for image-text retrieval (ITR). However, most methods rely solely on final-layer features, overlooking the rich semantic patterns embedded in intermediate layers. In this paper, we propose Semantic Consistency \textbf{\underline{D}}istillation via \textbf{\underline{K}}ey-layer \textbf{\underline{P}}re-alignment (termed as DKP), a novel paradigm that enhances cross-modal retrieval by leveraging intermediate knowledge of VLMs. Specifically, we introduce (i) Key-layer Pre-alignment (KPA) to identify and align the most semantically meaningful intermediate features across modalities, and (ii) Semantic Consistency Distillation (SCD) to regularize cross-modal learning via intra-modal structure. Extensive experiments on Flickr30K and MS-COCO validate DKP significantly boosts retrieval performance while requiring over 60\% fewer learnable parameters and significantly less computational cost, without introducing additional supervision or external knowledge. The anonymous code is available at: \textcolor{blue}{\url{https://anonymous.4open.science/r/DKP}}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents DKP, a parameter-efficient fine-tuning method for vision-language models in the context of image-text retrieval. The authors observe that in CLIP's transformer architecture, attention patterns stabilize in intermediate layers (around Layer 8). Based on this, they propose aligning features at this intermediate layer and adding a self-distillation loss to maintain intra-modal consistency. The method involves fine-tuning only the identified key-layer and the final layers of the model. Experimental results on Flickr30K and MS-COCO show that this approach can achieve performance comparable to or slightly better than full fine-tuning while using fewer trainable parameters.

### Strengths
Parameter Efficiency: The primary strength of the paper is the significant reduction in trainable parameters (~60%) compared to full fine-tuning, which can lower memory requirements for the training process.

Clear Presentation: The paper is well-written and easy to follow. The motivation behind the method is explained clearly with visualizations of the layer-wise attention mechanism .

### Weaknesses
Heuristic-Based Key-Layer Selection: The choice of Layer 8 as the "key-layer" is based on an empirical observation of attention map stabilization. This approach feels heuristic and lacks a strong theoretical foundation. It is not clear if this "semantic crystallization" is a generalizable phenomenon or a coincidental property of the specific CLIP architecture. This requires a manual, potentially expensive search for each new model, limiting the method's general applicability.

Incremental Contribution and Performance: The proposed method is an incremental variation on existing fine-tuning paradigms. The performance gains over standard full fine-tuning are marginal. Given the small improvement, the contribution appears more iterative than transformative.

Questionable Premise of Intermediate Alignment: The core assumption that forcing alignment in an intermediate layer is beneficial is not well-justified. There are no guarantees that this intermediate supervision leads to a better final representation. It could potentially impose an unnecessary bottleneck, constraining the model's flexibility to learn optimal features in its deeper layers, especially finetuned on top of CLIP-style pretrained model, may need to test from scratch. The paper does not provide a compelling argument for why this is fundamentally better than allowing the entire network to adapt via end-to-end training.

Limited Experimental Scope: The experiments are confined to fine-tuning on standard ITR benchmarks (Flickr30K and MS-COCO).  This setup does not adequately test the method's ability to generalize to more challenging, out-of-distribution datasets. The performance on in-domain data is not a sufficient indicator of the robustness of the learned representations.

### Questions
Please refer weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DKP, Semantic Consistency Distillation via Key-layer Pre‑alignment, for image‑text retrieval with CLIP‑like VLMs. The method (i) identifies a semantically "crystallized" intermediate layer (empirically around Layer 8) and performs Key‑layer Pre‑Alignment (KPA) there while still training the top layers, and (ii) adds Semantic Consistency Distillation (SCD) that aligns cross‑modal similarities with intra‑modal structure via KL divergence. On Flickr30K and MS‑COCO, DKP improves Recall@K and RSUM while using $\sim$21M learnable parameters ($\geq$60\% fewer than full fine‑tuning) and lowers training/inference time. The paper provides ablations (loss components, key‑layer choice, $\alpha/\beta$ weighting), qualitative visualizations, and complexity analyses.

### Strengths
- KPA leverages mid‑layer representations rather than relying solely on final‑layer features; SCD regularizes with intra‑modal structure. The architecture diagram is straightforward and the losses are precisely specified.

- On Flickr30K with ViT‑B/16, DKP reaches R@1 = 89.4 (I2T) and 77.22 (T2I), RSUM = 555.88; on MS‑COCO 5K, RSUM = 454.39, competitive or better than CLIP full fine‑tuning and CUSA under the same backbones

- Learnable parameters drop from $\sim$150M to 21.01M and per‑epoch/inference latency improves. This is attractive for resource‑constrained setups.

- Loss ablations show both KPA and SCD independently help, with the combination best. Key‑layer sweeps highlight a turning point at Layer 8. Learned weights $\alpha$, $\beta$ evolve sensibly over training.

- Visualizations show tighter intra‑modal clusters and sharper token‑image attention under DKP, aligning with the intended semantic‑consistency effect.

- The method is "plug‑and‑play" for CLIP‑style models; the appendix discusses asymmetric backbones (ViT‑L/14@336) and PEFT design considerations.

### Weaknesses
- While Fig. 3 motivates Layer 8 for ViT‑B, the universality of the "semantic crystallization" claim across architectures, datasets, and training regimes is only partially explored (Appendix A.9 is referenced but not deeply quantified in the main text). Stronger evidence across more backbones/sizes would help.

- Results are confined to Flickr30K/MS‑COCO. Cross‑dataset transfer (train on one, test on another), domain shift, or zero‑shot retrieval evaluations are not reported in the main paper, leaving external validity less clear.

- SCD uses batchwise N×N similarity matrices. The paper reports end‑to‑end wall‑clock improvements, but does not analyze memory/time scaling with batch size or sequence length, nor sensitivity to the temperature/softmax normalization choices.

- Although A.8 discusses PEFT, the main results lack direct, apples‑to‑apples comparisons against strong adapters/LoRA/prompt‑tuning baselines under identical compute and data. This blurs how much of DKP's gain is from KPA/SCD vs. parameter‑efficient tuning per se.

- The paper shows SCD helps (Table 3), but offers limited intuition/diagnostics, e.g., when intra‑modal neighborhoods are misleading, or whether SCD can amplify spurious clusters under noisy captions. Failure‑case analysis would strengthen the claim.

- Variance across seeds and significance tests are not reported for key tables; given small absolute gaps among strong baselines, CIs or paired tests would build confidence.

### Questions
- Beyond Fig. 3, can you report results for adaptive or learned key‑layer selection (even a simple gating over {8,…,11}) and show whether different samples prefer different layers? How does this compare to the fixed‑Layer‑8 choice in accuracy vs. overhead?

- What is the memory/time behavior of SCD as batch size grows (since it forms N×N similarities)? Any practical tricks (e.g., blockwise SCD, momentum queues) to keep SCD effective with large batches?

- Could you include (i) train‑COCO -> test‑Flickr30K (and vice‑versa) and (ii) zero‑shot retrieval from frozen DKP encoders to probe robustness under distribution shift?

- Under the same compute/data, how does DKP compare to LoRA/adapter/CoOp/VPT baselines on both backbones? A small table would clarify DKP's incremental value over standard PEFT. (Appendix A.8 hints at design, but results would help.)

- Can you provide sensitivity to temperature, similarity normalization, and neighborhood sharpness, plus qualitative failure cases where intra‑modal structure misleads cross‑modal matching?

- Table 2 reports wall‑clock gains. Could you also normalize by GPU model/count and report per‑sample or per‑token throughput to ease reproducibility across hardware?

- Appendix A.9 explores ViT‑L/14@336; any headline numbers you can bring into the main paper to demonstrate that the Layer‑8 phenomenon and DKP's gains persist at larger scales?

### Soundness
3

### Presentation
3

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
This paper introduces a lightweight cross-modal image-text retrieval method termed DKP, which maintains high performance while significantly reducing trainable parameters through key-layer pre-alignment and semantic consistency distillation. The innovative methodology design has been empirically validated across multiple datasets, demonstrating its efficacy.

### Strengths
This paper designs a dual-group innovative mechanism of KPA and SCD to achieve pre-alignment of the key layer and semantic consistency distillation. The experimental design is quite comprehensive, and the performance on public datasets Flickr30K and MS-COCO has been verified to be at the state-of-the-art level using multiple mainstream backbones. The ablation experiments effectively prove the effectiveness of each module independently. This has significant implications for efficient cross-modal learning.

### Weaknesses
1. No systematic comparison was made with more comprehensive parameter fine-tuning methods under the same settings.

2. Although the reduction of parameters and the decrease in training time were emphasized, the change in FLOPs during inference was not analyzed, which is particularly important for deployment on edge devices.

3. Although this paper provides a new idea for the efficient fine-tuning of VLMs, its applicability to resource-constrained scenarios still needs to be verified.

### Questions
1. Regarding the issue of "kite" and "paraglide" being grouped together, did the author attempt to combine the region-level visual features extracted by Faster R-CNN?

2. Analyze the correlation between the deceleration rate of different datasets and "dataset size" and "data complexity".

3. The paper claims to be applicable to resource-constrained scenarios. Please supplement the experimental data on mobile devices (such as inference latency and memory usage).

4. Compare the "performance - latency" trade-off relationship between DKP and CoOp (with parameters 7.12M) on edge devices.

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
This paper introduces DKP, a clever and efficient method for fine-tuning large vision-language models like CLIP for image-text retrieval. 
The authors found that semantic meaning crystallizes at a specific intermediate layer in these models. DKP leverages this by only fine-tuning the later layers and adding a pre-alignment objective at this key layer, which drastically cuts down on trainable parameters. To ensure the model maintains a coherent understanding, it also uses a self-supervised technique called Semantic Consistency Distillation (SCD). This forces the model to preserve the similarity relationships within images and within texts during training. The result is a model that achieves state-of-the-art performance while being significantly faster and cheaper to train, using over 60% fewer parameters than standard fine-tuning.

### Strengths
1. Their analysis revealing a "semantic crystallization" point in the intermediate layers of CLIP is a fantastic insight. It provides a strong, empirical reason for their parameter-efficient approach.
2. DKP achieves a brilliant trade-off. It gets better results than more computationally expensive methods (like full fine-tuning or CUSA) while needing way fewer trainable parameters and less training time.
3. The Semantic Consistency Distillation (SCD) is a neat trick. It helps prevent "semantic drift" and improves model robustness by using the structure already present in the data batch, without needing any external knowledge or labels.

### Weaknesses
1. It's Still a Heavy Model: While the training is efficient, the method still requires running the full, large VLM during inference. It makes training more accessible but doesn't reduce the final model's size or computational footprint for deployment on resource-constrained devices.
2. The method assumes a single fixed key-layer for all inputs. It's possible that for some simple images, semantics crystallize earlier, and for complex ones, later. A dynamic approach that adapts the key-layer based on the input could potentially be even better.

### Questions
1. Analysis on the asymmetric ViT-L model showed the image and text encoders crystallize at different depths (Layer 17 and Layer 8, respectively). Your experiment aligned these two different layers. What do you think would happen if you forced an alignment between the same layer index (e.g., Layer 8) for both, even though the image's semantics are not fully formed? Could this early "nudge" from the more mature text representation actually help guide the visual representation to form better?


2. The Semantic Consistency Distillation (SCD) uses the intra-modal similarity from the current batch as its target. This could be a bit noisy depending on the batch composition. Have you considered using a more stable target, perhaps by incorporating a moving average of the similarity structures from previous batches, similar to the momentum encoder in MoCo?

### Soundness
3

### Presentation
3

### Contribution
3
