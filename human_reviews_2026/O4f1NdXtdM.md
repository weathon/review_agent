# FaceMoE: Mixture of Experts for Low-Resolution Face Recognition

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Low-resolution face recognition (LR-FR) remains a challenging task due to poor
feature extraction and aggregation, as probe images often contain limited iden-
tity information resulting from extreme degradations such as blur, occlusion, and
low contrast. Additionally, the domain gap between high-resolution (HR) gallery
images and low-resolution (LR) probe images poses a significant challenge. A
single feature encoder struggles to generalize effectively across both domains when
fine-tuned on an LR dataset, and this issue is further magnified by catastrophic
forgetting. To address these challenges, we propose FaceMoE, a novel transformer-
based architecture enhanced with a Mixture of Experts (MoE) design. Specifically,
we introduce multiple specialized feed-forward network (FFN) experts and incor-
porate a top-k router, which dynamically assigns tokens to appropriate experts.
This design promotes specialization across experts for different semantic regions of
the face, which enables FaceMoE to perform resolution-aware feature extraction.
Moreover, the top-krouter facilitates sparse expert activation, enabling the model
to preserve pretrained knowledge when finetuned on a LR dataset, while increasing
model capacity without proportional computational overhead. FaceMoE is trained
with a combined face recognition loss, router z-loss, and load balancing loss to
ensure expert specialization and stable training. To the best of our knowledge, this
is the first work leveraging MoE for LR-FR. Extensive experiments across eleven
datasets, spanning HR, mixed-quality, and LR benchmarks, demonstrate that Face-
MoE significantly outperforms state-of-the-art methods, excelling in low-resolution
face recognition. Code and models will be made public.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces FaceMoE, a transformer-based face recognizer that inserts a mixture-of-experts MLP with a token-level top-k router into each block to specialize computation across facial regions and resolutions, stabilized by two auxiliary objectives (a router z-loss and a load-balancing loss) added to a standard CosFace recognition loss. The router learns to dispatch tokens from identity-rich landmarks, low-frequency skin regions, and high-frequency hair/background to different experts, yielding resolution-aware feature extraction while mitigating catastrophic forgetting when fine-tuning on low-resolution data. 

The main contributions are the encoder-level MoE design with token-wise routing for low-resolution face recognition, the stabilization losses for reliable expert utilization, and a broad evaluation demonstrating state-of-the-art or competitive results under surveillance-style conditions.

### Strengths
The paper is original in placing a token-level top-k MoE inside the transformer FFN to specialize by facial region/frequency and stabilize routing with z-loss and load-balancing; this goes beyond fusion-only or sample-level MoE and is backed by region-wise routing statistics.  

In quality, experiments span BRIAR, IJB-S, TinyFace and standard HR sets, with consistent improvements and ablations that tease apart routing, expert count, auxiliary losses, backbone choice, and data scale; random-routing and large-N collapse analyses bolster the causal claim.   

In clarity, the router, losses, and training algorithm are explicitly formulated and connected to semantic regions via quantitative tables, easing reproducibility.  

Finally, significance is strong for surveillance-style LR recognition: the method improves LR while preserving HR/mixed performance, uses conditional compute, and remains backbone-agnostic, suggesting practical adoption.

### Weaknesses
Despite solid engineering, the paper leaves several gaps that are correctable, limiting confidence in its claims. The “reduced forgetting” claim is under-substantiated; beyond trend plots, quantify selective drift via per-expert parameter change, Fisher overlap, or CKA between HR-pretrained and LR-tuned layers to show that MoE routing, rather than altered regularization, preserves HR/mixed performance.  

Baselines often overlook strong, parameter-efficient alternatives: adding head-to-head comparisons under matched budgets to quality-adaptive adapters/LoRA (e.g., PETALface) and recent fusion/sparsity methods (e.g., ProxyFusion) demonstrates that encoder-level MoE is necessary, not merely sufficient, given similar data and tuning regimes.  

Routing claims would benefit from stability and causality checks. This involves reporting region-wise assignment consistency across seeds and small perturbations (such as crop/blur/alignment), and intervening by freezing or swapping specific experts to test whether the observed gains truly derive from learned specialization rather than capacity.

### Questions
1. Reduced forgetting, measured not just observed: Beyond trend plots, can you quantify selective drift across layers/experts (e.g., parameter deltas per expert, Fisher overlap, CKA between HR-pretrained and LR-finetuned representations)? This would separate a MoE-driven effect from regularization or learning-rate artifacts.

2. Stronger baselines under matched budgets: Please add head-to-head comparisons with parameter-efficient baselines like quality-adaptive adapters/LoRA (e.g., PETALface) and recent fusion/sparsity approaches (e.g., ProxyFusion), controlling for pretraining data, compute, and tuning schedules. If omitted, justify why those methods are not applicable here.

3. Routing stability and causality: How stable are token-to-expert assignments across seeds and small perturbations (crop jitter, alignment error, blur)? Consider interventions (freezing or swapping specific experts) to test whether gains arise from learned specialization rather than extra capacity.

4. Expert semantics beyond visuals: You show regional/frequency tendencies; can you quantify whether experts capture complementary spectra (e.g., bandpass profiles) or identity-salient landmarks via controlled masking/occlusion tests? This would make “what each expert learns” more concrete.

5. Hyperparameter sensitivity: Please provide sensitivity analyses for top-k, router temperature, z-loss weight, and load-balancing weight across at least two datasets. If performance is brittle, suggest default ranges and an automatic tuning heuristic.

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
- Low-resolution face recognition (LR-FR) faces three core challenges: poor feature extraction and aggregation from degraded probe images (e.g., blur, occlusion, low contrast), significant domain gaps between high-resolution (HR) gallery and low-resolution (LR) probe images, and catastrophic forgetting when fine-tuning models on LR datasets. To address these issues, the paper proposes FaceMoE, a transformer-based architecture enhanced with a Mixture of Experts (MoE) design. The core method involves integrating multiple specialized feed-forward network (FFN) experts into transformer MLP layers and introducing a top-k router that dynamically assigns input tokens to k out of N experts based on input resolution, enabling resolution-aware feature extraction.
- For validation, the authors conducted extensive experiments across some datasets using two protocols: Protocol 1 (WebFace4M pre-training → BRIAR fine-tuning, evaluating BRIAR Protocol 3.1 and IJB-S) and Protocol 2 (WebFace4M pre-training → TinyFace fine-tuning, evaluating TinyFace and HR/mixed-quality datasets). Key metrics include TAR@FAR, TPIR@FPIR, and Rank-1/5/10 retrieval accuracy.
- The core conclusions are: 1) FaceMoE achieves state-of-the-art (SOTA) performance on LR datasets; 2) The MoE’s sparse activation mitigates catastrophic forgetting, maintaining minimal performance drop on HR/mixed-quality datasets; 3) The optimal configuration (N=3 experts, k=2 active experts per token) balances model capacity (2.17× increase over Swin-B) and computational cost (1.66× FLOPs increase, 26.29 GFLOPs vs. Swin-B’s 15.88 GFLOPs).

### Strengths
> 1. As a pioneering work to leverage MoE for LR-FR, it targets the field’s key points (domain gap, catastrophic forgetting) with a well-designed architecture. The top-k router dynamically assigns tokens to experts specialized in distinct facial semantic regions, directly solving the limitation of single FFN encoders that fail to adapt to both HR and LR domains. This design ensures resolution-aware feature extraction, critical for degraded LR images.
> 2. The paper conducts detailed ablations on critical components (MoE module, top-k router, auxiliary losses) in Table B.5. Additional analyses (expert specialization via Figure 2/Table B.1, resolution robustness via Table B.2, data scaling via Table B.8) confirm the model’s reliability and generalizability, strengthening the credibility of design choices.
> 3. The authors evaluated FaceMoE across diverse datasets, covering HR, mixed-quality, and LR scenarios, using two protocols that validate both LR performance and catastrophic forgetting mitigation. The results are enough for verifying the proposed method.
> 4. The ethics statement confirms compliance with data usage rules and discusses responsible deployment to avoid misuse. The reproducibility statement details implementation settings (PyTorch, AdamW optimizer, learning rate schedules) and promises code/model release upon acceptance, ensuring experiment replicability.

### Weaknesses
* 1. While the paper provides a bias analysis in Table B.3 (comparing SeR/DoB across age, gender, race on LFW/CFP-FF/AgeDB), it only contrasts FaceMoE with Swin-B and lacks deeper investigation into instrinsic reasons. 
* 2. The paper mentions failure cases on BRIAR (e.g., <8×8 pixel crops, extreme poses) but does not investigate how the MoE architecture responds to these failures—for example, whether expert activation becomes random (e.g., occluded landmark tokens misrouted to non-landmark experts) or if certain experts are deactivated. It also lacks heatmap comparisons of expert activation between successful and failed cases (e.g., Figure 6 only shows activation before/after fine-tuning, not failure vs. success). 
* 3. mplementation details specify warm-up epochs (1 for pre-training, 2 for BRIAR fine-tuning, 2/4 for TinyFace fine-tuning) and polynomial LR scheduling but do not explain how these parameters were tuned or their impact on performance. For example, it is unclear if reducing BRIAR warm-up epochs from 2 to 1 causes gradient instability or if a cosine LR scheduler outperforms the polynomial scheduler. This hinders experiment replication for other researchers.

### Questions
- Q1.  Why is the load balancing loss (Section 53) formulated with the product of "sum of routing probabilities (∑p_b,t,i)" and "sum of indicator functions (∑𝟙[i∈TopK(z_b,t)])" instead of simpler balancing metrics (e.g., variance of expert loads)?
- Q2. The final token output is a convex combination of k expert outputs, but the paper does not discuss how the model handles conflicts—e.g., if Expert 0 (landmark-focused) assigns high weight to a "nose" feature and Expert 1 (cheek-focused) assigns high weight to a "cheek" feature for the same token. Additionally, it does not report whether certain experts consistently dominate outputs for specific token types (e.g., 70% of landmark token weight from Expert 0). 
- Q3. How does FaceMoE perform on images with resolutions between 8×8 and 16×16 pixels (e.g., 10×10, 12×12), and does the top-k routing hyperparameter (k) need adjustment for these ultra-low scenarios? The resolution ablation study (Table B.2) only reports results for 16×16 to 96×96 pixels but ignores ultra-low resolutions (8×8, 12×12). The paper mentions <8×8 pixels as failure cases but does not address 12×12/10×10 pixels (a middle ground with potential utility). For example, would increasing k from 2 to 3 improve performance for 12×12 pixels by leveraging more expert information? 
- Q4. Does expert-specific initialization (e.g., initializing landmark-focused Expert 0 with weights from a pre-trained facial landmark detection model) improve LR adaptation speed and final performance compared to the current global initialization (WebFace4M pre-training)? The paper initializes FaceMoE globally on WebFace4M but does not explore targeted expert initialization—an approach that could accelerate semantic specialization (e.g., Expert 0 quickly focusing on landmarks). It is unclear if such initialization reduces fine-tuning epochs or improves performance on small LR datasets like TinyFace.

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
Facing challenges from degraded probe images and the resolution domain gap in low-resolution face recognition (LR-FR), FaceMoE is proposed which is a transformer model enhanced with Mixture of Experts (MoE). By employing specialized experts and a top-k router, FaceMoE achieves resolution-aware feature extraction while preserving pre-trained knowledge through sparse activation. This first MoE-based approach for LR-FR effectively addresses feature degradation and domain gap issues without significant computational increase.

### Strengths
A modified transformer encoder is proposed to use sparsely activated feed-forward network (FFN) experts. A top-k router directs tokens to specialized FFN experts, enabling resolution-aware feature extraction from distinct facial regions. This is a good attempt to apply MoE into this field.

### Weaknesses
The manuscript describes interesting progress, but several important issues need to be addressed to strengthen its validity and impact:
1. The mechanism by which different experts specialize in distinct facial regions requires more in-depth discussion. 
2. The complementarity between different experts is not clearly demonstrated, as the analysis reveals a high degree of redundancy in the regions they activate. A clearer distinction in their specialized roles is needed.
3. Could the authors please justify the use of a single coefficient for the last two losses in Line 279? The rationale for this design choice is unclear.
4. The authors should provide an ablation study to quantitatively demonstrate the contribution of each proposed module (e.g., the MoE layer, the top-k router) to the overall performance.
5. A thorough proofreading is required to address several formatting and typographical issues. A notable example is Table 1, where the text flows outside the designated cells, which affects readability.

### Questions
See above.

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
4

### Summary
**Problem.** Low-resolution face recognition (LR-FR) suffers from weak, domain-mismatched probe features (LR, degraded) vs. gallery (HR), plus catastrophic forgetting when fine-tuning on LR.  
**Idea.** Replace the single FFN in transformer blocks with a **Mixture-of-Experts (MoE) MLP**: $N$ FFN experts with **top-k** sparse routing at the token level. Auxiliary **z-loss** and **load-balancing** regularizers stabilize routing and encourage expert specialization.  
**Mechanism.** Sparse routing increases capacity without proportional FLOPs; selective expert updates during LR fine-tuning reduce forgetting while enabling **resolution-aware** feature extraction.  
**Evidence.** Strong improvements on LR benchmarks (BRIAR 3.1, IJB-S, TinyFace) and minimal drops on HR/mixed-quality sets; ablations over $(N,k)$ and compute trade-offs; qualitative expert activation maps.  
**Takeaway.** First focused MoE architecture for LR-FR with convincing, state-of-the-art results and a principled training recipe.

### Strengths
* **Originality.** MoE-FFN + top-k routing tailored to resolution-dependent cues; explicit anti-collapse regularizers; selective-drift argument is well motivated.
    
* **Quality.** Comprehensive benchmarks (BRIAR, IJB-S, TinyFace) with relevant baselines (ProxyFusion, PETALface, CAFace, etc.); competitive pretrain/fine-tune recipe and careful hyperparams.
    
* **Clarity.** Method section is concrete (router, losses, algorithm); compute/ablation plots aid intuition; training schedule is reproducible.
    
* **Significance.** Material SOTA gains where LR matters most, and minimal HR performance drift—useful for real surveillance/forensic pipelines.

### Weaknesses
* **Statistical rigor.** Most metrics are single-number; no confidence intervals, seed variance, or bootstrap CIs; a few improvements are significant, but some margins over strong baselines would benefit from uncertainty quantification.
    
* **Expert “semantics” not quantified.** Activation maps are qualitative; provide a measurable alignment between experts and regions/frequencies (e.g., mutual information with landmark masks, frequency energy, or SHAP on routing logits).
    
* **Degradation coverage.** LR comes with blur, compression, noise, occlusion, and illumination shifts; targeted stress tests (synthetic and in-the-wild subsets) are limited.
    
* **Efficiency on edge.** FLOPs are reported, but end-to-end latency/throughput and memory under varying batch sizes (edge/GPU constraints) are not; important for deployability claims.
    
* **Forgetting analysis.** The “selective drift” story is compelling; a more direct comparison against LoRA/adapter fine-tuning and LwF-style regularization would isolate MoE’s advantage.

### Questions
1. **Statistical robustness.** Can you report mean±std over ≥3 seeds (or 5-fold bootstrap CIs) for key metrics on BRIAR/IJB-S/TinyFace? Do the rankings hold under variance?
    
2. **Expert semantics.** Can you quantify specialization (e.g., Kendall’s $\tau$ between routing scores and (i) frequency bands; (ii) landmark/region masks; (iii) edge density), and report per-expert token distributions pre/post fine-tune?
    
3. **Ablations vs. PEFT/LwF.** How does FaceMoE compare to strong PEFT baselines (LoRA/IA³/adapters) and LwF/EWC in both LR gains and HR retention, at matched FLOPs/params?
        
4. **Degradation stress tests.** What happens under controlled blur/noise/compression sweeps and occlusion masks (eyes/mouth/cheeks)? Does routing adapt as hypothesized?
    
5. **Latency/memory.** Please provide inference latency (ms), peak memory usage, and throughput (fps) for (N,k) settings on an A6000 and a resource-constrained GPU, including router overhead. 
    
6. **Generalization.** Does TinyFace-tuned FaceMoE transfer to LR-like mobile video or body-cams without re-tuning? Any zero-shot observations?

### Soundness
3

### Presentation
3

### Contribution
3
