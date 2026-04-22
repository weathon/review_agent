# Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Multimodal large language models (MLLMs) have achieved remarkable progress in vision–language reasoning, yet they remain vulnerable to hallucination, where generated content deviates from the visual evidence. Existing mitigation strategies either demand costly supervision during training or introduce additional latency at inference. Recent vision-enhancement methods attempt to address this by reinforcing visual tokens during decoding, but they typically inject all tokens indiscriminately, leading to interference from background regions and distracting the model from critical cues.  To overcome this challenge, we propose an **A**daptive v**I**sual **R**einforcement framework for MLLMs, dubbed as **AIR**. AIR consists of two main components: prototype-based token reduction, which condenses the large pool of visual tokens into a compact subset to suppress redundancy, and OT-guided patch reinforcement, which quantifies the alignment between hidden state and patch embeddings to selectively integrate the most consistent patches into the feed-forward layers.  As a result, AIR enhances the model’s reliance on salient visual information and effectively mitigates hallucination. Extensive experiments across representative MLLMs demonstrate that AIR substantially reduces hallucination while preserving general capabilities, establishing it as an effective and independent solution for building reliable MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AIR, a training-free, selective reinforcement mechanism that (i) prunes redundant visual tokens and (ii) injects only OT-aligned patches during decoding, yielding reduced hallucination while maintaining general multimodal ability across representative MLLMs.  

## Motivation

Multimodal LLMs (MLLMs) still hallucinate—producing text that contradicts the image. Existing fixes either require costly fine-tuning or add inference overhead. “Re-inject all visual tokens” defenses help but indiscriminately amplify background noise. The paper proposes making visual reinforcement selective and adaptive, not blanket, and doing so training-free during decoding.  

## Methodology (AIR: Adaptive vIsual Reinforcement)

* A training-free decoding framework that reinforces only image regions most aligned with the current hidden states, rather than all tokens. 
* Visual evidence is integrated inside transformer feed-forward layers while generating, without auxiliary models. 
* Prototype-based token reduction: From the full set of visual tokens (Z), keep a compact subset by ranking tokens against a global prototype to suppress redundancy before reinforcement. 
* OT-guided patch reinforcement: Crop the image into patches, embed them, and compute optimal transport (OT) distances between selected hidden states and patch embeddings (Sinkhorn-regularized). Patches with stronger alignment (lower OT distance) are selectively injected to emphasize salient evidence.  
* Rationale for OT: Unlike pointwise similarity, OT compares distributions and captures global geometric structure, providing a finer criterion for semantic alignment at inference time.  

## Experimental Evaluation

* Benchmarks: POPE, CHAIR, MME and LLaVA-Bench.  
* POPE (LLaVA-1.5-7B): AIR achieves the best or near-best Accuracy/F1 across MSCOCO, A-OKVQA and GQA.
* LLaVA-Bench: GPT-4V-aided scores increase with AIR for all three models.  

## Analysis & Ablations

* Operating layers: Reinforcing in mid-to-deep layers is most effective; for LLaVA-1.5-7B, layers 26–32 yield the best CHAIR trade-off (CHAIR(_S)=18.4, CHAIR(_I)=5.7, BLEU=14.4). Very late or overly wide spans degrade hallucination metrics.  
* Component study: Both modules contribute; combining prototype reduction + OT patch reinforcement gives the lowest hallucination scores in CHAIR ablations.

### Strengths
* Clear motivation with empirical evidence. The paper diagnoses why 'inject all image tokens' can hurt—background tokens dilute salient cues—and supports this with similarity/attention analyses and a motivating figure.   
* Training-free, selective reinforcement that fits standard hooks. AIR stays within the FFN re-injection interface, but i) prunes redundant visual tokens via a prototype and ii) reinforces only OT-aligned patches—no fine-tuning or auxiliary models required.    
* Theoretically and empirically grounded use of OT. The paper argues OT is more sensitive than cosine for patch selection and shows larger patch-level separation and qualitative focus on salient regions.   
* Consistent gains on hallucination benchmarks across models. AIR reduces CHAIR$_S$/CHAIR$_I$ on three backbones and improves/near-improves POPE under Random/Popular/Adversarial.  
* General ability maintained. MME/MMBench remain competitive and GPT-4V–aided LLaVA-Bench scores rise.  
* Efficiency profile is reasonable. On A100, latency and memory rise slightly vs. baseline while hallucinations drop the most among compared methods.  
* Transparent ablations. Component ablation and hyperparameter sweeps clarify contributions and sensitivities; operating-layer study identifies effective mid-to-deep layers.

### Weaknesses
* 1. The evaluation benchmarks are limited. Adding more MLLM evaluation benchmark results is preferred (MM-Vet, HallusionBench, LLaVA Bench (in-the-wild)).
* 2. This novel approach seems to be hyperparameter sensitive. Performance depends on choices like TopQ and τ (showing a U-shaped trend), implying tuning is needed per setting/model. Do all models have the same performance variation when tuning the value of $Top_Q$, τ and the number of selected image patches, as show in Figure 3,4,5?
* 3. Non-zero inference overhead and added computation. Although modest, AIR introduces extra latency/memory and requires per-patch Sinkhorn OT plans during selection. Could authors provide theoretical analysis on the computation cost and inference latency of AIR?
* 4. Dataset/protocol breadth. In Table 1, CHAIR uses 500 MSCOCO images with a 64-token cap, why does the authors limit the max generation token? 
* 5. The FFN injection adds a term from selected visual tokens. Will this operation introduce disturbance to the model? how are feature scales matched to avoid over/under-amplifying the visual branch across layers?
* 6. Were MemVR/VAF/VCD run with author-recommended settings and the same decoding constraints and token caps across backbones? If yes, it is recommended that the authors should make a clearer statement on their settings in Appendix B.1.
* 7. Additional ablation study on the model's performance when injecting noised/averaged visual feature is recommended, to prove the effectiveness of AIR.

### Questions
Please refer to the weakness section. I would be glad to raise the score once authors have addressed my concerns.

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
This paper proposes AIR (Adaptive Visual Reinforcement) — a training-free framework to reduce hallucinations in multimodal large language models (MLLMs).
It combines prototype-based token reduction to remove redundant visual signals and optimal transport (OT)-guided patch reinforcement to re-inject only salient image regions during decoding. Experiments across LLaVA, Qwen-VL, and GLM-4V show consistent hallucination reduction (CHAIR, POPE) while preserving general performance (MME, MMBench).

### Strengths
1.The method achieves notable improvements across multiple models and benchmarks, showing robustness under both standard and adversarial conditions

2.The OT-based analysis is well-motivated and supported by proof and visualization, providing a clear justification for the proposed selection mechanism.

3.The paper is clearly written, visually well-presented, and includes detailed experimental settings for replication.

### Weaknesses
1.AIR assumes well-aligned hidden and visual spaces; if this alignment is weak, OT distance may emphasize irrelevant correlations, limiting reliability on misaligned models.

2.Despite its name, AIR uses fixed thresholds and token counts. Introducing data- or entropy-driven adaptation could further enhance robustness across tasks.

3.Experiments focus on standard datasets with clean imagery; robustness under distribution shifts or noisy visuals remains untested.

### Questions
1.Have the authors tested AIR on reasoning-oriented hallucination benchmarks (e.g., MM-SafetyBench, SPA-VL)?

2.Does prototype-based reduction influence the OT alignment results, and could the two be adaptively coupled?

3.Is the OT transport computation reused across layers, or recomputed at each reinforcement step?

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
This paper proposes a training-free framework to mitigate hallucination in MLLMs. The method combines (1) prototype-based token reduction, which compresses visual tokens to remove redundancy, and (2) optimal-transport (OT)–guided patch reinforcement, which selectively re-injects patches that are most semantically aligned with the hidden states. The goal is to emphasize salient visual cues while suppressing background noise during decoding.

### Strengths
1. The proposed method operates purely at inference, making it broadly applicable to existing MLLMs without retraining.
2. Integrating optimal transport to quantify alignment between hidden states and patch embeddings is a creative idea.

### Weaknesses
1. The central claim, that reinforcing salient patches directly causes lower hallucination, is not rigorously demonstrated. The supporting evidence is purely descriptive and does not establish a causal relationship between visual emphasis and reduced hallucination. The observed gains could equally arise from reduced visual redundancy or implicit regularization rather than genuine enhancement of visual grounding.
2. All experiments restrict generation to 64 tokens (Table 1), whereas hallucination severity is known to grow with output length. As a result, the evidence from short captioning tasks is insufficient to claim robustness in realistic scenarios involving long-form or multi-turn reasoning. Moreover, shorter captions are inherently more correlated with salient regions in the image, which may exaggerate the apparent benefit of saliency-based reinforcement, suggesting that the chosen generation length is tuned for conditions where the proposed mechanism performs best.
3. The paper compares only to a limited set of inference-time baselines.
4. The method depends heavily on empirically chosen parameters—such as the OT regularization strength, threshold, and the number of retained tokens—yet no principled analysis or sensitivity study is provided. Additionally, there is no discussion of computational scalability under higher-resolution inputs or longer textual outputs.

### Questions
How does AIR perform when generating longer outputs (e.g., 256–512 tokens) or in multi-turn dialogues, where hallucination typically escalates? It would be valuable to analyze how hallucination rates vary with generation length and whether the advantage of salient-patch reinforcement persists or diminishes as outputs become longer and less directly grounded in the visual context.

### Soundness
2

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
The paper proposes AIR, a training-free inference intervention to reduce hallucinations in MLLMs. AIR has two components: (i) prototype-based token reduction that retains the Top-Q most distinctive visual tokens before re-injection, and (ii) OT-guided patch reinforcement that computes an entropically-regularized optimal transport distance between hidden states and patch embeddings to select well-aligned image patches for FFN re-injection.

### Strengths
Efficiency-aware design. Prototype reduction + selective patch fusion yields small overhead, acceptable for many deployments. 

Clarity. Method is easy to implement in existing FFN-reinjection pipelines

### Weaknesses
Benchmark coverage is narrow. Evaluation centers on CHAIR (captioning) and POPE (binary VQA). Absent are harder hallucination suites probing language bias and visual illusions, such as HallusionBench, RLHF-v, and MMHal-Bench, V* etc; including them would strengthen claims of robustness.

Pure performance: The performance in incremental compared to VAF. May be provide curves for ε (entropic regularization), τ, Top-Q, and #patches on at least two models, and adversarial stress tests (noisy crops, cluttered backgrounds) for better explanation


Lack of novelty: While the paper positions AIR as a new training-free approach, its core idea—adjusting visual token utilization during inference to mitigate hallucination—is conceptually aligned with recent decoding- or selection-based techniques (e.g., Visual Contrastive Decoding, CLIP-guided decoding, or entropy-based token filtering). The novelty of AIR mainly lies in adopting optimal-transport–based patch selection combined with prototype-based token reduction, rather than introducing a fundamentally new mechanism. The contribution is thus incremental

### Questions
Please see weakness

### Soundness
3

### Presentation
3

### Contribution
2
