# Autonomous Knowledge Integration Enables Efficient Cross-Modality Prompt Transfer

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Cross-Modality Prompt Transfer (CMPT), which transfers prompts pretrained on a data-rich modality (e.g., text), seeks to address initialization instability and data scarcity in prompt tuning of pretrained transformers for tasks with limited data.
However, prior research in CMPT typically transfers all source prompt vectors without accounting for vector redundancy or incompatibilities.
This indiscriminate transfer may include irrelevant or even detrimental vectors, thereby negatively impacting the performance of the target task.
To mitigate this issue, we propose Selective Cross-Modality Prompt Transfer (S-CMPT), a method that automatically identifies and transfers only the most relevant prompt vectors.
S-CMPT employs a lightweight attention mechanism to select vectors that are most pertinent to the target task.
Furthermore, we introduce a simple regularization term to encourage the selection of diverse and non-redundant vectors.
The selected vectors are subsequently adapted to the target task through a linear projection layer.
As a result, S-CMPT achieves impressive simplicity and effectiveness, demonstrating significant accuracy improvements (+4.42\% over prompt tuning and +0.65\% over state-of-the-art prompt-based methods) while utilizing far fewer vectors, offering an efficient solution for data-scarce prompt tuning applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses limitations in Cross-Modality Prompt Transfer (CMPT), which transfers prompts pretrained on data-rich modalities (e.g., text) to improve prompt tuning for data-scarce tasks in other modalities (e.g., vision). The authors identify two key problems with existing CMPT approaches: (1) previous work has been limited to shallow architectures, and (2) conventional methods transfer all source prompt vectors indiscriminately without accounting for redundancy or incompatibilities between source and target tasks.

To address these issues, the authors propose Selective Cross-Modality Prompt Transfer (S-CMPT), which introduces an attention-based prompt selector that identifies the most relevant source vectors for each target layer, along with a diversity regularization term to encourage selection of non-redundant vectors. The selected vectors are then adapted via linear projection to the target task. The method is evaluated on the VTAB-1k benchmark across multiple ViT backbones (supervised ImageNet-21k, MAE, and MoCo v3 pretraining), demonstrating significant performance improvements over existing methods while using fewer prompt vectors (20 per layer vs. 49.3 for the state-of-the-art VFPT method).

### Strengths
1. The paper identifies an important limitation in existing CMPT approaches - the assumption of layer-wise correspondence and indiscriminate transfer of all prompt vectors. The random selection experiment (Figure 2) compellingly demonstrates that layer-wise correspondence is suboptimal, with random selection outperforming CMPT in 94.4% of trials.

2. S-CMPT introduces a conceptually straightforward approach (attention-based selection + diversity regularization) that significantly outperforms existing methods. The elegance of the solution is a strength, avoiding unnecessary complexity while addressing a genuine problem.

3. The paper provides thorough evaluation across multiple pretraining settings (supervised, MAE, MoCo v3) and VTAB-1k tasks, with detailed per-task results. The ablation studies on selection range and diversity regularization strength provide convincing evidence for the design choices.

4. By using only 20 prompt vectors per layer (compared to 49.3 for VFPT), S-CMPT achieves better performance with significantly fewer parameters, which is crucial for parameter-efficient transfer learning.

### Weaknesses
1. While the empirical results are strong, the paper lacks theoretical justification for why certain prompt vectors are more transferable across modalities. A deeper analysis of what properties make vectors transferable would strengthen the contribution.

2. The paper shows that diversity regularization helps but doesn't thoroughly analyze why the specific cosine similarity-based approach works better than alternatives, or provide deeper insights into how diversity affects transfer performance.

3. The paper focuses on positive results but doesn't thoroughly analyze scenarios where S-CMPT provides minimal gains or underperforms baselines. Understanding the boundaries of the method's effectiveness would strengthen the contribution.

4. The paper claims to be the first method for deep architectures in CMPT, but doesn't thoroughly establish why previous CMPT work was limited to shallow architectures or why this transition is non-trivial.

5. The paper mentions pretraining the source prompt on SNLI with RoBERTa but lacks sufficient details about the pretraining process (e.g., number of epochs, convergence behavior, specific hyperparameters beyond what's in the supplement).

6. The technical novelty is limitted in a simple prompt selection module and a diversity regularization scheme. The performance gains are also marginal in most cases.

### Questions
- The statistical significance of the random selection experiment (Figure 2) isn't quantified. While 34 out of 36 trials showing improvement is compelling, a formal statistical test would strengthen this evidence.

- The paper doesn't explore how the method performs with different source modalities beyond text (SNLI/RoBERTa). Testing with other source modalities would strengthen claims about cross-modality generality.

- The paper could have included computational efficiency metrics (training/inference time) alongside parameter efficiency analysis to provide a more complete picture of practical benefits.

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
4

### Summary
This paper proposes a new method called Selective Cross Modal Hint Transfer (S-CMPT), aimed at optimizing cross modal knowledge transfer by automatically selecting the most relevant source hint vectors. This method has its novel theoretical assumptions (challenge layer correspondence and validation source hint redundancy), excellent performance and parameter efficiency, and clear structure.

### Strengths
The core contribution of the article lies in clearly proposing and verifying two key assumptions in cross modal prompt transfer: redundancy in source prompts and suboptimal strict inter layer correspondence. This insight, challenges traditional beliefs through empirical analysis and emphasizes the superiority of selective knowledge adoption over indiscriminate reuse. The proposed method maintains excellent efficiency-efficacy trade-off compared with other SOTA methods.

### Weaknesses
Well-structured paper providing organized motivation but not enough innovation. One motivation of the author is that the original CMPT only used prompts from the corresponding layer, which is redundant and incompatible. Therefore, the selection range of prompts was extended from the corresponding layer to all layers; Although the results may seem effective, they are essentially an improvement in the selection range or hyperparameter configuration, without innovation in the logic of motivation or deeper theoretical exploration.


One of the motivations for the proposed "diversity regularization" is to "promote the use of different vectors by forcing more diverse attention distributions, thereby helping to obtain complementary information for the target task". Although this motivation is reasonable, it lacks specific examples or theoretical basis on how to measure or formalize "complementary information" in existing work, resulting in a lack of rigor in the contribution of this theory.

The performance improvement seems limited.

### Questions
The description in Figure 5 is confusing as to which two parts each row and column in the attention diagram are specifically calculated, and why there is a subscript j for Q in Eq. 5 but not in Eq. 2. How many Q are there in each layer, and what are the dimensions of each Q.

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
This paper addresses inefficiencies in Cross-Modality Prompt Transfer (CMPT), a technique for adapting pretrained models from data-rich modalities (like text) to data-scarce ones (like vision). The authors argue that existing methods suffer from indiscriminately transferring all source prompt vectors, which can include redundant or even detrimental information. This problem is compounded by a rigid layer-wise correspondence assumption, where vectors from a source layer are only transferred to the corresponding target layer, which the paper shows is suboptimal. To solve this, the authors propose Selective Cross-Modality Prompt Transfer (S-CMPT), a method designed for deep architectures that follows a "select-then-transfer" paradigm. S-CMPT uses a lightweight attention mechanism as a "prompt selector" to identify the most relevant vectors from the entire source prompt pool for each target layer. It also introduces a novel diversity regularization term to encourage the selection of diverse, non-redundant vectors. Experiments on the VTAB-1k benchmark demonstrate that S-CMPT achieves state-of-the-art accuracy, outperforming previous prompt-based methods while using significantly fewer parameters.

### Strengths
1. The paper's primary strength is its clear identification of a logical flaw in existing methods, vector redundancy and rigid layer-wise transfer and the proposal of an elegant, simple, and highly effective solution. The authors' hypothesis is convincingly validated before introducing their method via a pilot experiment, which demonstrates that even randomly selecting vectors outperforms the rigid layer-wise baseline. 

2. The proposed S-CMPT method achieves an efficiency-efficacy trade-off by setting a new state-of-the-art on the VTAB-1k benchmark while using 2.5x fewer prompt vectors than the next-best method. This claim is thoroughly supported by extensive ablation studies that systematically test the key design choices, namely the vector selection range (Local, Adjacent, or All) and the impact of the diversity regularization term, proving the robustness of their approach. 

3. Finally, the paper includes helpful visualizations that provide a clear intuition for why the diversity regularization is effective, showing how it forces the model to attend to a broader, less redundant set of source vectors

### Weaknesses
1. The related works are not fully discussed. Only a few works are incorporated and few state-of-the-art methods are included.
2. The paper has claimed that Prior work on CMPT has primarily focused on shallow architectures. However, the paper doesn't demonstrate how about the effects of shallow architectures? How does the proposed method overcome this?
3. The proposed method is relatively simple. A attention operation is adopted to select prompts from the source model, and a diversity loss is used to prompt source distribution.
4. The methods included for comparison are a bit old. Some state-of-the-art methods are not included. Only one work published in 2024 is included. More works should be discussed. For example:
[1] DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning, CVPR 2024
[2] M2PT: Multimodal Prompt Tuning for Zero-shot Instruction Learning, EMNLP2024
[3] FedMVP: Federated Multi-modal Visual Prompt Tuning for Vision-Language Models, ICCV2025
[4] PromptKD: Unsupervised Prompt Distillation for Vision-Language Models, CVPR2024
5. The performance improvement is marginal on most settings. For example, only +0.38%,+2.44%,and +0.18% on VTAB-1k with MoCo V3 pretrained weights.

### Questions
See above

### Soundness
3

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
The paper studies Cross-Modality Prompt Transfer (CMPT) in deep architectures and proposes S-
CMPT, a select-then-transfer pipeline. For each target ViT layer, an attention-based selector picks
vectors from a global pool of source prompt vectors (flattened across all source layers), and a single
linear projector adapts them to the target layer. During training, only selectors and projectors are
optimized; at inference they are discarded, and only the selected prompts remain. On VTAB-1k with
ViT-B/16, S-CMPT achieves strong performance compared to VPT, VFPT, and other parameter-
efficient tuning baselines on both supervised and self-supervised (MAE, MoCo) ViTs. A pilot study
shows that random cross-layer selection beats CMPT in 34 / 36 runs, suggesting redundancy in
source prompts and the limits of rigid layer correspondence.

### Strengths
- Clear motivation. The cross-layer pilot experiment convincingly demonstrates redundancy and cross-
layer compatibility, motivating a global-pool selection strategy.
- Simple and principled. The method uses a lightweight attention selector with diversity regularization to
reduce redundant picks; only selector + projector are trained, while inference keeps just the selected
prompts.
- Consistent gains. The approach performs competitively or better than VPT/VFPT across all VTAB-1k
groups and pretraining types, with fewer tunable parameters overall.

### Weaknesses
- Lack of Np sensitivity study. The method fixes Np = 20 without justification or sweeps (e.g., 4–64)
showing accuracy/memory/throughput trade-offs across task groups and pretraining types.
- Incomplete efficiency analysis. Although inference removes selectors/projectors, the paper omits end-
to-end wall-clock, memory, FLOPs, and throughput comparisons—especially the selector’s attention-
based cost and its scaling with Nl, Np, dh.
- Unclear robustness and negative-transfer boundary. While selection-scope/diversity ablations and
visualizations exist, there are no failure cases, temperature τ or entropy diagnostics, or confidence-
thresholded variants to handle worst-case selections.

### Questions
- Could you provide Np sweeps (e.g., 4–64) and accuracy curves per VTAB group？ Is there a stable
operating range?
- It would strengthen the paper to report end-to-end training/inference metrics versus VPT/VFPT/LoRA:
wall-clock time, peak memory, FLOPs, and throughput. Break down the selector cost.
- Do you observe harmful selections on “distant” tasks? Can τ-adaptation or confidence-thresholded
selection reduce worst-case drops? Please include failure cases and correlations between selection
entropy / max-attention and accuracy.
- The selector uses full-pool attention, which may be costly. Could lighter options—like sparse routing or
clustering-based selection—achieve similar adaptivity with lower compute?

### Soundness
3

### Presentation
3

### Contribution
3
