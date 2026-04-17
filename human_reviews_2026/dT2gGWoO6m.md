# VIBRA: Redundancy-Aware Information Bottleneck for Hallucination-Resistant Vision-Language Models

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Vision-Language Models (VLMs) have achieved impressive progress across a range of multimodal tasks but remain highly susceptible to visual hallucination, producing text that contradicts the visual input. Existing mitigation strategies often rely on additional large-scale VLMs or multi-stage decoding, which hinders efficiency and broad applicability. In this work, we identify redundant and noisy image features as a primary cause of hallucination, as they degrade the model’s ability to capture semantically relevant visual content. Correspondingly, we propose VIBRA (Vision-Language Information Bottleneck with Redundancy Awareness), a lightweight and plug-and-play module that adaptively filters out redundant visual information while preserving task-relevant semantics at both the token and feature levels. Specifically, VIBRA employs a multi-modal information bottleneck to retain image features aligned with textual input and introduces adaptive token filtering through spectral clustering and compression-aware pruning to eliminate instance-specific redundancy. Additionally, we design a Binary-Guided loss to sharpen the separation between informative and noisy features, enabling more effective visual information gating. Extensive experiments demonstrate that VIBRA consistently enhances visual reasoning and reduces hallucination across a variety of VLM architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper is motivated by the assumption that images processed by VLMs contain a lot of **unnecessary and redundant semantics, and the authors claim that it is the core reason leading to hallucinations**. To address this, the authors propose a Multi-model Information Bottleneck (MIB) module that filters out redundant information in image tokens and retains only text-relevant features. Tokens that pass through MIB are then clustered, and the largest cluster—assumed to be redundant—is removed. The core idea is that this reduces visual hallucination.

### Strengths
Despite the drawbacks discussed below, the Multi-modal Information Bottleneck is a solid contribution. Compared to simple image-token filtering, it offers a good technique for distilling text-relevant information, and the results are promising.

### Weaknesses
- **(Very major)** The paper’s motivation is that unnecessary and redundant information in a given image causes hallucinations. However, the only supporting evidence is a single simple experiment in Figure 1, and removing tokens changes performance by merely 0.3–0.4%. The core motivation of this paper lacks sufficient grounding, which is a critical issue.

- **(Very major)** On ScienceQA, VIBRA is combined on top of MM-CoT and MC-CoT. Since these CoT-style methods already outperform the baseline, claiming further gains by adding VIBRA makes a fair comparison impossible. All baselines should either be equipped with MM-CoT, or VIBRA should be paired with a simple CoT to unify the experimental setting.

- **(Major)** While criticizing prior methods for using a fixed token filter ratio (and thus lacking adaptability), VIBRA fixes the number of clusters to $K = 5$ for spectral clustering. It then treats the largest cluster as “visual redundancy” and removes it. This raises questions: that cluster could contain important information, and the second-largest cluster might be of comparable size. Overall, the process is coarse, yet there is no analysis or justification.

- **(Major ~ Minor)** Although the paper critiques the inference cost of prior work, it provides no analysis of VIBRA’s additional cost. VIBRA also incurs overhead for computing the information bottleneck, clustering, pruning, and then recomputing cross-attention with text tokens.

### Questions
- Since the MIB Compression Term is proposed around Intro line 62, it would be helpful to include at least a brief introduction to it there.

- In Table 1, the model-size reporting is unfair. Because these are plug-and-play methods, both the backbone model parameter count and the adapter-module parameter count should be reported for fair comparison.

### Soundness
1

### Presentation
1

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
This paper tackles visual hallucination in Vision–Language Models (VLMs) by reducing redundant and noisy image features, which the authors identify as a primary cause of hallucination. They propose VIBRA, a plug‑and‑play module that filters redundant visual features via a multimodal information bottleneck (MIB) objective and adaptive token filtering using spectral clustering with compression‑aware pruning. In addition, they introduce a Binary‑Guided (BG) loss that encourages near‑binary separation between informative and noisy features, improving visual information gating. In experiments, the method enhances visual reasoning and reduces hallucinations across several VLM architectures.

### Strengths
S1. Each component of VIBRA is described clearly and concisely. The MIB objective is well derived, with appropriate references supporting the upper bound on $ I(Z_v; X_v) $ and the lower bound on $ I(Z_v; X_t) $.

S2. The ablation studies support the contribution of each component (MIB, TFSC, and BG; Sec. 5.3). In addition, the distribution of the learned gating parameter $ \lambda_n $ with and without the proposed BG loss supports the validity of the objective.

S3. The authors provide qualitative results by visualizing the original images and their saliency maps derived from the MIB compression term (App. A.6).

### Weaknesses
W1. **Marginal novelty.**
The use of the information bottleneck for hallucination mitigation has prior art. For example, variational information bottleneck has been introduced for reducing object hallucination [1]. The paper’s novelty lies more in the system‑level combination (MIB + spectral clustering with compression‑aware pruning + BG loss) than in fundamentally new primitives.

W2. **Lack of token‑pruning baselines.**
The current comparisons focus on decoding‑time mitigation (e.g., VCD, OPERA, HALC, SID). Because the proposed method performs token filtering, pruning/merging‑based baselines should be included for direct head‑to‑head comparison under the same backbone and token budget (e.g., SimIgnore, PuMer, PruMerge, FastV).

W3. **Missing efficiency report.**
While the paper highlights a benefit of “no extra decoding stages,” the efficiency side is not yet quantified. It would strengthen the work to include an accuracy–efficiency Pareto—e.g., per‑image wall‑clock latency (ms) and images/sec—with the module on/off and across different  $k$ values (same hardware/backbone).

[1]:  Bai et al., Mitigating Hallucinations in Large Vision-Language Models by Adaptively Constraining Information Flow. Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

### Questions
Please see the weaknesses above.

### Soundness
3

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
4

### Summary
This paper proposed a  plug-and-play module to  mitigates hallucination and enhances reasoning in vision-language models (VLMs) by employing an IB-Net and designing an appropriate mutual information–based loss function to reduce redundant information between textual and visual modalities.

### Strengths
This paper is very well-presented, and the proposed method is quite novel.

### Weaknesses
1. This paper claims that *“we identify redundant and noisy image features as a primary cause of hallucination.”*
   However, the paper does not provide any statistically significant experiments to validate whether redundant and noisy image features indeed lead to hallucination.

2. I have some concerns about the design of Equation (3). Why does the input to $f_{\mathrm{IB}}$ depend only on $x_n^v$ rather than on both $x_n^v$ and $x_n^t$? Can such a design truly enable $Z_v$ to effectively remove (at least part of) the information from $X_t$?

3. Similar to Comment 1, it would be more convincing if the paper provided statistical results showing how different levels of $I(Z_v; X_t)$ (for example, by varying $\beta$) affect the final performance. Such an analysis would help demonstrate that the effectiveness of the proposed method indeed stems from the mechanism claimed in the paper.

4. What data was used to train VIBRA? Is there any overlap between the training data and the data used for evaluation?

### Questions
see Weaknesses

### Soundness
2

### Presentation
3

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
The paper proposes VIBRA—a plug-and-play, redundancy-aware information bottleneck for VLMs—to mitigate visual hallucination without multi-stage decoding or extra large auxiliaries. VIBRA works at two levels: (1) a Variational Multi-Modal Information Bottleneck (MIB) learns per-token gates to inject noise into text-irrelevant visual features while preserving text-aligned ones; (2) Adaptive Image Token Filtering via Spectral Clustering (TFSC) prunes instance-specific redundant image tokens using a compression-aware criterion rather than attention or cosine similarity. A Binary-Guided loss polarizes the gates toward near-binary keep/drop decisions. Integrated with MM-CoT/MC-CoT (ScienceQA) and with MiniGPT-4/LLaVA-1.5 (POPE/OPOPE/MME), VIBRA consistently improves reasoning accuracy and reduces hallucinations, while keeping inference simple.

### Strengths
- Clear plug-and-play design: Works with diverse VLMs (reasoning-centric and LVLMs) without extra decoding rounds; training only a lightweight module (e.g., 5k image–text pairs for LVLMs).
- **Principled objective:** MIB trades off compression against text-alignment with closed-form bounds and a simple per-token gating implementation.
- Adaptive token pruning: TFSC removes instance-specific redundancy; using the MIB compression term outperforms cosine-similarity based selection across retained-token budgets.
- Consistent empirical gains: +2.92 / +3.87 points on ScienceQA (MM-CoT / MC-CoT), and improved POPE/OPOPE/MME scores with MiniGPT-4 and LLaVA-1.5, indicating both better reasoning and reduced hallucination.

### Weaknesses
- Estimation assumptions & sensitivity: Mutual-information bounds rely on Gaussian/noise assumptions and inner-product density approximations; analysis of sensitivity to beta, delta, and gate dynamics is limited.
- Clustering overhead & robustness: Spectral clustering with fixed k=5 may raise compute cost and stability concerns on high-resolution or long-token inputs; ablations on k, approximations (e.g., Nyström), or batch variants are brief.
- Evaluation breadth: While ScienceQA/POPE/OPOPE/MME are strong, broader open-ended captioning or complex real-world scenes could further substantiate generality of hallucination mitigation.

### Questions
- Hyperparameters: How sensitive are results to beta(MIB trade-off) and the Binary-Guided loss weight? Is there a transferable rule of thumb across backbones/datasets?
- Token filtering design: How do different k values and similarity graphs (e.g., cosine vs. learned metrics) affect TFSC? Could lightweight eigen-approximations retain most gains?
- Latency & memory: What are end-to-end inference overheads (mean/percentiles) from MIB + TFSC on LVLMs compared to decoding-based methods?
- Interpretability overlap: When MIB-based importance disagrees with attention or Grad-CAM, what patterns emerge, and can the signals be combined for further gains?

### Soundness
3

### Presentation
3

### Contribution
2
