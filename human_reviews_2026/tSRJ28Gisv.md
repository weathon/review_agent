# MTRE: Multi-Token Reliability Estimation for Hallucination Detection in VLMs

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Vision–language models (VLMs) now rival human performance on many multimodal tasks, yet they still hallucinate objects or generate unsafe text. Current hallucination detectors, e.g., single-token linear probing (LP) and $P(\text{True})$, typically analyze only the logit of the first generated token—or just its highest-scoring component—overlooking richer signals embedded within earlier token distributions. We demonstrate that analyzing the complete sequence of early logits potentially provides substantially more diagnostic information.
We emphasize that hallucinations may only emerge after several tokens, as subtle inconsistencies accumulate over time. By analyzing the Kullback–Leibler (KL) divergence between logits corresponding to hallucinated and non-hallucinated tokens, we underscore the importance of incorporating later-token logits to more accurately capture the reliability dynamics of VLMs.
In response, we introduce Multi-Token Reliability Estimation (MTRE), a lightweight, white-box method that aggregates logits from the first ten tokens using multi-token log-likelihood ratios and self-attention. Despite the challenges posed by large vocabulary sizes and long logit sequences, MTRE remains efficient and tractable. Across MAD-Bench, MM-SafetyBench, MathVista, and four compositional-geometry benchmarks, MTRE achieves a 9.4% gain in Accuracy and a 14.8% gain in AUROC over standard detection methods, establishing a new state of the art in hallucination detection for open-source VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces MTRE, a lightweight white-box method for detecting hallucinations in Vision-Language Models (VLMs). Unlike existing approaches that rely only on the first token's logits, MTRE aggregates information from the first ten tokens using multi-token log-likelihood ratios and self-attention. The method is shown to outperform baseline detectors across multiple benchmarks, including MAD-Bench, MM-SafetyBench, MathVista, and arithmetic tasks.

### Strengths
1. This paper proposes a novel hallucination detection criterion, which captures reliability signals through the logits of multiple output tokens.
2. Extensive observational experiments are provided, particularly the analysis and discussion of different types of detection tasks.
3. The proposed method appears to be effective and exhibits promising performance.

### Weaknesses
1. As described in the limitations section, the number of open-source VLMs validated by this method is limited. I suggest conducting illusion detection tests on larger-sized VLMs, such as those larger than 32B, to see if the experimental observations also hold true for larger models.
2. Why start with 10 tokens instead of 20? What are the reasons and criteria for this choice?
3. In practice, the proposed method relies to some extent on supervised information. However, in real-world scenarios, data from unseen domains is often more common. Does the proposed method possess any cross-domain capabilities?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper targets VLM hallucination detection and argues that single-token probes (LP, P(True)) miss reliability signals distributed across early token sequences. Empirically, KL divergences between hallucinated vs. truthful continuations grow over several tokens, motivating multi-token analysis. The authors propose MTRE, a lightweight white-box detector that aggregates the first 10 tokens’ logits via multi-token log-likelihood ratios and a small self-attention module; it remains tractable despite large vocabularies.

### Strengths
1. The topic is interesting and tries to address an important problem.

2. The paper is well written

### Weaknesses
1. The baseline model should add some new models, like LLaVA 1.5, LLaVA NeXT, Qwen 2.5 VL.

2. Figure 1 seems unclear, I recommend the authors add more explanations.

3. For the benchmark discussion, note that several recent studies [1, 2, 3] address both hallucination and maintain performance (even some improvement) on general scenario. I recommend the authors add some benchmarks like OCRBench, MMMU, MME etc.

[1] Mitigating Object Hallucinations via Sentence-Level Early Intervention.

[2] A topic-level self-correctional approach to mitigate hallucinations in mllms.

[3] Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness.

### Questions
See above.

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
This paper addresses the hallucination detection issue in Vision-Language Models (VLMs) by proposing a novel method called Multi-Token Reliability Estimation (MTRE) and its variant, MTRE-$\tau$ . Departing from prior work of prioritizing the first generated token's logit, MTRE leverages the distributional shift across a sequence of early generated logits to extract richer diagnostic signals. The authors design a reliability classifier, $f_{\theta}$, trained on token-level labels, and aggregate its log-likelihood ratios (LLRs) over a dynamically determined or fixed number of early tokens. Experiments demonstrate MTRE's competitive performance, especially in scenarios where hallucinations emerge late in the sequence (Type II setting).

### Strengths
Clear Motivation and Rationale: The fundamental premise—that the full sequence of early logits contains more diagnostic information than the single first token—is clearly articulated and well-supported by preliminary analysis (Section 3). 

Sound Empirical Insights: The paper provides several valuable empirical observations (Section 3 and 5) that are useful for the VLM reliability community. For instance, the finding that hallucination divergence may emerge late in the sequence, and consequently, that single-token probing methods (e.g., Zhao et al., 2025) are suboptimal for Type II settings, is a significant takeaway.

Comprehensive Experimental Setup and Clarity: The authors provide a sufficiently detailed description of the experimental setup, particularly in the Appendix regarding model training and hyper-parameter choices. The experimental validation is thorough across various VLM architectures and hallucination settings (Type I and Type II).

Solution to Core Hyper-parameter Identification: The introduction of the MTRE-$\tau$ variant (Section 4.3), which employs cross-fitting for prior estimation (minimizing token-broadcast binary cross-entropy) and dynamic evidence length determination ($\tau_{s_{i}}$), offers a practical approach to tackling the sensitivity of the aggregation step to hyper-parameters.

### Weaknesses
Major Concerns

1. Limited Efficacy in Type I Setting: While MTRE significantly outperforms baselines in the Type II setting, its performance edge over the competitive Linear Probing (Lin. Prb.) baseline on Type I tasks (more common ones) is often marginal. 

2.  Incremental Gain of MTRE-$\tau$: MTRE-$\tau$ fails to demonstrate a clear and significant performance improvement over MTRE. Given that MTRE-$\tau$ introduces substantial additional complexity (cross-folding, parameter calibration, optimization for $C_{u}$ and $C_{b}$), the marginal utility of this variant is questionable. 

3. Extra Computational Cost Analysis: The MTRE method inherently requires training an auxiliary reliability classifier $f_{\theta}$, which increases complexity relative to post-hoc methods. While some performance gain justifies this, the paper lacks a detailed comparative analysis of the computational overhead.  A clearer comparison of the inference time complexity of MTRE against Lin. Prb. needs to be provided.

Minor Concerns

1. The labels OE and OEH in Figure 2 are not immediately clear from the main body text or caption. It would be better to explicitly explain  these abbreviations in the figure caption or the accompanying paragraph.

2. The definition of the $\sigma(\cdot)$ function used in the prior estimation loss (5) should be stated in Section 4.3 (though it is provided in the Appendix)

### Questions
Regarding the inference time reported in Table 3 (0.944 ms), please clarify whether this average inference time is measured per single generated token or per complete Q&A query (statement). Even if it is per complete query, the time consumption is highly significant, potentially negating the benefits for real-time deployment or subsequent hallucination mitigation steps (e.g., new inference, re-prompting, or post-hoc corrections).

### Soundness
3

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
This paper proposes MTRE (Multi-Token Reliability Estimation), a novel white-box hallucination detection framework for vision-language models (VLMs). Unlike existing single-token probing or P(True) methods that focus on the first generated token, MTRE aggregates information from the logits of multiple early tokens (typically the first 10) to estimate model reliability. By leveraging sequential log-likelihood ratios, attention-based classifiers, and calibration via cross-fitting, MTRE captures reliability dynamics that unfold across token generation. Experimental results on MAD-Bench, MM-SafetyBench, MathVista, and several arithmetic reasoning datasets show consistent gains over strong baselines like linear probing and TokenSAR.

### Strengths
1. The paper identifies a key limitation of previous approaches that rely solely on the first-token logit, showing through KL divergence analysis that hallucination-related divergence often arises in later tokens. This motivates the multi-token design in a theoretically grounded way.

2. MTRE introduces a lightweight yet principled multi-token aggregation method, formulated as a calibrated sequential log-likelihood ratio test. The design effectively balances interpretability and computational efficiency.

3. The authors conduct evaluations on four open-source VLMs (LLaVA-v1.5, mPLUG-Owl, LLaMA-Adapter V2, MiniGPT-4) and multiple benchmarks. The results demonstrate robust and consistent improvements, including on challenging self-evaluation (Type 2) settings.

4. MTRE adds negligible overhead (<1% VRAM and inference time) and does not require retraining large models, making it feasible for deployment in real-world systems.

### Weaknesses
1. The current comparison is restricted to models such as LLaVA-v1.5 (7B), mPLUG-Owl, LLaMA-Adapter V2, and MiniGPT-4, which may now be considered relatively early-generation VLMs. It remains unclear how MTRE performs on more recent and stronger models (such as LLaVA-Next, InternVL2 or Qwen2.5-VL), which exhibit lower hallucination rates. Incorporating these models would better demonstrate the generality and contributions of MTRE.

2. All experiments use 7B-scale models. It would be informative to analyze whether MTRE scales effectively to larger architectures (e.g., 13B) or smaller lightweight models (3B in scale).

3. The current datasets primarily involve English and relatively simple visual reasoning. Evaluating on multilingual or real-world datasets (e.g., OCR-heavy or video-based tasks) could further validate robustness.

### Questions
1. Include Stronger Baselines: Evaluate MTRE on newer VLMs such as LLaVA-Next, Qwen2.5-VL, or InternVL3 to verify whether multi-token reliability estimation remains beneficial when hallucinations are rarer but more subtle.

2. Ablation Across Model Sizes and Modalities: Analyze how performance scales with model size and whether similar gains appear for larger or multilingual models.

### Soundness
3

### Presentation
3

### Contribution
3
