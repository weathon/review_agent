# Hallucination Begins Where Saliency Drops

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 6, 8, 6, 4

## Abstract
Recent studies have investigated attention dynamics in large vision language models (LVLMs), yet existing methods remain limited in reliably distinguishing hallucinated from correct outputs — primarily because they rely solely on forward-pass attention, ignoring gradient-based signals that reveal how token influence propagates through the model. To bridge this gap, we introduce \textbf{LVLMs-Saliency}, an \textit{gradient-aware diagnostic tool} that quantifies the grounding strength of each output token by fusing attention weights with their gradients. Through analysis, we identify a decisive pattern: \textit{Hallucinations occur when prior output tokens shows low saliency to the next token prediction}, indicating a failure of contextual memory. Building on this insight, we propose a dual-mechanism inference-time framework: (1) Saliency-Guided Rejection Sampling (SGRS), which dynamically filters candidate tokens during decoding by rejecting those with saliency below a context-adaptive threshold, thereby preventing coherence-breaking tokens from entering the sequence; and (2) Local Coherence Reinforcement (LocoRE), a lightweight plug-and-play module that strengthens attention from the current token to its most recent outputs, actively counteracting the “forgetting” behavior identified by LVLMs-Saliency. Experimental results demonstrate that our method significantly reduces hallucinations across multiple LVLMs, offering a robust and interpretable solution to improve model reliability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
## Overview

This paper introduces LVLMs-Saliency, a gradient-aware diagnostic that fuses attention weights with their gradients to quantify how strongly the next token prediction is grounded in prior output tokens. The central empirical finding is that hallucinations emerge when saliency from recent outputs collapses. Building on this, the authors propose two inference-time mechanisms: Saliency-Guided Rejection Sampling (SGRS) to block low-saliency candidate tokens during decoding, and Local Coherence Reinforcement (LocoRE) to strengthen attention from the current token to its most recent outputs. Experiments across multiple LVLMs show reduced hallucinations and improved reliability. 

## Motivation

Existing mitigation methods often rely on forward-pass attention alone, making it difficult to reliably distinguish correct from hallucinated outputs or to explain *why* hallucinations arise in autoregressive generation. Recent “attention sink” analyses add perspective but still lack a token-level, causally informative signal tied to the next prediction. The paper frames this gap and motivates a diagnostic that incorporates gradient information to reveal how token influence propagates, thereby connecting low output-token saliency with hallucination.  

## Methodology

LVLMs-Saliency: For each layer/head, the method computes a saliency matrix as the Hadamard product of the attention matrix and its loss gradient (with causal masking), then aggregates across heads and layers to read saliency from previous outputs to current position. This provides a token-level grounding score for the next token. 

SGRS: During decoding, sample candidates for the next token and accept only those whose grounding saliency exceeds an adaptive threshold tied to recent outputs, thereby preventing low-saliency tokens from entering the sequence. 

LocoRE: After a token is accepted, multiplicatively reinforce attention from the next query to a short window of the most recent output tokens, helping the model maintain local coherence in subsequent steps. 

## Experimental results

Across LLaVA-1.5, Qwen2-VL, and Intern-VL variants, LocoRE and SGRS+LocoRE generally reduce hallucination metrics (lower $CHAIR(_S)$/$CHAIR(_I)$; higher POPE-Recall/F1/Acc) relative to baselines and several decoding-time methods. Representative tables show consistent gains on hallucination benchmarks and competitive performance on a series of VQA suites.  

## Analysis

Ablations on α (SGRS sensitivity) and β (LocoRE gain) indicate that (i) SGRS accounts for most hallucination reduction, (ii) modest LocoRE gains further improve POPE/CHAIR, and (iii) overly large β degrades results; the paper highlights workable settings. The authors also note that LocoRE is forward-only and thus lightweight compared with multi-pass or detector-based methods.

### Strengths
* 1. Clear, mechanistic diagnostic for hallucination.
   The paper isolates a simple, testable pattern—hallucination coincides with a collapse of saliency from prior output tokens to the next token—that standard attention maps miss. The gradient-aware *LVLMs-Saliency* definition is mathematically explicit (Hadamard product of attention and its gradient with causal masking) and tied directly to the next-token prediction.      

* 2. Strong empirical coverage across models and benchmarks and SOTA comparisons to strong baselines.
   Results span LLaVA-1.5 (7B/13B), Qwen2-VL (7B/13B/32B), and Intern-VL (7B/13B), with evaluations on MME (Hallucination), MM-Vet, VizWiz, ScienceQA, POPE, and CHAIR. Gains are consistent on hallucination metrics (lower $CHAIR(_S)$/$CHAIR(_I)$, higher POPE Recall/F1/Acc) and often maintain or improve general VQA ability.  
   On LLaVA-1.5-7B, LocoRE and especially SGRS+LocoRE outperform or match recent training-free methods on POPE/CHAIR while remaining competitive on MME; the tables include recent ICLR/NeurIPS/CVPR baselines.   

* 3. Ablations and hyperparameter guidance.
   The work analyzes the effect of α (SGRS sensitivity) and β (LocoRE gain), and discusses trade-offs among hallucination rate, recall, and latency, giving practical guidance for tuning. 

* 4. Compelling qualitative evidence aligned with the mechanism.
   Visualizations show cases where attention maps look similar for correct vs. hallucinated tokens, but saliency sharply diverges; with LocoRE, the same position flips from an incorrect token to a correct one alongside restored output-token saliency.

### Weaknesses
* 1. Quantitative validation of the saliency–hallucination link is missing.
  The saliency-guided hallucination detection has been shown through cases, but not quantitive analysis. Does such phenomenon always appear in no matter short or long generated content? and does low saliency always suggest hallucinated content, or hallucination content are more likely to be with low saliency? Authors are suggested to conduct quantitive analysis for their saliency-guided motivation.

* 2. SGRS behavior during inference is under-specified.
   Report per-benchmark stats: percentage of steps with at least one rejection, average resamples per token, distribution tails, and a false-rejection rate for correct tokens. Break out short QA vs. long captioning.

* 3. Hyperparameter portability is unclear.
   Seems that the hyper-parameters alpha and beta need to be adjusted for each new MLLM. Does it have to take several rounds of tests till an optimal setting comes out? 

* 4. Causality vs. correlation.
   Can authors demonstrate that low output-token saliency is not merely correlated with position/length, as late tokens naturally have lower influence?

* 5. Layer/head dependence.
   SGRS aggregates over “target layers”. Can authors provide sensitivity to layer/head selection and show whether the diagnostic holds with earlier vs. deeper layers?

* 6. Unclear Computation Cost.
    Many production stacks don’t expose backward passes. Authors are suggested to provide full memory cost analysis.

### Questions
Please refer to Weaknesses section. I would consider raising my score if authors can address my concerns.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper identifies an interesting pattern in LVLMs: Hallucinations occur when prior output tokens shows low saliency to the next token. Building on this, the authors propose two mechanism for mitigating halluciantion: (1) Saliency-Guided Rejection Sampling (SGRS), which  filters candidate tokens based on their saliency with respect to prior output context. (2)  Local Coherence Reinforcement (LocoRE),  strengthens attention from the current token to its most recent predecessors.  With extensive experiments, the proposed demonstrates
significant hallucination-mitigating performance across different LVLMs on image hallucination and generation benchmarks.

### Strengths
The paper offers a novel and well-motivated perspective on hallucination detection: leveraging gradients of attention weights to localize hallucination-prone tokens. To the best of my knowledge, this is the first systematic use of this signal, underscoring the work’s novelty. The two inference-time interventions—SGRS and LocoRE—are tightly coupled to this insight and translate it into practical, training-free improvements with minimal changes to the base model. The experimental study is thorough, and consistently shows gains over strong baselines, lending credibility to the approach. The paper is clearly written and easy to follow (I enjoy reading this paper); overall, the contribution is both original and useful.

### Weaknesses
My main concern is that the claimed “hallucination pattern” is supported largely by a few curated cases (Figs. 1–2), which risks selection bias. To make the claim compelling, the paper should provide population-level, statistically significant evidence to substantiate this claim.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

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
This paper proposes "LVLMs-Saliency," a diagnostic tool for quantifying token-level grounding in large vision-language models (LVLMs), aiming to better distinguish hallucinated from correct outputs by combining attention weights and their gradients. From the identified pattern—hallucinations arising where saliency from previous output tokens drops—the authors introduce a dual mechanism: Saliency-Guided Rejection Sampling (SGRS) for filtering low-saliency tokens during decoding, and Local Coherence Reinforcement (LocoRE) to strengthen attention among recent outputs, both at inference-time. Experimental results on major benchmarks and several popular LVLMs demonstrate significant hallucination mitigation and competitive or improved performance metrics.

### Strengths
1. The paper provides a compelling, interpretable explanation of LVLM hallucinations, showing through both empirical results and visualizations that attention-alone is insufficient and that joint attention-gradient saliency captures key failure modes (see Figures 1 and 2). 

2. The proposed SGRS (Algorithm 1) and LocoRE (Algorithm 2) do not require retraining, operate at inference, and effectively use saliency to dynamically control and reinforce generation. This “plug-and-play” aspect increases their practical value.

3. Tables 1 and 2 demonstrate consistently strong gains in hallucination reduction (e.g., substantial CHAIR-S and POPE improvements), with Table 3 and Figure 5 providing ablation studies on key hyperparameters ($\alpha$, $\beta$), establishing the contribution of each module.

4. The latency/efficiency trade-off discussion (Section 4.4.1, Figure 4) is practical and relevant for deployment scenarios, and the LocoRE-only alternative appears effective for large models or latency-critical settings.

### Weaknesses
1. The SGRS component's reliance on backward passes during inference imposes significant memory constraints, limiting its applicability to models up to 13B parameters and preventing scalability to larger LVLMs like Qwen2.5-VL-32B or 72B, which undermines the method's claimed broad generalizability and real-world deployment feasibility.

2. While the paper asserts a direct causal link between low saliency and hallucinations, the evidence is primarily correlational from observational patterns in figures and benchmarks, lacking rigorous controlled experiments or ablations to demonstrate causality, such as interventions that artificially manipulate saliency levels to observe corresponding changes in hallucination rates.

3. Despite strong aggregate benchmark results, the paper provides virtually no error analysis or qualitative examination of failure cases, such as specific hallucination types (e.g., knowledge-based or relationship errors) that the method fails to mitigate, domain-specific limitations like medical imaging, or edge cases involving rare objects or noisy inputs, which obscures the true robustness and variability of the approach.

### Questions
1. Given the memory-intensive nature of SGRS due to backward passes, which limits testing to models up to 13B parameters, could the authors explore or implement optimizations?

2. The paper claims a "direct causal link" between low saliency and hallucinations, but the supporting evidence appears largely correlational. Could the authors provide controlled experiments, such as synthetic interventions where saliency is artificially boosted or reduced in specific output tokens, to demonstrate causality (e.g., measuring resulting changes in hallucination rates on a subset of benchmarks like CHAIR or POPE)? 

3. The absence of error analysis leaves unclear when and why the method fails. Could the authors include a qualitative breakdown of failure modes?

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
Many previous studies have focused on reducing hallucination by leveraging attention-based information. In contrast, this paper innovatively employs saliency (i.e., gradient information) to mitigate hallucination. By incorporating the saliency of previous tokens to guide the prediction of the next token, the authors propose SGRS and LocoRE modules to reduce hallucination.

### Strengths
The proposed method in this paper is highly novel. While most previous studies have focused on leveraging attention mechanisms to reduce hallucination, this work explores the use of saliency, which represents a promising and valuable direction for further research.

### Weaknesses
1. The key finding and the basis for the proposed method in this paper is that “Hallucinations occur when prior output tokens show low saliency to the next token prediction, indicating a failure of contextual memory.” However, after reading the entire manuscript, I could not find any statistically significant validation of this claim. Figure 1 appears to be merely a case study and does not provide statistical evidence to support the finding.

2. From Table 1, the proposed method does not appear to demonstrate a statistically significant improvement over previous approaches. The performance differences are relatively minor and may fall within the margin of experimental variability.

3. The proposed method relies heavily on a large number of hyperparameters, which significantly limits its practical applicability. Moreover, the paper does not provide a comprehensive analysis of the impact of these hyperparameters—only a partial examination is presented. For instance, the parameters required in Algorithm 1 and Algorithm 2 are not thoroughly analyzed or discussed.

### Questions
see Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
