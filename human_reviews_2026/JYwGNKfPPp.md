# Unveiling Super Experts in Mixture-of-Experts Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Leveraging the intrinsic importance differences among experts, recent research has explored expert-level compression techniques to enhance the efficiency of Mixture-of-Experts (MoE) large language models (LLMs).
However, existing approaches often rely on empirical heuristics to identify critical experts, while lacking a deeper understanding into the heterogeneous importance of experts and the inner workings of MoE LLMs.
In this study, we report, for the first time, the discovery and systematic investigation of a distinct subset of experts that play a pivotal role in the model's forward inference.
These experts are prevalent in open-source MoE LLMs, and despite their extremely limited number, pruning them results in a substantial decline in model performance (e.g., prune just three out of 6,144 causes Qwen3-30B-A3B to generate repetitive and uninformative outputs).
We refer to these experts as Super Experts (SEs).
Our comprehensive analysis provides progressively deeper insights into SEs: 
(i) SEs are characterized by rare but extreme activation outliers in the output of the down\_proj, which give rise to massive activations in the
hidden states between decoder layers.
Moreover, the distribution of SEs is model-specific, data-agnostic, and remains unaffected by post-training processes.
(ii) By pruning SEs, we assess their significance across a variety of tasks, revealing their considerable impact on the model's overall performance, particularly in mathematical reasoning.
(iii) We further investigate why compressing SEs exerts such a pronounced impact. 
We show that, in MoE LLMs, SEs serve as the primary source of the systematic outlier mechanism in Transformers, and that compressing them profoundly disrupts this process, ultimately causing the collapse of attention sinks. 
These findings advance the understanding of the internal dynamics of MoE LLMs, filling an important gap in the current knowledge.
In addition, we developed an automated tool for rapid and accurate SE profiling.
The code is provided in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates “super experts", specialized experts within large Mixture-of-Experts language models that disproportionately influence performance. The authors identify such experts through gradient-based sensitivity analysis and routing statistics, arguing that a small subset of experts accounts for the majority of downstream quality.

### Strengths
[1] Introduces the super-expert hypothesis, offering a conceptual lens distinct from purely performance-based pruning. Connects findings to neural specialization, offering a biological analogy that enriches interpretability discussions.  
[2]  Strong empirical evidence across diverse MoE models (e.g., Mixtral-8×7B, DeepSeek-R1) and tasks (MMLU, GSM8K, CodeEval).   Good balance between theoretical framing and empirical results.    
[3] Clear motivation and well-structured narrative.

### Weaknesses
[1]  If “super experts” arise due to overexposure to specific domain tokens, the effect might reflect data imbalance rather than emergent specialization. The paper does not control for or analyze this possibility.    
[2] Results focus on held-out benchmarks similar to pretraining data. It is unclear if retaining only super experts maintains out-of-distribution robustness.   
[3]  There is little theoretical explanation for why certain experts become super experts.

### Questions
See weakness.  

Are “super experts” persistent across training epochs, or do they shift over time?

It would be valuable to provide per-domain heatmaps showing which experts dominate specific token types or tasks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce the concept of Super Experts (SEs) in Mixture-of-Experts (MoE) Large Language Models (LLMs), identifying them as the main source of activation outliers, a phenomenon analogous to the massive activations (MA) seen in dense LLMs. These SEs are few in number, yet their pruning severely degrades performance. The extreme activation outliers causing this behavior are found to arise specifically from the `down_proj` operation within the expert layers. A series of experiments confirmed that SEs share key characteristics with MA, such as being model-specific, data-agnostic, and having a relationship with Attention Sink (AS) tokens. Furthermore, their distributions are noted to remain unaffected after post-training, and their impact was demonstrated across both reasoning and non-reasoning tasks.

### Strengths
This work is the first to analyze the source of massive activations (MA) in Mixture-of-Experts (MoE) LLMs, providing a quantitative definition for this source by introducing the concept of Super Experts (SEs). The paper is well-written and easy to follow, and its core claims are supported by extensive experimental evidence.

### Weaknesses
See questions below.

### Questions
1. Since Super Experts (SEs) and Massive Activations (MAs) are hypothesized to function as an implicit, input-independent bias term, one would expect them to be primarily localized in global components, such as the shared expert in models like DeepSeek-R1, which is universally activated by all tokens. Therefore, could the authors provide a detailed explanation as to why SEs appear in both normal (token-specific) and shared experts, as demonstrated by the distribution shown in Table 7?

2. The original research on Massive Activations (MAs) in dense LLMs demonstrated that their presence could be eliminated by introducing an explicit, input-independent bias term to the hidden states. Have the authors investigated whether implementing a comparable, explicit bias term technique in MoE models can eliminate the appearance of Super Experts (SEs) or significantly mitigate their performance-critical reliance on them?

3. A key contribution of this work is the distinction between activation outliers in MoE and dense LLMs. The observation that "Unlike dense LLMs, where such behavior typically occurs within a single layer, MoE models exhibit the progressive formation of systematic outliers by SEs across multiple layers." (illustrated in Figures 7 and 19) is particularly intriguing but lacks detailed discussion. Could the authors elaborate on this "progressive formation" phenomenon, providing a comparative analysis of why this multi-layer progression is necessary or emerges in the MoE architecture, while the analogous behavior in dense LLMs appears to be self-contained within a single layer?

4. Could the authors include an extra section that summarizes the similarities and differences between the activation outlier phenomena in dense LLMs and MoE models (maybe in the appendix)? This section should detail their shared characteristics, such as both being model-specific, data-agnostic, and having a relationship with Attention Sink (AS) tokens, while clearly highlighting key distinctions, such as the outlier source transitioning from a single-layer event in dense models to a multi-layer progressive formation in MoE architectures, and so on.

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
4

### Summary
This paper investigates the heterogeneous importance of experts in Mixture-of-Experts (MoE) large language models. The authors report the discovery of a small, distinct subset of experts, which they term "Super Experts" (SEs). The central claim is that these SEs, despite being extremely rare (e.g., 3 out of 6,144 in Qwen3-30B-A3B), are fundamentally critical to the model's performance.

The authors provide strong empirical evidence showing that pruning just these few SEs causes a catastrophic collapse in model performance, including massive increases in perplexity and near-zero accuracy on reasoning tasks. In contrast, pruning a large number (e.g., 1000) of non-SEs has a negligible impact.

Mechanistically, the paper links SEs to the phenomenon of "massive activations" (MAs). It proposes that SEs are characterized by rare but extreme activation outliers in their down_proj output. These outliers propagate through residual connections, creating MAs in the hidden states. The paper further hypothesizes that these SE-generated MAs are the primary source of the systematic outlier mechanism in Transformers, and that their function is to create and maintain "attention sinks" (ASs). Pruning the SEs breaks this mechanism, causing the attention sink to collapse and, consequently, the model's performance to degrade, often resulting in repetitive and uninformative output. The authors also find that SE distribution is model-specific, data-agnostic, and stable through post-training.

### Strengths
1. **Interesting Empirical Finding**: The paper identifies a clear and striking phenomenon: a tiny, identifiable subset of experts is responsible for the vast majority of model stability. The empirical evidence, particularly the stark contrast between pruning SEs versus a large number of random experts (Fig. 1, Tables 3-5), is convincing.

2. **Plausible Mechanistic Explanation**: The paper provides a commendable in-depth analysis of why these experts are so critical. It successfully links the MoE structure to the known "massive activation" and "attention sink" phenomena in dense Transformers. 

3. **Thorough Investigation**: The authors analyze this phenomenon across multiple models (Qwen, DeepSeek, Mixtral) and datasets, demonstrating that the presence of SEs is a consistent pattern. The analysis of router scores (Fig. 6), showing that sink tokens are preferentially routed to SEs, is a key piece of evidence for the proposed mechanism.

### Weaknesses
1. **Is this a new discovery or a restatement?** The paper's core finding is that SEs cause MAs, which in turn create attention sinks. The phenomenon and importance of massive activation are already known in the established literature. The paper's main contribution seems to be locating the source of MAs within specific experts in MoE models. In appendix H, the author also provides analysis on locating superweights within super experts, causing massive activation. It's unclear to me whether "Super Expert" is a new concept or just "the expert containing the MA neurons." The framing suggests a novel discovery, but it might be more of a localization and restatement of an existing concept

2. **"Super” experts could be misleading**: My **most significant concern** is the framing of the discovery. The term "Super Expert" implies these experts are qualitatively "better," "stronger," or has a superior capability. However, the paper's own evidence strongly suggests they are highly specialized components. Their function appears to maintain the attention sink. Their "importance" stems from the fact that the model's internal mechanism is critically dependent on them; pruning them removes attntion sink, causing collapse. This is different from removing the "smartest" expert. This naming convention is misleading and obscures the paper's core finding, which is about a specialized mechanism, not capability.

3. **Expert-level or neuron-level phenomena**: The paper frames this as an expert-level discovery. However, the mechanism is attributed to "extreme activation outliers" which, as investigated in Appendix H, can be traced to specific "Super Weights" (SWs) or neurons. This raises the question: is this an expert-level phenomenon, or are we simply observing the location of a few critical neurons (which cause massive activation)? The findings in Appendix H, where pruning just the neurons causes a similar collapse, suggest the latter. This point significantly muddies the central claim that this is about "experts" as a unit.

4. **Excluding final-layer activations is unconvincing**: The paper notes (primarily in Appendix C) that other experts, particularly in the final layers, also exhibit extreme activation outliers. These experts are excluded from "SEs" because they "do not contribute to the formation of MAs" and pruning them does not impact performance (Table 6). This reasoning feels post hoc and incomplete. In my opinion, it seems that removing MA in the final layer does not destroy the attention sink and thus does not destroy the performance. But this also corresponds to weakness 2; the findings in this paper should not be framed as identifying the super expert in the model.

### Questions
1. The analysis in Appendix H on "Super Weights" (SWs) is fascinating and, in my view, central to the paper's claim. It shows that pruning just a few neurons can replicate the catastrophic failure. To de-conflate the "expert" vs. "neuron" contributions, have the authors considered two complementary experiments? a) prune only the SWs identified in Appendix H, but leave the rest of the SEs' parameters intact. b) prune the entire SE except for the identified SWs. These experiments would clarify whether the model relies on the entire expert or just on these specific activation-generating channels.

2. Would the authors consider a more descriptive and less misleading name for this phenomenon to more accurately reflect the paper's findings? 

3. Can the authors please clarify the definition of the layer set $L$ (layers responsible for MA formation) in Algorithm 1? How is this set determined a priori? How does this definition mechanistically exclude the final layers, even if they produce large outliers? Is the only reason for their exclusion the post-hoc empirical finding that pruning them does not hurt performance?

4. Could authors share insights on why the last layer produces more outliers in the expert output?

### Soundness
2

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
3

### Summary
This work identifies a rare subset of Super Experts in MoE LLMs, which generate extreme activation outliers that propagate through the model and trigger massive hidden state activations. The study further reveals that SEs serve as the primary source of systematic outliers by strongly activating attention sink tokens; compressing them severely disrupts attention sinks, ultimately causing repetitive and uninformative model outputs.

### Strengths
1. This paper presents the first rigorous identification and mechanistic explanation of "Super Experts" (SEs)—the primary source of systematic outlier phenomena, whose removal severely impairs performance, particularly in reasoning tasks.

2. By linking expert-level routing dynamics to Transformer-wide outlier mechanisms in MoE models, this work offers a principled account of known empirical phenomena and reveals critical vulnerabilities in current expert compression methods.

### Weaknesses
The identification of SEs heavily relies on an empirical threshold defined in Equation (6), where experts are selected based on exceeding the 99.5-percentile activation magnitude. This criterion appears rather extreme and ad-hoc, raising concerns that the observed stability of SEs may stem from the definition itself rather than reflecting an intrinsic property of the model.

### Questions
see weakness-1

### Soundness
3

### Presentation
3

### Contribution
3
