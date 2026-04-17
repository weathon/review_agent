# The Pitfalls of KV Cache Compression

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
KV cache compression promises increased throughput and efficiency with negligible loss in performance. While the gains in throughput are indisputable and recent literature has indeed shown minimal degradation on particular benchmarks, in general the consequences of compression in realistic scenarios such as multi-instruction prompting have been insufficiently studied. In this paper, we identify several pitfalls practitioners should be aware of when deploying KV cache compressed LLMs. Importantly, we show that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM. As a practical example of that, we highlight system prompt leakage as a case study, empirically showing the impact of compression on leakage and general instruction following. We show several factors that play a role in prompt leakage: compression method, instruction order, and KV eviction bias. We then propose simple changes to KV cache eviction policies that can reduce the impact of these factors and improve the overall performance in multi-instruction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the often-overlooked side effects of Key-Value (KV) cache compression in Large Language Models (LLMs), arguing that its consequences in realistic, multi-instruction scenarios are poorly understood. The authors demonstrate that performance does not degrade uniformly; rather, compression can cause "selective amnesia”, where certain instructions—particularly system prompts and safety guardrails—are disproportionately affected and silently ignored. This leads to critical failures, such as system prompt leakage. The study identifies that this vulnerability is influenced by the specific compression method, the order of instructions, and an "eviction bias" in many policies. To address this, the paper proposes two practical modifications: manually "whitelisting" critical tokens and implementing a "fair eviction" policy to ensure all instructions are compressed at a more equal rate.

### Strengths
- The paper studies the important and practical problem of how KV cache compression, while improving efficiency, can introduce unintended side effects that harm model alignment and safety.
- It moves beyond simple performance metrics to analyze a critical failure mode (system prompt leakage) that is highly relevant for deploying models in production.
- It provides practical recommendations for mitigating this degradation by modifying eviction policies to retain certain important KV cache entries.

### Weaknesses
- The core finding that performance degrades with a reduced KV cache budget is a well-known trade-off, and most existing literature on KV cache compression already documents utility degradation as the compression ratio increases.
- The study primarily uses the IFEval dataset, which is not a long-context benchmark. The argument would be significantly more convincing if it demonstrated these instruction-following failures on a benchmark explicitly designed to test long-context capabilities.
- The whitelisting solution relies on manual effort and user intuition to identify critical tokens. This approach is not scalable and cannot be generalized, as it would require bespoke tuning for every new set of system prompts or instructions.
- The proposed "fair eviction" policy raises questions. If eviction impacts all instructions equally, this could also be a negative outcome, as it might degrade critical instructions at the same rate as trivial ones, rather than intelligently prioritizing the most important information. The utility of this fairness seems questionable if it leads to uniform degradation.
- The relative impact of eviction across different layers/heads is not studied. The paper applies its eviction policies globally, but the effects of compression may not be uniform across all layers/heads.

### Questions
- How does the "selective amnesia" you observe differ significantly from the known performance/compression trade-offs already established in prior work?

- Could the instruction-following failures observed on IFEval be replicated or potentially magnified on dedicated long-context benchmarks?

- Regarding "fair eviction," could this policy be counter-productive by failing to prioritize critical instructions (like safety guardrails) over trivial ones, leading to a uniform, but still harmful, degradation?

- How could the "whitelisting" approach be scaled or automated? As it stands, doesn't it require manual, prompt-specific tuning that isn't generalizable?

- You note that policies are applied globally. Did you investigate the layer-wise sensitivity to compression? Is it possible that evicting tokens from specific layers (e.g., middle vs. final) has a disproportionate impact on instruction following?

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
3

### Summary
This paper raises awareness on the non-uniform behavior/performance of the KV cache compression algorithms. In particular, settings with multi-instruction prompting seem to be more affected. In addition, the paper conducts experiments that show the performance is influenced by the cache replacement algorithms used. As a use case study, the paper focuses on system prompt leakage (e.g., defensive prompts). They show several factors affecting the leakage, such as instruction order, compression algorithm and cache eviction methods. The paper also proposes fair eviction policies that enforce that the fraction of tokens kept from different partitions of the context/prompt is similar.

### Strengths
Extensive experiments to showcase the performance of KV caches in different scenarios

### Weaknesses
The methods proposed may require a priori knowledge on prompt structure and/or a prompt structure that is static throughout the lifetime of the LLM, which does not seem as a realistic setting

### Questions
Does the whitelisting and fair eviction require awareness of the prompt structure? What if the prompt structure is not fixed for the usage of an LLM and is dynamic, depending on the use cases (which is more realistic)?

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
This paper studies KV Cache compression and its effects on performance, with a focus on multi-instruction prompts. The paper performs a study which shows that in multi-instruction settings, some instructions degrade more rapidly with compression than others. As a practical example to this, the paper shows that system prompting as multiple instructions can lead to leakage of private and proprietary information. The paper proposes simple changes to KV cache eviction policies that reduce the impact of imbalance of instruction degradation to achieve more balanced and predictable performance in the presence of compression.

### Strengths
+ The paper highlights important problems such as prompt leakage due to KV cache compression
+ The paper does a good job of isolating the different problems with KV cache compression, such as eviction bias, and order of instruction, etc.

### Weaknesses
- While the paper does show different experiments that expose the different aspects of KV cache compression, most of the insights were expected and unsurprising
- It seems that token whitelisting is quite straightforward and not very hard, especially to be used for system prompts, and which can solve this problem. It would be good if there is discussion about some policy that cannot trivially include whitelisting of tokens, and where fair eviction is actually helpful due to its simplicity. 
- While the fair eviction solution proposed by the paper does well on the benchmark that the paper runs, it remains unclear whether the solution is good as a generic solution for multi-instruction prompting

### Questions
- Please try to answer as many questions as possible from the weakness section. 

- When it comes to the fair eviction policy that you propose, is the policy fully automatic? I imagine that even in your policy, somehow you have to partition the prompt into its corresponding instructions to be able to apply fair eviction? How do you do this automatically?

- Would blindly removing an equal number of tokens from each instruction be practical and accurate in realistic scenarios? Intuitively, it seems like different instructions may have different importance and you might want a different ratio in practice, which is much harder to achieve. Could you convince that fair eviction is a generic and always better policy for KV cache eviction?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper intends to investigate the failure modes of KV cache compression, arguing that performance degrades non-uniformly, causing LLMs to silently ignore parts of a prompt, a phenomenon the authors term "selective amnesia."  Using system prompt leakage as a case study for this security vulnerability, the work systematically analyzes several eviction-based compression policies. It identifies key contributing factors, including the compression method, instruction order, and a newly defined "eviction bias," where certain instructions are disproportionately targeted for eviction. The paper's main contributions are to identify and characterize these pitfalls and to propose and evaluate two simple mitigation strategies ("whitelisting" and "fair eviction") to counteract this bias.

### Strengths
Novel and Important Problem Framing: The paper's primary strength is reframing KV cache compression from a simple efficiency trade-off to a critical issue of model reliability and security. The identification of "selective amnesia" and its connection to system prompt leakage is an insightful and timely contribution, as such silent failures are a major concern for deploying LLMs in production.   

Systematic Empirical Analysis: The claims are well-supported by a rigorous set of experiments across multiple models and five different eviction policies. The analysis of instruction ordering and "eviction bias" provides convincing evidence that the observed failures are systematic rather than random.   

Clear Presentation: The paper is well-written and clearly structured. The core concepts are explained effectively, and the use of visualizations (e.g., Figure 1) makes the findings accessible and impactful.

### Weaknesses
Overclaiming of Scope: The paper makes broad claims about "KV cache compression" but exclusively evaluates eviction-based policies. It fails to consider other major paradigms like quantization or merging , which may not exhibit the same failure modes. This mismatch between the claims and the evidence is a significant limitation.   

Superficial Solutions and Incomplete Evaluation: The proposed solutions ("whitelisting" and "fair eviction") are simple heuristics that lack novelty. Whitelisting is not a scalable, automated solution, while "fair eviction" is a blunt instrument that may be suboptimal. The evaluation of these methods is also incomplete, as it omits performance at high compression ratios where their trade-offs would be most apparent.

Lack of Mechanistic Insight: The paper effectively describes what happens (e.g., eviction bias occurs) but provides little explanation for why it happens. A deeper analysis connecting the observed phenomena to the underlying mechanics of the attention mechanism and the specific logic of each eviction policy is missing.

Lack of Tradeoff Analysis: Will token recomputation mitigate the loss of accuracy and fairness? How does the proposed solutions affect inference speed?

### Questions
Regarding Scope: Your paper makes broad claims about "KV cache compression" but only evaluates eviction-based methods. Given that other paradigms like quantization and merging exist (which may not suffer from "selective amnesia" in the same way), would you consider reframing the paper to be more precisely about the "Pitfalls of KV Cache Eviction Policies"? This would make your claims more strongly supported by the evidence provided.

Regarding "Fair Eviction": Your "fair eviction" policy enforces a uniform retention rate across predefined instruction blocks. Have you considered scenarios where an "unfair" or biased eviction might actually be optimal for a given task (e.g., if one instruction is highly redundant or less semantically complex)? Could you discuss the potential trade-off between enforcing fairness and achieving optimal performance?

Regarding High Compression Ratios: In Figures 9 and 10, the performance of your proposed solutions is not shown for compression ratios above approximately 0.6. Could you provide these results and discuss how your methods perform in the high-compression regime? This is particularly relevant given that Figure 5 shows the baseline leakage problem naturally starts to decrease at very high compression ratios.

Regarding Mechanistic Causes: Your work clearly demonstrates that eviction bias occurs. Could you provide a deeper analysis of why it occurs for the different policies you tested? For instance, for position-based methods like StreamingLLM, is the bias simply an artifact of where the defense prompt is located relative to the attention sink and recent token window? For attention-based methods, what properties of the defense prompt's tokens (e.g., lower norm, different attention patterns) lead to them receiving lower cumulative importance scores?

Regarding the tradeoff: Will token recomputation mitigate the loss of accuracy and fairness? How does the proposed solutions affect inference speed?

### Soundness
2

### Presentation
3

### Contribution
2
