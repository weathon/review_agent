# PSRT: Accelerating LRM-based Guard Models via Prefilled Safe Reasoning Traces

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Large Reasoning Models (LRMs) have demonstrated remarkable performance on tasks such as mathematics and code generation. Motivated by these strengths, recent work has empirically demonstrated the effectiveness of LRMs as guard models in improving harmful query detection. However, LRMs typically generate long reasoning traces during inference, causing substantial computational overhead.
In this paper, we introduce $\textbf{PSRT}$, a method that replaces the model's reasoning process with a $\textbf{P}$refilled $\textbf{S}$afety $\textbf{R}$easoning $\textbf{T}$race, thereby significantly reducing the inference cost of LRMs. Concretely, PSRT prefills "safe reasoning virtual tokens" from a constructed dataset and learns over their continuous embeddings. With the aid of indicator tokens, PSRT enables harmful-query detection in a single forward pass while preserving the classification effectiveness of LRMs.
We evaluate PSRT on 7 models, 13 datasets, and 8 jailbreak methods. In terms of efficiency, PSRT completely removes the overhead of generating reasoning tokens during inference. In terms of classification performance, PSRT achieves nearly identical accuracy, with only a minor average F1 drop of 0.015 across 7 models and 5 datasets

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This submission pertains to large reasoning models (LRMs) used as guard models for detecting harmful queries. It proposes PSRT, a method that replaces the generation of reasoning traces with a fixed, prefilled "reasoning trace" composed of optimized embedding vectors. The paper presents an extensive evaluation of PSRT in terms of models (reasoning guard models, non-reasoning guard models, non-guard models trained to become reasoning guard models) and datasets (harmful, jailbreak, harmless, mixed). The overall finding is that PSRT preserves the detection performance of reasoning guard models while avoiding generation of reasoning traces, thereby substantially reducing the number of generated tokens.

### Strengths
- The main idea is a clever application of p-tuning (prefix tuning, prompt tuning) to reasoning traces, specifically to optimize a fixed reasoning trace for harmful query detection.
- Demonstrates that generation of reasoning traces is not needed to achieve almost the same detection performance (and in a few cases even better performance) compared to reasoning guard models

### Weaknesses
1. The most important shortcoming for me is the use of number of generated tokens as a proxy for computational cost. With PSRT (as I understand it), since the query varies and occurs before the fixed reasoning trace, it is still necessary to perform a forward pass on the tokens of the reasoning trace (computing all their internal representations, etc.). Simply reporting the number of generated tokens does not measure the cost of this forward pass, nor whatever computational savings are achieved by performing this forward pass on fixed tokens rather than generating a similar number of new tokens.
1. I am unsure about the significance of including the SFT-only models (Qwen3-8B, Llama-3.1-8B-Instruct, etc.), as well as the components related to them, namely the dataset construction and SFT in Section 3.1. Figure 1 shows that these models are Pareto-dominated by GuardReasoner (lower F1 score, more generated tokens). Moreover, it is not clear to me how novel is the method in Section 3.1 for training guard models, or how specialized it is for harmful query detection (please see the next point). Thus, the main significance that I see is to show the "generality of PSRT across diverse model architectures," but I am not sure that this warrants so much space in the main paper. I would have been more interested in seeing the additional results on GuardReasoner (Appendix A.1) in the main paper and discussed in greater depth, since GuardReasoner is a stronger model.
1. The paper limits itself to detecting harm in the query/model input and not in the model output. The reason for this limitation is not clear.
1. Section 2 cites prior work on shortening reasoning traces. It would have been good to see one of these methods used as an experimental comparison because it would be an intermediate approach that does not avoid generating reasoning traces completely.
1. The paper does not provide deeper insight into why PSRT works. Can the virtual reasoning trace be interpreted somehow? What are the relative contributions of the averaging initialization and subsequent fine-tuning?

More minor:
1. I find the term "safe reasoning trace" confusing because the predominant reading of this term is "a reasoning trace that is safe," i.e., free from harmful content, not "reasoning about safety." I think "safety reasoning trace" would be better.
1. Section 3.1 implies that DeepSeek-V3.1 is used as the judge of harmfulness. If this is correct, then this dependence on a single LLM could be acknowledged as a limitation.
1. Lines 261-262: Sections 3.1 and 3.2 are not experiment sections. Perhaps wrong references?
1. It would be good to perform the second ablation (omitting the average initialization) for GuardReasoner models also.
1. Line 724: Should Table 4 be Table 5? Table 4 is on mixed datasets.

### Questions
1. Number of generated tokens after PSRT: For the GuardReasoner models, is the number of generated tokens still around e.g. 17 in Table 2 because the model generates that many as answer tokens? Why are the corresponding numbers for the SFT-only models much higher, in the 70s or 80s or even higher?
1. Lines 360, 363: I do not see the exact numbers quoted here (99.26%, etc.) in Table 2. Are these numbers averages over the three sizes of GuardReasoner models?
1. In the ablation study, what initialization is used instead of the average embeddings?
1. In Table 5, why are the TPRs of the GuardReasoner models so uneven, and in particular, why is the original 8B one so poor?

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
This paper proposes PSRT, a method to accelerate Large Reasoning Model (LRM)–based guard models by eliminating explicit reasoning token generation during inference. PSRT introduces Prefilled Safe Reasoning Traces, represented as optimized “safe reasoning virtual tokens” in the embedding space, allowing the model to perform harmful-query detection with a single forward pass. The authors demonstrate the method’s generality across 7 models, 13 datasets, and 8 jailbreak attacks, showing comparable detection performance (≤0.015 average F1 drop) while completely removing reasoning overhead.

### Strengths
- Novel and practical contribution: The paper addresses a real bottleneck in LRM deployment, inference latency due to reasoning traces, and proposes an elegant solution by embedding “prefilled reasoning” directly into model inputs.

- Strong empirical validation: Extensive experiments across diverse models (Qwen, Llama, ChatGLM, Mistral, GuardReasoner) and datasets (StrongReject, JBB, SimpleSafetyTest, AdvBench, etc.) show consistent performance with drastically reduced computational cost.

- Well-motivated theoretical grounding: The connection between reasoning trace averaging and point-estimate optimality (Proposition B.5), and the ELBO interpretation for training objective, make the approach conceptually sound.

### Weaknesses
- Limited conceptual novelty: The idea is closely related to p-tuning and prompt embedding averaging, which have been explored for efficiency. The main novelty lies in the specific application to guard models rather than a fundamentally new optimization principle.

- Ablation insufficiently deep: The ablation (Fig. 3) focuses mainly on SFT and averaging initialization; it would be valuable to test different trace lengths, embedding dimensions, or virtual token counts to probe robustness.

- Evaluation scope: The paper is entirely focused on binary harmful query detection. Demonstrating that PSRT is effective on multi-class classification or structured safety reasoning tasks (e.g., toxicity type detection) is more impactful

### Questions
Please refer to the weakness section.

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
3

### Summary
The paper proposes Prefilled Safe Reasoning Trace (PSRT), a novel method designed to enhance the efficiency of Large Reasoning Models (LRMs) used for safety detection (e.g., harmful or jailbreak query classification). Traditional LRM-based guard models achieve strong safety performance but suffer from significant inference-time overhead due to long reasoning traces. PSRT addresses this issue by replacing explicit reasoning generation with prefilled “safe reasoning virtual tokens”, effectively compressing the reasoning process into a single forward pass.
The proposed framework introduces three key components:

1. Safe Reasoning Dataset Construction: A curated reasoning dataset is built using DeepSeek-V3.1 to generate reasoning traces and safe/unsafe labels for queries.

2. Safe Reasoning Token Initialization: Prefilled “safe reasoning tokens” r_s are initialized in the embedding space by averaging reasoning embeddings, replacing explicit reasoning sequences.

3. Single-Pass Binary Classification: The model leverages the prefilled r_s to directly classify queries as safe or unsafe without generating reasoning tokens.

### Strengths
The paper has two notable strengths:

**First,** it provides a clear and practical solution for accelerating safety reasoning in Large Reasoning Models (LRMs). By introducing Prefilled Safe Reasoning Traces (PSRT), the authors successfully remove the need for explicit reasoning generation while maintaining nearly the same detection performance. This represents a meaningful step toward efficient and deployable LRM-based safety systems, especially in latency-sensitive scenarios.

**Second,** the experimental evaluation is extensive and convincing. The authors validate PSRT across multiple model families (e.g., Qwen, Llama, GLM, Mistral) and a wide range of datasets (including harmful and jailbreak benchmarks), with detailed quantitative analysis and qualitative visualization. This comprehensive setup provides strong empirical evidence for the method’s robustness and general applicability.

### Weaknesses
Some concerns arise regarding the scalability and generalization of using a single r_s.

**First**, the current dataset construction heavily relies on existing safety reasoning datasets (e.g., GuardReasoner, ReNeLLM), which raises questions about the model’s cross-distribution generalization, whether it has truly learned generalizable safety reasoning logic or merely memorized dataset-specific patterns.

**Moreover**, as the scope and diversity of safety-related datasets continue to grow, it remains unclear whether a single global r_s can adequately cover the full spectrum of safety requirements, and whether its generalization performance can be maintained under larger and more diverse settings.

### Questions
See questions in the weakness part.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The topic of this paper is the LRM-based guardrail model. And the authors aim to reduce the computational costs of this kind of guard model by replacing the model's reasoning process with a prefilled safe reasoning trace. The comprehensive experiments demonstrate the effectiveness of the proposed method. The computational overhead of generating reasoning tokens is removed yet the performance doesn't drop.

### Strengths
1. The experiments are very comprehensive, e.g., the PSRT is evaluated on 7 models, 13 datasets, and 8 jailbreak methods. 

2. The code is provided, which ensures reproducibility. 

3. The paper is well-motivated and the topic is practical.

### Weaknesses
1. The color in Figure 1 is confused. For example, for the Qwen3-8B model, the line is blue, but the delta and the circle are black. Besides, it seems to be hard to identify GuardReasoner-3B and GuardReasoner-8B. In addition, it is not clear why these instruct models like LLaMA-3.1-8B-Instruct or base models like Qwen3-8B will generate more tokens than the LRM-based guardrail model, i.e., GuardReasoner.

2. The idea of the proposed method is similar to Coconut [1]. Please discuss it and identify the novelty.  

3. The efficiency experiments are missing, i.e., time costs and GPU memory costs of the LRM-based guard models and the proposed models. Please detail the inference process of the proposed method. Does it support vLLM?

4. Although the authors claim the proposed method can reduce the reasoning tokens significantly, it seems to reduce the explainability of the LRM-based models since the prefilled embeddings of safe reasoning traces are not readable.

5. Minor: missing discussion on an LRM-based guard model [2] in the related work part. The notation table is missing.


[1] Training Large Language Models to Reason in a Continuous Latent Space

[2] GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning

### Questions
1. How's the inference process of the proposed method? Does it support vLLM?

2. How can the proposed method keep the explainability of the LRM-based models?

### Soundness
3

### Presentation
4

### Contribution
3
