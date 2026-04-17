# Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Large language models (LLMs) are widely applied to data analytics over documents, yet direct reasoning over long, noisy documents remains brittle and error-prone. Hence, we study document question answering (QA) that consolidates dispersed evidence into a structured output (e.g., a table, graph, or chunks) to support reliable, verifiable QA. We propose a two-pillar framework, LiteCoST, to achieve both
high accuracy and low latency with small language models (SLMs). Pillar 1: Chain-of-Structured-Thought (CoST). We introduce a CoST template, a schema-aware instruction that guides a strong LLM to produce both a step-wise CoST trace and the corresponding structured output. The process induces a minimal structure, normalizes entities/units, aligns records, serializes the output, and verifies/refines it, yielding auditable supervision. Pillar 2: SLM fine-tuning. The compact models are trained on LLM-generated CoST data in two stages: Supervised Fine-Tuning for structural alignment, followed by Group Relative Policy Optimization (GRPO) incorporating triple rewards for answer/format quality and process consistency. By distilling structure-first behavior into SLMs, this approach achieves LLM-comparable quality on multi-domain long-document QA using 3B/7B SLMs, while delivering 2–4× lower latency than GPT-4o and DeepSeek-R1 (671B). The code is available at https://github.com/HKUSTDial/LiteCoST.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes LiteCoST, a two-pillar framework for long-document question answering that combines structured reasoning with small-language-model (SLM) fine-tuning. Pillar 1, Chain-of-Structured-Thought (CoST), uses schema-aware prompting to make a strong LLM generate an auditable reasoning trace and a serialized structured output (SSO), e.g., a table or graph, from which the answer is explicitly derivable. Pillar 2 transfers this structure-first behavior to compact models through two phases: SFT to instill schema and format discipline, followed by GRPO using a dual-reward setting targeting both answer correctness and process consistency. Empirical results on benchmarks on finance and legal domains (the Loong benchmark) show that 3B / 7B SLMs trained with LiteCoST substantially improve over base models and even approach GPT-4o performance, while achieving 2 to 4 times lower latency.

### Strengths
[S1] Valuable problem and novel idea. The problem of reliable QA over long, noisy documents is valuable because direct LLM prompting often fails. The idea of QA-by-structuring (introducing minimal schemas dynamically per query to improve interpretability and verifiability) is conceptually novel. 

[S2] Strong empirical results. On the Loong finance subset, both 3B and 7B LiteCoST variants outperform their base SLMs by +27.6 and +17.8 accuracy points respectively, and even slightly surpass GPT-4o-mini and DeepSeek-R1 while running 2 to 4 times faster. The consistent improvements across multiple subtasks (Spotlight Locating, Comparison, Clustering, Chain of Reasoning) demonstrate the effectiveness of transferring CoST behavior to small models.

[S3] The paper is generally well written and easy to follow.

### Weaknesses
[W1] Limited domain and benchmark scope. While the results on Loong (finance/legal) are strong, the framework has not been tested beyond these domains. Both training and evaluation rely on tabular data; it is unclear whether CoST and LiteCoST generalize to less structured settings (e.g., scientific literature QA, narrative multi-hop QA, or multimodal reports). When relevant evidence cannot be serialized, the "QA-by-structuring" assumption may not always hold.

[W2] Complexity and reliance on LLM supervision. Although the paper claims cost-efficiency, generating CoST traces still depends on large-LLM supervision (GPT-4o) for data construction and reward evaluation. The overhead of repeated calls during Stage A and GRPO reward computation could limit reproducibility and offset some of the efficiency gains. A more explicit accounting of total compute (LLM cost + SLM training) would strengthen the argument.

### Questions
[Q1] How sensitive is the GRPO process to the weighting between process and outcome rewards? The ablation indicates both are important, but does the balance depend on domain complexity?

[Q2] During CoST trace generation, does the model ever perform schema selection incorrectly (e.g., choosing a graph when a table is needed)? If so, how is this corrected or reflected in the reward?

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
4

### Summary
The paper proposes a method for distilling LLM traces and final structured responses (tables, graphs, etc.) into SLMs. This is in two key steps - (1) schema aware prompting that generates a structured output along with a step by step trace, (2) finetuning with SFT and RL using both outcome and process rewards. Authors demonstrate the performance of their method over other finetuned SLMs, LLMs on the Loong Benchmark legal and finance subsets, while maintaining a lower latency compared to LLMs.

### Strengths
1. The paper is well written and is easy to follow
2. At a high level the problem is important -- long document QA with SLMs is a longstanding challenge, and typically SLMs do not perform well on this task.
3. The method seems sound, there are no obvious technical flaws with the approach. 
4. Based on my understanding of the paper there are two key insights in this work. First, that training an SLM on LLM traces and a structured output that is easy parse significantly improves the SLMs ability to generate the correct answer on long document QA. Second is that a cold start SFT and GRPO with process rewards provides a better learning signal for the SLM. These seem to be sound, nothing unexpected.
5. The empirical validation is convincing - the reported results on Loong benchmark indicate a significant performance improvement over baselines.

### Weaknesses
1. The contributions of the paper lack novelty. Though well engineered, the contribution of the work lies in applying existing techniques to Loong benchmark and not fundamentally proposing a new strategy. For example: It is well known that cold-start SFT followed by RL training using fine-grained supervision is an effective mechanism for training SLMs (Luong et al., 2024). It might be misleading to present this as a core novel contribution of the paper. 

2. Similar to the first point, structured chain of thought prompting has been explored before. Prior works for example (Li et al, 2023) have tried it and confirmed its effectiveness.

3. Details of prompting LLM-based baselines (GPT-4o, GPT-4o-mini, DR1, etc) are not provided. For a fair comparison, both the proposed method and untrained baselines should leverage the same prompting strategy to confirm the effectiveness of training. The authors should also consider prompting LLM-based baselines with structured COT to establish the effectiveness of distillation.

4. Figure 7 compares latency primarily between LITECOST and large closed models. However, a latency comparison against other SLM-based baselines (e.g., Struc-Bench, IEPile) would clarify the efficiency gains.

5. It remains unclear whether the CoST schemas generalize beyond Loong. The paper’s own limitations section (Appendix E) acknowledges the narrow domain focus. Additional evaluation on non-financial datasets (e.g., academic papers or scientific QA) would strengthen the claim of generalizability.

6. The prompting technique in the work is hand-engineered. It should be compared against popular prompt optimization frameworks like dspy, promptim, etc. for a fair assessment. 

7. The paper lacks a comprehensive comparison against SLM based baselines on the legal subset of Loong. The main paper emphasizes financial QA results, deferring legal subset results to the appendix. Including a concise summary of those findings in the main text would improve completeness and support claims of cross-domain applicability.

8. The paper could benefit from a clearer comparison to previous “QA-by-structuring” systems such as StructRAG (Li et al., 2024) and StrucBench (Tang et al., 2024), beyond reporting numeric improvements.

Overall:

This paper addresses an important practical problem: enabling efficient, auditable long-document QA using small models. The experiments show that LITECOST narrows the performance gap between SLMs and LLMs. However, the novelty is primarily incremental, with several well-known components engineered into a working pipeline. Strengthening the baselines, clarifying prompt design choices, comparing the structured prompt with prompt optimization, and demonstrating cross-domain generalization would make the paper more compelling.

[1] Structured Chain-of-Thought Prompting for Code Generation. Li et al

[2] ReFT: Reasoning with Reinforced Fine-Tuning. Luong et al

### Questions
Kindly see weaknesses.

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
This paper studies document question answering (QA) that consolidates dispersed evidence into structured outputs to support reliable and verifiable QA. Based on this, the authors propose a two-pillar framework, LITECOST, aiming to achieve both high accuracy and low latency with small language models (SLMs). First, they introduce a Chain-of-Structured-Thought (CoST) template—a schema-aware instruction guiding a strong LLM to generate both step-wise CoST traces and corresponding structured outputs. Subsequently, compact models are trained on the LLM-generated CoST traces and structured data via supervised fine-tuning (SFT) and a reinforcement learning approach, GRPO.

### Strengths
The paper proposes a structure-first prompting template that leverages LLMs to elicit step-wise, schema-guided CoST traces and structured outputs from long, noisy documents, producing auditable supervision and machine-checkable results.

The two-stage training paradigm, involving supervised fine-tuning followed by GRPO, effectively transfers CoST behavior to compact models using reward signals on both answer/format correctness and process consistency.

### Weaknesses
The paper does not clarify how the process reward is computed.

Table 2 only compares a subset of the Loong benchmark with other state-of-the-art models, significantly limiting the validity of LITECOST’s effectiveness claims.

The novelty of LITECOST is limited; the paradigm of first applying SFT and then enhancing performance via GRPO is already well-established, and the paper does not sufficiently highlight any unique contributions.

It is unclear whether the reported parameter count of 200 billion for the GPT-4o model is accurate.

### Questions
See Weaknesses

### Soundness
2

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
3

### Summary
This paper introduces Chain-of-Structured-Thought (CoST) a structured prompting method for generated serialized structured outputs (SSOs) along with a verifiable reasoning chain. These outputs are further used to train LiteCost, a model (with two-stage SFT -> GRPO) that integrates this behavior within an SLM, allowing for more efficient inference without significant performance hit. The evaluation is performed on financial data (along with legal). Authors find that CoST -> SSO is an effective mechanism for enhancing reasoning in popular models for long/multi document reasoning.

### Strengths
Overall, this work offers a meaningful contribution for their domain, enhancing complex reasoning performance through several methods, both 1) in enhancing off-the-shelf methods via CoST prompting and improved SSO generation and 2) by effectively distilling this behavior into small language models.

### Weaknesses
- While the contributions are clear, I worry that the scope of the work is quite limited. The authors specifically explore the financial domain which has clear structure and a heavy emphasis on numeric reasoning. As noted in the intro, this structured reasoning approach doesn't extend to situations with information that is more nuanced. Similarly, as noted in Appendix C.4, the structured reasoning can be a lossy mapping, impacting performance on non-aggregative / numeric tasks. As such, it seems one must be quite careful in ensuring the applicability of this method to their task. 

- Similarly, I am curious if there are cases where a one output structure type isn't appropriate for an example (e.g. perhaps you ideally need a composition of tabular, graph, and textual data for one task), and if there any ways to address these by making the output structure more flexible. In this vein, I'd also like to see how CoST and LiteCoST performance scales with task complexity (e.g., table size, graph breadth/depth, input length).

### Questions
- Have you performed any failure mode analysis when applying this method to domains requiring 'fuzzier' reasoning? I'm curious if the text chunk representation acts as the fallback mechanism for these cases, or if there is still too much loss of information even in this format? Do you have any suggestions on how to make your method generalize to a broader set of applications?  

- Apologies if I missed this, but do you perform any analysis that plots task complexity (table size, document-set size, etc...) versus performance of your techniques (CoST, and LiteCosT)? Are the improvements more apparent over the baselines for harder splits, and have you observed any practical ceiling on what your method can support?

A few presentation notes:
- Some figures have typos (e.g. 'Strcture', 'Structurlizer')
- expand 'on-prem'
- Figure 5: define the axes somewhere
- Figure 6: perhaps report both values, and add explicit +Δ notations in parentheses.
- In general, the figures are pretty dense and tricky to follow. There are also four of them. Given your multi-faceted contribution this is fine, but I think some more work needs to be put into readability.
- It was also sometimes unclear when/whether your analysis was focused on CoST analysis (e.g. using this with off-the-shelf models) versus when your analysis was comparing your trained LiteCost model. The paper is fairly dense read, so it would be great if you can be extremely precise with your wording and the specific components that you contribute (e.g. "High-quality SSO improves LLM Reasoning" -- is the SSO high-quality because of CoST, or because of the model used to generate the CoST?)

### Soundness
3

### Presentation
3

### Contribution
3
