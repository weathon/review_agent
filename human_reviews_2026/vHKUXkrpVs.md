# The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 4

## Abstract
Enhancing the reasoning capabilities of Large Language Models (LLMs) is a key strategy for building agents that ''think then act''. However, recent observations, like OpenAI's o3, suggest a paradox: stronger reasoning often coincides with increased hallucination, yet **no prior work has systematically examined whether reasoning enhancement itself causes tool hallucination**. We address this gap with the central question: **Does strengthening reasoning increase tool hallucination?** To answer this, we introduce ***SimpleToolHalluBench***, a diagnostic benchmark measuring tool hallucination in two failure modes: (i) no tool available, and (ii) only distractor tools available. Through controlled experiments, we establish three key findings. First, we demonstrate a causal relationship: progressively enhancing reasoning through RL increases tool hallucination proportionally with task performance gains. Second, this effect transcends overfitting—training on non-tool tasks (e.g., mathematics) still amplifies subsequent tool hallucination. Third, the effect is method-agnostic, appearing when reasoning is instilled via supervised fine-tuning and when it is merely elicited at inference by switching from direct answers to step-by-step thinking. We also evaluate mitigation strategies including Prompt Engineering and Direct Preference Optimization (DPO), revealing a fundamental **reliability–capability trade-off**: reducing hallucination consistently degrades utility. Mechanistically, Reasoning RL disproportionately collapses tool-reliability–related representations, and hallucinations surface as amplified divergences concentrated in late-layer residual streams. These findings reveal that **current reasoning enhancement methods inherently amplify tool hallucination**, highlighting the need for new training objectives that jointly optimize for capability and reliability. Our implementation is provided at https://anonymous.4open.science/r/Reasoning_Trap-E5B6/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper investigates whether enhancing the reasoning capabilities of LLMs increases tool hallucination. By introducing the SIMPLETOOLHALLUBENCH benchmark, the study finds that RL significantly increases tool hallucination rates while improving task performance. This phenomenon is not limited to specific training data or methods but also appears across different models and training approaches.

### Strengths
Novelty: The first systematic exploration of the causal relationship between reasoning enhancement and tool hallucination, filling a gap in the field.

Experimental Design: The use of the SIMPLETOOLHALLUBENCH benchmark clearly isolates the impact of different failure modes.

### Weaknesses
The conclusion that reasoning enhancement leads to hallucination seems unsurprising and uninteresting.

The paper only tests a small subset of models, with the largest parameter count being 32B, which makes its conclusions appear less reliable.

The overall structure is loose and inappropriate; such as Table 1, Table 2, and formulas (1) and (2). The layout of different section makes it a bit difficult to read.

The use of "DeepSeek-R1 as the judge" is uncommon, and no sufficient justification is provided.

### Questions
I regret that I failed to see the unique advantages of this benchmark. For instance, could the use of this benchmark lead to conclusions that contradict those derived from other benchmarks(such as API-Bank)? Or does it merely test a different yet similar capability, ultimately yielding the same conclusion? I request the authors to provide a thorough explanation, supported by relevant citations. If the authors can fully address these concerns in their response, I would be willing to increase the score.

### Soundness
2

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
This paper investigates why reasoning reinforcement learning (Reasoning RL) often causes tool hallucination in large language models. Using Qwen2.5-7B-Instruct as the base model, the authors compare two RL frameworks: ReCall, which is tool-based RL trained on SynTool, and GRPO, a reasoning-focused RL trained on GSM8K. They find that Reasoning RL substantially increases the rate of tool hallucination, even when the model has never been exposed to tool-use data. Mechanistic analysis reveals two main effects: representation collapse, where post-RL models show reduced layerwise similarity to the pre-RL baseline on tool tasks, indicating disruption of tool-related representations; and residual stream amplification, where differences between correct and hallucinated behaviors accumulate gradually in later residual streams rather than emerging from a discrete failure. To mitigate hallucination, the authors test prompt engineering and preference optimization (DPO). Explicitly instructing the model not to use unavailable tools has minimal impact, while DPO significantly reduces hallucination but also weakens its tool-use ability. Overall, the study highlights an inherent reliability–capability trade-off in aligning reasoning-optimized models for tool use.

### Strengths
The paper tackles a timely and important question: how reasoning-oriented reinforcement learning (Reasoning RL) amplifies tool hallucination in large language models. It combines behavioral experiments, controlled RL training, and mechanistic analyses (CKA and residual-stream separability) to build a coherent causal story. The work is methodologically clear, empirically solid, and uses open models and benchmarks, enhancing reproducibility. Its identification of late-layer residual streams as loci of hallucination divergence provides novel mechanistic insight, and the study’s framing of a reasoning–reliability trade-off offers conceptual value for future alignment and safety research.

### Weaknesses
**1. Lack of cross-framework and cross-domain validation (Section 4)**:

The experimental design in §4.1–4.2 couples dataset and RL framework, where ReCall is only applied to SynTool and GRPO only to GSM8K. This coupling makes it difficult to determine whether the increase in tool hallucination stems from the reasoning RL framework or the used datasets. If possible, a full 2×2 design (SynTool × GRPO and GSM8K × ReCall) would provide a more rigorous disentanglement of (a) task domain effects and (b) RL framework biases.

**2. Potential domain-asymmetry in representation analysis (Section 5.1)**:

The CKA results are computed only on the GRPO-trained model, where the “in-distribution” is a reasoning task (GSM8K) and the “out-of-distribution” is tool usage. What about GRPO on the tool usage task and calculate the CKA results on both the reasoning task (ood) and the tool task (in-distribution)?

**3. Ambiguity in interpreting the residual-stream signal (Section 5.2)**:

The reported discrimination score (>0.14) for the residual stream is not large in absolute magnitude, and the residual stream itself is the aggregation of both the attention and MLP outputs. 

Also, the analysis is correlational: it identifies where divergence manifests, but not what causes it. Further mechanistic verification (e.g., activation patching) would substantiate the claim that hallucination behavior originates from late-layer residual dynamics.

### Questions
**1. Cross-framework consistency:**

Have you tested whether applying GRPO on SynTool or ReCall on GSM8K yields similar hallucination amplification trends?
This would confirm whether the effect is framework-agnostic or contingent on specific algorithmic biases (e.g., reward shaping differences between GRPO and ReCall).

**2. Symmetry in representation collapse:**

In §5.1, representation collapse is shown only for non-agentic RL (GSM8K + GRPO).
If a tool-specific RL model (ReCall-trained on SynTool) were analyzed with the same CKA pipeline, would the collapse pattern invert (stable for tool-domain, unstable for reasoning-domain)?
Such a symmetric test could verify whether tool representations are intrinsically fragile or if the phenomenon simply reflects domain shift.

**3. Interpretation of the residual-stream discrimination score:**

Since the residual stream equals the running sum of attention and MLP outputs, how should we interpret its higher separability (>0.14)?
Does it indicate that hallucination originates in the residual flow, or merely that this channel amplifies upstream micro-differences?Have you considered “delta-residual” or “activation patching” analyses to verify causal attribution?

**4. On scale and significance of discrimination score:**
Given that the discrimination difference (≈0.07 vs 0.14) is moderate, is it statistically significant across random seeds or sample batches?
Providing standard deviations or confidence intervals would help readers gauge robustness.

**5. Interaction with model size and types:**
Did you observe similar trends (representation collapse or residual amplification) across different backbone sizes (e.g., Qwen2.5-7B vs 14B) and across normal and reasoning models (e.g., Qwen2.5-7B vs deepseek r1)?

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
4

### Summary
The paper systematically studies if improving LLMs’ reasoning ability induces more tool hallucination and gives affirmative answer by several experimental studies. It first introduces a new benchmark the SimpleToolHalluBench to study if the LLM will hallucinate under settings with no tools or distractor tools. Using this benchmark, the authors experimentally shows that the task performance increases but so does tool hallucination. Next, the paper delivers a mechanistic analysis via standard metrics like Centered Kernel Alignment, showing that RL reasoning training will preserve representations relevant to the training task while destabilizing those for tool use. Finally, the paper confirms that there's no free lunch in such reliability-capability trade-off since both prompt engineering and DPO reduces hallucination only by hurting tool-use ability.

### Strengths
- The paper is clearly written and easy to understand.
- The experimetal results suppoort the claim from different perspectives.

### Weaknesses
- The experimental results are solid, but it will be better to provide theoretical insights for the studied reliability-capability trade-off.
- In Section 4.3, more reasoning models can be tested on SimpleToolHalluBench to strengthen the claim.

### Questions
- Why does the paper limit the scope on tool hallucination only? I think the similar experiment can be done under other settings to study such reliability-capability trade off more systematically.

### Soundness
3

### Presentation
4

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
This paper investigates a timely and important question for agentic LLMs. The authors introduce SimpleToolHalluBench, a diagnostic benchmark that isolates two abstention-centric settings: No-Tool-Available (NTA) and Distractor-Tool (DT). They found that both tool-based reasoning, non-agentic reasoning RL, and distillation from thinking models would increase tool hallucination downstream. Mechanistically, they report that post-RL, tool-related representations collapse (lower CKA similarity) in early/mid layers while in-distribution representations remain stable, and that linear discriminability between correct and hallucinated trajectories concentrates in late-layer residual streams. Finally, the authors propose to mitigate the issue by prompt engineering and DPO. However, prompt engineering offers limited gains, while DPO reduces hallucinations but significantly degrades tool-using utility.

### Strengths
1. Clear, timely, and focused research question with practical impact for tool-using agents.

2. Strong empirical story across three angles: 
   - Tool RL increases hallucination alongside reward.
   - Non-tool (math) RL increases hallucination, supporting causality tied to reasoning reinforcement.
   - Method-agnostic evidence via distillation and inference-time thinking mode.

3. Well mechanistic analysis. Layer-wise CKA highlights asymmetric stability, and linear probes localize discriminative differences to late residual streams, aligning with residual-as-accumulator intuitions.

4. Alternative solutions to the problem, although the solutions seem to be inferior currently.

### Weaknesses
1. The benchmark “guarantees” queries are impossible without the specific tool and counts direct answers as hallucinations. I think this is a strong assumption. Some queries might be partially answerable (e.g., templated guidance), so the label policy could conflate helpfulness with hallucination.

2. The proposed benchmark would be a bit small, just containing 296 samples. Additionally, it is created with the help of LLMs. I think there should be more descriptions about how you perform the quality control on the dataset, since it is the foundation of the entire work.

3. Enabling “thinking mode” typically increases output length and verbosity. Longer outputs may disproportionately increase the chance of mentioning tools, biasing the judge toward flagging hallucination. It is better to have length-controlled comparisons or comparisons under the same tool calls.

4. The mitigation suite is limited, just attempting prompting and DPO. Can it be solved with abstention-aware RL objectives (reward for correctly refusing when tools are absent, penalties for fabrications)? I think there should be many potential ways to mitigate the issues.

### Questions
1. Are the conclusions robust across seeds, temperatures, and decoding strategies (top-p, beam)?

2. Are the CKA collapses localized to particular attention heads or MLPs? Any evidence of module-specific drift (e.g., early attention heads linked to tool schema tokens)?

3. Would partial layer freezing during RL preserve tool pathways and reduce hallucination while keeping math gains?

### Soundness
3

### Presentation
3

### Contribution
2
