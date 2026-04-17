# Observe-Then-Think: Learning to Elicit Multimodal Understanding by Decoupling Perception and Reasoning

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Vision-language models (VLMs) have achieved impressive results in fusing visual and textual inputs, yet they often stumble on tasks demanding complex, multimodal reasoning. This imbalance arises from the inherent separation between perception—accurately interpreting sensory data, and reasoning—conducting multi-step, symbolic inference. To bridge this gap, we introduce a novel framework for multimodal reasoning, $\textit{OTT (Observe-Then-Think)}$, which includes a two-stage post-training process: supervised fine-tuning (SFT) followed by reinforcement learning (RL). During SFT, the model learns to decouple perceptual understanding from logical inference, mastering structured output formats and ensuring logical consistency. In the RL stage, our Perception-Guided Consistency Optimization (PGCO) algorithm, inspired by human cognition, enhances visual understanding through perception rewards and employs consistency rewards to align perception and reasoning steps, ensuring the final answer accuracy, eliminating logical contradictions without external tool support. Extensive evaluations across six challenging benchmarks demonstrate that our method consistently outperforms state-of-the-art baselines by an average of 3.8\% over the baseline models, delivering both stronger perceptual grounding and more reliable multimodal reasoning of VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **Obverse-Then-Think (OTT)**, a new visual reasoning strategy that decouples the perception and reasoning steps conventionally deployed by prior vision-language models (VLMs). The authors construct **OTT-20k**, a curated dataset of 20k examples distilled from a larger pool, and train the model to produce strictly structured observe/think/answer traces. Then, the authors apply a GRPO-style RL called **PGCO (Perception-Guided Consistency Optimization)** that rewards not just final answer’s accuracy but also perceptual fidelity and internal consistency of the intermediate observe/think’s outcomes. The authors show the entire OTT paradigm improves both perceptual grounding and multi-step reasoning on several hard benchmarks, such as MathVista and BLINK.

### Strengths
I personally find the work possesses the following highlights:

1. The method is easy to follow and intuitive to understand.
2. The original problem, i.e. the loss of visual grounding over long CoT,  is well demonstrated via the analyses in the manuscript, in particular those in the ablation study part.

### Weaknesses
I find the following aspects of this paper to be problematic.

**Non-LLM-as-a-Judge, i.e. Rouge-L, for Perception Reward.** Shouldn’t we use semantic-based metrics such as CLIPScore to evaluate the quality of \<observe\>, i.e. the Perception? Otherwise, we could risk having the generated Perception texts ‘overfit’ to the exact wording of the GT’s in OTT-20k. 

**A lack of sufficient ablation studies on PCGO’s reward hyperparameters.** Although some are already present in the original text, such as Table 7 & 8, we are not seeing enough and clear evidences that demonstrate the optimal mixture of the 4 PGCO rewards that can lead to the best-performing OTT-7B model. Why must it be lambda_acc = 0.7 in order to enjoy the best benchmarking performance? The current configurations seem to be relatively arbitrary and empirically determined.

**A lack of backbone model generalizability.** I believe the entire OTT paradigm is applicable to other backbone LVLMs, such as InternVL or LLaVA variants. Unfortunately, the authors have only tested with one backbone, Qwen2-VL-7B-Instruct, a moderately-sized variant of the large Qwen family.

### Questions
Please find my major concerns in the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes OTT (Observe-Then-Think), a two-stage framework for enhancing multimodal reasoning in vision-language models. The key idea is to decouple perception from reasoning by structuring model outputs into three sequential components: observation, reasoning, and answer. The approach involves supervised fine-tuning on a curated 20K dataset (OTT-20k) followed by reinforcement learning using Perception-Guided Consistency Optimization (PGCO).

### Strengths
The neurobiologically-inspired motivation for separating perception and reasoning is well-articulated in the introduction, making the rationale clear.

The two-stage training methodology (SFT followed by RL) provides a systematic framework that could potentially be adapted to other models.

### Weaknesses
**Severely limited experimental evaluation and baseline comparisons**: This is my primary concern with this submission. The experimental setup raises serious questions about the validity and significance of the reported improvements:

1. **Choice of base model**: The authors build upon Qwen2-VL-7B, which is puzzling given that Qwen2.5-VL-7B is publicly available and significantly stronger. The entire premise of the work becomes questionable when the base model is already outdated. What is the point of improving a weaker model when better alternatives exist? The authors should justify this choice or redo experiments on Qwen2.5-VL.

2. **Missing critical baselines**: The paper is missing comparisons with several highly relevant recent works that also use RL-based post-training for VLMs:
   - VL-Rethinker
   - MM-Eureka (only briefly mentioned in related work)
   - R1-VL-7B (appears in Table 1 but with limited benchmarks)
   - R1-Onevision-7B
   - OpenVLThinker-7B
   
   These omissions make it impossible to assess whether the proposed method offers any advantage over existing approaches.

3. **Incomplete ablation studies**: The ablation studies are scattered across multiple tables with inconsistent benchmark coverage. Table 2 only evaluates GRPO variants on MathVista and MME - why not all six benchmarks? Table 3 shows V*Bench and VisuLogic only. This selective reporting makes it difficult to understand the true contribution of each component. I would expect to see:
   - Base model + vanilla GRPO across ALL benchmarks
   - Base model + SFT only across ALL benchmarks  
   - OTT (full method) across ALL benchmarks
   - Ideally in a single comprehensive table

4. **Questionable improvements on some benchmarks**: The gains on BLINK are minimal (0.5-0.6%), and the paper doesn't adequately discuss why the method provides such marginal improvements on perception-heavy tasks, which contradicts the claimed strength in perception enhancement.

**Dataset curation lacks transparency**: The process of going from 260K samples to 20K for OTT-20k involves several filtering steps, but the criteria for "atomic answers" and "quality enhancement" are vague. How do you ensure this filtering doesn't introduce biases? What's the distribution of task types in the final 20K?

**Reward design appears ad-hoc**: The reward weights (λ_acc=0.7, λ_fmt=0.1, λ_perc=0.1, λ_conf=0.1) seem arbitrary. Was there any systematic search or justification for these values? The consistency reward using Qwen3-4B as a judge is interesting but raises questions about reliability and potential biases inherited from the judge model.

**Limited analysis of failure cases**: While the paper provides two case studies showing successes, there's no analysis of when and why the method fails. Given the modest improvements on some benchmarks, understanding failure modes would be valuable.

### Questions
1. **What is the computational cost?** Training involves sampling G=8 outputs per question during RL - how does this compare in terms of wall-clock time and compute budget to vanilla GRPO or other baselines?

2. **Consistency reward reliability**: You use Qwen3-4B to judge consistency between observe and think sections. Have you validated this judge against human annotations? What's the agreement rate?

3. **Why such small gains on BLINK?** Given that your method explicitly focuses on perception via R_perc, why are the improvements on the perception-intensive BLINK benchmark so minimal (0.5-0.6%)?

### Soundness
2

### Presentation
2

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
The paper proposes OTT, a two-stage post-training framework that explicitly separates Observe (perceptual sketch) from Think (reasoning) and Answer. Stage 1 uses SFT on a curated OTT-20k dataset to teach the XML-tagged structure; Stage 2 uses GRPO-style RL with Perception-Guided Consistency Optimization (PGCO) that combines accuracy, format, perception (ROUGE-L against “observe” ground truth), and observe↔think consistency rewards. Experiments on MathVista, WeMath, BLINK, V*Bench, VisuLogic, and MME report average gains of +3.8 over baselines, with larger jumps on math benchmarks.

### Strengths
Simple, modular recipe (format-aware SFT + GRPO with perception/consistency terms) that improves both math reasoning and general multimodal scores.

Interpretable traces via explicit observe→think→answer structuring, which also eases error localization and analysis.

Consistent gains across suites, with clear ablations showing the value of the Observe tag and the two reward components.

### Weaknesses
**1. Perception reward is text-similarity, not vision-grounded.** 

Using ROUGE-L between the model’s <observe> text and a reference description risks rewarding paraphrase fluency over true visual grounding. A calibration against region alignment or programmatic detectors would make the “perception” claim more credible.

**2. Template overfitting risk.**

 Format and accuracy rewards may push models to perfect tag hygiene without proportionate reasoning growth; multiple datasets here are choice- or short-answer-centric where structure alone helps.


**3. The reward ablation is incomplete.** 

The method uses four rewards (accuracy, format, perception, consistency) but only ablates two, so we can’t tell which components actually drive the gains. Without a full leave-one-out over all four and a light weight-sensitivity check, improvements might stem from stricter formatting or answer parsing rather than better perception or reasoning.

**4. Limited causal analysis for “decoupling” itself.**
 
We see that using \<observe\> helps, but it’s unclear if gains come from structural regularization, longer rationales, or genuinely better visual parsing.


**5. Ground-truth ‘observation’ source is under-specified.**

It’s unclear how the reference observation strings are derived across diverse datasets and how noisy they are. Noise could cap reward signal quality. Please document dataset-wise procedures and inter-annotator or automatic metrics for these references

### Questions
1. How are the “ground-truth” observation strings produced per dataset, and what is their measured noise level (e.g., human agreement, BLEU/ROUGE vs captions)?

2. Do observe-length and think-length caps affect accuracy or latency? Any trade-off curves?

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
3

### Summary
This paper proposes a new reasoning framework, observe-then-think (OTT), for multimodal reasoning. The core idea is to train the models to follow an explicit, structured outputs in first producing perception-related outputs (the observe stage), and then perform reasoning based on the perception outputs (the think stage), and finally answer the questions. Experiment results show improvements compared to existing baseline models.

### Strengths
1. The paper is written clearly and easy to follow. It includes details on methodology design and experiment setups.

2. The core idea of OTT is intuitive, and I do like the idea of separating perception and reasoning. It allows better understanding of where the model is falling short, and potentially allows scaling inference-time compute on these separate capabilities.

3. Experiment results show improvements compared to existing baselines. Qualitative results also show where current method like GRPO falls short (starting with wrong perception inputs). However, I have some concerns on the main comparison baseline to Qwen2-VL-7B in Table 1 (see below).

### Weaknesses
1. While the high-level intuition of OTT makes sense, the current design does not seem to handle cases where the observation may leave out necessary information in answering the question. Could the authors provide more insights on how to make sure the models output necessary information in the observation stage?

2. One of my concern is the baseline comparison to Qwen2-VL-7B in Table1. Since OTT is trained with additional 20k data, a fair comparison would be Qwen2-VL-7B trained also with these 20k data (with existing training approaches and data formats). The performance gain in this case may actually be smaller as alluded by the results in Table 3, where vanilla GRPO is very close to OTT.

3. When transforming existing datasets (Mulberry-260k) into OTT format, if we are able to extract and separate original model responses into "observe", "think" and "answer" format, does it actually suggest that existing models are implicitly using similar "observe" and "think" steps in its reasoning process (perhaps in a more interleaved fashion)? And if that's the case, why is it more effective to explicit separate these two stages, and not perform them in an interleaved fashion. Have the authors explored interleaving \<observe>\</observe>, and \<think>\</think> stages (which is related to Q1)?

4. Table 2, 3, and 4 should include all evaluation benchmarks as considered in Table 1. It is not clear how these ablation results are on these different benchmarks.

### Questions
1. In SFT data curation stage, specifically "Standardizing Response Format" stage, how do the authors transform original response into the target structured format?

2. Please see more questions in above.

### Soundness
2

### Presentation
3

### Contribution
3
