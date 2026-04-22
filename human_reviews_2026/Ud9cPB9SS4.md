# Caption as Reward: Enhancing Vision-Language Reasoning through Dense Visual Description

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 2, 6

## Abstract
Recent advances in reinforcement learning for large language models have demonstrated remarkable reasoning capabilities using simple question-answer supervision. A natural question arises: can we train vision-language models (VLMs) to reason over images through reinforcement learning alone, without explicit chain-of-thought annotations? Our investigation reveals a critical bottleneck: over 60\% of VLM reasoning failures stem from inadequate visual perception rather than logical errors. Furthermore, we find that standard RL approaches optimize reasoning chains without ensuring accurate visual understanding, leading to confident but incorrect answers. We argue that the key to effective visual reasoning is to explicitly evaluate whether visual descriptions actually improve task performance. Therefore, we propose Caption as Reward (CaR), a framework that assigns rewards to captions based on their downstream reasoning utility rather than linguistic quality. CaR uses a gain-based mechanism: captions that fix reasoning errors receive high rewards, while those that degrade correct predictions are penalized. Trained on 50K visual question-answer pairs without any CoT supervision, our 3B model outperforms strong baselines including Visionary-R1, TBAC-VLR1, and VLAA-Thinker on eight challenging visual reasoning benchmarks. Additional evaluation on MME-RealWorld confirms substantial improvements in visual perception, particularly for diagram understanding and OCR tasks. Code and checkpoints will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Caption as Reward (CaR), an RL post‑training method for VLMs that scores a caption by how much it improves the model’s downstream answer, compared with answering directly from the image. Instead of judging captions by linguistic similarity (e.g., BLEU, ROUGE), CaR measures the gain in answer accuracy that a caption gives to the same model for the task. If the answer with the given caption is correct compared to answering without the caption, then the caption gets a high reward. On eight visual‑reasoning benchmarks, CaR improves Qwen2.5‑VL‑3B and 7B and also boosts perception‑focused performance on MME‑RealWorld.

### Strengths
- The suggestion to reward descriptions that actually repair perception rather than optimize caption fluency (BLEU/ROUGE) is interesting.
- The analysis shown in Figure 1 is interesting and important.  
- Impressive performance on 8 reasoning benchmarks

### Weaknesses
- The paper is written very messy to the point that it hurts readability. I could not follow it much. It is also missing cross references (the paper repeatedly references unexisting material in the appendix (“Appendix ??). The paper is also not structured correctly. There is also a disordered narrative flow. It puts Related Work after Method and its training objective. The paper includes lists and it reads like notes rather than analysis. For example, the “Discussion and Limitations” spans roughly half a page as a list of five bullets. The figures are also hard to follow. Finally, there is contradictory in the reward weights. Section 2.4 says the composite reward uses weights (1.0, 0.1, 1.0) for (accuracy, format, caption‑gain) while the Reproducibility Statement lists (1.0, 1.0, 0.1). 
- In my opinion, the baselines are not comparable. The paper compares CaR‑3B on 30K training samples to SFT‑3B on 20K. This is a different data size and a weaker training recipe. The fair baseline would be SFT/GRPO with the same 30K data and comparable compute. It is unclear whether CaR’s gains come from more FLOPs / inference rather than the reward design. Moreover, MME‑RealWorld improvements are reported against 3B‑Instruct, not against GRPO/SFT trained on the same data/protocol, so it is hard to see CaR’s contribution. 
- The gain reward term scores captions only by whether the final answer becomes correct with the caption. There is no constraint of the caption being faithful to the image; a caption that hallucinates the correct answer receives high reward. The only other terms are accuracy and 0/1 format compliance, which do not mitigate this failure mode. This is a classic reward‑hacking problem not investigated in experiments.
- The paper should compare to simple and cheaper baselines such as SFT with the same structured template (<caption>/<think>/<answer>), and the simple self‑consistency or best‑of‑N caption selection without RL, and also GRPO on reasoning while always prompting the model to write a caption first. These are not reported, so we cannot attribute gains to the reward vs. to “just adding a caption step.”

### Questions
The paper isn’t ready in its current form. This is enough for me to reject this paper at this stage.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Caption as Reward (CaR), a reinforcement learning framework that evaluates visual descriptions in Vision-Language Models (VLMs) based on their contribution to downstream task performance rather than linguistic quality metrics. The key innovation is a gain-based reward mechanism that assigns rewards based on whether a caption improves, maintains, or degrades reasoning accuracy compared to direct visual reasoning. The authors evaluate CaR on Qwen2.5-VL models (3B and 7B parameters) across eight visual reasoning benchmarks, achieving significant improvements: the 3B model reaches 34.2% average accuracy (vs 22.9% for SFT baseline), and the 7B model improves from 36.5% to 38.1%. The method uses GRPO for training and requires no additional human annotations beyond standard question-answer pairs.

### Strengths
Strong empirical motivation: The manual audit of 1,200 samples showing 62.1% of failures stem from visual perception errors (after filtering) provides compelling justification for the approach (Figure 1, Section 2.1)

Novel reward formulation: The gain-based reward R(C|I,Q) that explicitly measures performance delta rather than caption quality is conceptually clean and well-justified through information theory (Equation 1, Section 2.2)

Consistent improvements across scales: Results show gains at both 3B (+11.3 points over SFT, +4.4 over base) and 7B (+1.6 points) scales, demonstrating robustness (Table 1)

Specific benchmark improvements: Particularly strong on MMK12 (+9.6 points for 3B-30K model), diagram understanding (+31.4 points), and OCR tasks (+8.1 points) where visual perception is critical (Tables 1-2)

No additional annotations required: Method works with standard QA pairs without requiring expensive caption annotations or chain-of-thought supervision

Clear ablations: Systematic ablation on data scale (10K/20K/30K), auxiliary evaluators (Qwen vs GPT-4o-mini), and reward components (Table 3, Section 4.5)

### Weaknesses
Single architecture family: All experiments use only Qwen2.5-VL models; generalization to other VLM architectures (InternVL, Deepseek etc.) is completely untested, limiting claims about method generality

External evaluator dependency: Semantic matching relies on either GPT-4o-mini (API costs, black box) or Qwen2.5-7B-Instruct (potential errors); reliability and failure modes of these evaluators are not analyzed. What happens when the evaluator makes mistakes?

Arbitrary reward values: The specific values (1.0, 0.7, 0.2, 0.0) appear heuristically chosen with limited justification; only brief mention of "based on ablation studies" without showing these ablations

Limited error analysis depth: While qualitative patterns are identified (enhanced detail, task-relevant focus, systematic coverage), only 100 samples were analyzed; no quantitative metrics for these patterns or failure mode frequencies

Missing theoretical analysis: The information-theoretic motivation (Equation 1) is presented but not rigorously connected to the reward design; no formal analysis of convergence properties or sample complexity

Incomplete appendices: Multiple references to "Appendix ??" suggesting missing content that would provide important implementation details
Baseline comparisons: Comparisons with Visionary-R1, TBAC-VLR1, and VLAA-Thinker use different data; not entirely clear these are fair comparisons (though results are still convincing)

### Questions
Same as weaknesses.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Caption as Reward (CaR), a RL-based framework for VLMs that assigns rewards to captions based on how much they improve downstream task accuracy relative to no-caption inference. The reward mixes accuracy, format, and visual description gain. The authors train the policy using GRPO and use external evaluators providing semantic correctness signals. Experiments show average accuracy gains and perception improvements.

### Strengths
1. The reward design is straightforward and intuitive. Especially the visual description term, it helps the model fix its own reasoning mistakes, while penalizing those that hurt previously correct answers. 
2. The training process is easy to understand, It builds on GRPO and uses an external evaluator for semantic checking. Notably, the external evaluators provides a reliable source of judging whether the model’s answer is correct, even when it differs from the ground truth.

### Weaknesses
[Major Weakness]
1. This paper’s writing is poor and seems unfinished. The references are sparse, and there are a lot of “Appendix ??” placeholders, and the appendix section itself is completely blank.
2. The novelty is also limited. CaR mostly uses the GRPO framework with an additional evaluator term, making it more like an extension or an application of GRPO.
3. The theoretical part in sec. 2.2 is disconnected from its actual method. The authors give a mutual information equation (line 152) and suddenly claim that the goal is to maximize I(C;A|Q) without deriving or estimating.
4. The performance gains over GRPO are marginal. CaR improves accuracy by only 0.6% on Qwen2.5-VL-3B with OpenData-10K and 1.6% on Qwen2.5-VL-7B with OpenData-20K. Moreover, as shown in Table 1, all CaR results rely on GPT-4o-mini as the external evaluator, which introduces additional cost. Given such small improvements, the benefit does not seem to justify the added complexity and overhead.

[Minor Weakness]
1. The hyperparameters are not consistent across different sections. For example, the reward weights are reported as (1.0, 0.1, 1.0) in line 249 but as (1.0, 1.0, 0.1) in line 386.
2. There are no qualitative examples provided to illustrate how captions actually influence the model’s reasoning.

### Questions
1. How sensitive is the performance to the reward weights? Did you perform any ablation of it?
2. How exactly is the external evaluator used during training? Is it queried for every sample, and what is the total computational cost?
3. The paper lacks clear ablations or qualitative results. Could you show examples or analyses that demonstrate when and how captions actually help reasoning?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Caption as Reward (CaR), a reinforcement learning framework for improving visual reasoning in VLMs. Unlike traditional caption evaluation methods that rely on linguistic metrics (e.g., BLEU, CLIPScore), CaR defines a gain-based reward that measures how much a generated caption improves downstream reasoning performance compared to direct inference without captions. The method integrates seamlessly with Group Relative Policy Optimization (GRPO) and uses no additional reward model. Experiments  demonstrate consistent gains. Additional evaluation on MME-RealWorld shows strong perception improvements, particularly in diagram understanding and OCR tasks.

### Strengths
- The “caption-as-reward” mechanism elegantly reframes visual description generation as a measurable contributor to reasoning success, grounding linguistic outputs in task performance rather than surface fluency.
- The mutual information analysis provides a solid theoretical basis linking CaR’s objective to maximizing task-relevant information gain 
- CaR integrates into GRPO with minimal architectural modifications and without training a separate reward model, making it a practical reinforcement learning strategy for VLMs.
- Results across eight reasoning benchmarks and perception-specific datasets (MME-RealWorld) show consistent improvements over strong baselines like Visionary-R1, TBAC-VLR1, and VLAA-Thinker.
- The paper includes ablation studies on data scale, evaluator choice, and reward composition, as well as qualitative and error analyses that clarify why CaR improves perception and reasoning alignment.

### Weaknesses
-  Experiments are restricted to the Qwen2.5-VL family. It remains uncertain whether CaR generalizes to architectures with different perception modules or alternative training pipelines. Demonstrating cross-architecture robustness would significantly strengthen the claim of task-agnostic applicability.
- The study focuses solely on visual question answering and reasoning tasks. While this choice ensures controlled evaluation, it limits the claim that CaR “enhances visual reasoning” broadly. Applying the method to captioning, grounding, or visual entailment tasks could validate its versatility.
- The comparison in Table 3 between GPT-4o-mini and Qwen evaluators shows small differences, but no analysis of inter-evaluator consistency or reward variance is presented. Measuring the reliability of semantic matching judgments could further validate CaR’s robustness.

### Questions
The qualitative error analysis is insightful but largely descriptive. Could the authors quantify how much of CaR’s gain stems from better perception (e.g., counting, OCR) versus better reasoning alignment?

### Soundness
3

### Presentation
3

### Contribution
3
