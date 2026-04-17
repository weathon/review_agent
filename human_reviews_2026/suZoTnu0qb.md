# Enhancing the Medical Context-Awareness Ability of LLMs via Multifaceted Self-Refinement Learning

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Large language models (LLMs) have shown great promise in the medical domain, achieving strong performance on several benchmarks. However, they continue to underperform in real-world medical scenarios, which often demand stronger context-awareness, i.e., the ability to recognize missing or critical details (e.g., user identity, medical history, risk factors) and provide safe, helpful, and contextually appropriate responses. To address this issue, we propose Multifaceted Self-Refinement (MuSeR), a data-driven approach that enhances LLMs' context-awareness along three key facets (decision-making, communication, and safety) through self-evaluation and refinement. 
Specifically, we first design a attribute-conditioned query generator that simulates diverse real-world user contexts by varying attributes such as role, geographic region, intent, and degree of information ambiguity. An LLM then responds to these queries, self-evaluates its answers along three key facets, and refines its responses to better align with the requirements of each facet. Finally, the queries and refined responses are used for supervised fine-tuning to reinforce the model's context-awareness ability. Evaluation results on the latest HealthBench dataset demonstrate that our method significantly improves LLM performance across multiple aspects, with particularly notable gains in the context-awareness axis. Furthermore, by incorporating knowledge distillation with the proposed method, the performance of a smaller backbone LLM (e.g., Qwen3-32B) surpasses its teacher model, achieving a new SOTA across all open-source LLMs on HealthBench (63.8\%) and its hard subset (43.1\%). Code and dataset will be released at \url{https://anonymous.4open.science/r/MuSeR-EC43}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MuSeR (Multifaceted Self-Refinement), a data-driven approach that enhances an LLM's context-awareness by (1) generating diverse, attribute-conditioned queries, (2) having the LLM self-evaluate and refine its own answers along three facets (decision-making, communication, and safety), and (3) using these refined pairs for supervised fine-tuning.

### Strengths
1. The paper convincingly argues that limited context-awareness impedes real-world medical deployment and positions MuSeR as a principled remedy.
2. The attribute-conditioned query generator is a practical way to overcome real-world data scarcity while stress-testing models across user roles, regions, intents, and ambiguity levels.

### Weaknesses
1. Notation is difficult to follow. Consolidate all symbols in a notation table; unify subscripts/superscripts 
2. Facet selection lacks empirical justification. Provide evidence for focusing on Decision-Making, Communication, and Safety (e.g., error taxonomy from baseline models, clinician interviews), and include an ablation showing the marginal gain of each facet and their interactions.
3. In Line 236, why "we consider a total of seven key attributes for query generation"?
4. Dependence on query-guided knowledge distillation (KD). Include an ablation of Base Model + MuSeR (no KD) versus MuSeR + KD to quantify KD’s incremental benefit; report compute/cost trade-offs and performance deltas.

### Questions
1. Why were the three facets and seven attributes chosen?
2. What ablations isolate MuSeR’s gains from knowledge distillation and refinement iterations, and how does it generalize across domains/OOD with cost/latency reported?

### Soundness
2

### Presentation
2

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
Authors propose MuSeR, a Multifaceted Self-Refinement framework designed to enhance LLMs’ context-awareness in medical tasks. Proposed framework first generates synthetic data by simulating real-world medical queries with controlled attributes (e.g., user role, region, disease, intent). It then performs self-refinement of the model’s responses along three key facets: (i) decision-making, (ii) communication, and (iii) safety. The refined dat used for supervised fine-tuning (SFT) and can optionally be combined with knowledge distillation from a larger teacher model (GPT-oss-120B). They evaluate their framework on HealthBench, where MuSeR achieves top performance using Qwen3-14/32B, even surpassing its teacher model and other much larger proprietary models.

### Strengths
- Authors point-out a well-motivated and timely issue in the medical LLM domain by emphasizing the distinction between exam-style benchmarks and real-world medical scenarios., where they focus on context-awareness makes the task notably harder yet more realistic and relevant for clinical applications.
- According to the authors’ results, MuSeR outperforms several top-priority models and achieves closer performance with gpt-5-thinking while using much smaller backbones (e.g., Qwen3-14B). Interestingly, it even surpasses its teacher model (gpt-oss-120b), which is a counterintuitive yet noteworthy finding that can be valuable for future research.

### Weaknesses
- The main limitation is the narrow evaluation setup. Although the authors conduct experiments with different model families and baselines for comparison, they evaluate solely on HealthBench, making it difficult to assess the framework's robustness. It would be beneficial to see results on at least 2-3 additional benchmarks. Moreover, HealthBench relies on GPT-4.1 as an LLM-as-a-Judge; such evaluations require statistical consistency tests, where the same experiments should be run multiple times with different LLMs, reporting alignment across models and standard deviations within the same model to better understand robustness.
- Catastrophic forgetting is a common problem in post-training domain-specific LLMs and is not addressed in the current manuscript. Since the authors conduct heavy fine-tuning on generated data, is there any evidence of catastrophic forgetting; if not how they overcome that?
- Finally, the "self-refinement" terminology seems confusing, as the proposed framework also includes a distillation step. Why do the authors frame their approach as self-refinement, and could they provide more details?

### Questions
1. Could the authors share a reference for lines 42-43?
2. The provided repository seems to include only the readme file.
3. In lines 192-193, the authors mention that “the formulations above represent our design goals rather than explicit optimization objectives.” What does this mean, and why formulate equations that are not followed in the framework?
4. It is interesting that a student model can outperform the teacher model; could the authors share theoretical insights on this?

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
5

### Summary
This paper aims to enhance the Medical Context-Awareness Ability of LLMs via Multifaceted Evaluation Metrics to refine the generation so that the model can learn to capture three key facets (decision-making, communication, and safety) on HealthBench. By this data-driven pipeline, they successfully improved the healthcare response generation ability of smaller LLMs and even surpassed the biggest competitor through continual distillation from gpt-oss even on the hard subset.

### Strengths
1. This paper focuses on a data-driven pipeline to use multifaceted eval metrics to refine the generation. This process ensures the diversity and quality of the data used for further reinforcement training.

2. Their results show significant improvements of their pipeline on improving the health capabilities of smaller LLMs, making the data potentially useful for this domain.

3. Health is a domain that requires more careful eval, and this paper focuses on decision-making, communication, and safety, which are essential to enhance the safety and usefulness of health questions.

### Weaknesses
1. Why only use GPT-oss-120B as the teacher? how about using other models? GPT-oss-120B might not be the best

2.  Why choose those three key facets (decision-making, communication, and safety) but not other dimensions? I want to learn more scientific justification for this choice.

### Questions
1. Why use DeepSeek-V3 for generating the queries but use smaller models ((Qwen3-14B/32B or OpenPangu-7B) as the multifaceted selfrefinement module? The inference cost is minimal, and I did not see any specific reason for using smaller models here. especially OpenPangu-7B is not that good.

2. The authors mentioned "capture the diversity and complexity of real-world medical queries". How do you measure diversity? quantitively?

3. What are the obvious failure cases after this pipeline? it would be good to provide a more thorough analysis to help us understand the limitations of this pipeline.

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
The paper introduces MuSeR, a data-driven framework to improve medical context-awareness in LLMs by synthesizing diverse, attribute-conditioned queries (e.g., user role, region, intent, information completeness) and training models to self-evaluate and refine their answers along three facets: decision-making, communication, and safety. For each synthetic query, the model produces an initial reply, generates facet-specific rationales, and directly refines the answer based on those rationales; the refined pairs then supervise SFT. A query-guided knowledge distillation stage precedes SFT to transfer medical knowledge from a stronger teacher, boosting downstream effectiveness. On HealthBench, MuSeR substantially improves multiple open-source backbones and achieves new open-source SOTA, with especially large gains on the context-awareness axis.

### Strengths
The results are strong and well-documented. MuSeR delivers consistent, sizable gains on HealthBench across multiple backbones, with Qwen3-32B and Qwen3-14B improved to 63.8% and 61.8%, surpassing a stronger teacher and setting a new open-source SOTA; detailed plots also show improvements on the hard subset and across evaluation axes/themes, emphasizing context-awareness. The paper backs these outcomes with granular analyses: stage-wise and multi-faceted self-refinement and facet ablations. Overall, the combination of broad backbone coverage, competitive headline scores, and thorough ablations/case study makes the empirical case both convincing and reproducible.

### Weaknesses
Weaknesses*: Backbone coverage is narrow; most results center on Qwen, so it’s unclear whether gains transfer to other families. Adding full runs on Llama-3/Mistral/Qwen2.5 (multiple sizes) would clarify generality. On novelty and positioning, the attribute-conditioned query synthesis overlaps with prior medically conditioned instruction tuning (e.g., AlpaCare’s diverse, synthetic medical instructions [1]); this prior work should be cited and contrasted to specify what is new here beyond the chosen attribute schema and prompts. Likewise, the use of rationale/CoT distillation follows earlier explanation-distillation work [2] and should be discussed. Finally, the empirical scope is largely HealthBench; adding at least one more public medical benchmark (e.g., MedQA/USMLE or PubMedQA) would help validate robustness outside HealthBench’s distribution.

References:
[1] Zhang et al. AlpaCare: Instruction-tuned Large Language Models for Medical Application.
[2] Li et al. Explanations from Large Language Models Make Small Reasoners Better.

### Questions
The current self-refinement loop is trained with SFT. Have you explored an RL variant (e.g., PPO/GRPO) where facet-specific rewards (decision-making, communication, safety) are used directly as the objective? In addition, for training data, any decontamination test of Healthbench is done?

### Soundness
3

### Presentation
2

### Contribution
2
