# Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for complex reasoning tasks with clear correctness signals such as math and coding. However, extending it to real-world reasoning tasks is challenging, as evaluation depends on nuanced, multi-criteria judgments rather than binary correctness. Instance-specific rubrics have recently been used in evaluation benchmarks to capture such judgments, but their potential as reward signals for on-policy post-training remains underexplored. We introduce $\textbf{Rubrics as Rewards (\textit{RaR})}$, an on-policy reinforcement learning method that extends RLVR beyond verifiable domains by using rubric-based feedback. Across both medical and science domains, we evaluate multiple strategies for aggregating rubric feedback into rewards. The best RaR variant achieves relative improvements of up to 31\% on HealthBench and 7\% on GPQA-Diamond over popular LLM-as-judge baselines that rely on direct Likert-based rewards. These results demonstrate that RaR-trained policies adapt well to diverse evaluation formats, performing strongly on both rubric-based and multiple-choice tasks. Moreover, we find that using rubrics as structured reward signals yields better alignment for smaller judges and reduces performance variance across judge scales.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents Rubrics as Rewards (RaR) that extends reinforcement learning with verifiable rewards.  It uses LLMs to generate checklist-style rubrics for each instance from the reference answer, and uses an aggregated score from these rubrics as the final rewards.  It also proposes 2 aggregation methods: explicit (with human assigned weights) and implicit (with LLM-as-a-judge to generate a score). Experiments show that their methods can out-perform the baseline methods such as likert-based rewards on benchmarks like HealthBench and GPQA Diamond.

### Strengths
RaR shows consitent performance gain over the baseline methods without needing further human annotation.  It well-aligns the design of HealthBench, which also evaluates the responses with its own rubrics.

This paper does some ablation studies and analysis, such evaluating RaR using human preference metric. On this metric, we can see that RaR can generate more human-preferred responses.

### Weaknesses
1. The performance improvement is marginal: The paper claims a 31% relative performance gain, which is partially true. On fig 2, not all baselines are fair comparison, and they might not be trained on HealthBench or not have access to the reference answers. Comparing RaR to reference-likert baseline, its performance gain is marginal on HealthBench (2.3 pts) and not significant on GPQA-Diamond (1.1 pts and error bars overlap).
2. Limited experiments: This paper only experiment on HealthBench and GPQA-Diamond. The former dataset is designed specifically with rubrics, thus being a good candidate dataset. But there are many multi-choice datasets similar to GPQA, and the authors should report results on these datasets to reduce the dataset noise.
3. The main point of this paper is that rubics-based reward is better than direct prompting. However, explicit aggragation of rubric scores cannot out-perform implicit aggregation, while the "implicit aggregation" can be considered as a more delicate prompt that inputs question, reference answer, and response to LLM to get an overall. It has no difference to the Likert-based LLM-as-a-judge method in nature.
4. More baselines can be introduced, such as fine-tuning with reinforcement learning and question-agnostic rubrics.

### Questions
1. If you do not use LLM to generate rubrics but directly use the rubrics of HealthBench, would that further improve the performance of RaR on HealthBench?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Rubrics as Rewards (RaR), a framework that extends Reinforcement Learning with Verifiable Rewards (RLVR) to domains where correctness is multifaceted and not easily verifiable by a single binary signal. The authors create two datasets, RaR-Medicine and RaR-Science, by generating rubrics from reference answers using strong LLMs. Evaluations on HealthBench (rubric-based) and GPQA-Diamond (multiple-choice) show that RaR, particularly the Implicit variant, outperforms strong baselines like Direct-Likert and Reference-Likert scoring.

### Strengths
The paper successfully bridges a critical gap between RLVR and preference-based RLHF by introducing a structured, multi-criteria intermediate: the rubric. The method demonstrates consistent and significant improvements over strong baselines across two challenging domains medicine and science. The relative gains of up to 31% on HealthBench are substantial.

### Weaknesses
The entire RaR pipeline's success is contingent on the quality of the synthetically generated rubrics. While ablations show the importance of using reference answers, the inherent limitations of LLM-based generation are not fully addressed. The performance ceiling is thus tied to the rubric generator's capabilities.
The method requires generating a unique, detailed rubric for each of the 20k+ training instances and then using an LLM judge to evaluate 16 rollouts per prompt. This is computationally expensive and may be prohibitive for larger-scale applications without significant optimizations.
The generalizability of RaR to more creative, open-ended, or highly subjective tasks (e.g., essay writing, dialogue, creative design) remains an open question.
The two aggregation strategies (Explicit and Implicit) are static. The paper does not explore more dynamic or learned weighting schemes that could potentially adapt during training.
While Implicit Aggregation performs best, it also makes the reward signal less interpretable. Did you observe any cases where the implicit judge failed to properly incorporate the rubric, and could this opaqueness be a risk in safety-critical domains like medicine?

### Questions
See weakness above.

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
5

### Summary
This paper introduces Rubrics as Rewards (RaR), an RL framework that uses rubric-based feedback as structured reward signals for policy training, extending RL beyond verifiable conditions like answer-only rewards. The authors construct instance-specific rubrics in medical and scientific fields and evaluate multiple strategies for translating rubric feedback into rewards. Experiments show that RaR achieves 31% and 7% relative improvements on HealthBench and GPQA-Diamond, respectively, and that rubric-based rewards enable more consistent alignment with human preferences as model scale increases.

### Strengths
1. The authors propose an innovative approach that uses rubric-based rewards to train models on tasks lacking verifiable reward signals in RL.
2. The experiments and analyses demonstrate that rubric-based rewards align well with expert human preferences and are effective when the LLM-as-a-Judge model is sufficiently capable.
3. The authors conduct experiments in the medical and scientific domains to demonstrate the effectiveness of the proposed rubric-based reinforcement learning paradigm.

### Weaknesses
1. The authors employ large-scale, instance-level rubric labeling for all training samples and use an LLM-as-a-Judge model (GPT-4o-mini) to generate rubric-based rewards in training. While this approach incurs substantial computational costs—since each rubric evaluation consumes additional tokens—the resulting performance gains, particularly on GPQA, are marginal (only 1.1 points compared to the Reference-Likert baseline). 

2. This paper uses only a single model, Qwen2.5-7B, as the policy model. However, experiments on a single model cannot fully demonstrate the robustness of the proposed method. Since the labeled rubrics are policy-agnostic, incorporating them into RL training for other policy models would be straightforward and could better show the method’s comprehensiveness and robustness.

3. The presentation of the paper could be improved — for example, Sections 4.1 to 4.5 contain many experimental settings that could be presented more intuitively. Using tables to summarize these details would significantly enhance the paper’s readability.

### Questions
1. How are the final checkpoints selected for evaluation in the RL-based baselines and RaR? Are they the final checkpoints, or are they chosen based on a validation set? If the checkpoints are selected based on intermediate validations, provide the intermediate validation plots could help clarify the effectiveness of the method.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Rubrics as Rewards (RaR), a reinforcement learning framework that extends RL with Verifiable Rewards (RLVR) to domains lacking clear correctness signals—such as medical and scientific reasoning—by leveraging structured, instance-specific rubric criteria as reward functions. The authors empirically evaluate strategies for aggregating rubric feedback and demonstrate strong improvements over conventional LLM-as-judge Likert-style baselines on HealthBench (medicine) and GPQA-Diamond (science) datasets. The paper details rubric construction, aggregation schemes, experimental outcomes, and ablation studies, highlighting the benefits of rubrics for judge consistency and alignment with human preferences.

### Strengths
1 The paper tackles a timely and important limitation of RLVR by moving beyond domains with verifiable, binary reward signals, introducing a pragmatic and interpretable alternative via structured rubrics for open-ended reasoning.

2 Empirical evaluation is thorough, utilizing large and diverse medical and science benchmarks (HealthBench, GPQA-Diamond), and benchmarking several methods including Direct-Likert, Reference-Likert, and filtered SFT, ensuring a fair assessment.

3 The paper offers the RaR-Medicine and RaR-Science datasets with rubric annotations, valuable resources for the community.

### Weaknesses
1 Limited Generalization and Scope: Experimental evaluation is confined to domains with relatively structured reasoning (medicine, science) and primarily question-answering tasks. The approach is not shown to generalize to less-structured or dialog/agentic settings noted in Section 9. 

2 Implementation Ambiguities: Category-to-weight mapping for explicit aggregation ("Essential": 1.0, etc.) is somewhat arbitrary, and no optimization or sensitivity analysis is provided. Table 2 suggests some robustness, but further technical depth would strengthen trust and generalizability.

3 Mathematical Ambiguities: The extension from binary to continuous $c_j(x, \hat{y})$ (as briefly suggested) is not developed; implications for policy learning, reward distributions, and judge reliability are not analyzed.

### Questions
1 For the implicit aggregation strategy (Eq. 2, Page 3), can the authors provide a deeper technical or empirical analysis of $f_\phi$ as an LLM judge? Specifically, how sensitive is this function to prompt design, rubric length, or underlying model drift?

2 Can the authors discuss the feasibility of extending rubric design and aggregation to dialog, tool-use, or more open-ended agentic tasks, and are there preliminary results or design recommendations for such settings?

3 Clarify how continuous (non-binary) rubric criteria (as alluded to on Page 3) would alter reward distributions and policy learning dynamics. Are there settings (Medicine/GPQA) where continuous-valued scores were tried, and if so, what were the findings?

### Soundness
3

### Presentation
2

### Contribution
2
