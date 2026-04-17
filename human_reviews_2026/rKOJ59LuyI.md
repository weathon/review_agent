# CompassJudger-2: A Holistic Approach Towards Generalist Judge Model

- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
Recently, the role of LLM-as-judge in evaluating large language models has gained prominence, emerging as an important method to partially replace costly human assessment. However, current judge models suffer from narrow specialization and limited robustness, undermining their capacity for comprehensive evaluations. In this work, we decompose the ability of a generalist and generative judge model into three levels: objective verification, subjective evaluation, and rubric refinement. 
We present CompassJudger-2, a novel generalist judge model that overcomes these limitations via a task-driven, multi-scenarios data curation strategy. We conducted large-scale data collection for each type of task and designed tailored rejection sampling strategies to filter the data, ensuring data diversity, accuracy, and effectiveness.
Empirically, CompassJudger-2 achieves superior results across multiple judge and reward benchmarks, demostrating its excellent robustness and generalization ability. 
These contributions advance robust, scalable LLM judgment and establish new performance and evaluation standards.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces CompassJudger-2, a new generalist judge model, alongside a comprehensive data curation strategy and two new benchmarks (JudgerBenchV2 and RubricBench). The authors decompose the capabilities of a judge model into three distinct scenarios: objective verification, subjective evaluation, and rubric refinement, and develop a tailored data collection and filtering pipeline for each. The resulting model is evaluated on a suite of existing and newly proposed benchmarks to demonstrate its robustness and generalization.

### Strengths
The paper presents a comprehensive investigation into the LLM-as-a-Judge paradigm. By systematically decomposing judge capabilities and addressing a wide range of evaluation scenarios (objective, subjective, and rubric-based), the work provides broad and valuable coverage of the problem space.

### Weaknesses
1. As the paper's core contributions include a new dataset and two new benchmarks, their availability is crucial for verification and for the broader research community. It is highly recommended that the authors make these resources publicly available, at least during the rebuttal phase, to allow reviewers to fully assess their quality and impact.

2. The manuscript could be improved in terms of clarity and structure. For instance, the transition and logical connection between Sections 3.2.1 ("Incorporating Verified Reward") and 3.2.2 ("JudgerBenchV2") are not immediately clear, which may cause confusion for the reader. Strengthening the narrative bridges between different methodological sections would enhance readability.

### Questions
1. In Line 260, you state that rejection sampling for the **subjective evaluator** is performed by matching predictions with the ground truth. Could you elaborate on this process? Given that subjective questions are open-ended and may not have a single "correct" answer, how is this ground-truth matching implemented, and how robust is this strategy to the inherent ambiguity of subjective data?

2. The paper presents an ablation study on the entire Judge-RFT dataset. It would be more impactful to see a more fine-grained ablation that isolates the contribution of the specific data components mentioned in Section 3.4 (objective, subjective, rubric, and general SFT data). For example, how does training on each data component individually affect performance on the corresponding benchmarks (e.g., RewardBench and RubricBench)? Such an analysis would be very helpful in demonstrating the effectiveness of your task-specific data curation strategy.

3. Generalization to State-of-the-Art Models: It is noted that several state-of-the-art LLMs (like GPT-OSS and Qwen3) were used during the dataset creation phase. However, the experiments in the paper are conducted primarily on what might be considered legacy models (e.g., Qwen2.5, Llama-3.1). To convincingly demonstrate the generalization ability of your proposed datasets and benchmarks, it is highly recommended to include experimental results on some of the latest publicly available LLMs.

I would be happy to raise my score if the authors can address my concerns.

### Soundness
3

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
5

### Summary
This paper proposes CompassJudger-2, an all-in-one LLM-as-Judge framework designed to generalize across three skill hierarchies: objective verification, subjective evaluation, and rubric refinement. The authors introduce two new benchmarks—JudgerBenchV2 (for accuracy + rank-fidelity evaluation with “Mixture-of-Judgers” ground truth) and RubricBench (for test-time improvement through rubric refinement). The work builds upon CompassJudger-1 by expanding data diversity via task-specific rejection sampling and verifiable reward guidance. Experiments show state-of-the-art results on multiple judge and reward benchmarks.

### Strengths
1: Clear task hierarchy. The decomposition into objective verification, subjective evaluation, and rubric refinement is well motivated and pedagogically intuitive.

2: Contribution in New benchmarks. JudgerBenchV2 and RubricBench contribute practical evaluation infrastructure and metrics that better capture ranking consistency and rubric-guided improvement.

3: Moderate innovation at model + data levels. The model integrates verified rewards and delta-based rubric generation, and the datasets are carefully curated with rejection sampling for reliability.

4: Strong empirical results. The system achieves solid gains across most judge and reward benchmarks and maintains good robustness across different judging styles.

5: Well-structured related-work section. The literature review is clean and logically organized, situating the work clearly in the LLM-as-Judge landscape.

### Weaknesses
1: Unsubstantiated claim of domain generalization. The introduction argues that existing judge models “fail on unfamiliar tasks or domains” and that CompassJudger-2 remedies this, yet all experiments are performed on common and popular benchmarks (MMLU-Pro, GPQA, RewardBench, ArenaHard, etc.). To validate generalization, the model should be tested on genuinely out-of-domain or unseen domains (e.g., non-academic reasoning, multimodal judgment, low-resource languages).

2: Pedagogical analogy lacks theoretical grounding. The paper draws inspiration from “the hierarchy of skills demonstrated by teachers,” but no pedagogical or cognitive theory (e.g., Bloom’s taxonomy, scaffolding theory) is cited. Strengthening this with supporting literature would make the conceptual framing more credible.

3: Dependence on extremely large teacher models. The pipeline heavily relies on Qwen3-235B-Instruct and DeepSeek-V3 (both are very big models)for data generation, verification, and judging. This raises concerns about computational cost, environmental footprint, and fairness of comparison with smaller baselines. A clear efficiency analysis is missing. 

4: Weak “student” baselines. LLaMA 3.1-8B (Grattafiori et al., 2024) and small Qwen variants serve as students; these are arguably too limited for 2025 standards, reducing the significance of reported rubric-refinement gains. Stronger student baselines in the same size would provide a fairer test.

5: Marginal conceptual novelty. The methodological core largely combines known components (rejection sampling, SFT + PG training, DPO-style mapping losses). The main contribution lies in benchmark engineering and data organization rather than new learning principles.

6: Incomplete evaluation of trade-offs. The paper does not report training time, inference latency, or computational resources, leaving the cost–benefit ratio unclear. Surely it can get better results in benchmarks, but the question is will the cost worth it.

### Questions
What is the computational cost of training with Qwen3-235B compared to other baselines?

Could smaller “teacher” models replicate the same gains?

What theoretical or empirical justification underlies the pedagogical “hierarchy of judging skills”?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper aims to improve the general and judgement capabilities of LLM-as-a-judge evaluations systems by a specific training paradigm for judge models called CompassJudger-2. The core part of the proposed system is the training data construction for three levels of evaluations tasks: objective verification, subjective evaluation and rubric refinement. The authors carefully design curation and sampling methods to produce high-quality training data for those tasks. This paper also introduce novel benchmarks called JudgeBenchV2 and RubricBench with more scenario coverage and less GT noise. The experimental results shows that the proposed CompassJudger-2 models perform excellently on multiple judge benchmarks including the JudgeBenchV2 and RubricBench.

### Strengths
1 This paper proposed a comprehensive data construction pipeline for evaluation model training. The task decomposition, sampling strategy and data curation methods are logically coherent. 
2 The authors makes great effort on data curation and sampling, including some dirty work such as outdated data refreshment and judgement data enhancement. I believe these data (if made public) could bring more contribution to the community than the proposed methods.
3 The introduced JudgeBenchV2 and RubricBench benchmarks bring more insight on reliable evaluation of LLM-as-a-judge systems.
4 The trained CompassJudger-2 models demonstrate excellent judgement capabilities on various scenarios.
5 The paper is generally well-written.

### Weaknesses
1 The training details in 3.2.1 and 4.1 are confusing. In 4.1, the paper claims the DPO and margin mapping losses are applied only on limited conditions while in 3.2.1 it seems that the mapping losses are only used for the subjective evaluation tasks. How the losses applying to different tasks and conditions are not clear. 

2 Minor: typo at line 113 in Figure 2: Unconsistent-> inconsistent.

### Questions
The paper trains a single model for CompassJudger-2 to do a wide range of evaluation tasks. Is it possible to train separate models for specific tasks and achieve better performance? e.g. train a model with only data for objective verification tasks.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces CompassJudger-2 (CJ-2), a generative 'all-in-one' LLM judge trained to handle three tasks: objective verification, subjective evaluation and rubric refinement. It contributes two evaluation resources, JudgerBenchV2 and RubricBench, and a training recipe that combines diverse data curation with a 'Judge-PG' mapping-loss family. In the main tables, CJ-2 reports higher scores than selected baselines across JudgeBenchV2, JudgeBench, RewardBench and VerifierBench. It also shows competitive performance on general benchmarks such as MMLU-Pro, GPQA, AIME-2025, LiveCodeBench v5, IFEval and Arena-Hard. The paper further claims that CJ-2 improves test-time 'rubric refinement' for several student models and generalises out of distribution.

### Strengths
* Unified evaluation system. Combining objective verification, subjective judging and rubric refinement into a single model is a clear systems contribution.

* Valuable benchamrk and dataset contributions. JudgerBenchV2 provides ranking-aware metrics and 'mix-of-judgers' references. RubricBench formalises rubric-driven test-time improvement with clear task guidelines.

* The training pipeline and ablations are well described. The paper details mapping-loss choices and includes data ablations that separate the effects of Judge-RFT and general SFT data.

### Weaknesses
* Limited stats. Results are reported as single numbers (without error bars, standard deviations, or significance tests). It is difficult to asses differences cannot for statistical meaningfulness. Including repeated trials and confidence intervals would strengthen claims.

* Baselines. The main tables skip several strong judge and reward-judge baselines (Skywork Critic/Reward models, GLM-judge family, InternLM reward judges, Nemotron-based judges) that rank highly on public leaderboards. While one Skywork model appears in the appendix, higher-ranked variants are still missing.

* Prompt-sensitivity analysis is limited. Only a small set of style prompts were tested, again with single number reporting. A more robust testing including paraphrases, role swaps and response-order randomisation woudl probably better assess bias and sensitivity, see [1,2]. 

* Sparse reproducibility details. Some important details such as the number of seeds, run-to-run variance or compute requirements are not present. 

Ambiguous references. Mentions of benchmarks such as VerifyBench seem to lack attribution or explanation. Clear citations and dataset descriptions are needed to allow independent verification.

[1] Wang, P., et al. (2024). Large language models are not fair evaluators. ACL. https://aclanthology.org/2024.acl-long.511.pdf
[2] Shi, L., et al. (2024–2025). Judging the judges: A systematic study of position bias in LLM-as-a-Judge. https://arxiv.org/abs/2406.07791

### Questions
1. Could you report error bars, confidence intervals, and significance tests for all main results (including RewardBench and JudgeBench subsets)?

2. Several strong judge baselines (e.g. Skywork Reward/Critic models, Nemotron, InternLM reward models, GLM-judge family) are missing from the main tables. Can you include direct comparisons to these top leaderboard models to better contextualise CompassJudger-2's performance?

3. How is the “overall score” in Table 5 computed? Please provide the exact formula, normalisation, handling of ties/negative scores, and aggregation unit.

4. Current OOD claims rely on only two held-out student models. Would it be possible to expand the OOD evaluation to include more families/sizes and report variance across runs?

### Soundness
2

### Presentation
2

### Contribution
2
