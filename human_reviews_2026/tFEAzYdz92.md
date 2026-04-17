# OR-PRM: A Process Reward Model for Algorithmic Problem in Operations Research

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 8

## Abstract
Large language models (LLMs) with Process Reward Models (PRMs) have shown strong reasoning ability, yet their potential in Operations Research (OR) remains unexplored. We present the first PRM tailored for OR, but find that directly training on mainstream datasets yields surprisingly weak performance.
To understand this gap, we conduct a systematic analysis and identify the primary bottleneck: the datasets themselves, where over 30\% of annotations are severely flawed.
To overcome these limitations, we first collect all existing synthetic datasets and apply a carefully designed filtering pipeline to construct a high-quality seed dataset. 
Building upon this seed, we then build OR-ProcessQA, the first large-scale dataset for OR with step-by-step supervision, where diverse solution pathways are generated via Monte Carlo Tree Search (MCTS) and each step is validated for logical consistency by GPT-4o.
Building on this foundation, we train OR-PRM, the first Process Reward Model in the OR domain, designed to evaluate and guide reasoning at every step rather than only the final outcome.
Together, these advances enable OR-PRM to substantially improve LLMs’ reasoning capability, achieving a maximum absolute improvement of 12.5\% over the base model in Best-of-N settings, and highlighting the power of process-oriented supervision for reliable problem solving in operations research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces OR-PRM, the first Process Reward Model (PRM) tailored for Operations Research (OR) reasoning tasks. The work targets a fundamental challenge: large language models (LLMs) often fail to produce reliable, logically consistent reasoning in OR problems involving optimization modeling, constraints, and solver code generation.

To address this, the authors propose a three-stage pipeline:

Data construction – Build a high-quality, process-annotated dataset (OR-ProcessQA) through careful seed curation, constraint validation, and semantic verification.

Monte Carlo Tree Search (MCTS) – Generate diverse reasoning trajectories and preliminary correctness labels automatically.

Process Reward Model (OR-PRM) – Train a generative PRM using Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO), enabling fine-grained, interpretable step-level feedback for OR reasoning.

Extensive experiments on multiple open-source (Qwen2.5, LLMOPT) and closed-source (GPT-4o) models show that OR-PRM substantially improves reasoning accuracy, robustness, and interpretability.

### Strengths
Well-motivated and novel contribution: Addresses a clear gap between generic reasoning PRMs and domain-specific reasoning in mathematical optimization. OR-PRM is the first reward model explicitly designed to evaluate reasoning steps in mathematical optimization, going beyond scalar scoring to produce structured, natural-language critiques and corrections.

Methodologically rigorous: Multi-stage pipeline (SFT + DPO) is logically constructed, with explicit checks (execution, constraints, semantics) to ensure data correctness.

Excellent technical clarity: The paper defines each component precisely (seed data, MCTS, PRM training), making it reproducible and interpretable.

Comprehensive evaluation: Benchmarks include both open- and closed-source models, with ablation studies isolating the effect of DPO alignment.

Strong empirical results and interpretability: The improvement is not just numerical but also interpretable, showing how OR-PRM critiques and corrects reasoning steps.

### Weaknesses
It is not clear to me whether the evaluation pipeline can serve as the ground truth. Since LLMOPT and GPT-4o may make mistakes. It is unclear how far the evaluation is from having an OR expert evaluate the reasoning steps.

### Questions
The pipeline heavily depends on LLMOPT. It would be nice to discuss if a different model were used to generate the data, how much impact that would have on the results.

It seems that the OR problems have data that appeared in the problem description in scalar form. What if the data is stored in CSV files? Would the pipeline still work?

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
3

### Summary
The paper introduces OR-PRM, a domain-specialized Process Reward Model for Operations Research (OR). The authors (1) diagnose high noise in existing OR datasets; (2) curate a cleaner seed set via a three-stage pipeline; (3) generate step-wise trajectories with MCTS; (4) have GPT-4o perform structured, step-level judgments and corrections; (5) train a generative PRM that outputs critiques rather than scalar scores; and (6) show gains under Best-of-N selection and a Modeling→Critique→Code pipeline, reporting up to +12.5 pp average improvement across benchmarks and model sizes (Qwen2.5 series, LLMOPT).

### Strengths
1. PRMs are a natural fit because OR solutions require step-wise logical validity (not only final objective values). The paper targets this gap explicitly.

2. Three-stage filtering for a seed set, MCTS exploration, and GPT-4o step audits with issue/judgement/correction fields—this is more structured than typical PRM pipelines in math reasoning.

3. Returning natural-language critiques and a corrected first error goes beyond scalar PRMs and mirrors trends in “corrective” PRMs in broader reasoning literature.

4. Best-of-N and model-agnostic critic usage show uniform gains; the Complex-LP results are especially notable (e.g., +24.2 pp on Qwen2.5-32B).

### Weaknesses
1. The paper should deeply contrast with OmegaPRM (automated process supervision via MCTS) which established scalable step-labeling at large scale in math reasoning (1.5M annotations) (https://arxiv.org/abs/2406.06592), and with recent generative/critic-style PRMs designed to explain and correct steps such as GM-PRM and VisualPRM/Athena-PRM (multimodal, but methodologically very close in “PRM generates critiques + refines BoN”) (https://arxiv.org/abs/2508.04088). The proposed “generative PRM” feels incremental w.r.t. these trends.

2. The paper claims >30% severe flaws in mainstream OR data and a rigorous three-stage filter, but I don’t see inter-rater reliability, error taxonomy distributions, or random spot-check protocols beyond illustrative examples. Given PRM sensitivity to label noise (documented in PRM surveys/lessons) (https://arxiv.org/abs/2501.07301) , stronger auditing is necessary.

3. Step labeling and final verification heavily rely on strong LLMs (GPT-4o/Qwen verifiers) inside the pipeline that also guide critique at inference. Without cross-model, cross-vendor checks and held-out validators, there is potential circularity (the critic agrees with the validator it was trained/selected with).

4. Since BoN/selection tends to give large lifts, the paper should compare against strong non-PRM baselines (self-consistency; majority vote with route-length filters; retrieval-augmented PRM or OOD-robust PRMs) that recent works show to be competitive, e.g., Retrieval-Augmented PRM (R-PRM / RAPRM) and R-PRM variants focusing on OOD and data bootstrapping. (https://arxiv.org/abs/2502.14361) Current ablations (e.g., “Major Voting”) look weak and not representative of the SOTA toolkit.

5. OR is a heterogeneous space (LP/MILP/NLP/CP-SAT with industry quirks). I don’t see OOD splits (new templates, new constraint families, solver switches) or robustness to noisy/problematic instances (e.g., infeasible but realistic specs). OOD weaknesses are a known pain point for PRMs. (https://arxiv.org/abs/2502.14361) 

6. The Modeling→Critique→Code pipeline may conflate several effects: (i) data curation, (ii) structured prompts, (iii) critic guidance. A factorized ablation (swap in a scalar PRM; blind the critic to code; remove “Corrected Step”) would clarify whether “generative PRM” itself is the key.

7.Important but under-specified: exact prompts, MCTS hyper-parameters per task, GPT-4o temperature and refusal handling, and the policy/critic decoupling at test time (who conditions on whom).

### Questions
1. How often does GPT-4o disagree with the MCTS label? Show confusion matrices and human audits on a random 500-step sample. How many “corrections” by GPT-4o were later proven wrong?

2. If you allow the critic to emit corrections (you already do), can you implement Refined-BoN like GM-PRM and report deltas? (https://arxiv.org/abs/2508.04088)

3. Evaluate with new solver backends (e.g., Pyomo→PuLP / OR-Tools), new template families, and noisy text to test OOD per RAPRM concerns. (https://arxiv.org/abs/2502.14361)

4. Ablate the “generative” aspect: Replace the critic with (i) scalar PRM, (ii) critique-only (no correction), (iii) correction-only (no explanation). Which component drives Complex-LP gains?

5. Add self-consistency, vote-with-length-normalization, and retrieval-augmented PRM. Current “Major Voting” is too weak to claim SOTA.(https://arxiv.org/abs/2502.14361)

6. End-to-end token/cost for data curation + training + inference (critic calls per step)? Compare with OmegaPRM’s cost per step label. (https://arxiv.org/abs/2406.06592)

7. Failure modes: Provide qualitative cases where OR-PRM gives confident but wrong constraints/objectives (classic OR pitfall), and whether the system catches feasible but semantically wrong models.

8. For 100 randomly sampled instances, have expert OR graders rate the usefulness and correctness of critiques (Likert & error-type taxonomy). Your current evidence relies mostly on automatic verifiers.

9.  As far as I know, the algorithm complexity of MCTS is relatively high, and I am interested in the impact of introducing MCTS on computational efficiency.

10. If we don't use GPT-4o, which is a relatively good model, but some relatively poor models, can your method still demonstrate robustness?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors develop the first Process Reward Model (PRM), which is specifically intended for Operations Research. Based on prior research that existing synthetic datasets in this area are significantly flawed, the manuscript proposes a filtering pipeline to initially construct a "seed" dataset of problems and their description in a formalized way as well as a solution. Each problem is validated in multiple ways by ensuring that the reference code runs and yields the correct result, that constraints are valid as well as that the mathematical model to solve represents the problem faithfully. This dataset is then used to construct several reasoning trajectories of multiple reasoning steps based on Monte Carlo Tree Search (MCTS), where each step is validated, to generate a dataset for training (called OR-ProcessQA). A process reward model, OR-PRM, is then trained on this dataset to provide dense rewards for guiding the reasoning effectively. The reward is not a single score, but instead provides feedback in natural and corrections in natural language, so is more like a critic model.

### Strengths
* new dataset seems to be addressing a need in this field, if existing datasets are so severely lacking
* writing seems to be mostly clear, especially the description of the dataset construction

### Weaknesses
* field of Operations Research is never properly introduced
* novelty is a bit diminished by over-claiming: abstract/introduction/conclusion give the impression that the analysis of the existing datasets is done by the authors of this manuscript, but judging from the related work section it was done by someone else
  * this directly hurts the motivation for the new dataset, since the problems in existing datasets are not discussed in detail
* dataset statistics are mostly missing, for example:
  * number of samples in the (seed) dataset
    * What is a sample?
  * average length of trajectories, i.e. number of steps
  * number of trajectories per sample
  * how many failed/successful trajectories
  * dataset composition from the original datasets (partially in Appendix C.2)
  * little empirical analysis of the dataset creation pipeline
* limited evaluation:
  * generation (not the primary focus of the manuscript): while different model sizes are used, they are only from one model family, leaving the question whether the approach generalizes to other model families
  * only a single model, small, trained as a reward model, so generalization is a big question
  * analysis focuses too much on the employed generative models and less on the capabilities of the reward model
  * no baselines for the first scenario (Best-of-N) besides the original model
  * unclear baselines for second scenario:
    * How does pass@1 with critic work? It considers the proposed corrected step/solution?
    * How does pass@8 without critic work?
  * potential baseline: Use of a general purpose model with the same prompt to act as an critic?
  * missing cost analysis

minor issues:
* it seems that the wrong cite command is used in general, so brackets around the references are missing
* related work:
  * title - seems to be not all caps like the other section titles
  * line 147: for some reason "offline" is incorrectly hyphenated
* Figure 2:
  * you might want to make the bubble (bottom row, roughly the middle) a bit bigger, so that "Data Diversification" does not touch the outline
  * same for the bubble inside "OR-PRM Model Training"
* titles of 3.1 and 3.1.1: You might want to properly capitalize the titles.
* 4.1:
  * line 327: "Specifically, Industry OR..." - not a proper sentence
  * line 331: "OR-PRM, We" - "we" should not be capitalized
  * line 346: reference for CoT?
* references:
  * cited differently than other arXiv references (can only be surmised from the URL):
    * [DeepSeek-AI 2024]
    * [Luo et al. 2024]
    * [Ma et al. 2024]
    * [Zhang et al. 2025a]
  * additionally consider capitalizing the titles to be consistent, especially abbreviations and proper names (which is sometimes done):
    * [Huang et al. 2025a/b]
    * [Ma et al. 2024]
    * [OpenAI 2025]
    * [Wang et al. 2025]
    * [Wu et al. 2025]
    * [Xiao et al. 2024]
    * [Xiao et al. 2025]
    * [Xie et al. 2023]
    * [Yang et al. 2025b]
    * [Zhang et al. 2025a]
    * [Zhou et al. 2025]
  * missing place of publication: [Wu et al. 2025]
  * [Xiao et al. 2025] was published IJCAI '25
* Appendix B:
  * NL4OPT: The number of samples (245) seems to differ from Table 3.
  * line 711: Wrong table reference (Table 3 instead of 4)?
* Figure 5, caption: "Example:LLM" - missing whitespace
* Appendix E, line 858: "etc.Three" - missing whitespace; "Three examples as follow:" - not a proper English sentence
* Appendix F: title sounds a bit strange

### Questions
* 3.1.1: What does $\hat x$ represent? The new solution? But if the expected output is already known, why not use that as ground truth?
* 3.1.2: What is $\mathcal{L}$? I suppose it is not the loss function?
* 3.2: How are Generative PRMs different from a general critic model employed for each reasoning step?
* Table 1: What would have been the improvements, if the best solution would have been selected based on the ground truth, i.e. what would have been the upper bound of improvement when generating additional trajectories?
  * This would give context how well OR-PRM selects reasoning chains or whether improvements "just" stem from the additional generation, i.e. more token use.
  * this sentence in 4.4 seems to suggest that such data exists: "our Best-of-N performance is strong, but it still falls short of the theoretical upper bound" (line 466)
  * Why was OR-PRM not used in this setting for the proprietary models
* Table 2:
  * What does Qwen2.5 (Zero Shot) do? Why are the results different than in Figure 3?
    * same for OR-PRM (Ours)
  * How does majority voting (filtered null) work?
  * Which scenario is used for the ablation study? probably the second one
* Did you try any out of distribution benchmarks?
* 4.4/conclusion: How would expanding the dataset help to create credible baselines?
* Appendix F, Critic Prompt: seems only for the full solution
  * Can we actually call this a Process Reward Model, if it does not look at individual reasoning steps?
  * What is difference between the two prompts on page 24? Where are they applied?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the first Process Reward Model tailored for Operations Research (OR-PRM), aiming to explore the potential of Large Language Models (LLMs) in this complex reasoning domain. The authors surprisingly find that direct training of a PRM on existing mainstream OR datasets yields weak performance. Through systematic analysis, they identify the primary bottleneck as data quality, noting that over 30% of existing annotations are severely flawed. To address this, the authors collect all existing synthetic datasets and employ a carefully designed filtering pipeline to construct a high-quality seed dataset, which is then used for model training and evaluation. This work not only lays the foundation for applying LLMs in the OR field but also provides a crucial warning and proposed improvement strategy for the quality of current OR algorithm datasets.

### Strengths
1. This is the first application of the Process Reward Model paradigm to algorithmic problems in the field of OR. OR is a domain highly dependent on structured reasoning and complex algorithms, making the integration of LLMs highly valuable for research and application.

2. The authors conduct a systematic analysis that clearly identifies a severe annotation quality issue (over 30% defects) in mainstream OR datasets. This is a significant contribution and warning to the wider community, as identifying data bottlenecks is a crucial step for field advancement.

3. The authors do not avoid the data quality issue but instead proactively address it by collecting existing synthetic datasets and applying a "carefully designed filtering pipeline" to build a high-quality seed dataset. This strategy of solving the problem at the data source is commendable.

4. If OR-PRM proves effective, it will greatly simplify the modeling and solving process for OR problems, providing new avenues for automated algorithm discovery and solution.

### Weaknesses
No serious weakness.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
