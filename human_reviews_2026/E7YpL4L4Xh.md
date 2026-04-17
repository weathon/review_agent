# EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Instruction-guided image editing has achieved remarkable progress, yet current models still face challenges with complex instructions and often require multiple samples to produce a desired result. Reinforcement Learning (RL) offers a promising solution, but its adoption in image editing has been severely hindered by the lack of a high-fidelity, efficient reward signal. In this work, we present a comprehensive methodology to overcome this barrier, centered on the development of a state-of-the-art, specialized reward model. We first introduce $\textbf{EditReward-Bench}$, a comprehensive benchmark to systematically evaluate reward models on editing quality. Guided by this benchmark, we develop $\textbf{EditScore}$, an efficient model to evaluate the quality of instruction-guided editing. Through meticulous data curation and filtering, EditScore effectively matches the performance of learning proprietary VLMs. Furthermore, coupled with an effective self-ensemble strategy tailored for the generative nature of EditScore, our largest variant even surpasses GPT-5 in the benchmark. We then demonstrate that a high-fidelity reward model is the key to unlocking online RL for image editing. Our experiments show that, while even the largest open-source VLMs fail to provide an effective learning signal, EditScore enables efficient and robust policy optimization. Applying our framework to a strong base model, OmniGen2, results in a final model that shows a substantial and consistent performance uplift. Overall, this work provides the first systematic path from benchmarking to reward modeling to RL training in image editing, showing that a high-fidelity, domain-specialized reward model is the key to unlocking the full potential of RL in this domain. Our code, models, and benchmark will be released publicly.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces EditReward-Bench (a benchmark for reward models) and EditScore (a family of generative reward models) for image editing. The paper first collects human preference and ranking data from annotators based on candidate images and asked to rank them based on certain qualities. This data is then used to train generative reward models, which are compared for agreeableness to the human rankers.

### Strengths
- this paper introduces a solid dataset and reward model for RL training.
- this paper is decently written, but some sections (eg, explaining how the scores of table 2 of EditReward-Bench) could be rewritten. Otherwise, it's clear.
- the methodology for data collection seems sound.

### Weaknesses
- I am concerned about the use of candidate images from current models as part of the data labeling process. Indeed, if there is an area where all the candiate models struggle, although we would be able to provide a relative ranking, the maximal reward would only be obtained from the model generalizing, not any supervision.
- section 6.4.1 compares two different models for optimization, and as OmineGen2 was chosen for high pass@k, a better comparison would be using the same base model. It is hard to tell if this truly is the result of a better reward model, or a more amenable base model.

### Questions
- how are table 2 scores computed exactly?
- how are the aggregation scores computed?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a critical bottleneck in applying reinforcement learning to image editing: the lack of a reliable and efficient reward model. The authors introduce **EditReward-Bench**, a comprehensive benchmark for evaluating reward models in instruction-guided image editing, and **EditScore**, a family of open-source reward models that outperform existing open-source and even some proprietary VLMs. Through extensive experiments, they demonstrate the utility of EditScore in both Best-of-N selection and online RL training, achieving stable and significant improvements over strong baselines.

### Strengths
**Novel Contribution:** The paper makes a clear and timely contribution by addressing the underexplored problem of reward modeling for image editing. The release of both the benchmark and the model fills a gap in the community.

- **Rigorous Benchmarking:** EditReward-Bench is well-designed, covering 13 diverse editing tasks and 11 editing models, with human-annotated multi-dimensional evaluations.
- **Strong Empirical Results:** EditScore consistently outperforms open-source VLMs and even matches or surpasses proprietary models like GPT-4.1 in some settings. The self-ensembling strategy is simple yet effective.
- **Practical Utility:** The paper convincingly demonstrates the real-world impact of EditScore through Best-of-\(N\) selection and online RL experiments, showing stable training and performance gains.
- **Insightful Analysis:** The study on reward variance vs. annotator accuracy is a valuable finding that could influence future reward modeling efforts.

### Weaknesses
- **Limited RL Scope:** The RL experiments are conducted only on OmniGen2. While justified, it would be beneficial to see how EditScore generalizes to other base models.
- **Computational Cost:** The self-ensembling strategy, while effective, increases inference cost. A more detailed discussion on the trade-off between performance and efficiency would be helpful.
- **Comparison with More Baselines:** While the comparison with Qwen2.5-VL is thorough, it would be useful to see how EditScore compares to other recent open-source reward models (e.g., those from HPSv3 or ImageReward).

### Questions
1. The training data for EditScore is constructed using a pipeline that involves generating instructions and candidate outputs from a pool of 11 editing models, including state-of-the-art proprietary ones. Could you please clarify the measures taken to ensure that **EditReward-Bench's test set does not contain images or near-identical instructions from the training data**? Given the overlap in the model pool and data sources (e.g., GEdit-Bench), a detailed description of the train/test split protocol is crucial to validate the benchmark's integrity and prevent overestimation of generalization performance.
2. The paper states that EditReward-Bench annotations "have passed rigorous inter-annotator agreement checks." For transparency and reproducibility, could you provide the **specific agreement metrics used (e.g., Kendall's Tau, Spearman correlation, or percentage agreement)** and their quantitative results? Furthermore, please detail the **number of annotators per sample** and the process for resolving disagreements to achieve the final consensus ranking.
3. The performance improvements presented in Tables 2 and 5, while substantial, are reported as point estimates. Have you conducted tests for **statistical significance** (e.g., paired bootstrap tests or t-tests over the benchmark subsets) to ensure that the gains from EditScore and the RL fine-tuning are not due to random chance? Reporting p-values or confidence intervals would strengthen the claims.
4. The paper excellently demonstrates EditScore's strengths. However, a thorough analysis of its **failure modes and limitations** is equally important. Could you provide examples or a discussion of edit types or instructions where EditScore consistently fails or provides unreliable rewards? For instance, does it struggle with highly abstract or compositional instructions ("make it look more joyful") ? A dedicated "Limitations" section would be valuable.
5. The self-ensembling (Avg@4) strategy is shown to be highly effective. However, it incurs a 4x computational overhead during inference. Could you provide a more detailed analysis of this **trade-off between performance and efficiency**, such as the actual increase in inference latency? This is critical for practitioners considering the deployment of EditScore in resource-constrained environments.

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
The paper tackles a critical bottleneck in applying online reinforcement learning (RL) to instruction-guided image editing: the lack of a high-fidelity, efficient reward signal. The authors argue that existing proprietary VLMs are too costly for online RL , while large open-source VLMs are not accurate enough, leading to training instability. To this end, the paper presents a comprehensive, three-part solution:
1. This paper introduces EditReward-Bench, a new, meticulously annotated benchmark for evaluating image editing reward models (RMs). 
2. Guided by this benchmark, they develop EditScore, a series of specialized generative reward models (7B-72B) that achieve SOTA accuracy, even surpassing proprietary models like GPT-5. 
3. This paper provides the first successful demonstration that a specialized RM (EditScore) can unlock stable and effective online RL training for image editing, whereas general-purpose VLMs fail. The final RL-tuned policy shows substantial performance gains on standard editing benchmarks.

### Strengths
* The authors constructed a benchmark for the image editing domain to evaluate Edit Reward Models. Based on this benchmark, they built a series of editing Reward Models, which were then used with Flow-GRPO to enhance the editing capabilities of the base model, OmniGen2.

* The authors commit to releasing the proposed benchmark, the weights of the Reward Models, and the weights of the optimized OmniGen2 model. This will provide definite value to the community.

* The tricks proposed in the methodology, such as self-ensemble, Reasoning-First, and using an RM with higher reward variance, are reasonable and lead to substantial performance improvements.

### Weaknesses
* I believe the experiments need improvement. The paper currently only reports experiments on OmniGen2. It remains unclear whether the method is effective for other editing models of different sizes and types, such as the simpler InstructP2P (SD1.5-based), the Unified Model-based Bagel, or even training a LoRA for the larger diffusion-based Step1X-Edit.

* The methodological innovation is limited. The Flow-GRPO paper [1] already demonstrated that its method can be used to improve editing models. The concept that high-variance RMs are preferable to high-accuracy RMs has also been proposed in prior work [2]. Furthermore, self-ensemble and Reasoning-First are common techniques in reward model training.

* The authors state they constructed "70,000 data samples for training the reward model and 60,000 samples for reinforcement learning training," but they do not plan to open-source this data. This makes it difficult for the community to fully reproduce the paper's results. Although the benchmark and model weights will be released, this lack of training data makes the contribution somewhat less exciting to me.

[1] Flow-grpo: Training flow matching models via online rl, NeurIPS 2025

[2] What makes a reward model a good teacher? an optimization perspective, arXiv 2025

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the missing piece for online RL in instruction-guided image editing by building a high-fidelity, domain-specialized reward model. It first introduces EditReward-Bench, a 3,072-pair human-rated benchmark that scores edits on prompt following, consistency, and overall quality. Guided by this benchmark, the authors train EditScore reward models at multiple scales (7B-72B) and use a simple self-ensemble at inference to boost accuracy, matching or surpassing proprietary VLMs on the benchmark. Finally, they show that EditScore unlocks stable and effective online RL, improving strong editors like OmniGen2 and outperforming open VLMs both for Best-of-N selection and policy optimization.

### Strengths
The paper is original in framing instruction-guided image editing as an online RL problem that is unlocked by a domain-specialized reward model and a purpose-built benchmark, plus a simple self-ensemble that improves reward fidelity. Experimental quality is strong, with a clear scaling study, competitive baselines for Best-of-N selection and policy optimization, and consistent gains on prompt following, consistency, and visual quality. Clarity is high: the problem setting, evaluation axes, and training pipeline are well motivated and easy to follow, and the empirical claims map cleanly to the stated goals. The work is significant because it removes a key barrier to applying online RL in generative editing, yielding practical improvements on strong editors and delivering reusable infrastructure that can benefit future systems and studies.

### Weaknesses
Despite solid results, the work’s evidence base and scope feel narrower than the claims. The benchmark is small and curated for perfect agreement, which can hide genuine preference diversity and inflate metrics; include disagreement analyses, release non-consensus pairs, and recruit a broader rater pool. Reproducibility is brittle because the candidate pool depends on proprietary editors; ship frozen artifacts or open substitutes so others can rerun everything. Generality is underexplored since online RL is shown mainly on one editor with one algorithm; add ablations across multiple editors, RL methods, and hyperparameters, plus stress tests on out-of-distribution edit types. The self-ensemble that yields top reward accuracy likely raises cost and latency; report cost-normalized accuracy, wall-clock time, and memory, and compare against single-model baselines at equal budgets. Finally, claims about reward variance aiding optimization are intriguing but not causally established; isolate variance from annotator accuracy with controlled studies and replicate the effect on additional datasets and editors.

### Questions
Benchmark: How were prompts/edit types sampled and how representative are they?
Raters: What was raw disagreement and how do results change without consensus filtering?
Reproducibility: Will you release frozen candidate sets or open substitutes for proprietary outputs?

### Soundness
3

### Presentation
3

### Contribution
3
