# Coder-R3: Recognize, Review and Repair Defective Code with Finetuned LLMs in Practice

- Decision: Reject
- Scores: 4, 4, 4, 4, 2

## Abstract
Large language models (LLMs) have emerged as powerful tools for software engineering tasks, demonstrating particular promise in code review activities. However, existing research on code review LLMs typically decomposes the review process into discrete subtasks, collecting data and fine-tuning separate models for each individual component. This fragmented approach overlooks the synergistic relationships between different tasks, necessitates multiple models with complex multi-stage invocations, and consequently exhibits limited practical applicability in real-world deployment scenarios. In this work, we advance beyond previous code review research by proposing a unified and comprehensive code review problem modeling approach. We focus on the complete code review process, including Recognize, Review, and Repair defective code fragments consecutively, and propose $Coder\text{-}R^3$, an approach that enables a single LLM to handle all code review-related subtasks uniformly. Additionally, we establish a practically feasible closed-loop iterative process for industrial scenarios, encompassing data construction, model evaluation, and operational feedback integration. We rigorously evaluate the effectiveness of various strategies, including input context selection, output format, and training methodologies. $Coder\text{-}R^3$ achieves state-of-the-art performance on the CodeReviewer benchmark, and demonstrates superior effectiveness in real enterprise scenarios. Our work provides valuable insights for enterprises seeking to leverage large language models to enhance code review efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Coder-R3, a unified framework that enables a single large language model to perform the three main subtasks in code review — Recognize, Review, and Repair — within one cohesive system. Unlike prior work that decomposes the process into independent subtasks, Coder-R3 integrates these within a hybrid training framework that balances learning efficiency and task specialization. The authors further introduce a data flywheel mechanism, wherein real-world code review data are collected from an internal industrial system, verified by human engineers, and iteratively fed back into model training. Experiments demonstrate: State-of-the-art performance on the public CodeReviewer benchmark; Significant gains on a proprietary internal dataset with over 18k annotated instances; Sustained performance improvement in real-world deployment through continuous feedback loops.

### Strengths
**Unified Modeling of Code Review Tasks** Integrating recognition, comment generation, and repair tasks within a single fine-tuned LLM is novel. This unified design effectively reduces pipeline complexity and inference overhead.

**Practical Deployment and Real-World Impact** The paper extends beyond academic benchmarks by demonstrating a real-world deployment with an active feedback loop. This significantly enhances the practical value of the contribution. The proposed method achieves state-of-the-art results on the CodeReviewer benchmark and shows robust performance in an industrial environment.

**Systematic Data Design** The iterative data collection and model refinement process is well-motivated and realistic. By leveraging an in-house closed-loop system to gather high-quality annotations, the authors mitigate common issues of noise and inconsistency present in public datasets.

### Weaknesses
**Underperformance in the Review Task** Despite the unified design, Coder-R3 underperforms previous methods (e.g., CoRAL, GPT-5) on the Review task, suggesting that unified training may fail to fully capture linguistic and stylistic quality.

**Lack of Theoretical Analysis** The paper is largely empirical, providing limited theoretical insight or interpretability regarding why hybrid training outperforms joint training.

**Reproducibility and Generalization Concerns** Since both the dataset and deployment environment are proprietary and not publicly available, the reproducibility of industrial results is limited. Moreover, the model’s generalization ability to unseen programming languages (e.g., Rust) remains unclear.

**Technical Novelty Relatively Low** The technical novelty of the proposed approach is relatively low.

### Questions
- The paper claims that hybrid training significantly outperforms joint training, but the underlying reasons remain unclear. Could you provide supporting evidence such as loss curves, learning dynamics, or error-type distributions?
- How are data proportions or task weights configured across the three subtasks (Recognize / Review / Repair)? Are all validated samples included equally in the training?
- As the dataset evolves across iterations, does the model suffer from forgetting earlier knowledge? Over a longer horizon, do you observe any performance saturation or regression?
- Have you compared Coder-R3 with three separately trained task-specific models (using the same data)? How large is the performance gap between the unified and the multi-model setup?
- Compared with traditional multi-model pipelines, what are the computational costs (both training and inference)? What are the latency differences across tasks? Does the unified model offer measurable efficiency or cost advantages in production?
- How consistent is Coder-R3’s performance across programming languages? Does it exhibit any zero-shot generalization on languages unseen during training (e.g., Rust)?
- Beyond automatic metrics like BLEU, have you considered incorporating human evaluation or LLM-based assessments to better gauge comment quality?

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
The paper introduces Coder-R3, a single LLM fine-tuned to handle the full code-review loop: Recognize defects, Review them with an explanation, and Repair the code. It builds a data flywheel: deploy the model in an internal review system, capture human accept/reject feedback plus public data, filter/clean it, and retrain on a regular cadence. Through ablations, the authors find a hybrid setup (invoking the same model sequentially per subtask) with diff + language context and generating full repaired snippets works best. On the CodeReviewer benchmark it reports state-of-the-art results for recognition and repair, and in a real company deployment the metrics improve steadily across iterations, suggesting practical viability.

### Strengths
Originality: Frames code review as a single, unified task, "Recognize → Review → Repair", handled by one fine-tuned model rather than separate models/pipelines, which prior work typically used. This integrated formulation is novel and more practical for deployment. 

Quality (technical rigor): Provides concrete design choices and ablations on input context (language metadata, diffs), output format (full code vs diff), and training setups; findings are backed by measurements and useful diagnoses (e.g., overfitting when diffs are truncated). 

Clarity: The paper presents a clear system picture (Figure 1) showing inputs (snippet + context like language, diff, related code) and unified outputs across the three stages, making the overall workflow easy to understand. It also explains how “related code” is sliced via Tree-sitter and call relations, which improves reader comprehension of context construction. 

Significance (practical impact): Goes beyond benchmark gains to show a deployable data flywheel: internal integration to collect human accept/reject signals, public-data supplementation, quality filtering, and regular retraining, with steady real-world improvements over iterative deployments. This makes the contribution meaningful for teams aiming to productionize code-review LLMs.

### Weaknesses
Benchmark dependence & label noise. Most public results hinge on CodeReviewer, which the authors acknowledge has “some quality issues,” including mismatched conventions that make even strong base models underperform and diff truncation that induces overfitting under certain setups. This undermines how confidently we can read reported gains. Consider adding a second public benchmark and a manual audit of a stratified sample. 

Reproducibility/data access gaps. Internal data central to the “flywheel” will not be publicly available, limiting external verification; artifact availability is promised but not yet present in the submission. Release minimally a de-identified internal evaluation set and the full preprocessing scripts. 

Eval concerns. On CodeReviewer dataset eval, baselines are largely non-fine-tuned pretrained LLMs, while Coder-R3 is fine-tuned under a task-specific I/O format raising fairness concerns. And the CodeReviewer model was produced in 2022, which is certainly not SOTA for code review task anymore. You should at least compare to GPT-5 and Sonnet 4.5 with proper prompt tuning. Also BLEU score is not a good metrics for code related tasks, execution-based eval would be much more desired to validate code fix.

Limited external validity of deployment study. The real-world evaluation is within a single company’s stack and language mix, so generalization to other ecosystems is unclear. Replicate with some open-source repos and report cross-project transfer. 

Human-centered quality evaluation is thin. While production metrics (OAR/RP/RR) are useful, there’s no reader study on comment helpfulness, clarity, or correctness. Add a blinded developer study with task time and fix success as outcomes.

### Questions
1. The paper notes that the CodeReviewer dataset has “some quality issues” and that truncated diffs may cause overfitting. Could the authors provide results on an additional public dataset or a manually audited subset of CodeReviewer to confirm that the reported gains generalize beyond this benchmark?

2. Since Coder-R3 is fine-tuned on task-specific I/O while most baselines are frozen pretrained LLMs, how do the authors ensure fairness in comparison? Would they consider releasing fine-tuned checkpoints or training scripts so that external researchers can reproduce the hybrid-training results under identical conditions?

3. The industrial deployment study focuses on automated metrics (OAR/RP/RR) but lacks direct human-judged measures of comment helpfulness or fix correctness. Could the authors add or plan a user study, e.g., developer task-time or fix-success rates, to validate whether Coder-R3 actually improves review quality in practice?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Good paper, but Limited Novelty

### Strengths
The paper presents Coder-R3, a model that learns to recognize, review, and repair code jointly. They implore supervised finetuning on top of the baseline Qwen2.5-Coder-7B. 

++ The qualitative analysis via real-world deployment is interesting and shows the practical impact of this work.

### Weaknesses
- The paper could dive a bit more into the technical implementation. I am a bit unsure about the novelty in the implementation that led to higher performance.
- Missing evaluation on competitive models on the curated internal dataset.
- Missing comparison with the latest SWE-Agents, as I assume they are also capable of completing the three tasks separately.
- Missing citation to a relevant work: Review4Repair: Code Review Aided Automatic Program Repairing (Huq et al, 2021)

### Questions
n/a

### Soundness
3

### Presentation
2

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
The paper addresses the fragmentation in LLM-based code review research by proposing a unified framework, Coder-R3, which models the review process as three steps: Recognize (detect defects), Review (generate explanations), and Repair (correct code). By fine-tuning a single large language model to handle all tasks, the authors explore joint training (generating all results at once) and hybrid training (handling each subtask separately). The paper systematically analyzes the impact of contextual inputs. A key contribution is the "data flywheel," collecting real-world human feedback to iteratively improve the model. Experiments show that Coder-R3 achieves state-of-the-art results on public and proprietary datasets, and its performance continues to improve in real-world deployment via the data flywheel mechanism.

### Strengths
The paper addresses automated code review deployment by integrating a unified model and iterative data strategy, directly targeting industrial software challenges.

The "Recognize-Review-Repair" (R3) framework captures the synergy between defect identification, explanation, and correction, avoiding fragmentation of isolated tasks.

Achieves SOTA on two key tasks in the CodeReviewer benchmark, confirming the reliability of the fine-tuning and data processing methods.

### Weaknesses
Coder-R3 underperforms dedicated models in generating review comments—a limitation noted but insufficiently analyzed. It remains unclear whether this stems from intrinsic trade-offs in multi-task objectives or limitations in training data/methodology. This analytical gap weakens the paper’s credibility.

The data flywheel, meanwhile, is painted in rosy hues. What about the messy realities—spam feedback, shifting code styles across teams, or the growing compute bill every time you retrain?

### Questions
Your results show that "hybrid training" (three invocations) outperforms "joint training" (one invocation). Could you elaborate on the practical advantages of your hybrid approach compared to using three distinct, potentially smaller, specialized models? 

As mentioned above, the data flywheel relies on human feedback. How do you plan to address the challenge of noisy or low-quality feedback (e.g., reviewers lazily accepting all suggestions) as you scale the system? Are there automated quality checks in your "Data Filtering" step to mitigate this?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents Coder-R3, a finetuned LLM that unifies defect detection, review generation, and bug fix a single model. In experiments and ablations, a hybrid invocation of the same model with code-diff context and full-snippet outputs consistently beats joint training and diff-style outputs, yielding clear best practices. The model reaches state-of-the-art on the CodeReviewer benchmark and shows steady real-world gains when deployed inside a company code-review system.

### Strengths
1. Coder-R3 shows substaintial performance gain on top of the origianl model (Qwen-2.5-Coder-7B) being fine-tuned.

2. A rather comprehensive collection of baselines are introduced and compared, making it informative for new readers in this area.

### Weaknesses
1. It seems that exsting works already have solution for unified role [1] .

2. The technical novelty and contribution of this paper is not clear, I fail to see any actual novel methodlogy 

3. In Table 2, performance metrics of significant part of the baselines are omitted, and there's no reason to explain that.

4. There lacks sufficient details to the data construction, for example, what sources are considered, are they overlapping with the test cases you used? How does this dataset differs from the datasets from previous studies. What is the detailed statistics.

5. One important motivation seems to be the efficiency concern, but throughout this paper I cannot see any experiment justified this.

[1] CodeAgent: Autonomous Communicative Agents for Code Review, EMNLP 2024.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1
