# Persuade with Reason: Enhancing Debate Persuasiveness through Accurate Persuasion Feedback Derived from Weak Supervised Labels

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2, 2

## Abstract
Existing methods for debate generation often struggle to provide convincing proof, lacking critical persuasiveness. More challengingly, directly fine-tuning or using RLHF on large language models (LLMs) can decrease the persuasiveness of the generated text, making it difficult to leverage advancements from state-of-the-art LLMs. We identify two key biases underlying this issue: reward hacking and reward sparsity. Reward hacking blurs the model's training objectives, causing the model to focus more on linguistic style and rhetoric while neglecting the essential logical reasoning and value shaping. Reward sparsity reduces the generalization and robustness of the reward model. To address these two problems, we propose a novel persuasiveness enhancement training method: $\rm P^{3}$. Firstly, we introduce \underline{\textbf{P}}ersuasive reward estimation and modeling by separating persuasiveness scores from surface cues, addressing the reward hacking problem. Secondly, we solve the reward sparsity issue by employing \underline{\textbf{P}}ersuasive sample mining to extract persuasive annotation information from weakly supervised labels. Lastly, we design a new DPO algorithm tailored for \underline{\textbf{P}}ersuasiveness generation optimization, which modifying the objective function to mitigate the divergence problem on debate generation task. Extensive experimental results demonstrate that $\rm P^{3}$ effectively alleviates the aforementioned issues, significantly enhancing the model's performance in debate and persuasion tasks, surpassing state-of-the-art closed-source commercial models, such as Gemini and Claude, in both automatic and human evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the lack of persuasiveness in the debate generation, especially after fine—tuning and RLHF. The main reason for that is reward hacking and reward sparsity. The authors propose P^3, a three-stage pipeline to improve LLM persuasion in debate-style generation: first, separate persuasiveness from surface/literal cues via an EM + Bradley–Terry inspired estimator, then mine persuasive samples from weak supervision (upvote−downvote scores) and filter by the estimated persuasiveness score, and finally, optimize with a modified DPO called PAPO to avoid DPO’s small-sample divergence. Experiments on ChangeMyView (CMV) show improved automatic (GPT-4 o1) and human evaluations; authors claim their 13B model outperforms much larger closed-source models.

### Strengths
- The paper is well-structured and easy to follow
- The paper includes both automatic and human evaluations, and shows the correlation between those metrics

### Weaknesses
- Lack of a more comprehensive analysis of the generated output. There is only one example described in the case study (section 4.3).
- Unnecessary and redundant equations in section 2 that could have gone to the appendix and free up some space for the analysis of the results (the point above)

### Questions
* Why are the human evaluation scores and the o1-score on different scales? Wouldn't it be easier to report the automatic o1-score on the same scale as the human evaluations to facilitate direct comparison?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PAPO, a persuasion-aware preference optimization framework. Using the CMV dataset, the authors disentangle persuasiveness from style by modeling two latent components and optimizing through a score-weighted objective that addresses divergence issues in standard DPO. The model is evaluated both automatically and with limited human assessments, showing improved alignment with human judgments compared to existing preference optimization methods.

### Strengths
The study addresses a gap between language fluency and genuine persuasiveness in LLM outputs. Its attempt to separate logical argument quality is conceptually novel. The integration of theoretical justification and practical large-scale experiments make the paper methodologically grounded.

### Weaknesses
Comment 1. The authors fit two MLPs (sd for persuasion, ss for surface) and learns a Bernoulli-mixture with EM. It’s unclear what sd truly captures. Does sd represent causal persuasiveness (logic, evidence) or merely represent latent artifacts?
Using a small subsample, the authors might test the following.
(i) hold content constant while changing style; (ii) hold style constant while changing content. Check that sd is stable in (i) and changes in (ii); ss should do the opposite.

Comment 2. The Reddit-based supervision signal (upvotes minus downvotes), as the authors note, can conflate persuasiveness with unrelated noise, such as popularity and timing effects. This weak supervision may cause the model to reward factors other than genuine argumentative strength. This is particularly important as the entire algorithm hinges on this “weak” supervision.
The authors can probably include fixed effects for subreddit, posting time, and author karma, and then re-estimate the learned persuasion scores to test robustness.
Furthermore, the authors can use CMV “delta” awards as clean persuasion labels, train on that subset, and compare whether PAPO still outperforms the traditional DPO in that subset.

Comment 3. I am not particularly convinced by the reliability of o1 scores. The authors’ argument is based on limited validation (only 100 human-rated samples).
Potentially, the authors can consider increasing the human sampling stratified by topic, stance, and argument length. Importantly, it will be interesting to re-prompt the evaluator with alternative rubrics (logic-only vs. style-only) and show that model rankings remain stable.

Comment 4. The CMV dataset is known to have topic imbalances. To ensure that the authors’ results can generalize across different topics, the authors can split CMV data by topic clusters and show whether improvements hold uniformly or only in high-frequency topics. It will be worthwhile to investigate under which conditions (by topic or by any other systematic characteristics) the algorithm performs relatively better or worse.

Comment 5. The title of paper is currently missing in your submission.

### Questions
See the above "Weaknesses."

### Soundness
2

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
2

### Summary
This paper tackles the challenge of generating persuasive debate text with large language models. The authors identify two major issues that cause existing methods, especially supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF), to fail in producing truly persuasive arguments. To address these, the authors propose P3, a three-stage training framework.

### Strengths
Clear Problem Identification: The paper convincingly argues that traditional RLHF pipelines are poorly aligned for debate/persuasion tasks due to reward misalignment (hacking) and data sparsity.

Well-Motivated Framework: The P3 pipeline logically decomposes the overall goal into reward estimation, data selection, and strategy optimization. Each component addresses a specific, well-defined issue.

### Weaknesses
1. Limited Dataset Scope: All experiments are restricted to the CMV dataset. While appropriate for persuasion, results on other argumentative or dialogue datasets (e.g., ConvAI, PersuasionForGood) would demonstrate generalization.

2. Using GPT-4-o1 as the primary automatic evaluator introduces circular dependence, since GPT-4’s reward alignment may favor stylistic fluency over genuine logical rigor. The correlation coefficient (0.67) is reasonable but not perfect; more robust human-only evaluations would strengthen the conclusion.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper focus on persuasion generation, creating text that convinces a specific audience. The authors attribute the limitation of current persuasion generation to 2 main reasons: Reward hacking (objective design) and Reward sparsity (training data).
To tackle the deconstructed issues, the enhance pipeline, $P^3$, is proposed. The pipeline scores generations along literal and persuasiveness dimensions and trains using weakly supervised labels. Experiments on CMV demonstrate its effectiveness in improving persuasive quality.

### Strengths
1. Problem decomposition: deconstructing the weaknesses of persuasion generation along two axes, objective design and data scarcity, to diagnose failures and guide targeted remedies.
2. Fine-grained scoring: decoupling the reward into literal fidelity and persuasiveness, enabling fine-grained optimization of persuasive quality.
3. Data construction pipeline: propose a scalable data-construction pipeline that alleviates data scarcity and provides a practical recipe for weak/cheap supervision.

### Weaknesses
1. Fig. 1 inconsistency (objective vs. paradigm).
The paper attributes failures to objective design, but Fig. 1 varies training paradigms (SFT vs. RL) rather than the objective itself. As a result, the figure does not isolate objective mis-specification. 

2. Limited evaluation scope (single dataset).
The primary evaluation is on CMV; broader experiments on additional persuasion/debate datasets would strengthen external validity and generality of the conclusions.

3. Baseline parity (backbone vs. fine-tuned).
In Tab. 1 and Fig. 4, comparisons appear to pit the proposed fine-tuned system against general-purpose backbones. For a fair test of $P^3$’s contribution, including equally fine-tuned baselines can consolidate the final conclusion

4. Strong model comparison.
Given the reported high correlation between o1-score and human annotations, GPT-4o seems to be a strong baseline models which is missing in the performance comparison.

5. Missing specific metric was used for human evaluation.
The paper lacks a clear description or example of the human scoring process, making the evaluation criteria ambiguous. Providing concrete examples or a defined scoring rubric would clarify how the human judgments were obtained and interpreted.

6. The title in the manuscript is inconsistent with the title provided in the submission page.

### Questions
1. In Fig.2, more detailed statement is required. Why do such scatter plots demonstrate the deviation?
2. In formula 3, what is the $x$ in $f_s(x)$ standing for
3. same one in formula 5
4. in Fig.3, given that the o1-score shows the highest correlation with human judgments, would a simple pipeline that uses GPT-4o1 to automatically annotate data and then trains SFT/RL models without any task-specific tailoring yield promising gains on persuasion generation?
5. For automatic evaluations, what are the prompt templates (scoring rubric, temperature, top-p, max tokens).

### Soundness
2

### Presentation
2

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
The paper presents a persuasiveness-enhanced preference optimization framework that extends DPO to model persuasive reasoning under weak supervision. It leverages social feedback signals to learn from large-scale debate data and employs a dual-reward mechanism to separate genuine persuasion from superficial style. Experiments on persuasion and debate benchmarks show consistent gains in both automatic and human evaluations. Overall, the approach provides a simple yet effective way to align language models toward persuasive and content-driven generation under weakly supervised settings.

### Strengths
1.The paper introduces a novel persuasiveness-aware extension of DPO that separates logical persuasion from surface-level style through dual reward modeling.
2. The paper effectively leverages large-scale social feedback as weak supervision, demonstrating how noisy real-world data can be adapted for persuasive language modeling.

### Weaknesses
1. The paper’s exposition is often unclear and difficult to understand.
2. The experiments compare P³ mainly with general instruction-tuned LLMs (e.g., LLaMA, Claude, Gemini), but omit specialized baselines that also target persuasion or argumentation, such as: SFT-based methods, DPO or similar methods are focusing on reward hacking and reward sparsity.
3. The paper reports only the O₁-score (GPT-4 evaluation) and limited human judgments. Even though some previous metrics have limitation, it can reflect something by using Bert score, Rouge/BLEU. Such as semantic alignment and style or phrasing pattern.
4. The paper lacks both statistical and qualitative analyses of generated outputs. There is no examination of response length, lexical diversity, or argument structure, nor any examples of failure cases. Without such analysis, it is unclear why P³ performs better—whether due to improved reasoning or merely longer, stylistically refined responses.
5. No enough ablation for hyper parameter. For example, α in expression (3) controlling the trade-off between persuasiveness and surface cues, yet no ablation or sensitivity study is provided to justify its selection or stability.
6. Lack of clear definition of persuasiveness: The paper does not explicitly define what builds persuasiveness (e.g., length, strength of style, number of supporting arguments), making it unclear what aspects the model actually learns or optimizes.

### Questions
1. Clarification on Figure 2
The paper would benefit from a clearer explanation of Figure 2. Specifically, please specify what the x- and y-axes represent in the scatter plot, and clarify the interpretation of the plotted metrics. It would also help to explicitly state whether higher values indicate better performance for each metric .

2. Bias Mitigation
The paper briefly mentions several sources of bias but does not provide sufficient methodological detail on how they are addressed.
(i) Data bias (Reddit): Please elaborate on how cultural and popularity biases are mitigated when training the MLPs, given that popularity signals (upvotes, downvotes) are not equivalent to genuine persuasiveness.
(ii) Accumulated bias: The training pipeline may compound multiple biases — Reddit community bias → flawed supervision signals → MLP learns spurious correlations → biased reward f_d → misguided PAPO optimization.  It would strengthen the work to clarify what specific mechanisms are applied to break or mitigate this propagation.
(iii) Evaluation bias: Since o1_score relies on GPT-based judgments, can you do some statistic analysis to show o1_score is diverse and would not give high score for some specific topic? What's could you please explain how to reduce the bias in the human evaluator during your experiments.

### Soundness
2

### Presentation
1

### Contribution
2
