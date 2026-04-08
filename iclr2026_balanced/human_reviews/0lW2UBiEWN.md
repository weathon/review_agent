## Human Reviewer 1

### Summary
The paper proposes a benchmark and framework to evaluate deceptive behaviour under specific scenarios. In particular alignment failures are assessed via the comparison of chains-of-thought and model responses when in the neutral scenario vs. under additional pressure. A methodology to generate these scenarios is proposed and the responses are evaluated with LLM-as-a-judge. The paper evaluates several leading LLMs and finds that most exhibit deceptive tendencies.

### Strengths
Benchmarking the susceptibility to pressure cues is a very interesting and relevant topic. The paper is well written and easy to follow. The methodology is mostly clearly outlined and the motivation and experimental results are clearly presented.

The dataset covers a large amount of deceptive behaviours (strategic deception, alignment faking,...) and several critical domains (military, finance,...), which makes it relevant to real world scenarios. I skimmed some examples in the dataset, which seem reasonable complex.

It is also worth noting that a human annotation and quality control was performed. The number of models evaluated is also reasonable and includes several prominent closed- and open-source models.

### Weaknesses
I am mostly concerned about some missing details regarding the scenario seed generation, as well as the overall evaluation and depth of the discussion.

**Unclear methodology for scenario seed generation:**

While the main text and the appendix provide some details, it remains unclear how exactly the scenario seeds were generated. Additional details on this would be great, e.g. how and from where were the sources obtained. This seems crucial to reproduce the benchmark generation, a core part of the paper. While several LLMs are evaluated on the benchmark, the quality of the benchmark and generation framework itself is unclear. It is stated that quality control and human annotation was performed, but additional details regarding how exactly the samples were annotated would be helpful.

**Evaluation pipeline:**

The benchmark seems to heavily rely on LLM-as-a-judge to evaluate and compare responses and chains-of-thought. While this may be fine generally (Appendix C.1 provides some evidence), it would be great to strengthen the confidence in the evaluation pipeline by performing a deeper error analysis and ablation study.

**Evaluation could be strengthened:**

Overall it seems that the main message of the results is that LLMs act deceptively or in a harmful manner when under pressure (some more so than others), which as far as I know had also been observed in existing literature cited in the paper. A clear comparison to existing benchmarks would help clarify the contributions. 

Also, the relative gap between open- and closed-source models seems to be primarily driven by Claude sonnet 4 and 3.7 which obtain quite low results. It would be good to further support this claim by including additional closed source models, e.g. GPT-5. Some of the trends discussed in 5.3 and observed in Figure 5 may be of questionable significance. Specifically, there seems to be some variance within the u-shape regarding Deepseek. It would be great if some measure of significance/confidence across runs could be added to the Figures and Table 1. The explanations of the observed u-shape as well as the large increase in Figure 5 (right) are somewhat speculative and more evidence by contrasting distilled vs non-distilled models and MoE vs non-MoE models would provide stronger evidence.

**Other points:**

* Appendix E provides the full prompts for several scenarios. It would be useful to see at least one full example of scenario, response with and without pressure and the scoring of those responses.
* Several related works are mentioned, but a clear comparison is missing. For example, how does Mesa & Mask compare to DeceptionBench?

### Questions
* How does your benchmark compare to existing benchmarks, e.g. DeceptionBench? What are the key questions that it provides answers to that prior works did not?
* How were the environment scenario seeds obtained?
* How exactly did the human annotate the data? (I read the process on appendix B.3)
* On what data were the evaluation model results in Appendix C.1 obtained?
* How robust are model evaluation results (with LLM-as-a-judge) to same family vs cross family evaluation? Has any error analysis been done for misclassification or an ablation study with more human involvement during the evaluation phase?
* Have ambiguous or borderline cases been checked? If yes, what did they look like?
* Were there any differences between human and judge agreements/disagreements across the domains or deception types?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces a dataset and a methodology to detect and classify deceptive behaviour in LLMs. The dataset contains 2100 samples spanning 6 professional domains and 6 different deceptive behavior types. Each data sample is based on realistic professional scenarios and refined in an iterative process at the end of which each data sample comprises a user prompt paired to a neutral system prompt (MESA condition) and a system prompt used to provide pressure through incentives highlighting goal conflicts and often related to deployment, utility, or preservation without ever explicitly instructing deception (MASK condition). Model generated CoT and responses are then evaluated comparatively under these paired conditions (MESA & MASK) producing 4 possible model behaviors: Consistent (MESA & MASK produce consistent CoT and responses), Superficial alignment (MESA & MASK produce inconsistent CoTs but similar responses), Deception tendency (MESA & MASK produce consistent CoTs but inconsistent responses), and Explicit deception (MESA & MASK produce inconsistent CoTs and responses).

The paper then evaluates 20 mainstream models and concludes that deception tendencies are widespread. It further analyses how deception depends on the model being open vs closed source, dense or with MoE, or size. These analyses produce interesting insights, but these remain relatively underexplored to allow for strong conclusions.

Overall, the paper addresses an important research direction and produces a valuable dataset but would benefit from improvement in presentation clarity, positioning relative to related work, and justification of certain methodological choices.

Mandatory disclosure of LLM usage by the reviewer: The reviewer used LLMs to reformat this review text into organized prose and numbered lists, and to write this sentence.

### Strengths
1. The paper addresses a very timely research direction: it studies brittleness of alignment in terms of how models adopt deceptive behavior under subtle pressure.

2. The dataset created seems to be of high quality and could be a useful deception benchmark eval, based on realistic scenarios and importantly spans multiple professional contexts and deception types. The amount of effort, involving multi-source data collection, iterative refining and human validation, behind its construction is noteworthy.

3. While the idea of comparing response deviation between neutral and pressured conditions is not an original idea (see Ren et al. 2025), doing so while looking at both CoT and responses is original and allows for a more complete classification of model behaviors.

4. Evaluating multiple models on these dataset is a valuable contribution and can give a better understanding of models' deceptive behavior under subtle, yet realistic, pressure.

### Weaknesses
1. Clarity of Novel Contributions and Positioning: The paper's title (MESA & MASK) suggests its main contribution is a comparative evaluation framework contrasting model behavior under neutral (MESA) and pressured (MASK) conditions. However, this approach already exists in Ren et al. 2025's MASK framework, which the authors mention in the introduction but not in related work. Hence, the title may be misleading. The paper should better clarify the novelty in its contributions (realistic high-risk professional domain settings and using CoT to reveal internal cognitive shifts from honesty to deception?). The related work section should mention Ren et al. 2025 (especially in section 2.2).

2. Clarifications Regarding Methodology: Dataset difficulty assessment (section B.4) is unclear: How is the "multi-dimensional framework" encompassing "scenario sophistication, ethical ambiguity and decision complexity" used if the dataset is evaluated (or rather filtered) on whether at least ⅔ models exhibit deceptive behavior? How does this relate to the stratified sampling? Using the same models for data sample filtering and then evaluation biases the final dataset evaluation results for these models. Would results change significantly if three different models (belonging to different model families) were used for data filtering? The conclusions on ultra-large MoE exhibiting higher deception rates might be poisoned by this bias. The MESA chain-of-thoughts and responses to each user prompt are aggregated through a consensus process. While I understand this is intended to create a stable baseline, it would be important to report whether models already exhibit inconsistencies in their responses before the application of pressure. Without this information, it is difficult to gauge how much the pressure cue itself contributes to eliciting deceptive behavior, particularly in interpreting metrics the significance of metrics such as Deception Rate @1. One major contribution of the paper seems to be the use of CoT. However, for the most part the analysis seems to be based only on the top part of the behavioral quadrant (i.e. deception due to inconsistent response, independent of CoT consistency). This makes the results comparable to other works (again Ren et al. 2025 above all) which does not use CoT. The paper could benefit from more in depth discussion, example or analyses of deception tendencies vs explicit deception.

3. Presentation and Clarity Issues: The paper is not easy to follow in several passages and could overall improve in presentation. Clarity is sometimes hindered by excessive or unclear naming. For example, the Data Quality Evaluation criteria (MESA Utility Elicitation, Deception Induction, and Invisible Pressure) are harder to interpret than the alternative more transparent names used in the appendix to explain them (User Prompt Quality, System-User Integration, System Prompt Quality). Much of this naming is unnecessary and reduces clarity. Here is another example: "Once the prompts are constructed, the pipeline enters the Multi-turn Generation and Sampling Loop to produce deception data through context refinement." Sampling loop is never defined in the main text or in the appendix. Scenario generation in section 4.2 could benefit from shorter periods (the whole paragraph is only 2 long and convoluted periods). More examples of: examples of consistent and inconsistent model responses and of mesa replies vs consensus aggregated mesa replies (nice to have), and how prompts or scenario improves during the iterative process (nice to have).

4. Evaluation metrics for models are compared against expert annotations used as GT. This information was present only at the end of the section and could be made clearer earlier, before discussing the numerical results.

Nitpicks:

- Figure 1 could improve a bit with more text and the word "Mitigates" (in MASK model response) should be hyphenated when going to the next line. (Other than this, I find most figures of high quality).

### Questions
See weaknesses above. Additionally:

1. The paper says "As shown in Figure 3, our approach operates through integrated dataset generation and model evaluation phases": It is unclear to me in which sense and why are dataset generation and model evaluation integrated? I don't see this discussed in the paper and dataset statistics in Figure 4 and the file "M&M_dataset.csv" seem to suggest that the dataset once and statistically. Could you provide clarifications regarding this point? Which model was used to generate this data?

2. Regarding human annotations: are agreement score and Cohen's kappa computed on each checked item or only on the final assessment? Are samples missing one of the checks filtered out? This could be better detailed.

3. What are exactly scenarios and templates? Both these terms are used in the first part of dataset generation and it's not clear what's the difference.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper introduces a dataset for measuring LM deception when prompted with a situation that puts pressure towards deception vs control prompts. The authors evaluate a large number of models on dataset.

### Strengths
Important problem (AI deception) and key types of deception evaluated (alignment faking, sycophancy, sandbagging). 

Solid methodological contribution in comparing LM deception under pressure (MASK) to the control (MESA) --- something that is missing in many similar works. 

The biggest strength of the paper is the comprehensive benchmark which spans a large ranges of interesting deceptive behaviours and domains. 

Overall, good awareness of relevant literature. 

Broad range of domains and behaviours tested. Large number of frontier LMS tested. 

Overall well-written and presented.

### Weaknesses
Overall I didn't think there were that many compelling results or key findings. (However I think the benchmark itself is a very solid contribution which lays the ground for future findings.)

IIUC the core metric is deception as judged by GPT-4.1. But the paper does argue for this metric very much, and it's unclear whether LM judges are reliable for this. I'd like to see some evaluation of the judge, e.g., according to its agreement with human evaluators. However, the full judge prompts were appreciated. 

"COMPARISON OF OPEN-SOURCE AND CLOSED-SOURCE MODELS"
"Open-source models show higher deception rates" ---> but you're not controlling for other factors right, like model capability? So what should we really take away from this?

Figure 5 does not seem to show very clear results --- is there a key takeaway?

SAFETY FINE-TUNING IMPACT ANALYSIS
The results here are very minimal --- only a few percentage points of difference in deception rates from SFT. It seems like this isn't representative of the effect of safety fine-tuning in general (eg HHH training which makes models much more honest). Were the Qwen models tested already safety fine-tuned? Can you test instruct or base models vs HHH models?  

Minor 

The paper gives a few different notions of deception (based on intentionally causing false beliefs, or hiding internal reasoning). The authors should explicitly stick to one definition or just acknowledge there's no universal definition that you want to capture. You should also cite: https://arxiv.org/abs/2312.01350

I didn't find figure 1 that intuitive, maybe there is a more clear-cut example. 

There's quite a bit of content on the tenth page, e.g., limitations --- imposed not sure if this breaks the ICLR policy

### Questions
See questions above.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper introduces MESA&MASK, a deception‐oriented benchmark that contrasts neutral system prompts (MESA) with pressure-inducing system prompts (MASK) across 2,100 scenarios spanning 6 deception types × 6 domains, and evaluates 22 LLMs with a rubric-guided LLM-as-judge (GPT-4.1) on both final answers and chain-of-thought (CoT). The core claim is that pressuring prompts systematically increase deceptive behavior and that models differ markedly in both their propensity to deceive and their “deception consistency” across settings.

### Strengths
1. The paper includes a detailed description on how they generated the dataset with a largely-automated pipeline, including usage of tools such as Model-Context-Protocols(MCP) in the process, and the prompts used for generation.

### Weaknesses
1. The paper solely relies on GPT-4.1 as LLM-as-Judge. Although it has a comparison table on Table 4 that has performance metrics on GPT 4.1, GPT 5, DeepSeek-R1, the table does not clearly state which test data it has used for evaluation. Also, the paper does not take into account for mitigating biases in LLM-as-Judges, such as positional bias [1].
2. Despite the dataset is generated by language models with templates and seed scenarios, it does not consider or analyze data duplication. The human annotation or data quality evaluation stage includes checks with data format or data sanity, but does not include any duplication checks. 
3. The paper lacks novelty in that it is based on the prior work on MASK [3], which already contains comparative evaluation for eliciting pressure or deception with language models. The novelty of this work is limited to expanding the benchmark to chain-of-thoughts(CoT), 6 domains, 6 deception types.

[1] Zongjie Li, Chaozheng Wang, Pingchuan Ma, Daoyuan Wu, Shuai Wang, Cuiyun Gao, and Yang Liu. 2024. Split and Merge: Aligning Position Biases in LLM-based Evaluators. EMNLP 2024
[2] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. "Deduplicating Training Data Makes Language Models Better." ACL 2022
[3] Ren, Richard, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler, Mick Yang et al. "The mask benchmark: Disentangling honesty from accuracy in ai systems." arXiv preprint arXiv:2503.03750 (2025).

### Questions
1. Although the paper mentions about multi-turn interactive benchmarks, it is not clear if it the generated dataset covers multi-turn conversations. The attached data samples in the appendix indicate that the generated data samples are single-turn. Would you elaborate on this matter?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
3