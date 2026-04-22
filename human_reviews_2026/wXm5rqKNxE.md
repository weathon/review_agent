# Look Before you Leap: Estimating LLM Benchmark Scores from Descriptions

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Progress in large language models is constrained by an evaluation bottleneck: build a benchmark, run models, then iterate. We ask a question: can we forecast outcomes before running any experiments to inform earlier study design? For example, a team building an AI assistant for a certain task can estimate whether expected performance is around 50 or closer to 80, evidence that supports whether to proceed to a pilot study, how to scope it, and how to allocate resources. We study text-only performance forecasting, where a model predicts a score from a redacted task description and intended configuration, with no access to dataset instances. 
To support systematic study, we curate PRECOG, a corpus of redacted description–performance pairs spanning diverse tasks, domains, and metrics. We scrape task and configuration descriptions from arXiv, yielding 2,290 instances covering 1,519 papers, and construct a leakage-free test split using papers published after the evaluated models’ knowledge cutoff. Experiments show the task is challenging but feasible: reasoning models achieve moderate prediction performance with well-calibrated uncertainty, reaching mean absolute error as low as 9.9 at high-confidence thresholds. We further test a zero-leakage setting—forecasting on newly released datasets or experiments before their papers are indexed—where GPT-5 with built-in web search still attains non-trivial prediction accuracy.
Overall, our corpus and analyses offer an initial step toward open-ended anticipatory evaluation, supporting difficulty estimation and smarter experiment prioritization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores the possibiltiy of predicting an LLM's performance on a new benchmark, solely based on a textual description of the evaluation setup. The authors introduce a corresponding corpus, datamined from arxiv. On that corpus, zero-shot GPT-5 clearly outperforms a constant baseline, while Qwen models generally perform close to the constant baseline in terms of MAE. Allowing models to search arxiv papers (excluding the paper containing the prediction target) has inconsistent benefits, with minor improvements for GPT-5, but no clear trend for Qwen models.

### Strengths
- The paper considers an interesting setting, and the results look promising
- The authors seem to have put care into making their crawling responsible and complying with the arxiv access policies

### Weaknesses
- Despite some efforts to combat task leakage, this remains a major concern: If I understand correctly, the corpus includes papers from 2023 to 2024, while GPT5's knowledge cutoff is in June 2024, suggesting that it might have "seen" most of the papers used in the main benchmark during training.
- Reproducibility is generally low. For example, the specific parameters for GPT-5 queries do not seem to be documented. Similarly, it is not completely clear whether Qwen3 thinking was ran with greedy decoding (which is explicitly discouraged in the HF docs and might explain performance issues), or the default parameters. 
- Given the brittleness of LLM benchmark scores, the prediction task seems to be potentially underspecified. For example, the specific way in which classification scores are extracted from LLMs is known to cause large differences in benchmarks scores (https://huggingface.co/blog/open-llm-leaderboard-mmlu), but does not seem to be included in the example dataset items. Similarly, in many cases, it seems plausible that up/downweighing "harder" items within the boundaries of the description could strongly affect benchmark performance. 
   - While it is perfectly fine for a benchmark to be underspecified in this sense, the relatively strong performance despite these issues exacerbates concerns about spurious features and contamination. 
- Additional ablations and experiments would be useful to evaluate the paper's cliams. 
   - To isolate the impact of reasoning, GPT-5 and Qwen3 could be evaluated with and without reasoning, rather than comparing older, non-reasoning models with newer models. 
    - The streaming experiment could be rerun without search enabled (on the same papers)
    - Evidence against contamination could be provided by evaluating models' ability to predict the dataset name from the provided description

### Questions
- "Because source fidelity is critical, any remaining errors are corrected via targeted human edits." Was this done for *all* papers, or a subset?
- "we focused on curating diverse and high- quality examples using the GPT-5 (OpenAI, 2025) API ($200) " Does this simply refer to the process detailed in the paper, or was there additional "curation"?
- For the streaming setup, what steps were taken to ensure papers were processed before being indexed by GPT's web search? 
- Regarding the text-encoder pipeline, what was the value of k for the cross-validation, and how were the baselines tuned? Also, how was E5-Mistral-7B chosen as the encoder?
- Do I understand correctly, that most benchmarks in the dataset only have a single model evaluated on them? 
- Is there a reason to lower-bound MAE at 10 in Figure 3?

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
2

### Summary
This paper studies text-only performance forecasting: the task of estimating a model’s benchmark score solely from a natural-language description of the task and configuration, without running the experiment. The authors introduce PRECOG, a dataset of redacted description–performance pairs built from arXiv papers, and evaluate several LLMs (GPT-5, Qwen3-32B, etc.) as regressors.
They show that only GPT-5 achieves meaningful accuracy, outperforming humans and classical baselines, while open-source models remain close to simple predictors (test-set average, text-encoder based regressors).

### Strengths
- The paper presents an interesting framing of model evaluation as a forecasting problem.
- The PRECOG dataset appears well documented and carefully constructed, with redaction, leakage control, and validation steps.
- The authors develop their own retrieval-based tool that allows models to access relevant literature before making predictions, which both enriches the setup and helps ensure a fairer comparison across models.

### Weaknesses
- While the framing of “evaluation as forecasting” is conceptually interesting, the concrete benefits of text-only performance prediction are not fully demonstrated.  The reported errors (e.g, 14.0 MAE with GPT-5 + Search on a $\[ 0, 100 \]$ scale) suggest forecasts may be too coarse to guide real experimental decisions.  The work would be stronger with a case study illustrating how such forecasts could meaningfully inform benchmark design, compute allocation, or early-stage experiment planning.
- Current non LLM baselines (mean predictor, simple regressors) are rather weak, yet only GPT-5 manages to outperform them. This makes it hard to assess the usefulness of the benchmark itself, since improvements may only reflect pretrained knowledge rather than genuine reasoning ability.
- The human baseline is interesting but too limited to show practical utility. The sample is small, lacks calibration, and participants do not have the same background information as GPT-5. This makes the gap hard to interpret and insufficient to support claims of useful forecasting ability.
- The analysis of retrieval and reasoning patterns (e.g., number of queries, token diversity) remains mostly descriptive. More thorough inspection could reveal whether retrieval genuinely contributes to improved forecasts, what kinds of evidence are most useful, and how reasoning strategies differ between models

### Questions
- How do you envision these forecasts becoming practically useful for researchers? For example, what level of accuracy or reliability would make them meaningful for experiment planning or benchmark design?
- You mention several leakage-control measures in constructing PRECOG, but it remains challenging to determine whether an LLMs’ forecasting accuracy reflects genuine reasoning about task descriptions or residual background knowledge from pretraining. Could you elaborate on how your controls address this issue and whether there is room for stronger evaluation of reasoning beyond memorized benchmark patterns?
- Could you elaborate on how retrieval influences forecasting performance and what evidence supports its contribution to improved results?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework to predict a model’s benchmark performance without explicitly evaluating the model or viewing the model's ability in source papers. They are able to accurately forecast model abilities. They also validate their findings on newly released datasets and experiments.

### Strengths
I like Table 1 with clear differentiation from prior work. I also think the systems seem quite useful for accelerating the development and research of LLMs. I was also impressed by the streaming prediction and human baseline.

### Weaknesses
Redundant related work section. I’d also prefer this to be near the beginning if kept. 

I’d like to see more cost/resource analysis presented. What are the resource requirements in terms of time and money for all the different baselines? How does this compare to running the actual model on the baseline? 

I’d like more discussion of the weaknesses and limitations. Do I never have to run benchmarks again? Where and when should I use this system? On all models for all benchmarks? There is some discussion in the "further analysis across data subsets" part; this should be clarified to see the big picture

### Questions
The highest gains are seen with GPT-5 + search, with reasoning. This is a somewhat expensive model to run. How does this compare with actually running the benchmark for standard benchmarks? 

Is there any heterogeneity predicting the performance across different benchmarks and different models (ie, open vs closed source models, reasoning vs knowledge benchmarks, recent vs old models, small vs large, etc)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates text-only performance forecasting: estimating a model’s score using only the task and configuration description. The authors collect a corpus and run experiments across multiple LLMs. Although I remain skeptical and the results largely support that skepticism, the authors argue that the task is feasible. They present this work as an initial step toward open-ended anticipatory evaluation, while I hope it is the final step.

### Strengths
- The paper studies a bold question.

### Weaknesses
- The task is unrealistic to me. An experiment's score depends on the test data, the model, and the testing configuration. I don't see how one could compress billions of parameters and gigabytes of data into a few sentences and still predict the resulting scores. The paper's results do not support the authors' claim, given a mean absolute error above 14.0%.
-  I see no practical use for so-called open-ended anticipatory evaluation. Most new methods and models aim for small, incremental improvements. An MAE of 14.0% conveys little meaningful information about performance gains or losses. If the forecasted score is 50%, the real score could lie anywhere between 36% and 64%.

### Questions
- Did you try your methodology on your own experiments? What's the estimated performance of all experiments in this paper?

### Soundness
1

### Presentation
2

### Contribution
1
