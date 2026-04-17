# Pitfalls in Evaluating Language Model Forecasters

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4

## Abstract
Large language models (LLMs) have recently been applied to forecasting tasks, with some works claiming these systems match or exceed human performance. In this paper, we argue that, as a community, we should be careful about such conclusions as evaluating LLM forecasters presents unique challenges. We identify two broad categories of issues: (1) difficulty in trusting evaluation results due to many forms of temporal leakage, and (2) difficulty in extrapolating from evaluation performance to real-world forecasting. Through systematic analysis and concrete examples from prior work, we demonstrate how evaluation flaws can raise concerns about current and future performance claims. We argue that more rigorous evaluation methodologies are needed to confidently assess the forecasting abilities of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors discuss many issues with regards to forecasting benchmarks for LLMs. They mainly argue that LLMs can potentially not be trusted for real world forecasting due to these isssues.

### Strengths
- Very good arguments about current issues with using LLMs for forecasting.
- Provides empirical evidence of the discussed issues.
- They propose solutions to the discussed ideas, although only as future work directions to be explored.

### Weaknesses
- While the authors do have evidence for their claims, it is somewhat limited. As the authors mention themselves: "We also do not have proof that the benchmark issueswe uncoverwould lower the performance claims of LLM forecasters". More damning evidence would strengthen the claims.

### Questions
- Page 9 weirdly ends with empty vertical space. Have the authors added a page break or vertical spacing?
- Have the authors changed the default font of the ICLR latex file? This is not something that should be changed.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper argues that evaluating LLMs as forecasters has two primary failure modes. First, backtests are hard to trust: they can leak information via “logical leakage”, unreliable date-restricted retrieval (search engines yield future-informed or misdated pages), and over-reliance on reported model cutoff dates that are only loose guidelines. Second, the paper explains why even “clean” benchmark gains may not translate to real-world forecasting skill. For example, LLMs can ingest or retrieve human crowd forecasts and then be credited with matching humans. 

The paper offers practical guidance for the field moving forward, such as penalizing memorization and being cautious about turning backtests into training objectives. It also suggests that research in the field should constrain retrieval to well-dated corpora, add buffers around model cutoffs, report multiple metrics, and, where possible, include live market evaluations to substantiate claims

### Strengths
I find the paper very well written and the arguments are presented cleanly and convincingly. In particular, the main argument is thoughtful. It systematizes failure modes for evaluating LLM forecasters into two categories—temporal leakage in backtests and misinterpretation of benchmark gains—which is a novel perspective in the literature.

For each of the pitfalls, the paper provides concrete examples. For instance, on logical leakage, the authors show that the very fact that a question is being graded at time T rules out some outcomes (e.g., “Jan 6” queries biased toward post-2021 associations) and argue that many published results likely overstate LLM forecasting ability because models may implicitly access post-T information.

These contributions are, to my knowledge, original. Beyond critiques, the paper offers mitigation guidance.

Finally, the paper offers a comprehensive view of the field and provides a strong critique of the status quo. I believe this kind of critical examination and revisiting is healthy for the area to move forward.

### Weaknesses
While I find the arguments convincing, the paper could be stronger in offering more systematic and empirical studies. For example, regarding logical leakage, I wonder if the impact of that could be precisely measured (in the context of any of the previous benchmarks). Similarly, for retrieval leakage, the paper offers a few examples. However, a broader measurement could strengthen the argument. 

For each pitfall, the paper suggests various potential solutions. It would be nice to implement & test some critical ones and compare them end-to-end to demonstrate improved robustness.

Finally, I wonder if the author(s) have examined https://futurex-ai.github.io/ and have any comments.

### Questions
Two recent benchmarks emerged after ICLR deadline: https://mirai-llm.github.io/ and https://futurex-ai.github.io/. I wonder if the author(s) could offer any comments and whether these new eval candidates suffer any of the pitfalls.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper analyzes and synthesizes the evaluation of LLM forecasters. It identifies two main challenges: (1) various forms of temporal leakage make it difficult to trust evaluation results; and (2) performance in evaluations is hard to extrapolate to real-world forecasting settings, thereby affecting the effectiveness of LLMs as forecasters. Drawing on concrete cases from existing work, it illustrates how these challenges impact the evaluation of LLM forecasters.

### Strengths
1. The paper explores an interesting topic on the evaluation of LLM forecasters. The problem is well scoped and the structure is clear. It groups the challenges into two main categories and then discusses more specific cases under each.

2. The challenges identified by the authors are valid and important, and experiments in the subsequent work [1] further confirm them.

[1]FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction

### Weaknesses
1. Overall, I consider this paper closer to a position paper: its core contribution lies in raising the problem, summarizing the challenges, and offering methodological recommendations, while lacking new methods, benchmarks, or rigorous, controlled experimental evaluation to substantiate the main claims.

2. The paper focuses primarily on backtesting or retrodiction, and the authors contend that “The gold standard for evaluating a forecaster involves running it on unresolved questions, waiting until the questions resolve, and then scoring the predictions” is impractical for rapid model evaluation; however, benchmarks of this type already exist [1].  

[1]FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction

### Questions
Please refer to Weakness.

### Soundness
2

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
This is a meta paper which presents a critical analysis of various evaluation approaches to LLM forecasting. Main theme is that most claims about LLMs outperforming humans on forecasting may have flawed evaluation.

### Strengths
S1: I think this paper is timely is quite relevant given the surge of papers in forecasting.This is an essential paper for the community. The primary focus of the paper is to bring forth the flaws and issues in evaluation and benchmarks.

S2: Paper demonstrates various issues with prior work’s evaluations such as model cut-off date, bias in retrieval, leakag. The paper reads well. And is structured for ease of understanding.

### Weaknesses
W1. My main concern is that this is a meta analysis paper, where main contribution is the analysis and synthesizing prior work from the lens of evaluation. It is not a new artifact in a traditional sense like algo, data, methods etc.

W2. Some claims are supported through prompts/evidence, while others, like LLMs gaming the benchmarks, etc, are extrapolated/opinionated about.

### Questions
NA

### Soundness
3

### Presentation
4

### Contribution
2
