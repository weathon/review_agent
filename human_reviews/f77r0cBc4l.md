# GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3

## Abstract
With the rapid advancement of large language models (LLMs), there is a pressing need for a comprehensive evaluation suite to assess their capabilities and limitations. Existing LLM leaderboards often reference scores reported in other papers without consistent settings and prompts, which may inadvertently encourage cherry-picking favored settings and prompts for better results. In this work, we introduce GPT-Fathom, an open-source and reproducible LLM evaluation suite built on top of OpenAI Evals. We systematically evaluate 10+ leading LLMs as well as OpenAI's legacy models on 20+ curated benchmarks across 7 capability categories, all under aligned settings. Our retrospective study on OpenAI's earlier models offers valuable insights into the evolutionary path from GPT-3 to GPT-4. Currently, the community is eager to know how GPT-3 progressively improves to GPT-4, including technical details like whether adding code data improves LLM's reasoning capability, which aspects of LLM capability can be improved by SFT and RLHF, how much is the alignment tax, etc. Our analysis sheds light on many of these questions, aiming to improve the transparency of advanced LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work comprehensively evaluates 10+ leading LLMs, such as OpenAI's GPT series models, Claude 2, and Llama2, on 20+ curated benchmarks across 7 carefully chosen capability categories, including knowledge, reasoning, comprehension, math, code, multilingual, and safety. The comparison results offer valuable insights into the evolutionary path from GPT-3 series to GPT-3.5 series and GPT-4, and partially answer some important questions that are of curiosity to the community.

### Strengths
**Significance**: This work provides a high-quality and comprehensive benchmark for LLMs research, which may provide a good foundation for LLMs development and comparison. 

**Quality**:  Each dimension of the evaluation benchmark (e.g., metric, model, used prompt, black-box evaluation vs white box evaluation, etc) is carefully chosen. The analysis about the evaluation results is well conducted and deliver some useful information. 


**Clarity**: The paper is well-written, and the structure and figure is very clear.

### Weaknesses
**Originality**: l have seen that the authors clearly compare this work with previous LLMs benchmark work (in the penultimate paragraph in the introduction section), and it appears to be the first benchmark that consistently evaluates so many LLM models across multiple capability dimensions. I am curious to know if there are any novel evaluation dimensions proposed in this work.

### Questions
1. Have the authors open-sourced the benchmark work, and what costs are associated with evaluating a newly trained model? 
2. Are the current capability dimensions sufficient to systematically evaluate current Low-Level Models, or is there any important metric that this work is missing?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new benchmark GPT-Fathom for comparing the performance of closed-source and open-source LLMs. The benchmark is comprehensive and consistently compares different LLMs. The paper has shown the impact of evolution of closed-source LLMs, inclusion of coding datasets during training, as well as alignment methods such as SFT and RLHF.

### Strengths
- Creating a well-designed benchmark for LLMs is an important problem statement
- Consideration of both open source and closed source LLMs in the evaluation
- Focus on reproducibility, consistent settings, and ablation of methods/prompts.
- Extensive experiments and analysis of LLMs
- Explanation of why the black box evaluation was considered for all benchmarks and LLMs.
- Interesting analysis of evolution of OpenAI LLMs.

### Weaknesses
The research questions addressed by the paper is unclear. For a research publication focused on benchmarking, it is insufficient to study a new set of LLMs and explain the results. The paper needs to explain the benchmark design and how it has resulted in a substantial improvement over existing benchmarks. Listing a few points below. 
- The paper claims results are based on "aligned" settings, but still includes numbers from other papers (in brackets) and optimized results (with a star). Instead, it will be useful to compare the numbers in existing papers, and show the impact of the aligned evaluation settings. Did the results change? If so, what was the reason? Such an analysis would confirm their posed hypothesis that aligned evaluation settings lead to more insights than those already published.
- Similarly, it would be good to understand each new feature introduced by GPT-Fathom compared to prior benchmarks, and show why it led to a better evaluation outcome not just for LLMs considered but also for future LLMs that will get evaluated.
- The paper claims that they have picked representative LLMs, benchmarks, and settings. Why are the choices made representative? No explanation have been provided. Without an explanation, the benchmark looks like a collection of benchmarks from other papers, and the benefits of the proposed benchmark is not clear. 
- The paper acknowledges that the categories in the benchmarks are chosen based on their "own interpretation". But no justification is provided to explain why this  interpretation should be adopted by the research community as the best benchmark to use for LLMs. Some analysis on how these benchmarks cover the range of tasks that LLMs are used for will be useful. 
- The paper repeatedly states that it is trying to answer open questions from the community. The open questions that these benchmarks provide answers for is not clearly stated. 
- Prompt sensitivity is an important issue. Two prompts show there is an issue, but it is unclear if LLMs work well for these two prompts that the sensitivity issue is resolved. A better design to evaluate sensitivity with an appropriate metric will be more useful. 
- In consistency in the number of shots across benchmarks and types of ablation does not show "aligned" settings claimed by the paper.

### Questions
Some clarification questions:
- How are you automating the testing of web-version of LLMs? Is that done manually or through some web toolkit?
- I did not understand what is meant by "pass@K". Do you pick the best answer out of K retries? 
- Why does Table 5 and 6 not use zero-shot prompting? 
- Why do different benchmarks use different shots?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper curates a benchmark suite to evaluate the performance of LLMs.

### Strengths
The proposed benchmark covers a range of aspects to study, including knowledge, math, coding, etc. 

It also provides performance and analysis of several popular LLMs on the proposed benchmark.

### Weaknesses
I appreciate the experiments and analysis, but I am mostly concerned with mismatched claims and unclear novelty.

1. mismatched claimed: The paper underscores that the paper sheds light on "the evolutionary path from GPT-3 to GPT-4," several times in the abstract, intro, and conclusion. However, after reading the main text, I could not find enough evidence and/or analysis on the evolutionary path. Figure 1 gives a visualization of OPENAI's announcements of different features/models over time, which the authors defined as evolutionary path. But how is it related to the proposed benchmark?

2. unclear novelty: The proposed benchmark, GPT-Fathom, is effectively a selection/collection of (subsets of) existing benchmark datasets (MMLU, Bigbench, etc). Prompting and evaluation metrics are also quite standard. The analysis seems to resonate many well-known assertions, e.g., proprietary models are more performant. It is unclear to me what new message this paper brings in.

### Questions
Please see my question in the weakness parts.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
