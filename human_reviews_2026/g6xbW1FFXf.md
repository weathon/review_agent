# First Try Matters: Revisiting the Role of Reflection in Reasoning Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Large language models have recently demonstrated significant gains in reasoning ability, often attributed to their capacity to generate longer chains of thought and engage in reflective reasoning. However, the contribution of reflections to performance improvement remains unclear. 
In this paper, we systematically analyze the rollouts of eight reasoning models on five mathematical datasets. We focus on reflective behaviours where the model has already produced an answer but continues reflecting before finalizing its output. Our analysis reveals that reflections are predominantly confirmatory and rarely alter the model's initial answer, a pattern consistent across models and datasets. 
To understand the role of reflections in training, we construct supervised fine-tuning (SFT) datasets with varying amounts of reflection steps. We observe that training models on rollouts with more reflection steps primarily enhances first-answer correctness rather than the ability to correct initially wrong answers through reflections. This motivates us to propose a question-aware early-stopping method that enhances inference-time token efficiency by stopping the reasoning process once a few plausible candidate answers are generated, thereby reducing unnecessary reflection steps. Motivated by this, we further propose to dynamically truncate the reflections after a candidate answer has appeared during generation, which reduces reasoning tokens by 24.5% across five mathematical datasets, within a 2.9% drop in accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a systematic and quantitative investigation into the reflection behavior of reasoning models. By disentangling forward and reflective reasoning through an LLM-based extractor, the authors provide empirical insights into how reflections impact both training and inference. The work is novel and thought-provoking, particularly in questioning the common belief that reflections primarily serve as a self-correction mechanism. However, certain experimental designs and interpretations could benefit from further clarification or control.

### Strengths
1. The paper addresses a novel and underexplored topic — the reflection behavior in reasoning models — and formulates two fundamental questions that are both insightful and significant to the field.  

2. The authors conduct a comprehensive experimental study across a wide range of reasoning models and five mathematical benchmarks of varying difficulty, demonstrating both breadth and depth in evaluation.  

3. The proposed LLM-based method for extracting intermediate answers and identifying reflection segments is well-motivated and technically sound. The validation of extraction accuracy through manual annotations further strengthens the reliability of the analysis.  

4. The paper provides valuable insights into the distinction between confirmatory and corrective reflections and introduces a practical inference-time early-stopping strategy that effectively balances computation cost and performance.

### Weaknesses
1. When investigating the influence of reflection during training, the authors rely solely on supervised fine-tuning (SFT) to control reflective reasoning behavior. However, recent studies have shown that reinforcement learning (RL) can alter reasoning patterns in ways that differ substantially from SFT. It remains unclear whether the findings from the controlled SFT experiments would generalize to RL training. A discussion or empirical verification in this direction would be valuable.  

2. According to Figure 5, models tend to exhibit more reflections on easier problems and fewer on difficult ones, suggesting a misalignment between reflection behavior and task difficulty. It would strengthen the paper if the authors could analyze specific cases or provide plausible explanations for this counterintuitive trend.  

3. The paper reports a negative correlation between reflection amount and problem difficulty. However, in constructing training data for controlled reflection experiments, the authors only select samples with a reflection amount greater than or equal to six. This design choice might introduce a difficulty bias in the training distribution. The paper should clarify whether such bias exists and, if so, describe any measures taken to mitigate its impact.  

4. The conclusion that “training on reflection-rich rollouts yields higher accuracy and longer generations” might be confounded by token-length differences. Longer rollouts naturally contain more tokens, which could encode richer information beyond reflection alone. Thus, the observed performance gains might partially stem from larger token exposure rather than the reflection mechanism itself. It would be helpful if the authors could control for token length — for instance, by comparing responses of similar length but differing reflection amounts — to more rigorously isolate the reflection effect.

### Questions
Please see weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper do a analysis on CoT rollouts and find they mostly confirm the first answer rather than fix errors. Then authors proposed a question-aware early-stopping method that detects candidate answers to truncate reflections, cutting tokens by about 24.5% with about a 2.9% accuracy drop.

### Strengths
1. very interesting findings on the large-scale cot first-try answer correctness
2. the proposed method is straightforward and intuitive

### Weaknesses
1. all analysis and the efficiency method are evaluated only on five mathematical datasets, so generalization to other domains is unclear
2. early stopping cuts tokens but incurs non-trivial accuracy drops on hard sets
3. CAD and QRC are trained on LLM-generated labels and depend on a hand-crafted candidate-extraction prompt, which may be sensitive to formatting, symbols, and units
4. the CAD+QRC early-stopping demos are shown on Qwen3-8B only; portability to other reasoning models is not demonstrated.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper challenges the common assumption that "reflection" in large language models—the reasoning steps that occur after a model has already produced a candidate answer—serves to correct mistakes.
The authors' large-scale analysis reveals two key findings:
1. Inference-Time Reflections are Confirmatory, Not Corrective: When models "reflect," they overwhelmingly just confirm their initial answer, whether it was right or wrong. True self-correction (changing a wrong answer to a right one) is extremely rare (less than 2% of the time).
2. Training-Time Reflections Boost First-Try Accuracy: While reflections are ineffective for correction during inference, training on data that includes long reflections is beneficial. However, the benefit does not come from teaching the model to fix its mistakes. Instead, it significantly improves the model's ability to get the answer right on its very first try.
Based on these findings, the authors propose an adaptive early-stopping method. By cutting off these unnecessary and computationally expensive reflection steps during inference, they can reduce token usage by 24.5% with only a minimal 2.9% drop in accuracy.

### Strengths
1. Significant and Counter-intuitive Research Question:
This paper challenges a widely held assumption in the field of LLM reasoning that "reflection" (the steps after generating an answer) is an active, human-like self-correction mechanism. The authors question this "reflection = correction" intuition, which serves as a novel and high-impact entry point. In a community where "longer chain-of-thought = better reasoning" is a common belief, this paper's focus is especially timely and important.
2. Robust Large-Scale Empirical Analysis:
The paper's core argument is not based on speculation or anecdotes but on solid quantitative data. The authors systematically analyzed 8 different advanced reasoning models (covering both RL and SFT) across 5 math benchmarks. The quantitative findings such as "over 90% of reflections are confirmatory" and "less than 2% are truly corrective" ($F \rightarrow T$)—provide exceptionally strong evidence for the paper's claims, demonstrating that this is a general phenomenon across models and datasets.
3. Clever Experimental Design:
  - "Candidate Answer Extractor": The authors designed an ingenious methodology to clearly partition a model's thought process (rollout) into "forward reasoning" (up to the first candidate answer) and "reflective reasoning" (everything after). This is the key methodological contribution that enables the entire analysis.
  - "Cut-at-i" Supervised Fine-Tuning (SFT) Experiment: This experiment is particularly outstanding. By controlling the total training token budget while varying the ratio of "reflection length" to "number of problems," the authors successfully isolated the true role of reflection data in training. This experiment powerfully demonstrates that the value of reflection data is not in teaching the model "how to correct" but rather in significantly improving its "first-try accuracy" by providing diverse reasoning paths.

### Weaknesses
1. Potential Data Contamination in Evaluation

The paper uses Math500 and other datasets to evaluate reflection behavior, but Math500 (published in 2021) likely overlaps with the training data of the evaluated models. This contamination could artificially inflate first-try accuracy, making reflections appear less useful than they actually are. The analysis would be more convincing if conducted primarily on temporal holdout sets (e.g., AIME2024/2025) or with explicit discussion of data leakage impact.

2. Biased Training Data in SFT Experiments

The SFT experiments (Section 3.1) only use rollouts with correct final answers, which fundamentally biases the training distribution toward first-try success patterns rather than error correction patterns. This design prevents models from learning self-correction behavior and directly causes the observed outcome that "reflections improve first-try accuracy." 
Furthermore, the paper evaluates improvements using absolute values rather than relative growth rates. Calculating from Figure 7, the reflection improvement shows a 33% relative increase (0.9%→1.2%) compared to only 19% for first-try accuracy (18.5%→22.0%), suggesting reflections may be more impactful than claimed.

3. Unverified "Diverse Reasoning Paths" Hypothesis

The paper hypothesizes that reflection-rich rollouts improve generalization by exposing diverse reasoning paths (Section 3.1), but provides no experimental validation. This remains speculation without analyses such as: measuring reasoning path diversity, ablation studies comparing single long rollouts vs. multiple diverse rollouts, or visualization of intermediate reasoning representations.

4. Missing Computational Efficiency Analysis

The proposed early-stopping method claims efficiency gains but only reports token reduction (24.5%). However, the method requires running two additional models (CAD and QRC), with CAD being invoked serially multiple times during generation. The paper provides no measurements of total FLOPs, wall-clock latency, throughput, or memory overhead. Without these comprehensive efficiency benchmarks, it is unclear whether the method achieves genuine computational savings or if the overhead from auxiliary models offsets the token reduction benefits.

### Questions
1. The efficiency analysis relies on token reduction. As token count can be an imperfect proxy for computational cost (e.g., prefill vs. decoding), could the authors provide an analysis in terms of FLOPs to more accurately quantify the computational savings?
2. The paper posits that reflections improve first-try accuracy by exposing the model to 'diverse reasoning paths.' This is a key hypothesis. Could the authors provide more direct evidence to support this claim, perhaps by quantifying this 'diversity' and correlating it with the downstream performance improvements?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the role of reflective reasoning in large language models, showing that most reflection steps reinforce an initial answer rather than correcting it. The authors further demonstrate that training on reflection-rich rollouts primarily improves first-attempt accuracy, and they propose an early-stopping strategy that limits unnecessary reasoning while largely preserving model performance.

### Strengths
1. The paper reveals an interesting finding: training models on rollouts with more reflection steps primarily improves their ability to answer correctly on the first attempt, rather than enhancing their capability to correct mistakes through reflection.

2. The presentation is clear and well-structured.

### Weaknesses
1. It would be better to expand the experiment in Section 3.1 to larger models, such as a 14B version.

2. Regarding the statement *“One possible explanation is that richer reflections expose the model to diverse problem-solving approaches, improving generalization and boosting initial answer quality rather than simply correcting mistakes.”*, it would be valuable to provide further discussion or empirical evidence to support this point. In addition, exploring this phenomenon from an interpretability perspective and examining the actual training dynamics would make the paper more comprehensive and insightful.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
