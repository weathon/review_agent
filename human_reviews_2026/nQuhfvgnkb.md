# Generation Space Size:  Understanding and Calibrating Open-Endedness of LLM Generations

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Different open-ended generation tasks require different degrees of output diversity. However, current LLMs are often miscalibrated. They collapse to overly homogeneous outputs for creative tasks and hallucinate diverse but incorrect responses for factual tasks. We argue that these two failure modes are unified by, and can both be addressed by, the notion of effective generation space size (GSS) --- the set of semantically distinct outputs a model considers for a prompt. We present GSSBench, a task suite of prompt pairs with ground-truth GSS relationships to assess different metrics and understand where models diverge from desired behavior. We find that hallucination detection metrics, particularly EigenScore, consistently outperform standard diversity and uncertainty quantification metrics, providing interpretable insights into a model's internal task representations for the open-endedness of different prompts. We demonstrate three applications of GSS: (1) detecting prompt ambiguity and when models ask clarification questions for better grounding, (2) interpreting overthinking and underthinking in reasoning models, and (3) steering models to expand their generation space to yield high-quality and diverse outputs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper “Generation Space Size: Understanding and Calibrating Open-Endedness of LLM Generations” introduces the concept of Effective Generation Space Size (GSS) — the number of semantically distinct outputs a large language model (LLM) considers for a given prompt. The authors argue that many generation failures (e.g., overly homogeneous outputs in creative tasks or excessive hallucination in factual tasks) stem from miscalibration of this generation space: models either over-expand or over-collapse their internal search space.

To study this phenomenon, the paper proposes GSSBench, a benchmark consisting of prompt pairs with known ground-truth GSS relationships. Using this framework, the authors systematically compare various metrics (diversity, uncertainty, and hallucination-based measures) to estimate a model’s GSS. They find that hallucination detection metrics—especially EigenScore—consistently approximate true GSS better than traditional diversity or uncertainty measures and provide interpretable insights into model behavior.

### Strengths
1. Defines Effective Generation Space Size (GSS) as a unifying measure of output open-endedness in LLMs.

2. Methodological: Introduces GSSBench, the first benchmark for evaluating GSS calibration across tasks.

### Weaknesses
While the paper claims that GSS estimation can improve hallucination detection and creative text generation, it does not clearly quantify how much GSS-based calibration outperforms existing baselines. The experiments mainly focus on correlation analyses rather than downstream task gains. Demonstrating concrete improvements (e.g., reduction in hallucination rate or increase in creativity scores) would strengthen the argument that GSS provides practical benefits beyond theoretical interpretability.

### Questions
as weakness

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper considers an intuitive setup to the issue of model output diversity —  the space valid generation candidates (and its size — Generation Space Size, aka GSS). The author proposes GSS mismatch, where the lack of diversity is covering lower GSS than allowed, and hallucination, too high. 
To that end, the paper proposes GSS bench, where prompt pairs with relative GSS relationship known available for comparison, and various assess the best proxy of model GSS coverage.

### Strengths
- Using multiple datasets, this works made a good effort to cover various topics for measuring model GSS coverage.
- Compared many existing signals from lexical level to semantic one as a proxy for GSS coverage, including improved variants. 
- Studied GSS’s correlation with various effects like question ambiguity, reasoning lengthen, solution paths, and value as a loss.

### Weaknesses
Unfortunately, this paper has some fundamental issues
- No explicit mention of any utility evaluation of generated candidates. For all we know, models can be generating non fluent text and still be considered high GSS.
- The paper assumes prompt-pair orderings can predict GSS. However, such automated construction might only reveal surface-form notion of distinct outputs. So the generalizability is of concern - investigations of these multiple metrics might be fitting for assumptions more than actually GSS.
- The paper says it is “currently impossible to access the model’s generation space … unless we sample infinitely many times” but then proceeds to estimate all metrics from K=10 samples. Yet, the paper itself assumes one can work with small-K samples. Hence, the motivation to find proxy metrics seems conflicted. I
- It is not informative to simply assumes both error terms are “small enough” so metric orderings transfer. Nowhere do we see, for at least one dataset/model, the rate of violations of these assumptions, which can be a reliability test for GSS Bench.
- Some claims are post-hoc. To “predict” clarifications or ambiguity or #solution paths, the method has already sampled multiple responses from the model — at that point you can already see whether the model asked for clarification.
- EigenScore, using more samples, is more costly than other metrics. Hence all comparison were made fair.

### Questions
- Please edit “all reported values have ±0.02 margin of error”
- I realized you used Llama 3.1 only from the citation. Please actually say so in the paper instead of just saying "Llama 8B".

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper hypothesizes that the homogenization of outputs in creative tasks and the occurrence of hallucinations in factual tasks both arise from a miscalibration of the Generation Space Size in LMs. Since direct quantification of this space is tricky, the authors propose, GSSBench, to assess how well existing metrics approximate GSS. They identify EigenScore as the most effective proxy and demonstrate some applications of GSS including detecting prompt ambiguity and analyzing divergence from optimal reasoning token budgets.

### Strengths
- The paper has some interesting insights that are corroborated by literature in diverse generations; For instance, the paper demonstrates that larger models show higher miscalibration in GSS from a different angle (something that’s already anecdotally reported in literature already).

### Weaknesses
- The use of Eigen (Avg + Output) for calibrating the GSS for a wider variety of prompts where GSS might be conflated or constrained due to other reasons is unclear: 

I am not entirely convinced about the soundness of the design decision that the output space of X is more constrained than Y (agnostic to the type of constraint that's introduced). For the motivating example -  “Generate an email that contains the word Sam” having a smaller GSS  than “Generate an email” seems plausible but is seems strongly conditioned on the inclusion of a proper noun constraint. But if the same constraint is replaced by ‘climate’ - the generation space may or may not become constrained depending upon the topical associations seen in pretraining/post-training for such a constraint. Empirically speaking - For Figure 2: the small proportion of overlapping buckets between complements and originals is not addressed - perhaps such prompts elicit such overlapping score ranges and why ? 

Figure A2 also shows that Eigen Variant has complete overlap with the original distribution for the Random Choice dataset so it’s further unclear which of the 3 Eigen variants are ultimately useful. 


- GSS reasoning-length correlation is weak at best: Ln 334 (Experiment 2) claims that GSS is predictive of reasoning complexity. Figure 3 demonstrates this weakly. The consistent winner for the previously winning method E-Average shows a maximum correlation of 0.17 and several of the sizes/datasets show negative to no correlation. It also seems a lot of the other methods show even higher correlation with the reasoning length so it's unclear if the takeaway is supposed to be that EigenScore is independently useful for such prediction or the entire class of metrics can mostly do the job. 

- In general, picking a single method among the EigenScore variants is also recommended since keeping a track of which of the variant’s outperforms is cumbersome.

### Questions
- The claim that GSS measures prompt ambiguity is insufficiently validated. The results demonstrate population-level separation but not instance-level classification. Why not report metrics such as F1 or ROC-AUC to quantify per-sample discriminability?
- Ln 161: Missing space.
- Figure 2: Font size is too small to be legible.
- In the LOOE experiment (Section 4.3), could the authors clarify how the threshold parameter p (e.g., p = 0.6 in Table 5) was selected and whether this choice was validated on a held-out set or tuned on the test data? This detail is essential for assessing the robustness of the reported improvements.

### Soundness
2

### Presentation
3

### Contribution
2
