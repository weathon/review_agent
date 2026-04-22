# Chain-of-Thought Degrades Abstention in Large Language Models, Unless Inverted

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
For Large Language Models (LLMs) to be reliably deployed, models must effectively know when not to answer: *abstain*. Chain-of-Thought (CoT) prompting has been gained popularity for improving model performance by ensuring structured outputs that follow a logical sequence. In this paper, we first investigate how current abstention methods perform with CoT outputs, finding that direct use of reasoning traces can degrade performance of existing abstention methods by more than 5%. As a result, we introduce a new framework for thinking about hallucinations in LLMs not as answering a question incorrectly but instead as LLMs answering the *wrong* question. Based on this framework, we develop a new class of state-of-the-art abstention methods called **Trace Inversion**. First, we generate the reasoning trace of a model. Based on only the trace, we then reconstruct the most likely query that the model responded to. Finally, we compare the initial query with the reconstructed query. Low similarity score between the initial query and reconstructed query suggests that the model likely answered the question incorrectly and is flagged to abstain. We perform extensive experiments to find impressive performance gains with our Trace Inversion methods. The code is publicly available at: https://anonymous.4open.science/r/trace-inversion-9EE0/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores the performance of abstention methods when including Chain-of-Thought (CoT) traces. The authors evaluate several baseline abstention methods on 8 datasets and using 5 open-weights models, and find that including CoT traces generally worsens abstention. They then introduce Trace Inversion, a new abstention method that regenerates a prompt given the current reasoning trace, and then uses distance between the original prompt and the regenerated prompt as an abstention metric, where high distances should be abstains. The authors test three different ways of measuring prompt distance, and find promising results for their approach overall.

### Strengths
* The paper's empirical observation that passing CoT reasoning traces into abstention methods worsens their performance is important.
* The idea behind "Trace Inversion", i.e., to regenerate the query from the sampled reasoning trace and compute its distance from the original query as an abstention signal, is novel, and the results seem promising overall.
* The authors select a wide variety of baseline methods, and evaluate on 8 datasets and with 5 open-weights models.

### Weaknesses
* The authors present very large results tables and only limited aggregated statistics, which I fear may obfuscate the scale of the problem. Table 1, for example, has no mean accuracy over models or methods, while Table 2 has the mean per method. Particularly given the variability of the results (i.e., there's no clear winner), finding a more appropriate way to present the high-level results---either as a table of aggregated results, or as a figure---would make this paper much stronger.
* The authors test three different implementations of Trace Inversion using different distance measures. However, there is no clear winner as to which distance measure works best. In Figure 4, for example, the best-performing method for GPT-OSS on Misconceptions is the worst for DeepSeek-Distill-Qwen on MMLU. How should someone who wants to apply Trace Inversion to a model decide which to use?
* "Reliable Accuracy" seems improperly specified, and would reward over-abstention (a trivial example would be abstaining for all but one answerable questions). A more appropriate choice, given the threshold-based formulation in section 3.1, would be something like an Accuracy-Rejection Curve.
* No large-scale, API models are evaluated. While this is perhaps understandable given cost constraints, several of the abstention methods provided could be used on closed models. This makes it hard to understand whether the results hold on large-scale frontier models, and should at least be discussed as a limitation.

### Questions
1. Kirichenko et al. [1] have previously reported that including CoT traces can inflate the performance of an LLM-as-judge abstention detector. How do you reconcile this result with your finding that including the trace worsens abstention, in particular for the (relatively similar) REFLECT method?
2. How is the correctness of final answers verified for the non-multiple-choice benchmarks? Exact match or some other approach?
3. A threshold parameter is introduced on line 179. How is this applied for methods that don't emit a probability, such as REFLECT?
4. On the datasets which only contain answerable questions, the authors state that "the model is expected to abstain when it does not have the knowledge to answer" (line 259). How is this determined, and when should the models actually abstain for these 6 datasets?


References

[1] https://arxiv.org/abs/2506.09038

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a new method called Trace Inversion to determine whether a large language model (LLM) answered the intended question in order to abstain from providing a response. The authors also benchmark several baseline abstention methods including confidence-based and answer reviewing approaches, showing including reasoning traces in the model responses degrades abstention. The authors benchmark eight datasets including commonly used benchmarks such as MMLU across several open models including DeepSeek R1, Qwen, and GPT OSS.

### Strengths
The authors explore the important problem of abstention, particularly for models that produce chain-of-thought reasoning traces. The authors include a reasonble set of baseline methods includign confidence based as well as answer-reviewing methods. The authors confirm prior findings that reasoning degrades abstention. 

The proposed method, Trace Inversion, is clear and well presented. The method covers the issue of models responding to the wrong query using self-review method and response similarity. The authors show Trace Inversion, depending on the choice of similarity between the original and reconstructed query, can outperform existing methods

I appreciate the authors’ inclusion of the code, integration with VLLM, and straightforward README.

### Weaknesses
# Scope of claim is far out of what is reasonably supported by experiments

- The author's frame abstention as "“Hallucinations are the result of models answering a different question that the intended one.” While this is part of abstention, models can also hallucinate context or facts while answering the intended query. The authors inappropriately recast all of abstention under this narrow umbrella. 

# Experimental Setup

- Table 1 missing baseline abstention without abstention methods. This is key to verify that these method do in fact boost abstention and that they provide reasonable baselines against which to compare Trace Inversion.
- Choice of datasets is suspect? Table 3 shows only 2 out of the 8 datasets used in evaluation have unanswerable questions, which is not the best setting for evaluating abstention (the ability to recognize unanswerable questions). 

# Trace Inversion's gains are not systematic or robust

- Claim on line 453 “Trace Inversion provides a systematic and robust enhancement to abstention strategies” is not reflected in experimental results. The best Trace Inversion method varies by model and by benchmark—and trace inversion is in fact not always the best depending on the similarity metric compared to baselines as shown in table 1. In some cases inversion methods are worse than uncertainty or answer reviewing baselines. For example, Figure 4: Trace Inversion Ground is tied for best for GPT-OSS but is worse than baselines for DeepSeek R1

# Computational cost 

- The method is quite costly as we have to run inference then process all the chain fo thought again to regenerate the question adn compare the similar fo the generated and original query.  This isn't discussed or measured in the paper.

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates how Chain-of-Thought (CoT) reasoning affects LLM abstention and proposes "Trace Inversion", a method that reconstructs the query from its reasoning trace, then flags abstention when this differs from the original query. They evaluate across 8 datasets and 5 models, showing improvements in 28/40 settings.

### Strengths
1. Strong Empirical Results: Consistent improvements across diverse models and domains, with significant gain in reliable accuracy across 40 evaluation settings.
2. Comprehensive Evaluation: Thorough testing on multiple model families (7B to 32B parameters), diverse domains (math, reading comprehension, bias detection), and various abstention baselines.

### Weaknesses
1. The observation that CoT harms abstention is intuitive and not novel. Prior work cited and not cited already demonstrated that reasoning models degrade in abstention ability significantly.
2. The paper's two parts (CoT degrades abstention and Trace Inversion improves abstention) lack logical connection. The authors appear to have concatenated two separate observations without establishing why observing CoT degradation motivates the specific design of Trace Inversion. The method would work regardless of whether CoT helps or hurts.
3. The paper shows that Trace Inversion improves performance but provides no insight into why. More alabtion studies or failure / success case analysis would be helpful.

### Questions
See weakness.

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
3

### Summary
The paper investigates how reasoning traces in large language models (LLMs) affect their ability to abstain, that is, to decide when not to answer. The authors demonstrate that while Chain-of-Thought (CoT) prompting improves reasoning accuracy, it consistently harms abstention reliability by making models more prone to overconfident or misleading answers. Across eight established benchmarks and five model families, they show that adding CoT reduces both reliable accuracy (accuracy when answering) and abstention accuracy (correctness of abstain or answer decisions). To address this, they introduce Trace Inversion, a new abstention framework that treats hallucination as the model “answering the wrong question.” Instead of relying on confidence scores, Trace Inversion reconstructs the question implicitly answered by the model’s reasoning trace and measures its similarity to the original query; if misaligned, the model should abstain. Three variants of this method are explored: embedding-based, LLM-based, and grounded detection. Experiments show that these methods outperform all existing abstention baselines and improve reliability

### Strengths
The paper correctly identifies a key issue overlooked in prior work: improvements in reasoning performance can sometimes come at the cost of reduced abstention reliability. To address this, it introduces an elegant and effective method called Trace Inversion. The approach is conceptually simple yet broadly applicable, demonstrating strong performance across multiple models and benchmarks. Trace Inversion is particularly compelling because it frames model errors as instances of “answering a different question,” offering a meaningful bridge between interpretability and uncertainty estimation.

### Weaknesses
- The approach relies heavily on the reconstructed query and on the hypothesis that hallucinations arise when a model answers an incorrect question. However, this assumption may not always hold. The reconstructed query itself can be inaccurate or incoherent, leading to spurious or unjustified abstentions.

- Using semantic similarity to compare the original question 
𝑞
 and the reconstructed question 
𝑞
′
 is a fragile proxy for alignment. Similarity metrics may conflate paraphrasing with correctness and fail to capture deeper logical or causal mismatches. How is this handled? 

- Trace Inversion still depends on the model’s Chain-of-Thought, making it unable to detect or correct flawed reasoning that appears coherent on the surface. If the CoT itself is wrong, inversion merely reflects that error.

- The paper does not analyze how model accuracy changes with or without Chain-of-Thought. Exploring this trade-off could reveal when one might prefer a model that is slightly less accurate but more cautious or better at abstaining.

- All the models experimented upon are trained with instruction tuning to generate a reasoning chain. My assumption is that they may struggle to generate an answer directly. Wouldn't experiments with base models work better? 

- The presentation of the results and experimental variants is somewhat difficult to follow. Clearer tables, visualizations, or structured explanations of the baselines and Trace Inversion variants would make the work more accessible.

- Overall this is an interesting paper and I'm happy to reconsider once my concerns have been addressed

### Questions
In weaknesses

### Soundness
2

### Presentation
2

### Contribution
3
