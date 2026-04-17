# Sample Lottery: Unsupervised Discovery of Critical Instances for LLM Reasoning

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Reinforcement Learning with Verifiable Reward (RLVR) has equipped large language models (LLMs) with the capability of reasoning over complicated logical problems through policy optimization. However, conventional methods require complete annotation of the entire dataset and allocate computation uniformly over all samples. We articulate the lottery sample hypothesis in policy optimization of LLMs: a large training set contains a small subset that, when trained alone, yields performance comparable to that of the full dataset. This paper therefore explores the following question: How can we identify these lottery-winning samples from the original dataset without access to answers? Unlike prior efforts that analyze the effect of different samples in the training set with complete annotation, this paper focuses on the unsupervised discovery of critical instances for LLM reasoning and proposes a novel framework termed Complementary Conformal Selection (CONST). Specifically, CONST evaluates the importance of samples by considering two complementary components: procedural volatility and outcome volatility. Procedural volatility measures the potential variations during the LLM’s reasoning process, while outcome volatility captures inconsistencies in the final answer. Subsequently, conformal prediction is used to obtain a prediction set whose cardinality serves as the criterion for selecting the lottery-winning samples for annotation. We also provide a theoretical analysis, showing that CONST can effectively approximate the optimal policy. Extensive experiments on various LLMs across different datasets demonstrate the effectiveness of CONST.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on identifying data subsets that achieve performance comparable to using the full dataset. The proposed method, CONST, leverages procedural and outcome volatility to construct conformal prediction sets, enabling unsupervised discovery of critical instances. Experimental results demonstrate that CONST is both efficient and effective.

### Strengths
- The problem addressed is important and compelling. Identifying significant samples within datasets can reduce computation, conserve resources, and clarify how data-collection choices influence training. 

- Notably, the method requires minimal human annotation, which enhances its generalizability and practical applicability.

### Weaknesses
The paper would benefit from clarifying its scope earlier. My understanding is that the method is developed and evaluated specifically for the RLVR setting. To help readers set the right expectations, I suggest explicitly stating this focus in the title and/or abstract. Doing so would also avoid any impression that the approach covers broader RL or non-RL problems.

I also have several additional questions/suggestions, which I listed in the Question part. I did not examine the theoretical part in depth and just provisionally assume its correctness; I’m happy to discuss if other reviewers raise concerns.

### Questions
How should  n_p be selected in practice? For each truncated CoT, how many final answers are sampled? Only one or multiple ones?

Does outcome volatility functionally overlap with procedural volatility? In other words, is outcome volatility a special case of procedural volatility? 

What are instances? Is that data samples? Like 4 instances means 4 questions in the math dataset.


Could you also report the experimental results where you (i) generate multiple answers per test question, (ii) filter them by the scoring function and the threshold (from Conformal Prediction) to form a candidate subset, and (iii) measure the pass rate of the final answer set? A comparison against self-consistency and entropy-only baselines would be helpful for understanding the motivation for using the calibration set.

In addition, using a calibration set incurs a cost as well: you use m = 1024 instances (vs. 4 or 8 for training). Are these 1024 instances annotated? Finally, the choice of  m likely matters—why fix m=1024? A sensitivity analysis over m would be informative.

Assumptions are central to the validity of your claims; please state them explicitly and tie each result to the assumption it requires: permutation tests and exchangeability. Also, please discuss the practical applicability. Why can we use the MMLU subset as a calibration set for the BM set?

### Soundness
3

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
The paper investigates whether a very small subset of training problems can drive reinforcement-learning-with-verifiable-reward (RLVR) fine-tuning of large language models to the same accuracy obtained from the fully-annotated corpus. It fuses procedural volatility (how unstable an answer is to reasoning truncation) and outcome volatility (how inconsistent full answers are across rollouts) into a single sample-utility score using conformal prediction. This mechanism is intuitive, computationally light compared to full uncertainty modeling, and agnostic to the base model or reward function—making it broadly applicable.

### Strengths
1. If validated at larger scale, CONST could substantially reduce reward annotation costs in reasoning RL pipelines, which is an increasingly important direction for sustainable LLM alignment.

2. Novel formulation: Using conformal prediction to combine procedural and outcome uncertainty for sample scoring is an elegant, model-agnostic idea.

3. The figures and algorithms are cleanly presented; the method’s intuition is easy to follow even for non-specialists.

### Weaknesses
1. Similar ideas have appeared under names like self-consistency filtering, or uncertainty-guided selection. CONST’s originality mainly comes from framing these within conformal prediction and the RLVR objective.

2. The “ε-approximate lottery-sample” assumption is interesting but unverifiable. The authors don’t measure ε or show gradient proximity, so the bound doesn’t really illuminate why CONST works.

3. Procedural and outcome volatility both depend on stochastic decoding variance. Without separating linguistic noise from reasoning uncertainty, the method could mis-rank samples for harder domains.

4. Compared mainly to classification-style active learning. RL- or reasoning-specific selection strategies (e.g., self-consistency filtering, entropy-weighted sampling, or value-driven selection) are absent.

### Questions
1. Why not incorporate log P(Y | X) from π₀ into fπ₀(X,Y)? Did likelihood-based scoring perform worse?

2. Which embedding and K were used? How sensitive is CONST to these settings? Provide ablations with identical budgets but varying clustering to isolate diversity effects.

3. Could CONST operate in an active loop where the fine-tuned policy re-selects new samples? Any preliminary results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes CONST, a framework for identifying lottery-winning samples that are most critical for RLVR in LLMs. Instead of requiring full annotations for all training data, CONST selects a very small subset of informative instances without access to ground-truth answers. It combines two complementary measures: procedural volatility (instability of reasoning paths) and outcome volatility (variability in final answers), and leverages conformal prediction to quantify uncertainty through the size of prediction sets. Samples with higher uncertainty are selected for annotation and used for RLVR optimization. Theoretical analysis under the lottery sample hypothesis shows CONST can approximate the optimal policy, and experiments on several mathematical reasoning benchmarks demonstrate that CONST achieves near full-dataset performance with less than 0.5% of annotated samples, outperforming existing active learning baselines.

### Strengths
1. The method is valuable because it helps us understand which samples truly improve model performance and which do not, providing insight into data efficiency for RL-based reasoning.

2. The experiments are comprehensive and convincing, covering multiple models, datasets, and ablation settings to clearly show the method’s effectiveness.

3. The paper is easy to follow and well organized, with a clear narrative from motivation to theory and experiments, making complex ideas accessible.

### Weaknesses
1. I want to know whether CONST is suitable for logic reasoning datasets with discrete or small answer spaces such as multiple-choice tasks with only four options, where outcome volatility may be artificially low and conformal prediction less informative.

2. Theoretical results justify that an optimal subset can approximate full-data training, but the analysis stops short of proving that CONST reliably finds such a subset, the connection between the proposed selection criterion and the theoretical gradient proximity assumption remains heuristic.

3. Is the method sensitive to the number of instances in the calibration dataset?

4. The ablation (V2) skips the clustering step in Algorithm 1, but the paper never explains how clustering is done—what features or metrics are used and how the number of clusters is chosen.

### Questions
See weakness.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper describes a new data subset selection method called CONST. CONST identifies the critical instances from training data by measuring their procedural volatility (variations in the reasoning chain) and outcome volatility (inconsistencies in the final answer), combining these metrics using conformal prediction to determine sample importance. Experimental results demonstrate that training on the small subset selected by CONST achieves comparable performance to full-dataset training, showcasing its effectiveness for data-efficient policy optimization.

### Strengths
- The problem is important and interesting. It is good that the proposed method requires minimal annotation from experts and also does not use any LLM annotations.

- The authors analyse their algorithm theoretically.

- The authors show that using a small number of examples gives almost as good a performance as the whole dataset.

### Weaknesses
- The main drawback is that the authors do not discuss the whole of data subset selection and valuation literature. Example papers include:

Paul, Mansheej, Surya Ganguli, and Gintare Karolina Dziugaite. "Deep learning on a data diet: Finding important examples early in training." Advances in neural information processing systems 34 (2021): 20596-20607.

Guo, Chengcheng, Bo Zhao, and Yanbing Bai. "Deepcore: A comprehensive library for coreset selection in deep learning." In International Conference on Database and Expert Systems Applications, pp. 181-195. Cham: Springer International Publishing, 2022.

Das, Soumi, Manasvi Sagarkar, Suparna Bhattacharya, and Sourangshu Bhattacharya. "CheckSelect: Online Checkpoint Selection for Flexible, Accurate, Robust, and Efficient Data Valuation." IEEE Transactions on Artificial Intelligence (2024).

- I am not sure about the validity of assumption 3. The authors cite another paper, calling it a standard assumption. However, in my opinion, this assumption is very strong and is often invalid. Also, I am not sure if the proof is novel under this assumption.

- The authors did not show results on bigger reasoning models.

### Questions
Why are the authors not reporting results on qwen 2.5 7 b or qwen 3 8b ?

### Soundness
3

### Presentation
2

### Contribution
1
