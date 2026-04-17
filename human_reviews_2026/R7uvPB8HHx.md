# Which LLM to pick? Online Active Model Selection for Large Language Models

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Large Language Models (LLMs) are increasingly applied to process streaming data, with practitioners relying on benchmarks to select the best model even though these signals only approximate real performance. While oracle annotations can provide reliable feedback, they are often costly and difficult to obtain at scale. To address this challenge, we propose Online LLM Picker, the first framework for active model selection for LLMs in online settings. Given an arbitrary stream of queries and a limited annotation budget, Online LLM Picker  selects the most informative prompts for annotation to identify the best LLM among candidate models. Across multiple tasks including 10 datasets, for over 130 language models, we show that Online LLM Picker saves annotation cost by up to $71.67$\% while reliably identifying the best or near-best model for the stream. We also show that using the returned model for sequential generation on unannotated prompts across the stream reduces regret by up to a factor of $2.51\times$, indicating that Online LLM Picker can identify the best or near-best model well before processing all streaming prompts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Picking which LLM to serve in an online setting is an important problem inspired by real-world use cases. LLM Picker attempts to solve this problem with guarantee by using an algorithm based on the EW algorithm with a modified adaptive learning rate rule. Experiments to demonstrate the advantage over baseline algorithms are provided.

### Strengths
- Well-defined metrics for the problem.
- Good performance on several benchmarks with a variety of models.

### Weaknesses
- The writing could be much better. Sec 3.2 reads like a laundry list of the things the author did without a clear story and explanation for the various designs of the algorithm.
- Unclear notation: $p_t$ is introduced on line 133 before definition.
- Section 3.2: the authors should formally state why the EW algorithm is chosen to minimize regrets.
- There is actually a related work using the EW algorithm for the same problem [1].
[1] Feng, Y., Khare, A., Nguyen, N., & Sengupta, S. B. Boss LLM: Adaptation via No-Regret Learning. In ICLR Scaling Self-Improving Foundation Models without Human Supervision Workshop.

### Questions
- In Sec 3.1, what are the verifiable advantage of the new adaptive learning rate rule vs pure EW rule?
- In Eq 10, what are the final contributions of the different design choices to the final performance? Is there an ablation study?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes an online LLM model router that selects which model is best for a certain input.

### Strengths
The paper compares their online LLM picker against several baselines and strategies. They show that their approach is best among all in a set of single-turn standard benchmarks.

The models selected for the experiment include both open-source and proprietary LLMs (see lines 345-358). This makes the experimental setup thorough

### Weaknesses
The paper claims that this is the first framework for LLM routing,  there is work on this area e.g., https://arxiv.org/abs/2309.15789, https://arxiv.org/abs/2402.05859, https://proceedings.neurips.cc/paper_files/paper/2024/file/7a641b8ec86162fc875fb9f6456a542f-Paper-Conference.pdf. The paper does not cite these and others.

The paper only tackles a collection of public standard benchmarks which makes the scope and impact of the experiments quite limited.

Given that the paper claims that this is an online routing, it would be great to showcase this in the context of multi-turn trajectories (this paper could be a good start https://arxiv.org/pdf/2505.06120). Maybe each turn can be sent to a different LLM.

### Questions
How do you account for potential contamination of the routing (i.e., the model used to route knows about the benchmark) and the underlying models have been trained on the benchmarks. It would be great to show datasets that are unique for this task.

How would this work in a conversational setting with multiple trajectories (see e.g., https://arxiv.org/pdf/2505.06120) ?

How would this work in an agent environment with react framework, would it be possible to also pick LLMs in that context?

I saw this comment in line 354: Prompt engineering is also used to improve performance on our benchmark datasets by reducing unnecessary verbosity and avoiding irrelevant explanations in the output.
could you elaborate? does this make the results not comparable with other methods using teh same benchmarks?

In Figure 2, what are the results of always taking the top performing model?

### Soundness
3

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
This paper tackle the problem of active model selection for LLMs in a online streaming settings. It proposes a new framework named online LLM picker, which picks next prompt for oracle annotation based on the information gain to identify the best LLM. The proposed approach combines an Exponential Weights algorithm with adaptive learning rate based on loss variance, together with a query probability that balances hypothetical variance across model responses with posterior entropy. The paper performs experiments on 10 datasets and 130+ LLMs show up to 71.6% annotation savings and 2.51x regret reduction compared to selected baselines.

### Strengths
The paper addresses the problem of active model selection in an online streaming fashion which is well formulated and resonate well with real world applications, especially given the proliferation of available LLMs and the cost of annotations. The proposed framework is model agnostic (no assumptions about the LLM architecture or require log likelihood) and it works with both black and white box LLMs. This ensures the general applicability of the approach. The paper performs experiments on 10 datasets among diverse tasks over 130+ LLMs, the results show quite strong and consistent improvements over baselines in terms of the annotation efficiency (up to 71.6% savings) and identification probabilities, which proves the great value of the proposed approach in the real world settings.

### Weaknesses
1. Missing baselines in the comparison. Although the authors claim they are the first paper deal with active model selection in an online streaming fashion, I would still argue the baselines are still quite simple and it should have compared with other active model selection methods. For example, the paper cites Ashury-Tahan et al. 2024 (any many others) in the related work, these method can be easily adapted to the online streaming fashion by applying sliding window to convert from pool based method to the streaming based method. Comparison with these SOTA methods will certainly informative. 

2. Missing results analysis and ablation study. The paper introduces some variable components in the framework, such as the adaptive learning rate (vs. fixed learning rate), variance only vs. variance + entropy and various of similarity metrics for loss calculation. What is the impact for each of them? How sensitive will the final results be due to the change of these choices? These questions are not discussed in the paper.

3. The experimentation setup for the online streaming simulation is limited. Part of the novelty for the paper comes from the online streaming model selection, but I don't think we are simulating some unique patterns potentially from the stream. For example, what if the underlying tasks from the stream is too diverse? How will that impact the performance? What if the stream has significant distribution shift over time? How will the method generalizes to these distribution shift? I would recommend to design some experiments to show specific characteristics w.r.t the streaming vs. a fixed pool.

### Questions
1. We are using pretty "old" similarity based metrics such as ROUGE/BERTScore etc. in the paper. These metrics are proven to less correlated with human performance. What is the reason we select these metrics? For example, LLM based metrics correlated much better with human, what is the reason we didn't select these better metrics? Have we tried these metrics and how that impact the model performance comparison?

2. Do we have any theoretical analysis to show the bound of the online LLM picker, for example, on the regret? I think the work can get strengthened even if we provide some simple analysis there. 

3. Have we compared the method with other pool based active model selection method? How does it compare with these SOTA models? I think some analysis there will be very helpful for the community to understand the difference and the value provided in the paper.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces ONLINE LLM PICKER, a framework for active model selection among multiple large language models (LLMs) in online settings where data arrive sequentially and annotation budgets are limited.
At each time step, the algorithm decides whether to query an oracle for the reference output, updates model posteriors using exponential-weights (Hedge) with an adaptive learning rate based on variance, inspired by AdaHedge ( Rooij et al., 2013 ), and uses a variance-entropy criterion to determine which prompts are most informative to label.
Compared to the current state-of-the-art active model selection methods, the proposed method follow a streamlined approach without relying on a given pool of labeled data. 
Earlier methods rely on discrete class labels, making informativeness easy to define via label entropy. In contrast, this paper defines a variance-based informativeness measure over generated text, approximating the space of possible references using pairwise model outputs.
That allows the approach to be applied to open ended generative tasks like question answering, grammar correction, and arithmetic reasoning settings where prior active selection methods don't apply
The author conducts experiments on several generative tasks.
Across ten generative datasets, the method reportedly reduces annotation cost by up to 70.69 \% and reduces the regret compared with baselines such as Random, Disagreement, KL-divergence, and Uncertainty sampling significantly.

### Strengths
1. Prior active-selection work assumes pool-based settings or classification tasks; this is the first explicit treatment of streaming LLM selection with generation-style outputs.

2. The combination of Hedge-style posterior updates, adaptive learning rate $\eta_t$, and a joint variance-entropy query rule is mathematically clear and connects online learning with Bayesian experimental-design intuition.

3.	Comprehensive empirical validation.
Experiments span ten diverse generative datasets and more than 130 models. Metrics include regret, identification probability, and annotation efficiency, showing consistent gains (up to 2.5 $\times$ lower regret and $\approx$ 70 \% fewer annotations).

### Weaknesses
1. Although inspired by AdaHedge, the paper stops short of giving formal regret bounds for stochastic or adversarial streams. The adaptive $\eta_t$ and variance entropy querying are heuristic extensions without proofs of convergence or sample complexity guarantees.

2. The author uses several baselines (Random, Disagreement, KL-divergence, Uncertainty sampling) but does not give some details on their implementation. For example, how is uncertainty sampling defined for generative outputs? More clarity on baseline implementations would help assess the empirical gains.

3. Each query step requires evaluating $m \times m$ pairwise distances to estimate hypothetical variance (Eq. 8), which could be quadratic in the number of models. What is the computational overhead compared to baselines?

### Questions
Same as the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
