# Keep the Beam on Track: Stabilizing Reward Trajectories in Guided Decoding

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Decoding algorithms play a central role in enhancing the performance of large language models (LLMs) on complex reasoning tasks. 
A common approach incorporates Process Reward Models (PRMs), which estimate the quality of intermediate reasoning paths and guide the selection of possible continuations. 
In this setting, our analysis reveals two notable phenomena: reward estimates tend to decline as reasoning progresses, and the reasoning paths exhibit distinct volatility patterns across decoding steps
depending on whether the paths lead to correct or incorrect final answers. 
In particular, correct reasoning tends to be associated with stable reward trajectories, while incorrect reasoning often shows high volatility. 
Motivated by this observation, we propose Volatility-Scaled Guided Decoding (VSGD), a decoding algorithm that prioritizes candidate paths with lower volatility by jointly considering the magnitude of PRM-estimated rewards and the volatility of these rewards across decoding steps. 
Experiments on datasets including GSM8K and MATH500 indicate that VSGD reduces the volatility of selected reward trajectories and improves the accuracy of the final answer. 
These findings suggest that considering the temporal dynamics of reward values, in addition to their magnitude, provides a potential direction for enhancing guided decoding in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Volatility-Scaled Guided Decoding (VSGD), a decoding algorithm for large language models that stabilizes Process Reward Model (PRM) guidance by accounting for the volatility of reward trajectories during reasoning. The authors observe that correct reasoning paths exhibit lower reward volatility than incorrect ones and design VSGD to prioritize more stable trajectories. Evaluations on GSM8K, MATH500, and an MMLU subset demonstrate consistent accuracy improvements and reduced incomplete reasoning, showing that modeling temporal reward dynamics enhances guided decoding efficiency and reliability.

### Strengths
The proposed method itself is coherent and easy to follow. The authors conduct experiments on several benchmarks to validate the effectiveness of the proposed method.

### Weaknesses
1.	Overall, the contribution of this paper, while meaningful, is somewhat incremental relative to prior PRM-guided decoding research. The method reuses standard components such as beam search and PRM scoring, with limited theoretical advancement beyond empirical validation.
2.	Across datasets, performance gains over the strongest baseline are relatively small (about 1–1.5% on average). Such improvements may not justify introducing an additional volatility computation layer in practical systems. Moreover, the improvements vary across domains, with some categories (e.g., Virology and GSM8K) showing minimal or even negative differences, indicating limited robustness.
3.	The authors employ only one LLM backbone in their experiments. To strengthen the evaluation, recently proposed LLMs, such as Mistral and Qwen should be included as the backbone models for comparation.
4.	The paper does not investigate how sensitive VSGD is to its hyperparameters, such as the stability constant ϵ or the aggregation function used in ranking candidates. It also lacks ablation experiments isolating the effects of volatility scaling from other algorithmic factors.

### Questions
1.	How sensitive is VSGD to the choice of the stability constant ϵ and the aggregation function Agg?
2.	Would volatility normalization still help when PRM rewards are poorly calibrated or highly correlated with token length?
3.	Could volatility be exploited during PRM training rather than only inference?
4.	Have the authors tested whether volatility correlates with interpretability or logical consistency of reasoning paths beyond correctness?

### Soundness
3

### Presentation
2

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
This paper addresses the instability of reward signals in Process Reward Model (PRM)–guided decoding for large language models. The authors observe that correct reasoning paths exhibit stable (low-volatility) reward trajectories, while incorrect ones fluctuate significantly. To exploit this, they propose Volatility-Scaled Guided Decoding (VSGD), which adjusts reward values based on their temporal volatility to favor stable reasoning paths. Experiments on GSM8K, MATH500, and MMLU subsets show that VSGD improves reasoning accuracy and reduces incomplete reasoning, demonstrating that incorporating reward stability enhances guided decoding performance.

### Strengths
The paper’s strength lies in introducing Volatility-Scaled Guided Decoding (VSGD), which leverages reward stability to guide search. By prioritizing reasoning paths with low reward volatility, it offers a simple yet effective way to stabilize decoding and improve reasoning accuracy.

### Weaknesses
1. The paper only observes the fluctuation of PRM rewards empirically without providing a theoretical explanation.
It remains unclear why reward volatility naturally emerges or correlates with reasoning correctness.

2. All experiments rely solely on LLaMA-3.1-8B and VersaPRM, making the results model-specific.
It is uncertain whether stronger models like GPT-5 would show the same volatility behavior or performance gains.

3. The study omits stronger decoding and test-time scaling baselines such as MCTS or Q-function-based methods. Moreover, the reported accuracy improvement is small (around +1.1–1.4), limiting the practical significance.

4. The paper introduces several hyperparameters but does not analyze their sensitivity or robustness. Without such evaluation, it is difficult to assess the stability and reproducibility of the proposed method.

### Questions
see weaknesses

### Soundness
2

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
3

### Summary
The authors propose a new decoding algorithm for Large Language Models (LLMs) called Volatility-Scaled Guided Decoding (VSGD), which prioritizes candidate paths with lower volatility by jointly considering the magnitude of PRM-estimated rewards and the volatility of these rewards across decoding step. The technique is derived from the observation that correct reasoning paths usually exhibit high rewards but low volatility across reasoning steps. Thus, instead of selecting candidates with maximum reward in each step, the authors select the ones with the maximum ratio of reward and standard deviation. Experimental results demonstrate the merits of the proposed method.

### Strengths
- In terms of clarity, I think the authors did a great job. From pilot study to algorithm, everything is crystal clear.
- I think exploiting the dynamics of reward for better decoding is interesting and promising. The use of std as a damping factor avoids myopic emphasis on high-reward steps.
- The experiments are comprehensive.

### Weaknesses
- The technical depth is limited. The major contribution is two folds: (i) the empirical observation of PRM dynamics of both correct/incorrect reasoning paths; (ii) replacing the selection criterion in beam search to incoprate variability with no theoretical result, which seem to be straightfoward. Also, it is unclear to me that whether the proposed method can be generalized to other decoding algorithms like MCTS.
- The experimental results are not strong and convincing enough. The authors only compared their methods with vanilla beam search which is obviously not the SotA methods for test-time scaling. The datasets used are also a bit old. I suggest the authors considering popular benchmarks like AIME or AMC. Finally, the improvement is marginal, which sheds some doubts on the significance of the proposed method.

### Questions
See Weakness section

### Soundness
3

### Presentation
3

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
This paper focues on guided decoding with a process reward model. Firstly, the authors observe that the reward distribution for different steps has a close correlation with correctness. I.e. a declining reward across steps and a high volatility patten of the rewards normally hint a final false prediction. Inspired by this observation, the authors propose Volatility-Scaled Guided Decoding (VSGD), which prioritizes candidate paths with lower volatility by jointly considering the magnitude of PRM-estimated rewards and the volatility of these rewards across decoding steps. 

Experiments on three benchmarks (GSM8K, MATH500 and MMLU) with a process reward model show that VSGD outperforms other baselines. Extensive ablation also justify the design choice.

### Strengths
1. The observation is interesting. I.e. reward stability and volatility distribution is correlated with the prediction correctness.
2. The proposed method is closely related to the observation, making the paper coherent.
3. Extensive experimemts and ablation show the benefits from VSGD.

### Weaknesses
1. Lack of process reward models and decoding models. Only one process reward model (VersaPRM) and one decoding model (Llama-8B) are verified here. It cast doubts on the generalization of the observation and results.
2. Limited improvement. From Table 1, we can see that the improvement from VSGD is very limited compared, +0.6 or +0.7 on MATH500 and -0.2 or +0.1 on GSM8K. The most improvement comes form MMLU, with +1.1 or +1.4. This shows the limitation of the proposed method on various domains.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2
