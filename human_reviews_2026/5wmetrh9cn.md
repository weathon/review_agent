# Mirage or Method? How Model–Task Alignment Induces Divergent RL Conclusions

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Recent advances in applying reinforcement learning (RL) to large language models (LLMs) have led to substantial progress. In particular, a series of remarkable yet often counterintuitive phenomena have been reported in LLMs, exhibiting patterns not typically observed in traditional RL settings. For example, notable claims include that a single training example can match the performance achieved with an entire dataset, that the reward signal does not need to be very accurate, and that training solely with negative samples can match or even surpass sophisticated reward-based methods. However, the precise conditions under which these observations hold—and, critically, when they fail—remain unclear. In this work, we identify a key factor that differentiates RL observations: whether the pretrained model already exhibits strong *Model-Task Alignment*, as measured by pass@k accuracy on the evaluated task. Through a systematic and comprehensive examination of a series of counterintuitive claims, supported by rigorous experimental validation across different model architectures and task domains, our findings show that while standard RL training remains consistently robust across settings, many of these counterintuitive results arise only when the model and task already exhibit strong model-task alignment. In contrast, these techniques fail to drive substantial learning in more challenging regimes, where standard RL methods remain effective.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This papers presents an empirical study on some reported phenomenas in RL training such as: 1) training with random erroneous rewards (random reward), 2)  self rewarding LLMs via Test time RL i,e using the majority vote answer from a rollout as a true one , 3)  one shot , i,e trainig with a single selected example 4) GRPO training with only positive or negative sampling. Some recent work Wu et al 2025 explained some of these phenomenas as due to the contamination of the training set of some of LLMs with the test set. This paper probes another hypothesis that the contamination is not the only reason behind such phenomenas but the alignment between the task and the LLM at hand. The alignment between task and LLM is measure with pass@k of the LLM on the test set of this task. Authors focus on Qwen and Llama family of models and divide task and models to three areas: strong alignment and strong contamination (red), strong alignment and weak contamination (green) , and weak alignment and no contamination (gray). Throughout extensive experimentation authors find that pass@k on a task is a predictive of the performance of the model with standard RL, but also its robustness to the phenomenas listed above. A high pass@k wether contaminated or not pretaining translates into a lift in performance for standard RL and the variants above (TTRL, one shot RL, random reward, positive , negative sampling), and low pass@k leads to a bad performance across the board.

### Strengths
The paper is very well written and a very nice and easy read with all experiments well articulated and well presented, I personally enjoyed reading the paper. 

The fact that pass@k of the reference model is a predictive of the performance of standard RL is not surprising and it has been even shown theoretically in multiple works showing that best of n is an approximation of the RL optimal policy see for example best of n through the smoothing lengths or in Asymptotics of Language Model Alignment or in  information theoretic gaurantees of alignment in LLMs. It is interesting to develop similar results to robustify these results to data and reward perturations as presented in this paper.

### Weaknesses
One interesting point the paper is not studying in the analysis is to follow the KL(pi|| pi_{ref}) when using standard RL versus the non standard RL (TTRL, one shot RL, PSR, NSR). 

It would be interesting to compare the E_{test set }KL(pi|| pi_{ref}) , to see how the perturbations of the rewards/ sampling impacts the deviation from the reference models, in the green, red and gray area. Note that using DAPO we don't have the regularization to pi_{ref} so this is a fair study.

### Questions
* what is the group size you used in DAPO? 
* do you think the conclusions will change if we add the KL regularization to pi_{ref} in the training of these RL ? 
* can you study the KL of those perturbed RL to see how the perturbation of the reward impacts the KL emprically ?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper examines various hypothesis and claims about the performance of RL methods for language models. In particular, they investigate numerous phenomena around RL -- training with noisy rewards, training on single examples etc. and show how their performance is strongly correlated with "model-task alignment".

### Strengths
* the paper is well organized and generally easy to follow. the purpose of each section is clearly defined.
* the experiments cover a number of different hypotheses for why certain phenomena appear
* The authors consider multiple model types.
* The insights provided by the paper are generally useful
* using pass@k for model-task alignemnt is an interesting idea, but digging more into the realm of RL I think this might really be an indicator of how much exploration is necessary for RL to work.

### Weaknesses
The paper claims that "model-task alignment" measured by pass@k is predictive of whether or not several of these RL-based phenomena occur. However, the authors only use a handful of benchmarks and bin the "model-task alignment" into either strong or weak. It would be interesting to see if the authors could plot the pass@k (the proposed proxy metric) versus the increase in performance to actually see the correlation. From the results pass@k does seem predictive, but it's hard to know for sure without a ton of information. Doing more tasks for all methods would be excessive, but it could be interesting to see at least one representative.   For example, one could consider a coding domain.

### Questions
not any at this time.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper identifies Model-Task Alignment, measured by pass@k accuracy on the evaluated task, as a key factor that explains the counterintuitive phenomena that have been reported in LLMs but not observed in traditional RL settings. These phenomena include single training example matching the performance on the entire dataset, the reward signal not needing to be very accurate, training only with negative samples matching the performance of sophisticated reward-based methods, efficacy of test-time RL, etc. The paper demonstrates across multiple model-task combinations, spanning two language models (Llama and Qwen), and multiple mathematical and logical reasoning domains, that strong model-task alignment explains the counterintuitive phenomena observed in LLMs but not in traditional RL. In settings with weak model-task alignment, standard RL approaches work while the counterintuitive techniques fail

### Strengths
The paper critically evaluates existing claims in RL for LLMs, such as the effectiveness of spurious rewards, one-shot training, and negative-only signal sufficiency, and identifies Model-Task Alignment as a key factor determining if these claims hold true.
Additionally, the paper provides a comprehensive and systematic examination of these counterintuitive reinforcement learning (RL) phenomena in large language models (LLMs), supported by rigorous experimental validation across different model architectures and task domains. Furthermore, they show that dataset contamination is not the underlying cause, as was proposed in earlier works, by demarcating model-task combinations into those potentially contaminated and those not potentially contaminated in their experiments. Moreover, the paper is well-written and easy to follow, with dedicated sections for each of the reported counterintuitive phenomena.

### Weaknesses
The experiments primarily focus on Qwen and Llama models (of similar parameter scale, 8b), which may limit the generalizability of the results to other LLMs. Further experiments encompassing more LLMs and model parameter scales would help to understand if the results are more general. Additionally, while the Model-Task Alignment hypothesis is well-supported, the paper could explore other factors that might also contribute to the observed phenomena, such as the role of pretraining data, model sizes, or architectural differences.

Additionally, the paper does not explicitly mention if hyperparameter search was performed for each model-task combination and the counterintuitive RL phenomenon being explored (eg: negative samples, spurious rewards, single example training, test-time RL etc.).  Different combinations might need different hyperparameters to elicit their best possible performance

### Questions
1) Do you have the model-task alignment results for other LLMs (eg: mistral, deepseek, gemma) or different model sizes?

1) Was hyperparameter tuning performed for each model-task combination and the counterintuitive RL phenomenon being explored?

2) Why does self-rewarded RL seem to work for the Llama-math combination in Table 2, which has no contamination with weak model-task alignment? Any factors that may explain this?

3) Similar to the previous question, Negative Sampling Reinforcement (NSR) also seems to work for the Llama-math combination in Table-5. What explains these observations?

### Soundness
3

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
This paper argues that many of the recent, slightly “too good to be true” RL-for-LLMs findings (spurious/noisy rewards still work, one-shot RL, test-time RL, negative-only signals) only work when the base model is already well aligned with the task, and that this alignment can be read off from pass@k on that task. When alignment is weak, they say, those tricks collapse, while plain RLVR/DAPO still helps.

### Strengths
This works has studied different claims in RL literature for LLM to show their effectiveness.

### Weaknesses
1. The experimental evidence is drawn almost entirely from Qwen2.5-7B and Llama-3.1-8B on math and KOR-style reasoning tasks (AIME24, MATH500, AMC, Operation, Counterfactual, Puzzle, Cipher). The conclusion that model–task alignment is a *fundamental* determinant therefore generalises beyond the evaluated model/task scope.

2. The paper equates model–task alignment with pass@k, but no prior work is cited to justify this choice of proxy. It would strengthen the argument to motivate pass@k more formally or to report an additional alignment signal (e.g. a divergence-based measure between base and trained models) as a complementary view.

3. The description in Appendix D.3 of “incorrect” and “format” rewards is underspecified: the exact criteria for each reward type and the RL setup used in that subsection should be made explicit, and the authors could briefly justify why they chose this particular RL recipe over newer GRPO-style variants (e.g., Dr.GRPO).

4. The set of weakly aligned models is fairly narrow (mainly Llama-3.1-8B in low-pass@k regimes). Including another family with weaker out-of-the-box performance on these tasks (e.g. Gemma 3) would make the alignment–effectiveness pattern more convincing.

5. The paper should state earlier and more prominently that DAPO is the default RL method used in the experiments, so readers do not have to infer it from later sections/appendices.

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
