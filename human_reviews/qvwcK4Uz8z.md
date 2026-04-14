# CASE: Challenger Arm Sampling for Efficient In-Context Reasoning

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
The in-context learning paradigm with LLMs has been instrumental in advancing applications that require complex reasoning over natural language. An optimal selection of few-shot examples (exemplars) is essential for constructing effective prompts under a limited budget.
In this paper, we frame the problem of exemplar selection for In-Context Reasoning (ICR) as a top-m best arms identification problem. A key challenge in this context is the exponentially large number of arms that need to be evaluated to identify the m-best arms. We propose CASE (Challenger Arm Sampling for Exemplar selection), a novel selective exploration strategy that maintains a shortlist of ``challenger'' arms, which are current candidates for the top-m arms. In each iteration, only the arms from this shortlist and the current top-m set are pulled, thereby reducing sample complexity and, consequently, the number of LLM evaluations. Furthermore, we model the scores of exemplar subsets (arms) using a parameterized linear scoring function, leading to a stochastic linear bandits setting. In this setting, CASE identifies the top-m arms with significantly fewer evaluations than existing state-of-the-art methods. CASE effectively works with black box LLMs and selects a static set of few-shot examples, resulting in an extremely efficient scheme for in-context reasoning. The exemplars selected with CASE show surprising performance gains of up to 15.19% compared to state-of-the-art exemplar selection methods. We release our code and data (https://anonymous.4open.science/r/CASE_exemplar_bandits-7403).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Due to the financial and computational costs associated with prcessing large contexts, providing all training examples for in-context learning is often impractical in many settings. This work aims to address this issue by designing an efficient and optimal selection of exemplars to include in the prompt. The authors frame the task of constructing an optimal prompt as a multiple examplar subset selection problem, where, given a training set, the goal is to pick a subset of size m such that the corresponding prompt which is generated maximizes the total validation accuracy. As this is a discrete optimization problem over an exponentially-large search space, naive approaches are intractable. 

The authors therefore model this problem as a top-m arm selection problem from the literature on stochastic linear bandits, where they use a linear function to approximate the loss of each arm (i.e. prompt). While the problem of identifying the top-m arms can be solved using gap-index based algorithms, the total number of arms is exponentially large, and so applying these approaches off-the-shelf is computationally infeasible. Instead, the authors propose a new algorithm (CASE) to solve this problem. At a high level, CASE iteratively creates a low-regret set of selected “challenger” arms from uniformly sampled arms. An off-the-shelf algorithm is then used to pick an arm from the union of this set and the current estimate of the top-m arms. The authors prove a high-probabiliyt upper bound for the sample complexity of their algorithm. 

The authors apply their algorithm on the GSM8K and AquaRAT datasets for commonsense reasoning, StrategyQA for tabular reasoning, and FinQA & TabMWP for numerical reasoning. They compare their method to zero-shot and manual few-shot baselines, and other instance-level selection measures which use diversity & similarity-based measures to select exemplars for each test example. Across all datasets, the authors find that CASE consistently outperforms random, zero-shot, and few-shot exemplar selection methods. Moreover, CASE (sometimes paired with an existing selection measure) outperforms the other instance-level selection measures on their own. The authors also find that CASE is more computationally-efficient than other stochastic linear bandit-based approaches, and observe that the number of LLM calls made by CASE is significantly less than the number made by LENS (another exemplar selection method). Finally, the authors show that using their exemplar selection methods, smaller methods can be made to perform well compared to larger LLMs.

### Strengths
In-context learning is now a ubiquitous task for large language models. Methods for exemplar selection have the potential to produce better results for in-context learning without dealing with the large computational overhead which often comes with feeding the entire training set into the LLM in-context. While the authors are not the first to study the exemplar selection problem, their solution (or some variant of it) outperforms all baselines and other methods across a wide variety of tasks.

### Weaknesses
It was unclear to me how the theoretical assumptions in Theorem 1 and Lemma 1 translated to the setting of in-context learning. 

While the experimental results are comprehensive, it appears to me that several important baselines are missing. (See Questions for more details.)

### Questions
Could you clarify how the assumptions in Theorem 1 and Lemma 1 translate to the setting of in-context learning?

Why didn't you compare to LENS+MMR+SC and LENS+KNN+SC? This is especially relevant given that CASE+MMR+SC and CASE+KNN+SC often performed best across different instances.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies how to select a few shot examples (termed exemplars) to be passed with the prompt for the chain-of-thought prompting. They use bandit selection strategies to select the most informative prompts as exemplars. The key issue in this setting is the exponential size of the arm/action set (which is the total set of all available exemplar prompts).  The known bandit strategies for finding the top-m best arm in an exponential action set require the computation of gap indices between all currently estimated top-m arms and the remaining arms. This is impractical in an exponentially large arm set and therefore they propose a new approach to prune the candidate arm set. They basically run a UCB-LCB algorithm over a k-sized subset $S$ which is similar in spirit to algorithms in Chen et al. (2017), and Kaufmann & Kalyanakrishnan (2013). They provide a theoretical guarantee for their algorithm which uses a standard PAC bayesian analysis of fixed confidence top-m bandit theory. They empircally validate their approach on multiple datasets.

### Strengths
1. The problem of choosing the best set of informative prompts in-context for reasoning and mathematical analysis for an LLM is an open and interesting problem.
2. Their approach of selecting the top-m best exemplars to be padded with prompt is a theoretically sound approach. However, their practicality for natural language generation needed more validation.
3. They theoretically analyze their algorithms and also empircally validate their approach on multiple datasets.

### Weaknesses
1. The writing needs to improve significantly. For example, consider the definition of the UCB, $W_t(i,j)$ definition below eq (3). It is not clear what are $\left(\left\|E\_i\right\|\_{\hat{\Sigma}_t^\lambda}+\left\|E\_j\right\|\_{\hat{\Sigma}_t^\lambda}\right)$. Several such areas require more explanation and definitions need to be defined more clearly.
2. My main concern is that their approach looks mainly like a bandit selection approach without taking into consideration how this affects the natural language generation. For example, their synthetic experiments consider only the bandit case and they show that their approach performs well. However, they then progress to complex reasoning tasks, and it is not clear to me how this bandit approach actually helps the LLM to solve the complex reasoning task. See questions for more.
3. It is not clear to me how the featurization of the actions $\mathcal{X}$ and the $k$-subsets of $\mathcal{X}$ are obtained.
4. How do you learn the $\alpha_i$ in eq(3)? Is it pre-specified?
5. While the $\phi_u(a)$ is estimated using a pre-trained transformer, how do you estimate the featurization $x_i \in \mathcal{X}$?

### Questions
1. One of my main concern is that the algorithm does not take into account the reasoning aspect of an LLM while selecting the exemplar. Consider the synthetic experiment vs the reasoning experiment. How do we actually understand the chosen exemplar leads to better reasoning? Consider the example on pg 23 where they show the rationale selected by CASE in the exemplar. Their closest competitor LENS actually chooses rationales that are more concise (and to me look better) than CASE. Also strangely they show different questions and rationales while comparing against their competitor. I suggest showing a table where the same questions show what CASE vs LENS choose as exemplars/rationales.
2. The bigger point I wanted to highlight is that in no part of their algorithm, I found a way that the natural language generation being taken into account to choose the exemplar. It is also not clear how they factor in the original prompt in the arm features (in context) and how this influences the design matrix in line 22 of the algorithm. Do you initialize with the feature of the original prompt? 
3. Some more questions are mentioned in the weakness section.
4. Some small things: Chen et al. (2017); Kaufmann & Kalyanakrishnan (2013) are not uniform sampling algorithms for top-m arms. Quite similar to yours, they rely on UCB-LCB mismatch to figure out the top-m arms in stochastic bandits. 
5. How do you choose $\epsilon$ in the algorithm? Should it be given as an input to the algorithm?
6. In figure 2d, I observe that LinGapE is almost similar in simple regret to CASE. This goes back to my earlier doubts. How do you actually differentiate how well one bandit algorithm selection is leading to a better reasoning exemplar selection than another bandit algorithm selection.

### Soundness
2

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
4

### Summary
This paper considers the problem of exemplar selection for In-Context Learning and In-Context Reasoning using LLM’s. The authors consider a setup where they have access to n (labeled) examples and a collection of subsets of k of of these. They want to find the m subsets that lead to the highest m accuracies on a fixed validation set. They postulate a model for the accuracy of ICL as a function of similarities between a validation set and the input. They then apply best-m pure exploration linear bandits algorithms and demonstrate empirical gains across several methods.

### Strengths
I thought the paper touched upon an interesting topic in a novel way combining ICL in LLM literature with bandits. The experimental section seemed well done and they demonstrated signficant gains.

### Weaknesses
My main concern (and see below) was the couching of the work in the existing bandit literature and theoretical results given. I also had some concerns about the mapping of the ICL problem to bandits.

### Questions
A. I thought the mapping of the problem of finding good exemplars to linear bandits to be an interesting one. However, I had some confusions/questions

a) Can you say more about why pi needs m sets of exemplars, rather than  just a set of k examples? Maybe having an example pi in the text would help this could even be in the experiments section.
b) Are S1…Sm allowed to overlap? Why? It feels like this could really alter the approach (see below).
c) How does the objective in (2) turn into the top-m problem? I.e., there seems to be an assumption that finding the top-m individual sets of size k with high accuracy also identifies the collection of best m sets to go into pi?
d) I liked the use of similarity between the training prompts and the validation. In practice, why not just use this directly instead of trying to find a set of m exemplar sets for all prompts? I.e., why go after the average over the validation set? (Overfitting issues aside of course)
e) I think a diagram on page 4 would help

B. I had a few comments about the linear bandit comparisons. 
a) The authors are missing some important papers in the pure exploration linear bandit literature, eg, Fiez '19, Jedra'20, Li'23. One key discussion that is missing is that unlike Xu, and Reda, in the m=1 case, all these works provide algorithms that have matching upper and lower bounds. I believe that Degene et al. 2020 (and maybe Li'23?) could be extended to an algorithm that could solve the top-m problem and achieve an optimal lower bound. However both of these algorithms would still be computationally infeasible. It would be good if the authors could comment on this.
b) I didn’t really understand what Theorem 1 says.  What does epsilon-delta PAC means here? How does this sample complexity scale with a function of m, k, n? What is K? This theory section is woefully lacking.
c) As a comment, is the reduction to linear bandits necessary? You are trying to find the top m sets of size k. It feels like you could do this in a more direct manner. In particular, if a1, a2, …., an were an ordering of the individual arms by rewards as given in 3), you just need to return a1, .. a_{mk} appropriately segmented (if there are no overlaps - otherwise it would be a_{k+m}). Could you employ existing works on coarse ranking, eg Katariya '18?

C. Comments on selective exploration
a) I don’t really see the connection to SETC. I went and read this section of the bandit book again, and the comparison is very unclear. Please clarify. 
b) As a result, I struggled with the discussion at the start of 3.4. How does the regret of selective exploration factor in?

D. Experiments:
a) The choice of S seemed to be restricted to 100 rather than all subsets of size k. I found this to be a bit confusion (line 3 and 4)

Fiez '19, Sequential experimental design for transductive linear bandits
Jedra '20, Optimal best-arm identification in linear bandits
Wang '21, Fast pure exploration via Frank-Wolfe
Li' 23, Optimal exploration is no harder than Thompson Sampling
Katariya '18 Adaptive Sampling for Coarse Ranking

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a best k-arm identification algorithm to identify the k-examples that needs to be added into the context of a LLM such that it maximizes the downstream performance. They use GSM8K and Aqua datasets to show decent improvements over other state of the art example selection algorithms. They use relatively smaller/older models such as llama2-7b and mistral 7b as well as newer models such as gpt-4o mini.

### Strengths
+ The paper provides a reasonable algorithm and the improvements over other SOTA methods are substantially higher that its unlikely to be noise.
+ The results are well supported by real-world experiments on real-world models.
+ There is extensive discussion about reusing the examples from one LLM to another LLM and thus verifying the transfearability of these methods.

### Weaknesses
- The paper only compares against other ICL methods. It would be nice to know what happens when compared against full mode finetuning. Especially for models like llama2-70B this is important, since its open model and can potentially be fully FT-ed on the training data. 
- There is no sufficient information about contamination of the training data with evals. In particular, where exactly does the improvements come from: is it the algorithm, is it that in the experiments the learning algorithm learns to identify examples in training data that is present in the test set, or something else? There is very little details and discussion about the exact data setup in the paper.

### Questions
- Could you please help address both my questions above?

### Soundness
3

### Presentation
3

### Contribution
2
