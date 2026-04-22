# Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Inference-time scaling techniques have shown promise in enhancing the reasoning capabilities of large language models (LLMs). While recent research has primarily focused on training-time optimization, our work highlights inference-time reward model (RM)-based reasoning as a critical yet overlooked avenue. In this paper, we conduct a systematic analysis of RM behavior across downstream reasoning tasks, revealing three key limitations: (1) RM can impair performance on simple questions, (2) its discriminative ability declines with increased sampling, and (3) high search diversity undermines RM performance. To address these issues, we propose CRISP (Clustered Reward Integration with Stepwise Prefixing), a novel inference-time algorithm that clusters generated reasoning paths by final answers, aggregates reward signals at the cluster level, and adaptively updates prefix prompts to guide generation. Experimental results demonstrate that CRISP significantly enhances LLM reasoning performance, achieving up to 5% accuracy improvement over other RM-based inference methods and an average of 10% gain over advanced reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper first presents observations regarding failure modes of inference time methods that use reward models (RM): (1) RM method perform can produce worse performance on easier questions defined based on whether the model can already likely answer them, (2) RMs' tendency to attribute higher scores to rate, incorrect responses, and (3) excessively high search diversity may lead to worse RM performance.

Based on these observations, the paper proposes a method (CRISP) that (a) generates reasoning paths and cluster them based on answers, (b) aggregates reward signals based on clusters and selects the highest score paths from the top cluster to generate prefixes for the next iteration, (c) terminates the iterations early if only a small number of clusters are left. The paper empirically shows that CRISP outperforms other RM methods on various reasoning benchmarks.

### Strengths
- The paper is well-written, with observations on key issues around RM-based inference time methods first presented before outlining the CRISP methodology with components that attempt to address each of these key issues: (1) Early termination addresses poorer performance on easy questions, (2) Cluster aggregation is motivated to mitigate the inverse long-tail issue, and (3) path generation attempts to constrain search diversity. 

- The paper addresses an important problem of how to better design performant inference time methods while also reducing token costs.

- The paper presents extensive experimental results, with empirical studies on various failure modes of RMs, comparisons with different RMs, base models and benchmarks.

### Weaknesses
- The paper has relatively weak theoretical justification. While the paper attempts to motivate the method's various components loosely by linking them with the observed empirical observations, the design of CRISP remains largely heuristics-based.

- The authors may want to justify some of their experimental design more clearly on why the results are meaningful. 
    - For example, for assessing question-difficulty they used the model's own responses to bin the question. As the 'easier' questions were already found to be answerable by the model itself, it is no surprise that applying additional RM methods will not result in better performance. 
    - The inverse long-tail point could be better substantiated and made clearer, especially if it is a central claim of the paper. It is not clear whether Fig.5 is sufficient evidence. If the claim is that high rewards of RMs should be distrusted when sample frequency from the base model is low, this seems counter-intuitive since the sample frequency depends on the base model being queried rather than be a property of the RM. Showing that among high RM score samples, frequency of the samples is a good classifier for actual accuracy, could  potentially help, though there should be more thorough analysis. 
    - For the illustration on accuracy versus coverage, without normalization it may not be surprising that the count of the number of times 'correct-to-wrong' transitions occur is increasing with N. Directly assessing diversity metrics of sampled responses and performance may be more meaningful too.

- The performance gains presented (e.g. Table 1 and 2) sets to not be that large for many cases (especially given the lack of error bars). The authors may want to discuss some of these results more thoroughly. Table 2 seems to indicate that non-math tasks may have different performance trends, but benchmark method results were not presented for them in Table 1. 
   - The performance gains should be contrasted with the additional computational costs. E.g. CRISP takes close to 60% more time compared to BoN for GSM8K but seem to achieve the same performance (0.95 accuracy).

- For cost comparisons, an important factor is whether the proposed method scales well. The authors could consider including results on whether the proposed method scales well over N and number of iterations, compared to other baseline methods.

- The proposed approach combines the benefits of 'self-consistency'-type sampling-based approaches and RM scoring, but there may be issues related to it that the paper could more fully analyze especially given the weak theoretical justifications. For example, given two clusters, (1) with many samples, each with middle-level RM scores, and (2) with fewer (but still not rare) samples each with very high RM scores, the proposed approach may select (1) instead of (2) which intuitively may not be the right approach. The specific CRISP approach may seem sensitive to such thresholding. Exploration of different aggregation strategies within the framework could be useful.

- The paper could benefit from a fuller discussion of related work, to contrast its contributions to existing work.  For example, the authors could be more explicit in discussing among the key failure modes of RM methods, whether there have already been existing work that have investigated similar problems and have proposed methods [1,2]. There could also be comparisons/discussion against works that optimizes prompts/prefixes to improve sampling diversity of reasoning paths, including those that work with RMs [3,4].



[1] Chen et al, When is Tree Search Useful for LLM Planning? It Depends on the Discriminator

[2] Stroebl et al, Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers

[3] Hu et al, Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning Tasks

[4] Du et al, Improving factuality and Reasoning in Language Models through Multiagent Debate

### Questions
Please see questions already embedded in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper identifies several issues in RM-based inference-time scaling, proposes a new scaling algorithm, and validates its effect through empirical studies.

### Strengths
1. The observations in limitation of BoN are solid.
2. The proposed algorithm is strong and efficient.

### Weaknesses
1. Limited insights into the observed flaws of BoN.
1. Limited analysis on the proposed algorithm.

### Questions
1. Weakness 1. Can you explain possible reasons why the 3 issues happen? Which of them are inherent in reward modeling, and which of them are tractable (but not yet tackled) flaws?
2. Weakness 2. Are there evidences other than the ablation studies that validate the effectness of the proposed algorithm?
3. Can similar issues be observed in general tasks?

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
3

### Summary
This paper systematically diagnoses three major failure mechanisms of reward models (RM) during inference: RM degrades performance on simple problems, RM misjudges solutions as sampling increases due to an "inverse long-tail" phenomenon - where low-frequency incorrect answers receive high scores, and RM loses discriminative power under high search diversity. To address these issues, the paper proposes the CRISP framework, which clusters solutions by final answer and aggregates cluster-level rewards to suppress low-frequency misleading samples, controls search diversity through progressive prefix extraction, and introduces an early-stopping mechanism for simple problems. Experiments demonstrate that CRISP delivers stable performance improvements and higher efficiency (in both token and runtime cost) over baselines such as BoN and MCTS on benchmarks including MATH-500 and GSM8K.

### Strengths
The paper conducts an in-depth diagnosis of three core failure modes of reward models (RM) during inference - the "inverse long-tail" phenomenon, diversity-induced degradation, and failure on easy problems. The proposed CRISP algorithm tightly aligns its design components (cluster-level aggregation, early stopping, and prefix-based control) with these diagnoses ("inverse long-tail / diversity harm / easy-problem failure"), resulting in a novel yet pragmatic approach. The five-stage pipeline is clearly decomposed and highly modular, making it friendly for future engineering integration. The authors further validate the method across multiple models and datasets, thoroughly covering common baselines such as BoN, MCTS, Beam Search, and Self-Consistency.

### Weaknesses
The most glaring issue with the paper is its disorganized structure. The overview of the core method (Figure 8) is placed on page 6, well after the methodology section; meanwhile, the related work section (Section 5) appears after the experiments, which severely hinders the reader's ability to understand the CRISP framework and situate its novelty. Although CRISP relies on clustering by final answer, the paper does not clearly explain how this strategy generalizes to open-ended long-form or multi-solution tasks, nor does it discuss system robustness and failure modes when the RM is distribution-mismatched with the target domain.
On the experimental side, the paper lacks statistical significance analysis: key performance tables (e.g., Tables 1 and 2) report only single-run results, without mean and standard deviation over multiple runs to support claims of stability. In addition, CRISP introduces a substantial number of new hyperparameters (e.g., cluster threshold, prefix steps, per-round sample size n), yet the paper provides no sensitivity analysis or tuning methodology. Finally, although the paper compares token usage and runtime, it does not provide a systematic (compute, latency) vs. accuracy trade-off curve, making it difficult to evaluate realistic deployment trade-offs under different resource budgets.

### Questions
1、Could the authors explain why the core flowchart (Figure 8) is placed on page 6 rather than adjacent to the method description in Sections 2 or 3? Presenting it earlier would help clarify the core framework pipeline. Likewise, why is the related work section (Section 5) positioned after the experiments?
2、Could the authors supplement the key tables and curves with results over ≥5 random seeds, reporting mean ± variance / confidence intervals, and additionally provide the flip-rate as a function of the sample size n?
3、For non-multiple-choice or long-form generation tasks, how is equivalence between answers determined automatically? Is there a robust solution involving "scorer + normalization + semantic entailment," and can the authors provide an analysis of failure modes?
4、How sensitive is CRISP's performance to the prefix length k and per-iteration sample size n? Could the authors provide a default configuration card (recommended n, cluster threshold, prefix steps, temperature, and number of rounds for different models/tasks) as well as Compute-Return curves to facilitate practical reproduction?

### Soundness
2

### Presentation
2

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
The paper investigates the main factors that hinder the performance of reward-model-based reasoning algorithms, including diversity, number of samples and difficulty. Guided by the findings, they propose a new inference time reasoning algorithm CRISP. The method uses a combination of early stopping, output clustering and aggregation, and prefixing, and is show to outperform baselines in terms of accuracy and cost.

### Strengths
1. The new method CRISP shows a strong performance-per-token performance compared to baselines.
2. Experiments are diverse and consider both accuracy and the cost 
3. Some useful insights in the investigation of RM weaknesses. I have some criticisms which I will mention in the next section.

### Weaknesses
**W1**) In my opinion, the main weakness of the paper is the investigations of RM limitations. While there are some insights, I believe we should be more careful in drawing conclusions. Here are my thoughts on each claim.

Effect of question difficulty: I don’t have any issues with this one. I think it is a cool observation and nicely motivates early stopping. One detail that is important to be accurate is that “simple” is defined based on the generative model accuracy. If we had defined it by reward model’s accuracy, the result would be different. Perhaps saying “problems easy for the generative model” is more accurate. 

Sampling number: I found the statement to be too vague and hard to find a reasonable takeaway. The claim is against the fact that the performance of most algorithms increase with more samples. There are many ways to measure the effect of number of samples and the measurement in Fig 4 seems extremely unnatural to me. By definition, the introduced metric is increasing as a function of N. Why not just look at final performance of the algorithm, or fraction of correctly judged correct-incorrect output pairs?

Inverse Long-tail Phenomenon: I think the methodology to show this is flawed. The histogram shows that usually, the high scoring incorrect answer has low frequency. It is the concluded that “ RMs struggle to correctly score incorrect responses with low occurrence frequencies”. Another reason for the histogram could be that usually, all incorrect answers have low frequency. In that case, the observation has nothing to do with reward models and is just about the generative models. 

Search Diversity: The term diversity is very loosely used here and the intended meaning is not clear. In this section we focus on the temperature that is much more specific. These results are used to motivate CRISP path generation mechanisms to limit diversity, but I find the connection to be weak. Also the statement might contradict [1] (I am not aware of details though)

**W2**) Discussion on hyperparameter selection: There is no discussion on the impact of max steps m, sampling numbers n, and k. Plots  that show the scaling of the algorithm compare to other methods would be nice. 


**W3)** I could not find the comparison to BoN and MCTS on common sense, social, and logical reasoning. 

[1] Representation-Based Exploration for Language Models: From Test-Time to Post-Training. Jens Tuyls, Dylan J. Foster, Akshay Krishnamurthy, Jordan T. Ash

### Questions
1. Could you please formalized what you mean by “normalizing” rewards both for aggregation in CRISP and weighted BoN? I find it non trivial and perhaps important.

### Soundness
2

### Presentation
3

### Contribution
2
