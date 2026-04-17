# Mixture of Heterogeneous Grouped Experts for Language Modeling

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Mixture-of-Experts (MoE) offers superior performance over dense models. However, current MoEs impose a critical limitation by enforcing uniform expert sizes, restricting the model's ability to dynamically match computational resources with token-specific requirements. Despite several attempts on heterogeneous experts have been made, they struggle either with limited performance and inefficient parameter utilization or unbalanced GPU utilization, there is still a lack of general heterogeneous MoE architecture.
To this end, we present Mixture of Heterogeneous Grouped Experts (MoHGE), an innovative MoE architecture that introduces a two-level routing mechanism and enables more nuanced and efficient expert selection tailored to each input token's characteristics. We also propose a Group-Wise Auxiliary Loss to enhance efficient parameter utilization without compromising model performance.
To address the resulted workload imbalance challenges, we develop: (1) an All-size Group-decoupling Allocation  strategy and (2) Intra-Group Experts Auxiliary Loss, collectively ensuring balanced GPU utilization.
Extensive evaluations on multiple benchmarks demonstrate that MoHGE achieves comparable performance to state-of-the-art MoE architectures while reducing total parameter count by approximately 20\% and maintaining balanced GPU utilization. Our work establishes a new paradigm for resource-aware MoE design, better aligning computational allocation with actual inference demands.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Traditional Mixture-of-Experts LLM design assumes all experts to be of the same amount of parameters. Recent findings show that this could lead to a waste of computation, as different tokens have different difficulties, hence require different amounts of computation to process. Existing explorations in this direction either assume all experts should be treated equally during routing, or suffer from GPU utilization imbalance. The author proposes MoHGE, which introduces a two-layer routing schema, to allow better GPU utilization balance and more diverse routing. Empirical results show that the proposed method outperforms the baseline dense model and training model.

### Strengths
- The design is intuitive and well-motivated, making the core idea easy to grasp.

- The method section (sec 3) is clearly written, which is helpful for the reader to understand the approach.

- The paper provides appropriate background material, achieving a good balance between necessary context and conciseness without digressing into unrelated details.

### Weaknesses
- Insufficient experiment and baseline selection. Only the dense model and traditional MoE model was selected for comparison. In my opinion, the author should **compare against the two baselines** mentioned on line 47, which are **MoDSE and HMoE**. However, there is not a direct comparison against these two baselines in the experimental section. The main result only shows that the proposed method is better than the traditional MoE or a dense model, yet the advantage of heterogeneous experts has already been described in aforementioned prior works. Without a clear comparison against these baselines, it's hard to convince the reader that it is the proposed change that improves the performance. Specifically:
  - What if we utilize a global load balancing loss, like in MoDSE? Is there a row in table 3 that is equivalent to such a setting? If so, why is it not clearly mentioned?
  - How bad would the load balance become if the author chooses to adopt the expert setting in HMoE? How would that affect the hardware utilization in your settings? What would be the difference in terms of the **batch size without OOM/GPU utilization/per iteration latency**?

- Insufficient details regarding the hardware and other training/inference setup. The author describes the GPU used as "NVIDIA GPUs", and other details are also unclear, such as the batch size chosen for training or inference. Is any inference specific acceleration technique being used? What is the communication kernel used for MoE routing? **Code has not been provided.**

- Minor writing issues:
  - Font is way too small at many places.  This creates difficulty for the readers and please consider improving the readability. I found it very difficult to read after printing it out and I had to use a large screen to read this paper.
    - All equations
    - Figure 1 & 3 Caption
  - Line 156: incorrect linewidth.
  - Missing "," on line 297.

### Questions
See weaknesses for a plethora of questions. Besides:
- Why were two epochs being trained on the LLM? Doesn't the second epoch lead to overfitting?

- What is the inference/training time for the dense model? Given that the dense models are typically faster than MoE when the amount of parameters are equal, I wonder whether there exists a significant advantage to choose MoE model under the described scenario.

- One interesting design choice is that for each token, the final activated expert is selected top-$K_e$ experts globally, instead of always selecting a certain number of experts from each group, which theoretically has an even better hardware balance. Is there any empirical evidence for the design?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To enable models to dynamically match computing resources based on token-specific needs, the development of Heterogeneous MoE is of great significance. Existing studies on Heterogeneous MoE either suffer from limited performance and low parameter utilization efficiency, or face the issue of unbalanced GPU utilization, with a universal heterogeneous MoE architecture still lacking. This paper proposes the Mixture of Heterogeneous Grouped Experts (MoHGE) — an innovative MoE architecture that introduces a two-level routing mechanism, enabling more refined and efficient expert selection based on the characteristics of each input token. This paper also proposes the "Group-Wise Auxiliary Loss", which improves parameter utilization efficiency without compromising model performance, including the All-size Group-decoupling Allocation strategy and Intra-Group Experts Auxiliary Loss.

### Strengths
This paper proposes the Mixture of Heterogeneous Grouped Experts (MoHGE) — an innovative MoE architecture that introduces a two-level routing mechanism, enabling more refined and efficient expert selection based on the characteristics of each input token. This paper also proposes the "Group-Wise Auxiliary Loss", which improves parameter utilization efficiency without compromising model performance, including the All-size Group-decoupling Allocation strategy and Intra-Group Experts Auxiliary Loss.

### Weaknesses
The predefined grouping of experts in the authors' proposed Mixture of Heterogeneous Grouped Experts (MoHGE) leads to a significant reduction in the combinatorial diversity of experts during routing.

All comparative experiments conducted by the authors lack comparisons with existing research on Heterogeneous MoE (including HMoE), only comparing against MoE models and Dense models—among these, the comparison with Dense models is unnecessary. Furthermore, in experimental design, the authors should strive to ensure consistency in total parameters and activated parameters; currently, these two critical factors exhibit significant discrepancies, making the experiments insufficiently rigorous.

When scaling the model, will the grouping mechanism of MoHGE affect the expert parallelism (ep) strategy?

There is a lack of rigorous ablation experiments on MoHGE.

It is anticipated that refining the experimental section would result in a higher rating.

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Mixture of Heterogeneous Grouped Experts (MoHGE), a variant of the Mixture-of-Experts (MoE) framework that partitions experts into groups of varying sizes and introduces a two-level routing mechanism (group selection followed by intra-group expert routing). Two auxiliary objectives are added to encourage balanced expert usage: a Group-Wise Auxiliary Loss and an Intra-Group Experts Auxiliary Loss.
 Experiments on several language understanding and reasoning benchmarks (MMLU, GSM8K, SIQA, LAMBADA, PIQA, TriviaQA, MATH) across 1B, 3B, and 14B models show similar or slightly better accuracy compared to homogeneous MoE baselines, with roughly 20% fewer parameters.

### Strengths
- Addresses a practical bottleneck in large-scale MoE training.
- Method is straightforward and compatible with existing frameworks.
- Writing and figures are clear and professional.
- Experiments span multiple scales and include ablations on loss terms.

### Weaknesses
- Missing quantitative evidence of GPU utilization or efficiency.
- Reported improvements are small and lack statistical validation.
- No theoretical  analysis of routing dynamics or convergence.
- Experimental reporting omits critical hyperparameters and setup details.
- Contribution is incremental relative to prior heterogeneous MoE work.

---
1. Quantify “Efficiency” Claims (Page 3–4)
- Provide actual GPU load statistics, throughput (tokens/sec), and communication cost.
- Report training time per step and FLOPs vs. MoE baseline.
- Showing real efficiency data would validate the motivation and significantly raise the Quality and Significance scores.
2. Report Statistical Significance (Page 6–7, Table 1)
- Repeat experiments with multiple random seeds.
- Include standard deviation or confidence intervals.
- Apply paired t-tests to confirm that improvements are not noise.
3. Ablate Grouping vs. Routing (Page 5)
- Isolate the effect of heterogeneous expert sizing and two-level routing separately.
- Provide visualization of token-to-group assignment entropy.
- Helps clarify where gains actually come from, strengthening Originality.
4. Provide Theoretical or Analytical Justification (Page 4)
- Offer a short analysis on expected load variance under two-level routing.
- Discuss whether routing convergence differs from standard MoE gating.
5. Expand Experimental Reporting 
- Include training setup: batch size, gradient accumulation, GPU count, communication strategy, optimizer, etc.
- Report training hours and memory footprint.
6. Broaden Evaluation Scope (Page 7–8)
- Add large-scale pretraining or autoregressive datasets (e.g., C4, Pile).
- Report cost per token to demonstrate real-world scalability.
7. Tone Down Overstatements (Page 1 & 9)
- Replace “establishes a new paradigm” with more measured phrasing like “offers a simple variant that modestly improves efficiency.”

### Questions
Although the goal—improving parameter efficiency and GPU load balance in MoE systems—is relevant, the evidence provided does not convincingly support that MoHGE achieves this goal in a meaningful way.
1. The motivation is not empirically substantiated.
 The paper claims that homogeneous experts lead to “severe GPU imbalance” and “inefficient utilization,” yet no quantitative evidence (e.g., utilization variance, expert activation frequency, or throughput) is reported. The claim remains speculative.
2. Efficiency claims lack measurements.
 The reported 20% reduction in parameters simply reflects reduced hidden dimensions for some expert groups (Page 6, Table 1). There is no analysis of actual computation cost, latency, or memory use. Without such data, it is unclear whether MoHGE is truly more efficient.
3. Empirical improvements are small and inconsistent.
 Across benchmarks, improvements over MoE are ≤1 point—likely within training variance (Table 1, Page 6). On GSM8K, inference is slower. Without multiple runs or variance reporting, these gains are not statistically credible.
4. Limited analysis of mechanism and behavior.
 The two-level routing and auxiliary losses are introduced heuristically, without analyzing how they affect routing stability or specialization. Figures 2–3 show routing distributions but offer no interpretation.
5. Incremental novelty.
 Similar ideas appear in HMoE and MoDSE. MoHGE reorganizes them hierarchically but does not introduce a fundamentally new concept or analytical perspective.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new Mixture-of-Experts (MoE) architecture with different expert sizes. A two-level routing strategy is used to select experts based on task difficulty. Multiple techniques (group-wise auxiliary loss, all-size group-decoupling allocation strategy, and intra-group experts auxiliary loss) are utilized for addressing GPU load imbalance, routing imbalance issues. Extensive experiments are conducted to show the effectiveness of the proposed approaches.

### Strengths
1. The idea of the two-level routing strategy is intuitive and interesting.
2. Extensive experiments (1B, 3B, 14B models with 0.58T tokens) are conducted to show the effectiveness of the proposed methods.

### Weaknesses
1. Multiple loss functions are introduced in the pretraining stage, which makes the hyperparameter selection costly.
2. The design of group-wise auxiliary loss is not that clear to me. Why is routing to the larger experts a problem? The model performance improvement introduced by the group-wise auxiliary loss seems to be marginal. The authors claim that group-wise auxiliary loss reduces the number of activated parameters. Could the author provide more qualitative results?

### Questions
1. What is the intuitive reason for the design of equation 5?
2. What is the final loss function? Please list it explicitly.

### Soundness
3

### Presentation
2

### Contribution
2
