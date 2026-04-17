# Random Scaling of Emergent Capabilities

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Language models famously improve under a smooth scaling law, but some specific capabilities exhibit sudden breakthroughs in performance.
While advocates of ``emergence" view breakthroughs as unlocked capabilities, others attribute them to thresholding effects on noncontinuous metrics. We propose that breakthroughs are instead driven by continuous changes in the \textit{probability distribution} of training outcomes when performance is bimodally distributed across random seeds. In synthetic length generalization tasks, we show that different random seeds can produce either highly linear or emergent scaling trends. We reveal that sharp breakthroughs in metrics are produced by underlying continuous changes in their distribution across seeds. In a case study of inverse scaling, we  show that even as the probability of a successful run declines, the average performance of a successful run increases monotonically.
We validate our distributional scaling framework on realistic settings by measuring MMLU performance in LM populations. Our observations hold true even under continuous loss metrics, confirming that random variation must be considered when predicting a model's performance from its scale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents the observation of the bimodal distribution of performance of models trained under different random seeds. Some seeds show emergent abilities, while some seeds display smooth improvement. The bimodal distribution can be used to explain some training dynamics. For example, the emergent abilities of LLM might be a manifestation where a smaller model falls into the left side of the distribution (bad performance) and a larger model falls into the right side (good performance). Also, the minimum capability can be identified when the performance distribution goes from unimodal (all on the left side) to bimodal. Experiments are conducted on two synthetic algorithmic tasks, MMLU, and CSQA.

### Strengths
1. The proposed observation of bimodal distribution is interesting and makes sense as a potential explanation for emergent abilities.
2. The emergence from unimodal to bimodal distribution as a sign of possessing minimum capability is an interesting and well-explained observation.
3. The paper is clear and easy to follow.

### Weaknesses
1. I would suggest changing the title of Section 2 from "Experiment" to "Experimental Setup." You only introduce the setup there.
2. Typos in lines 246-247: "we see that the probability (Figure 3.2 (bottom left) and mean (bottom right) of such “successful”." Throughout the paper, you seem to regard Figures 3 and 6, which have 4 subfigures, as being displayed as a 2*2 layout.
3. In lines 359 & 368, Figure 3.5 is mislinked to Figure 6.
4. In line 414, incorrect citation format. ("...process the multiple-choice format Hu and Frank (2024).")
5. Results of the synthetic task in Section 3 may not transfer well to benchmark datasets. Specifically, the experiments of MMLU and CSQA do not convince me:
    (1) The emergent abilities of MMLU happen at the emergence threshold, where model performance rapidly increases from 25% acc to 40% or so, as displayed in Figure 1 of [a]. In contrast, your experiments are on small models that are before the emergence threshold.
    (2) Figures 8 and 15 do not show a clear bi- or multi-modal distribution.

My understanding is that a slight perturbation on model weights (through different randomizations) affects its performance, leading to a steeper/flatter performance gap between two adjacent models. However, the effect of perturbation is eliminated as the model size grows (trained on more samples).

6. I would suggest that authors discuss this work's limitations in an independent paragraph or section.

a. [U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models](https://openreview.net/forum?id=jjfve2gIXe)

My main concern is 5.; MMLU and CSQA do not exhibit clear bimodal distribution in my opinion.

### Questions
1. [a] seems relevant to some of your arguments, such as the observation in Section 3.5 ("Competing solutions can lead to either monotonic or U-shaped trends in emergence likelihood."). They found that emergent abilities can be decomposed into a U-shaped trend and a double descent trend, which cancel out each other before the emergence threshold.

2. Can you explicitly state the formulas of overall mode, overall mean, success probability, and mean of successful runs in Figures 3 and 6?

3. Do you have any hypotheses for the cause of bimodal distribution?

4. Some works [a, b] argue the predictability of emergent abilities. Does this observation provide any new insights to help answer this long-lasting debate?

a. [U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models](https://openreview.net/forum?id=jjfve2gIXe)

b. [Predicting Emergent Capabilities by Finetuning](https://arxiv.org/abs/2411.16035)

### Soundness
2

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
3

### Summary
Instead of studying emergent capabilities using a single training run or average of a few runs, this paper studies emergent capabilities using ~200 training runs of different random seeds. The authors attributed that "breakthroughs are instead driven by continuous changes in the probability distribution of training outcomes when performance is bimodally distributed across random seeds".

### Strengths
The paper studies emergent capabilities from a novel perspective, from a distributional perspective of many training runs instead of a single training run. This is important and helpful because neural network learning is inherently stochastic.

### Weaknesses
1. I could not agree with the paper's explanation that emergent capabilities are driven by the binomial distribution in capabilities, that "This variability is precisely what causes some model runs to appear as breakthroughs while others follow a more linear progression." I believe the causality should be the other way around. Some training runs show breakthrough, so that the capability performance improves abruptly from one mode to another. And other training runs show linear improvement. When these two kinds of training combine, they give the multi-modal distribution shown in Figure 2.

For example, thresholding is a mechanism that gives rise to discontinuity and that might cause emergent capability. I understand this as a valid cause driving emergent capability. But I believe how the distribution of capability performance changes during learning is more of a result/consequence of linear/emergent learning, rather than as cause/driving factor.

I acknowledge that the distribution of different training runs, rather than one single training run, is worth study regarding emergent capability. My concerns are regarding the causality.

2. The authors conduct experiments using reinitializations rather than training from scratch for computational cost constraints. Reintializations involves reinitialize the final attention layer and the subsequent LM head, while keeping most other layers as trained. I believe this experimental setup differs significantly from a from-scratch training, and might change the learning behaviour. Could the authors provide evidence that support the eligibility of such approach for studying emergent capabilities? for example, are there other works on studying emergent capabilities that use similar reinitialisation instead of training from scratch?

3. The Section 4.3 explains Figure 8 as roughly bi-modal. I think it is ambiguous from reading the figure alone. It also looks reasonable to me that the MMLU ratio = 7.5% figure and the MMLU ratio = 20% figure are unimodal. Could the authors more rigorously use a standard to test whether they are bimodal, for example statistical tests?

4. Finally, I feel the paper is quite dense in terms of experiments and explanations and I feel personally challenging to grasp the main take-away. The authors are encouraged to improve the ease of reading.

### Questions
See above

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
2

### Summary
This paper studies "emergent" capabilities--sharp breakthroughs/increases in model performance at a sufficiently large model size--and provides empirical support for the view that model performance across scales is bimodally distributed across random seeds. Through experiments on both synthetic and natural language tasks (MMLU), they show that some seeds exhibit linear trends, while others show emergent trends; they argue that when reporting results on individual seeds, results are likely skewed to the modes of the underlying distributions of model performances at scales. They find that these results hold for both discrete and continuous metrics (loss).

### Strengths
- The paper provides a careful conceptual lens through which to view the well-studied phenomenon of emergent capabilities. This viewpoint is empirically well-supported.
- Experiments are extensive, showing the robustness of results for a wide range of metrics (continuous vs discrete, mode vs mean) across seeds and datasets (synthetic and real-world).

### Weaknesses
- Potential for impact: Although the finding that not all individual seeds themselves exhibit non-linearity in emergent capabilities is interesting, it is not clear what the impact of the empirical findings in the work are. If what appears as emergence is that the mode of the performance distributions sharply increases, is this not a form of emergence? What are the implications of this work for how we study and evaluate models?
- Some analysis decisions are arbitrary: For example, why is 20% exact match accuracy used as the threshold for success in Section 3.2?
- Limited models: All analyses on real-world tasks are with Qwen models, not models from other families.

### Questions
Minor Notes:
- Figure 3 is referred to as Figure 3.2 in the paper, but there is only a single figure. Line 247: It is also unclear to me what "bottom left" and "bottom right" mean here.
- Line 247: What do depths 2 and 3 correspond to in the figure?

### Soundness
3

### Presentation
3

### Contribution
2
