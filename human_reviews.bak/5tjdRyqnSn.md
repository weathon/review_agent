# PhiNets: Brain-inspired Non-contrastive Learning Based on Temporal Prediction Hypothesis

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Predictive coding has been established as a promising neuroscientific theory to describe the mechanism of information processing in the retina or cortex.
This theory hypothesises that cortex predicts sensory inputs at various levels of abstraction to minimise prediction errors.
  Inspired by predictive coding, Chen et al. (2024) proposed another theory, temporal prediction hypothesis, to claim that sequence memory residing in hippocampus has emerged through predicting input signals from the past sensory inputs.
  Specifically, they supposed that the CA3 predictor in hippocampus creates synaptic delay between input signals, which is compensated by the following CA1 predictor.
  Though recorded neural activities were replicated based on the temporal prediction hypothesis, its validity has not been fully explored.
  In this work, we aim to explore the temporal prediction hypothesis from the perspective of self-supervised learning (SSL).
  Specifically, we focus on non-contrastive learning, which generates two augmented views of an input image and predicts one from another.
  Non-contrastive learning is intimately related to the temporal prediction hypothesis because the synaptic delay is implicitly created by StopGradient.
  Building upon a popular non-contrastive learner, SimSiam, we propose PhiNet, an extension of SimSiam to have two predictors explicitly corresponding to the CA3 and CA1, respectively.
  Through studying the PhiNet model, we discover two findings.
  First, meaningful data representations emerge in PhiNet more stably than in SimSiam.
  This is initially supported by our learning dynamics analysis: PhiNet is more robust to the representational collapse.
  Second, PhiNet adapts more quickly to newly incoming patterns in online and continual learning scenarios.
  For practitioners, we additionally propose an extension called X-PhiNet integrated with a momentum encoder, excelling in continual learning.
  All in all, our work reveals that the temporal prediction hypothesis is a reasonable model in terms of the robustness and adaptivity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces PhiNet and X-PhiNet, where the input is processed in three parallel pathways with 0,1,2 predictors and with two pathways having the stop-gradient operation. The authors report favorable results over SimSiam and explore online/continual learning effectiveness in their models.

### Strengths
The idea is well motivated and the paper is thoroughly cited. The analysis is rigorous and convincing.

### Weaknesses
The majority of my weaknesses stem from my lack of understanding about certain claims, and I believe that this paper will be improved if the following points are made clearer:

- Can the authors explicitly explain, with specific reference to any important terms, why stop-gradient leads to any temporal lagging? I understand how stop-gradient works, but I fail to see how it is related to time in any way.

- I am not able to see any claims in section 5.1 in Figure 5, even though it is referenced in the text. What kind of improvement am I supposed to see and what effect am I looking for?

I will update my review once I gain a better understanding of these results.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This papers takes another step in building non-contrastive self-supervised learning algorithms by taking inspiration from the architecture and supposed function of the hippocampal circuit. This PhiNets presented in this work are an extension of the SimSiam algorithm and the authors compare their method extensively to SimSiam, showing that it outperforms SimSiam in terms of robustness of low weight decay settings and continual learning. Their algorithm is competitive with many well-known contrastive and non-contrastive SSL methods (BYOL, Barlow Twins, etc.). Taken together, this is a nice paper showcasing the inspiration-from-neuroscience to usefully-deployed-ML-algorithm possibility.

### Strengths
Originality
- As far as I could tell, their conversion of a hippocampal circuit process to a ML algorithm is novel. It does go beyond the closest match, SimSiam, in both architecture and abilities.

Quality
- In addition to empirical results, the analytical treatment is super useful to make us understand *why* their algorithm might outperform SimSiam in terms of robustness to low weight decay (among other things). 

Significance
- As I wrote in the summary, "this is a nice paper showcasing the inspiration-from-neuroscience to usefully-deployed-ML-algorithm possibility."

### Weaknesses
I have two core concerns:
1. The authors link their algorithm to the temporal prediction hypothesis. While I can see how in principle their setting (with image augmentations and the StopGradient) is similar to what temporal prediction would entail, results from this project cannot be used easily to validate their success as a success of temporal prediction SSL solely due to the reason that they do not use a temporal sequence. Yes, in a temporal sequence like a video subsequent frames could be thought of as augmentations but it is unclear if this algorithm will necessarily generalise to those settings because it is unclear if the class of augmentations used here resemble "natural" augmentations in videos.
2. Keeping the hippocampal inspiration aside for a moment, the addition of g and Sim-2 is what distinguishes this algorithm from SimSiam. However, Fig.5. suggests that setting g=I doesn't influence the performance too much - it would be good to know if the continual learning results and other results do require that g is trainable and not I. Similar concern for Sim-2 - what happens when Sim-2 is left out? These ablation analysis are important to understand what makes PhiNet special. Also removing Sim-2 makes the algorithm similar in essence to SimSiam - I'd be curious if this version already outperforms the original SimSiam implementation - this will help us establish a stronger "baseline" for the full PhiNet.

Clarity suggestion:
- In the analytical Section 4, it'd be great if you'd give us an intuition for what phi and gamma in Eq. 1 actually denote - how are they related to g and h, etc. This will truly help tie in the section to the rest of the paper (although it is fine as is too).

### Questions
What is the physiological relevance of low weight decay - why is that the important analysis here? (the authors spend a lot of space on it)

### Soundness
3

### Presentation
2

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
The manuscript links the temporal prediction hypothesis observed in the brain's hippocampus-cortex interactions to a novel self-supervised learning structure (PhiNets). PhiNets is largely based on SimSiam but incorporates one more predictor layer and StopGradient module inspired by neuroscience observations. This model demonstrates greater stability and resilience against representational collapse compared to SimSiam and shows superior performance in online and continual learning scenarios. Additionally, the study presents X-PhiNet, an extension incorporating an exponential moving average encoder for improved long-term memory retention in continual learning.

### Strengths
- A clever observation between the temporal prediction hypothesis and the StopGradient module, making this architecture more brain-like. However, the definition of temporal steps was poorly articulated and needs more work. (See weakness)
- Rigorously proved the benefits of adding one more predictor layer through the analysis of gradient flow in a linearized case 
- Demonstrated improved performance in terms of robustness to weight decay in terms of image classification

### Weaknesses
- Presentation of the paper needs more improvement: For example, it would be nice to add the diagram of SimSiam to Figure 1 and this is the main work the current manuscript was based on. 
- From the derivation, we know that adding one more layer of signal prediction could prevent from representation collapse. But is this going to slow down the convergence rate? What are the drawbacks of having one more layer of prediction. 
- In the performance evaluation plots: Phinet showed sufficient improvement compared to its original version SimSiam but not much improvement compared to other methods such as Barlow twins. And it's unclear why mse loss is better than cos loss in L_{NC}.

### Questions
- The weight decay term appeared out of sudden and definite needs more explanation about how it approximates modern optimizers and its relationship with learning rate etc.  because the robustness about decay is where PhiNet showed most improvements in the later numerical experiments. 

- The definition of time steps in the network and its link to temporal prediction is not well articulated. I roughy get the general idea how StopGradient could lead to synaptic delays but how are these two linked mathematically is still vague. Does one time step refers to one gradient update? Is gradient update performed in an online manner?

### Soundness
3

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
In this paper, the authors propose PhiNet, a self-supervised learning (SSL) method partly inspired by the temporal prediction hypothesis and a previous SSL method SimSiam. PhiNet can be seen as an extended version of SimSiam with an additional predictor and a stable encoder which is an exponential moving average of the main encoder. The authors analyzed the learning dynamics of PhiNet in a linear setting and found that the network is less prone to collapse compared to SimSiam. The method is then evaluated using the CIFAR benchmarks and is shown to outperform classical baselines in SSL, especially in online learning and continual learning settings.

### Strengths
1. The hypothesized correspondence of the algorithm to the learning and memory mechanisms in the brain is interesting to both ML and neuroscience communities.
2. The authors conducted a detailed analysis of the learning dynamics which nicely explains the advantage over the baseline.
3. There are extensive experiments that clearly illustrate the strengths and weaknesses of the models.
4. The presentation is clear and a lot of details can be found in the appendix.
Overall this is a solid and interesting contribution that bridges predictive coding and SSL algorithms known to the ML community. I thus support acceptance provided that the authors can reasonably address my concerns.

### Weaknesses
1. While this is understandable for a more brain-inspired algorithm, there is little performance gain on classical SSL benchmarks.
2. The link from the model and task setting to the temporal predictive coding idea is not that strong, see below.

### Questions
1. One major concern is that the proposed method is not really doing "temporal predictive coding", instead, the predictor is predicting the embedding of an augmented image. The way to introduce the model is thus somewhat misleading. It also seems more natural to use some video datasets if the goal is to build a model for temporal predictive coding. I wonder if the authors can explain more about the motivations here.
2. When connecting X-PhiNet to the brain as in Fig 1(c), we are basically assuming that layers II and III in EC share the same encoder and the NC encoder is EMA of the one in EC. I wonder if the authors can provide more rationale for these assumptions.
3. While there are some qualitative comparisons of the learning dynamics between PhiNet and SimSiam, I think the result would be stronger if there were also quantitative comparisons, e.g., something showing that more random seeds would result in collapse for SimSiam than for PhiNet.
4. For empirical performance, the errorbar is missing from Fig 5 and Fig 7 so it's a bit hard to measure the margin of error.
5. When comparing models on online learning and continual learning benchmarks, it seems that the EMA encoder is critical, but at least for continual learning, this is not very surprising since EMA essentially prevents the representations from drifting away too much. As baselines for comparison here (POYO, SimSiam, etc) are not designed for continual learning, it might make sense to include more baselines from the field of continual learning, otherwise, the claim that X-PhiNet particularly excels at online and continual learning seems a bit unfair.

### Soundness
2

### Presentation
3

### Contribution
2
