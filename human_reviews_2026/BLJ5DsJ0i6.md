# GraphPFN: A Prior-Data Fitted Graph Foundation Model

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Graph foundation models face several fundamental challenges including transferability across datasets and data scarcity, which calls into question the feasibility of graph foundation models at all.
However, despite similar challenges, the tabular domain has recently witnessed the emergence of the first successful foundation models such as TabPFNv2 or LimiX.
Many of these models are based on the prior-data fitted networks (PFN) framework, in which models are pretrained on carefully designed synthetic datasets to make predictions in an in-context learning regime.
Recently, G2T-FM has made the first step towards adopting PFNs for graph tasks, yet it is limited to hand-crafted features and was never pretrained on graph data.
In this work, we make the next step by proposing GraphPFN, a PFN-based model designed and pretrained specifically for graphs.
Following the PFN framework, we first design a prior distribution of synthetic attributed graphs by using a novel combination of multiple stochastic block models and a preferential attachment process for structure generation and graph-aware structured causal models for attribute generation.
Then, we augment the tabular foundation model LimiX with attention-based graph neighborhood aggregation layers and train it on synthetic graphs sampled from our prior.
On diverse real-world graph datasets with up to $50{,}000$ nodes, GraphPFN shows strong in-context learning performance and achieves state-of-the-art results after finetuning, outperforming both G2T-FM and task-specific GNNs trained from scratch on most datasets.
More broadly, we hope that GraphPFN shows the potential of PFN-based models for building graph foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GraphPFN, which augments the tabular FM LimiX with attention-based message-passing adapters and pretrains them (PFN style) on synthetic graphs drawn from a novel prior that mixes degree-corrected SBMs, preferential attachment (to induce core-periphery), and graph-aware SCMs (MLP/GNN-mixed, optional LapPE). It shows promising ICL on some datasets and SOTA after finetuning on graphs up to ~50k nodes, though several settings hit OOM and TFMs’ class-count limits remain.

### Strengths
1. The paper proposes a clear architectural idea: lightweight graph adapters (masked attention over adjacency) added to each LimiX block; retains PFN ICL while enabling learned structural reasoning.
2. The paper combines multi-SBM with preferential attachment to mimic community + core-periphery; attributes from graph-aware SCM (MLP/GNN mix, optional LapPE).
3. The pretraining scale is large with 4M synthetic samples and 500k steps.

### Weaknesses
1. The efficiency is still the bottleneck. The model processes the whole graph jointly; OOM prevents FT on multiple datasets and the paper itself flags poor scaling. Provide rigorous memory/throughput curves and practical remedies.
2. Task breadth is narrow. Despite an MGM head, the method remains node-only; no on-main link/graph-level results, so the “foundation” scope is limited.
3. The baselines are limited. Not enough recent graph transformers or foundation models are compared. This could make the sota claim less effective.
4. ICL trails G2T-LimiX on heavy, structurally different graphs (e.g., hm-prices, city-roads-M), and authors attribute this to prior mismatch—but evidence and how to solve the mismatch is lack of discussion.

### Questions
1. Can the pretrained model be extended to other graph-related tasks? And what kind of changes are needed for the pretraining datasets and methods.
2. Can you integrate sampling (random walk) or chunked/sparse attention to avoid whole-graph processing?
3. What is performance vs. number/selection of labeled context nodes and class imbalance?
4. Could you please ablate MGM weight, LapPE during pretraining vs. downstream, and adapter placement per block; quantify where LapPE helps/hurts?
5. Could the author compare the method with more up-to-date baselines?

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
4

### Summary
This paper proposes GraphPFN, a graph foundation model for node-level prediction that extends the tabular foundation model LimiX by adding attention-based message-passing adapters to each transformer block. The model is pretrained on 4 million synthetic graphs sampled from a prior that combines multiple stochastic block models with preferential attachment for structure generation, and augments tabular structured causal models with message-passing mechanisms for attribute generation.

### Strengths
Motivation:  The proposed method combines multiple degree-corrected SBMs to address the "too clean" cluster problem in single-SBM generations, and the preferential attachment augmentation naturally to create the core-periphery structure in social and information networks, which sounds interesting. 

Method: Rather than training a graph model from scratch, the authors initialize from LimiX and add sparse message-passing adapters that respect the per-feature tokenization. The motivation seems reasonable since it lets the model learn what to aggregate rather than relying entirely on hand-crafted features like G2T-FM does.

Empirical results: The overall performance is not bad if fully fine tuned.

### Weaknesses
Scalability: The scalability issues are frankly concerning for something claiming to be a foundation model. GraphPFN hits out-of-memory errors on 3 of 11 datasets during finetuning, that's failing on 27% of your own benchmark. What's worse is that these failures happen at 148K-168K nodes, I don't think these graph datasets are big enough comparing with OGB datasets. While the authors acknowledge this in Appendix A, but this isn't just a limitation to note. It fundamentally undermines the claim of "foundation model"  which should be scalable. The paper needs a more concrete solution (even if preliminary).

Prior analysis: The prior coverage problem reveals a deeper issue with the core contribution. When GraphPFN performs dramatically worse than G2T-LimiX on datasets like hm-prices or city-roads-M, and the explanation is "The proposed graph prior does not cover graphs from specific domains". That's a problem with the prior design, not just an implementation detail. The manuscript provides no quantitative validation that synthetic graphs match real-world statistics like degree distribution, clustering coefficient, or assortativity. The prior appears tailored to social/information networks without demonstrated generalization to physical/spatial networks. The core claim that pretraining on synthetic graphs from a well-designed prior is effective lacks sufficient support when systematic failures occur on specific graph types.

Insufficient ablation of critical design choices: The manuscript lacks ablations for several key decisions. The rationale for adding adapters to every LimiX block rather than fewer layers is unexplored. The choice to freeze LimiX layers during pretraining instead of joint training lacks justification through empirical comparison. The individual contribution of the masked graph modeling objective versus supervised PFN loss remains unclear, and the 0.1 coefficient for MGM appears arbitrary without sensitivity analysis. Performance sensitivity to the number of pretraining graphs (4M) and steps (500K) is unexplored despite Figure 2 showing continued improvement at 500K steps. The effect of prior hyperparameters like mixing probability p or LapPE inclusion frequency lacks systematic investigation. Furthermore, the compared GNN models are just so limited and many current state-of-the-art models are not included especially graph transformers and heterophilic GNNs.

Model comparison: G2T-FM uses Laplacian positional encodings (LapPE) in all experiments while GraphPFN do not use them at all, creating uneven comparison where G2T-FM accesses global positional information unavailable to GraphPFN base variant. G2T-FM employs neighborhood-aggregated features computed via explicit aggregation while GraphPFN relies on learned message passing. These differences prevent isolating whether improvements stem from trainable message passing specifically versus superior feature engineering. Further ablation study on this aspect should be conducted.

Weak in-context learning performance without systematic analysis: GraphPFN's ICL performance is substantially worse than G2T-LimiX on multiple datasets, sometimes by large margins (e.g., hm-prices: 68.02 vs. 74.96). This indicates the model has not fully learned to perform ICL on graphs, this really undermines a key promise of the PFN framework. If ICL performance is unreliable, GraphPFN's value as a zero-shot foundation model is limited, reducing it primarily to a pretrained initialization for finetuning. The manuscript provides no systematic analysis correlating ICL success with graph properties like size, density, or community structure to understand when the approach is applicable.

Validation of graph prior quality: The manuscript lacks quantitative validation that synthetic graphs match real-world distributions. Figures 3-4 provide only qualitative visualizations without comparing metrics like degree distribution, clustering coefficient, or graph diameter against real datasets. An analysis demonstrating that real datasets fall within the statistical distribution of synthetic graphs would substantiate claims about prior quality and coverage. The absence of such validation makes it difficult to assess whether poor performance on certain datasets stems from fundamental approach limitations or simply inadequate prior design.

Presentation: The overall presentation has many problems. 1) Captions are sloppy and not informative. 2) Some terms are not defined such as "OOM" even though I understand what is stands for but it should be clearly defined. 

Overall, I feel that this paper is only half-finished, as many important aspects, especially those related to scalability, are not thoroughly addressed. More importantly, while the manuscript claims to focus on graph foundation models, it only evaluates node-level tasks, which makes its scope far too narrow.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose GraphPFN, a method for adapting pre-trained tabular foundation models (TFM) to graph data by augmenting a TFM with a message-passing layer, and pre-train it on a large variety of synthetic graphs with causal, structurally-correlated node features. They demonstrate that GraphPFNs perform largely on par with or better than G2T-FMs across tasks in both in-context learning and fine-tuning setting, while avoiding expensive pre-processing required by G2T-FMs.

### Strengths
1. The core idea is interesting and potentially impactful. Leveraging pre-trained tabular foundation models (TFM) for GFMs is a sensible idea.
2. The integration of the graph adapters is a smart way of injecting structural inductive biases into TFMs effectively, without resorting to costly hand-crafted features (the avenue taken by G2T-FMs).
3. The results are overall convincing to the extent that the resulting models can compete with G2T-FMs and GNNs.
4. The paper is well-written, despite requiring substantial background on several areas the bare minimum for the background is mostly provided, and the overall presentation is clear and coherent.

### Weaknesses
1. The graph structure generation process seems mostly guided by an “eye test” on what “realistic” graphs _should_ look like. While I appreciate the authors’ efforts to go beyond simple synthetic graphs (SBM/ER/BA graphs etc.) and provide a generation algorithm that outputs a rather rich distribution of graphs, I’m not fully convinced that pre-training only on graphs from a single family of distributions is sufficient to capture real-world graph complexity. The authors clearly acknowledge this in section 5.2 and Appendix A (which I appreciate), where this phenomenon is reflected in the limited performance of GraphPFN (ICL) on certain datasets.
2. GraphPFN results are largely _on par with_, rather than consistently beating G2T-FMs across the benchmarks tested. I acknowledge that G2T-FMs typically perform better on downstream tasks where the datasets fit the graph prior better, however. Given comparable performance with G2T-FMs, I think the advantage of GraphPFN is the lack of reliance on hand-crafted features, and as a (likely result) faster evaluation, though the authors have not conducted any experiments (see Suggestion 1).
3. There are several issues with the evaluation procedures:
   - The GNN benchmarks are not provided with LapPE features, unlike GraphPFN,  resulting in an unfair comparison (see Suggestion 2).
   - Another factor that prevents fair comparison is that no information is provided on model sizes and training/pre-processing/inference efficiency. This is especially crucual in evaluating GraphPFN since it’s clear that the model is quite large, taking 6 days with eight A100s to pre-train _only_ the graph adapter layers, and leading to OOM errors in fine-tuning on certain datasets. When discussing foundation models parameter efficiency is an important metric to pay attention to, and it seems in this paper this dimension is wholly omitted (see Suggestion 1).

### Questions
**Questions:**
1. During pre-training, all layers except the graph adapters are frozen. Was a non-frozen model considered, and would there be there any potential benefits in pre-training the whole model?
2. Given the reliance of GraphPFN on synthetic graphs in pre-training despite the need for a realistic graph prior, can we not consider leveraging existing graph datasets either instead of or alongside synthetic graphs?
3. How does GraphPFN (with/without LapPE) compare with G2T-FMs in terms of any pre-processing and inference speed? I would expect GraphPFN (without LapPE) to perform inference faster since they don’t need to precompute encodings or structural statistics.

**Suggestions:**
1. More information regarding parameter counts and training/pre-processing/inference efficiency is required to better compare the GraphPFN and the baseline models (particularly the G2T-FMs) in a fair manner. I would imagine this would also help the authors make the point that not relying on hand-crafted features makes GraphPFN faster than G2T-FMs (assuming this holds; if not, please explain why).
2. Considering that some GNN benchmarks already outperform GraphPFN (ICL, without LapPE) on some datasets (`hm-prices`, `city-roads-M`, `questions`), GNN results with LapPE would provide a fairer comparison against stronger benchmarks; I suggest the authors provide them as well. Note that I am aware that the GraphPFN (ICL) results are with no additional training on the downstream dataset, so comparable results against GNNs are already impressive in their own right.
3. Referring to the UniMP model (Shi et al., 2021) as GT is confusing, given that acronym’s use for _graph transformers_ which typically leverage global attention over the graph. A minor suggestion is to use the UniMP convention.

**Conclusion:** I am very on the fence with this paper; the proposed idea is nice and seems to be implemented well. However, it seems to me the authors have failed to capitalize on the proposed method’s key advantage over G2T-FMs in not relying on costly hand-crafted features, and with performance against the baseline models being rather on par rather than a clear win, it’s hard to evaluate the true potential impact of the paper. I therefore recommend a tentative acceptance, and encourage the authors address the aforementioned concerns to arrive at a stronger paper.

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
4

### Summary
This paper introduces GraphPFN, a prior-data fitted network for node-level prediction that builds graph foundation models (GFMs) through synthetic pretraining. It generates diverse, realistic graphs using a hybrid of stochastic block models and preferential attachment, combined with graph-aware causal models for attributes and targets. By augmenting the tabular foundation model with attention-based neighborhood aggregation and training on these synthetic graphs, GraphPFN achieves strong performance.

### Strengths
1. This paper studies an interesting and practical research question of applying synthetic data in large-scale pretraining.
2. The proposed data generation pipeline seems reasonable.

### Weaknesses
1. The writing of this paper is confusing and the core contribution of the work is not well presented. In the introduction (line 066), the authors mentioned "but such models still depend heavily on hand-crafted features and, as a result, are limited in their ability to capture complex graph patterns." This challenge is addressed by the proposed message-passing layer which replaces the structural features. However, the motivation of utilizing synthetic data pretraining is not discussed, which has no clear connection to the challenges mentioned. 
2. Efficiency concern: the proposed method builds on existing TFMs and requires post-training involving massive compute resources(8*A100*6days). At inference time, it involves dataset-specific fine-tuning to achieve the claimed SOTA performance. Given the lack of efficiency comparison, I doubt the real gains of the proposed method over baselines. 
3. Misleading experimental results: Table 2 compares GraphPFN with baselines. However, the gains of GraphPFN are inconsistent, with OOM issues which were not observed in baselines. Since a core edge of GraphPFN is that it does not rely on hand-crafted features, it is not reasonble to include the results for the LapPE enhanced version. As a result, the original version of the proposed method is less competitive, especially when compared with G2T. 
4. The data generation process involves many hyperparameters. Though the authors claim that they can be sampled from some distribution, the details are missing and the method still heavily relies on trial-and-error, limiting its real-world applicability.

### Questions
1. What is the initial motivation of applying synthetic data pretraining? Can it be replaced by real-world data, or a mix of both? 
2. How does the properties and amounts of pretraining data affect performance?

### Soundness
1

### Presentation
2

### Contribution
1
