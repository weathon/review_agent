# PRES: Toward Scalable Memory-Based Dynamic Graph Neural Networks

- Decision: Accept (poster)
- Scores: 6, 6

## Abstract
Memory-based Dynamic Graph Neural Networks (MDGNNs) are a family of dynamic graph neural networks that leverage a memory module to extract, distill, and memorize long-term temporal dependencies, leading to superior performance compared to memory-less counterparts. However, training MDGNNs faces the challenge of handling entangled temporal and structural dependencies, requiring sequential and chronological processing of data sequences to capture accurate temporal patterns. During the batch training, the temporal data points within the same batch will be processed in parallel, while their temporal dependencies are neglected. This issue is referred to as temporal discontinuity and restricts the effective temporal batch size, limiting data parallelism and reducing MDGNNs' flexibility in industrial applications. This paper studies the efficient training of MDGNNs at scale, focusing on the temporal discontinuity in training MDGNNs with large temporal batch sizes. We first conduct a theoretical study on the impact of temporal batch 
size on the convergence of MDGNN training. Based on the analysis, we propose PRES, an iterative prediction-correction scheme combined with a memory coherence learning objective to mitigate the effect of temporal discontinuity, enabling MDGNNs to be trained with significantly larger temporal batches without sacrificing generalization performance. Experimental results demonstrate that our approach enables up to a 4 $\times$ larger temporal batch (3.4$\times$ speed-up) during MDGNN training.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to offer a scalable training method for Memory-Based Dynamic Graph Neural Networks (MDGNNs) by mitigating the temporal discontinuity issue, thus training MDGNNs with large temporal batch sizes. It consists of two main contributions: 1) conducting a theoretical study on the impact of temporal batch size on the convergence of MDGNN training, and 2) proposing PRES based on the theoretical study, an iterative prediction-correction scheme combined with a memory coherence learning objective to mitigate the effect of temporal discontinuity. The evaluation shows that the proposed approach enables up to 4X larger temporal batch sizes and achieves up to 3.4X speedup during MDGNN training.

### Strengths
+ It targets an emerging and important GNN, MDGNNs, and the proposed designs generally make sense. 
+ The problem definition is easy-to-follow.
+ The introduced concept of memory coherence is interesting.
+ The code is publicly available.

### Weaknesses
- The trained graphs seem small. It is unclear how PRES performs on large-scale graphs.
- No absolute execution time is reported.
- It is tested on four GPUs. Its scalability to multi-nodes (with more GPUs) is somewhat unclear.
- In many cases, PRES still sacrifices some precision for performance gains.

### Questions
Overall, this is a solid study with clear innovations. The theoretical study on the impact of temporal batch size on the convergence of MDGNN training is extensive and helpful. My major concerns focus on the evaluation aspects. It would be extremely helpful if the authors could offer more information about these questions:

1. PRES is mainly evaluated on four graph datasets (Reddit, Wiki, Mooc, and LastFM). It seems these graphs are not very large with around 1K to 10K vertices and 400K to 1.3M edges. It would be helpful to justify that these graphs are large enough or PRES’s performance is not affected by the graph size.

2. It would be helpful to report the absolute execution time as well rather than relative speedup only.

3. It would be helpful to discuss if this method can be extended to multi-nodes with more GPUs.

4. It seems Table 1 shows that PRES still sacrifices some precision for performance gains in many cases. Please correct me if I have any misunderstanding here.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work conducts a theoretical study on the impact of the temporal batch size in MDGNN training. This shows that there can be a significant gradient variance using a small temporal batch, which in turn sheds light on an unexpected benefit of large batch sizes. Next, the authors define memory coherence, which represents the similarity of gradient directions within a temporal batch. Memory coherence is then used to model the upper boundary of gradient. 
With these theoretical insights, the authors present PRES with two main components: 1) iterative prediction-correction scheme 2) memory coherence smoothing. The former uses a GMM (updated with MLE) to predict newest memory states and fuses it with the calculated memory state to obtain the final ‘corrected’ state. The latter uses a new learning objective to promote larger memory coherence. 
Using PRES, the authors were able to increase the temporal batch size without compromising overall accuracy.

### Strengths
-	The authors provide theoretical results on the influence of temporal batch size on MDGNN training.
-	With memory coherence, the authors effectively define new methods to compensate for the accuracy drop of naively increasing the temporal batch size.

### Weaknesses
- The tradeoff of improved speed at the cost of lower accuracy does not seem to be appealing.
- Comparison with prior work on increasing temporal batch size is insufficient. 
-	In a similar manner, there are only a small number of baselines in the experimental results. 
-	The specific results of a dataset (LASTFM) is excluded.

### Questions
The paper is overall well written. The introduction on MDGNN was easy to follow. Insights from theoretical analyses were well presented. It was also evident how these insights became the main building blocks of PRES. However, my main concern comes from the experiment section.

-	**Is it really useful to gain speed at the cost of accuracy?**

So far this is my main concern. At first I thought the authors were trying to achieve SOTA accuracy.
However, what the authors are doing is gaining speedup of around 2x to 3x, at the cost of decreased accuracy (~1.0%).
I am not so sure about this, considering the effort the community is putting to gain higher accuracy.
Especially on the tested datasets, the number of vertices is only around few thousands, which wouldn't take terribly long to train.
I believe this partially comes from not reporting the training time (only the speedup is reported) and there are not enough baselines to compare. But the bottomline is that a strong justification is needed for this.

-	**What is the consensus on the ‘optimal’ temporal batch size?**

	In Figure 3, the authors show the performance of baselines by increasing the batch size up to 100. In the figure the ‘small batch size’ seems to be ~50. My question is do the majority of MDGNNs use a batch size smaller than 50, or are they already using approximately 500 (which seems to be the optimal size in Figure 4)? If the latter is the case, then personally the insight from theorem 1 (variance of the gradient for the entire epoch can be detrimental when the temporal batch size is small) loses some of its shine. Thus, the authors should try to first do a comprehensive overview on the currently used batch sizes.

-	**How does PRES differ from other baselines?**
	Two related works came to my mind which are missing in the current paper. “Efficient Dynamic Graph Representation Learning at Scale” (arXiv preprint, 2021, https://arxiv.org/abs/2112.07768) and “DistTGL: Distributed Memory-Based Temporal Graph Neural Network Training” (arXiv preprint, 2023, https://arxiv.org/abs/2307.07649). Both try to increase the temporal batch size without harming the accuracy. The former also uses prediction to utilize data-parallelism, while the latter tries to push the temporal batch size to the extreme for distributed GPU clusters. In my opinion, both (and any other baseline that shares the same goal with this work) should be compared methodologically and speedup-wise (in the current setting). Also, it would be interesting to see if these can also benefit from PRES. 

-	**Why is performance with/without PRES not shown with the LASTFM dataset?**
LASTFM stands out in that 1) the AP is the lowest 2) the speedup of PRES is the lowest. However, I was unable to find a figure like figure 4 for this dataset. Is there a reason for only leaving this dataset out?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
