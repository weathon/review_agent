# Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Data selection plays a crucial role in data-driven decision-making, including in large language models (LLMs), and is typically task-dependent. Properties such as data quality and diversity have been extensively studied and are known to enhance model performance. However, it remains unclear whether there exist other quantitative and general principles of data selection that can consistently improve performance, especially for complicated tasks.  In this paper, we demonstrate that selecting more uniformly distributed data can improve training efficiency while enhancing performance. Specifically, we establish that more uniform (less biased) distribution leads to a larger minimum pairwise distance between data points, denoted by $h_{\min}$, and prove that a smaller $h_{\min}$ can slow down the training dynamics of gradient descent (GD). Moreover, we theoretically show that the approximation error of neural networks decreases as $h_{\min}$ increases. Our analysis introduces a convergence framework for GD beyond the Neural Tangent Kernel (NTK) regime, applicable to a broad class of architectures, including transformers, without requiring Lipschitz smoothness. This framework further provides theoretical justification for the use of residual connection and function composition in deep neural architectures. In the end, we conduct comprehensive experiments for supervised fine-tuning across various settings, including different optimization strategies, model sizes, and training datasets. The results consistently demonstrate that selecting data by maximizing pairwise distance significantly accelerates training and achieves comparable or better performance in LLMs across diverse datasets. Code and Datasets are available at the link: https://anonymous.4open.science/r/data-uniformity-1A5C.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper analyzes how data uniformity—measured by the minimum pairwise distance among samples—affects neural network training efficiency and approximation ability. It develops a convergence framework beyond the NTK regime and proposes a simple greedy data selection method that maximizes pairwise distances. Experiments on several datasets and LLaMA models show faster convergence and comparable or better performance using fewer tokens.

### Strengths
* The paper establishes a clear link between geometric data uniformity and training convergence beyond the NTK regime.
* This work demonstrates that more uniform subset selection can reduce training cost while maintaining or improving performance.

### Weaknesses
* The performance gains of the uniform over random selection are relatively minor (Figure 1), raising doubts about the practical significance and scalability.
* The evaluation scope is limited to only a few benchmarks and llama-1 models; broader and more diverse downstream tasks and base models would be necessary to substantiate the generality of the approach.
* The proposed data-uniformity procedure requires computing or approximating all pairwise distances among samples---an $O(N^2)$ operation---but the paper does not analyze its computational overhead.
* In Figure 4, the training and validation losses of the 16K full dataset display abnormal behavior.

### Questions
pls see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper explores whether selecting more uniformly distributed training data improves neural network training. It formalizes uniformity via the minimum pairwise distance $h_{\min}$, proves that larger $h_{\min}$ accelerates gradient descent beyond the NTK regime and reduces approximation error, and introduces a greedy distance-based sampling strategy to increase uniformity. Experiments on both toy regression tasks and LLaMA fine-tuning (e.g., WizardLM, LESS) show that uniform subsets can reach similar or better performance than larger random subsets while achieving faster convergence.

### Strengths
Strengths

* The paper tackles data selection, an increasingly important topic for efficient LLM training.
* Comprehensive theory analysis, provides a general convergence result beyond NTK assumptions and links data geometry to dynamics and approximation error.

### Weaknesses
Weaknesses

* The paper uses max-min distance sampling as the core uniformity criterion. However, pure maximum distance does not necessarily guarantee globally uniform coverage. For example, if the data contains two distant dense clusters, the greedy selection may oscillate between these clusters and ignore other regions of the space. Please correct me if this interpretation is incorrect.

* The proposed selection strategy is closely related to prior work on distance-based uniform sampling. For example, the method in [1] shares many similarities with this paper, which also claim uniformity is important and also improve data uniformity via greedy distance selection while introducing extra constraints to alleviate the cluster oscillation problem mentioned above. It would strengthen the paper to explicitly compare and discuss differences from this line of work.

* In several settings, the accuracy of uniform sampling is very close to or worse than random sampling. Also the loss curve of uniform sampling also similar to random sampling. 

* When comparing partial and full datasets, are the models trained for the same number of iterations? If training the full dataset longer, does it eventually surpass the uniform subset? Clarification of additional results would be helpful.

* Some curves (e.g., Figure 4) show that the full dataset appears to stop training after only a few epochs while others run to 100 epochs. I assume this is due to instability or large loss spikes. Could the authors provide an explanation for this behavior?

* How does the method behave when rare but important examples exist? Uniformity-based selection may inadvertently discard such samples. A discussion or experiment on rare scenarios would strengthen the paper.

* The pairwise distance computation in greedy selection is potentially $O(n^{2})$, which may be expensive for very large corpora. A runtime analysis would be useful.

[1] An effective negative sampling approach for contrastive learning of sentence embedding

### Questions
Please refer to the weaknesses part.

### Soundness
3

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
This paper proposes a new theoretical framework for analyzing convergence of neural networks based on data uniformity. To avoid the standard, but often impractical, Lipschitzness assumption in previous literature, the authors propose and use a new Poly-smoothness assumption which is weaker and compatible with empirical deep neural networks such as transformers and residual networks.The theoretical convergence analysis extends beyond the NTK regime. The authors also provide a new perspective on how residual connections help with neural network training from non-degeneracy of Jacobians.

Based on the theoretical analysis, which shows that the uniformity of data (measured by minimum distance between data $h_{\min}$) increases convergence speed of GD, the authors propose a new data selection metric encouraging uniformity. Empirical results show that this method achieves on-par or better performance compared with SOTA results.

### Strengths
1. This paper provides a new perspective on how data uniformity helps with training, justified with theoretical analysis. The effectiveness of the proposed approach is validated through empirical results.

2. The proposed Poly-smoothness condition aligns better with neural networks used in practice, compared to standard Lipschitzness. This might be helpful for future analysis of deep neural networks.

### Weaknesses
For the theoretical part:
1. It is unclear why the minimum pairwise distance $h_{\min}$ is a good characterization of data uniformity. Specifically, when the data distribution is fixed, $h_{\min}$ will decrease as the sample size increases. This means that the convergence speed in Theorem 2 becomes slower with more samples and becomes $0$ when the sample size tends to infinity. Is this an intended behaviour? What if we consider infinitely many data points sampled from a continuous distribution (population loss)?

2. Figure 2 is a good illustration of the proof sketch, but no other parts of the main text has explained how the proofs are constructed. Can you include more explanations on the theoretical ideas behind the proof, how does the proof connects convergence speed with data uniformity, and how does it go beyond the NTK regime?

For the empirical part:
1. In figure 1(b), it is claimed that the 10k Uniform subset outperforms the 10k Random subset, but the random subset actually has higher accuracy on TruthfulQA MC (even higher than full training).

2. In figure 4, why does the Z-core method take a longer training time than uniform selection, with about the same number of iterations and samples?

3. The data selection method (Algorithm 1) has a computational cost depending quadratically on the dataset size. This can be burdensome when $N$ is large.

### Questions
See the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes "data uniformity" (maximizing the minimum pairwise distance, $h_{min}$) as a principle for efficient LLM data selection. The authors present a theoretical framework, claiming to go "beyond the NTK regime", to argue that uniform data accelerates gradient descent training for a family of non-linear architectures. Empirically, they select a "uniform" subset of data using Word2Vec embeddings and show that they can fine-tune LLaMA models significantly faster (e.g., 2x) while achieving comparable accuracy to the full dataset.

### Strengths
The paper's key strength is its strong and practically relevant empirical result: that a small, uniformly-selected data subset can fine-tune LLMs significantly faster while matching the performance of the full dataset. Moreover, the paper is theoretically ambitious (I'm not sure if the results do actually imply what the authors claim, see weaknesses), tackling the important problem of data selection by attempting to build a convergence framework for non-linear architectures.

### Weaknesses
**Presentation**: The paper is very densely written, making the theoretical arguments difficult to read and understand. The overall presentation could be significantly improved for clarity.

**Beyond NTK Claim**: The 'beyond NTK' claim is not fully convincing. In standard NTK analysis, a PL-like inequality is proven where the constant is the minimum eigenvalue of the kernel at initialization. This paper seems to follow a similar structure, proving a PL-like inequality (Figure 2) where the PL-constant ($\mu_{low,s,X}$) is now dynamic and path-dependent, and global smoothness is relaxed to a local smoothness. It remains unclear from Theorem 2 how this framework guarantees that feature learning (i.e., weights moving far from initialization) can actually occur, rather than just describing a different form of local convergence.

**On Corollary 3**: Corollary 3 appears to contain a significant logical leap. It first establishes a general, interesting bound on the convergence rate parameter ($\mu_{low,s,X}$) based on the density of local data clusters (the $\sqrt{\sum h_{ij}^2}$ term within a radius $H$). However, it then arbitrarily sets $H = h_{\text{min}}$, which means the cluster $D_{i,H}$ is empty for almost every point $x_i$. The only time it is non-empty is when selecting the two points that are exactly $h_{\text{min}}$ apart. The paper connects this to Theorem 1 (biased sampling $\to$ small $h_{min}$) to claim that more data uniformity implies faster convergence. 

This reduces the claim to the specific and well-known case that having a single pair of near-duplicates is bad for convergence. This specific case does not provide a sufficient theoretical justification for the main goal of the paper “data uniformity speeds up convergence”, making their actual result feel like a big over claim. 

**Confounded Definition of "Uniformity" in Experiments**: The practical definition of 'uniformity' in the experiments is confounded. As stated in Section 5.1, distances are measured in the embedding space of an external, pre-trained Word2Vec model. This means the selected "uniform" subset is an artifact of this specific and dated embedding choice, not a fundamental, intrinsic property of the data itself. Is this dependent on the embedding choice? What if you choose some other model? 

**Theory-Practice Disconnect**: The experiments in Section 5 do show a clear and valuable empirical finding: the uniformly-sampled subset converges significantly faster and achieves comparable performance (e.g., Figure 1, 5). This empirical contribution is good, but it is not convincingly explained by the provided theory—which seems to be the main point of this work.

I am keeping a low score because of the above issues. I would be more than happy to engage with the authors during rebuttal and rethink my score.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
