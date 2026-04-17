# Inconsistency Biases in Dynamic Data Pruning

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Dynamic data pruning accelerates training by focusing on informative samples. However, comparing importance scores across different model states introduces inconsistency (score context drift), and variable selection rates bias gradient dynamics over time (temporal gradient bias). We introduce RePB (Resolving Pruning Biases), a framework addressing these issues. RePB performs pruning decisions within local windows (short sequences of batches) during training, using loss scores computed with a near-constant model state within each window to ensure valid comparisons. These decisions determine the data subset used in the subsequent training phase. To counteract temporal gradient bias arising from non-uniform sample inclusion, cumulative temporal rescaling reweights sample losses during training based on their historical selection frequency. We provide theoretical grounding for RePB's consistency in score comparison and gradient alignment. Experiments show RePB achieves near-full-dataset accuracy using reduced data (most above 30%) across 16 datasets, 17 models and 13 tasks, offering a robust and scalable approach to efficient deep learning. Code is available at https://github.com/mrazhou/RePB.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Deep learning models rely their success mainly on the variety of data used in vast quantities. 
This dataset expansion has direct impact on the training efficiency. This work focuses on data selection which aims to train models on smaller but carefully chosen data subsets. There are dynamic methods that perform such data selection during the learning process, called online data pruning. Giving wrong importance to data samples can introduce biased gradients. This submission identifies two fundamental consistency issues called: score context drift and temporal gradient bias. 

Score context drift focuses on the parameters drift during training whereas temporal gradient bias takes into account the bias introduced by non-uniform sampling distribution over time (due to different data selection).

### Strengths
Strengths:
- Identify fundamental issues of inconsistency biases in dynamic data pruning
- Propose a framework (RePB) to mitigate the inconsistency bias due to parameter drift during training and non-uniform sampling distribution over time
- RePB maintains data diversity and prevents sample pool collapse
- Theoretical foundations are provided that justifies comparison of scores collection within local windows
- Comparisons are reported over different methods.

### Weaknesses
Weaknesses:
- Comparisons with the most recent SOTA (Salaun et al.2025a) is missing
- How sensitive is RePB to the local window size?
- How the method ensures that computing scores over a small window does not accumulate error over time?
- Missing citations: 

Recent work by Salaun et al. 2025a, b, develop better online importance mechanisms driven by Multiple importance sampling (MIS) and Optimal MIS. The paper seems to completely ignore these references.

@misc{salaun2025a,
      title={Online Importance Sampling for Stochastic Gradient Optimization}, 
      author={Corentin Salaün and Xingchang Huang and Iliyan Georgiev and Niloy J. Mitra and Gurprit Singh},
      year={2025},
      eprint={2311.14468},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2311.14468}, 
}

@misc{salaun2025b,
      title={Multiple Importance Sampling for Stochastic Gradient Estimation}, 
      author={Corentin Salaün and Xingchang Huang and Iliyan Georgiev and Niloy J. Mitra and Gurprit Singh},
      year={2025},
      eprint={2407.15525},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.15525}, 
}

### Questions
- How sensitive is RePB to the local window size?
- How the method ensures that computing scores over a small window does not accumulate error over time?
- How the method compares to Salaun et al. 2025a,b?
- Does it make sense to combine RePB with Salaun et al.?

### Soundness
3

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
This paper introduces RePB (Resolving Pruning Biases), a framework that tackles two fundamental consistency problems in dynamic data pruning: score context drift (where importance scores computed at different training stages aren't comparable) and temporal gradient bias (where non-uniform sample selection alters training dynamics). RePB addresses these issues through Local Window Pruning, which restricts score comparisons to short windows (a few batches), and Cumulative Temporal Rescaling, which reweights samples based on their historical selection frequency to align gradient trajectories with full-dataset training. The authors provide theoretical guarantees for both components and demonstrate RePB's effectiveness across 16 datasets, 17 model architectures, and 13 diverse tasks spanning classification

### Strengths
The paper’s main strength is its comprehensive and diverse experimental validation, covering many datasets, architectures, and tasks. RePB seems to consistently deliver good performance and hence is a promissing approach.

### Weaknesses
Overall, I found that the paper needs further refinement and clarification before it is ready for publication.

The main weaknesses of the paper lie in its theoretical clarity and methodological presentation. The theoretical analysis lacks rigour: propositions are not clearly stated, assumptions are introduced informally within the proofs, and the proofs rely on approximations rather than precise derivations. This makes it difficult to assess what is actually proven and under what conditions the results hold. 

In addition, the methodology section presents the pruning criterion using a fixed mean-based threshold $\mu_k$, which limits flexibility in controlling the pruning or compression rate $r$. While most experiments appear to use this fixed rule, some dynamic data pruning benchmarks suggest a variable rate, creating inconsistency between the method’s description and its implementation.

### Questions
- What are the formal statements of the propositions ?
- How does the framework deal with flexible pruning rate $r$ ?

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
5

### Summary
This paper identifies two important problems in dynamic data pruning: inconsistency biases in the scoring context, and temporal gradient dynamics. It proposes to use local window pruning and cumulative temporal rescaling to address these two problems. The authors evaluate its effect on various datasets and demonstrate its effectiveness. Overall, this is an insightful and meaningful work.

### Strengths
1. This work identifies two important problems in current dynamic data pruning methods and solves them. The insight is quite accurate, and the meaningful improvement further enhances the robustness and generalization of dynamic data pruning.
2. The experimental evaluation is comprehensive. This work extends the application to a lot more scenarios, including other vision and vision-language tasks, and semi-supervised learning.
3. The presentation is good.

### Weaknesses
1. According to the ablation experiment demonstrated in Tab. 9, RePB relies on resampling more than other dynamic pruning methods. It is worth a little bit more discussion: whether it is because of the inner window score stability, or cumulative temporal rescaling (rescaling factor too high in some cases)?
2. Currently, the pruning factor $\rho$ is used, but this factor has no ablation, and there is no discussion of its value and tuning characteristics.
3. For proof of 4.2, a critical problem is that in this scenario $E[1/X] \neq 1/E[x]$, the substitution may not hold.

Minor:
1. The citation of "Scale efficient training for large datasets" would be better to use the accepted version.

### Questions
1. In the original InfoBatch, there was a trick mentioned: one can further downsample the lower-loss samples (E.g., keeping only 25% samples for 25% low-loss samples). Is this trick also compatible with RePB? It could further enhance the saving ratio. This is a complementary question, out of my curiosity, but not required.
2. CTR weight could encounter a more extreme value than InfoBatch's rescaling factor (which is fixed to 1/(1-r)). Is there any observation on this potential problem? 
For example, an update with a large factor on a sample not sampled for many epochs could lead to a large update step, leading to higher loss, and the subsequent updates cannot immediately reduce the CTR weight, which will cause this sample to be updated with much higher importance than sampling it earlier.

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
4

### Summary
This paper identifies two fundamental consistency issues in dynamic data pruning: score context drift (incomparability of importance scores computed under different model states) and temporal gradient bias (distortion of gradient dynamics due to non-uniform sample selection over epochs). To address these, the authors propose RePB (Resolving Pruning Biases), a framework that combines (1) Local Window Pruning (LWP) to ensure valid score comparisons within short training windows where model parameters are nearly constant, (2) Uniform Probability Resampling to maintain data diversity, and (3) Cumulative Temporal Rescaling (CTR) to reweight sample losses based on historical selection frequency, thereby aligning long-term gradient expectations with full-dataset training. The method is theoretically motivated and evaluated across 16 datasets, 17 models, and 13 tasks, showing strong performance—often matching or exceeding full-dataset accuracy while pruning 30% or more of the data.

### Strengths
1. The paper clearly articulates two underappreciated but critical pitfalls in dynamic data pruning and provides a principled, theoretically grounded solution.

2. RePB’s design is elegant: LWP directly tackles score inconsistency by restricting comparisons to stable model contexts, while CTR offers a practical, history-based inverse weighting scheme that avoids the need to model complex instantaneous selection probabilities.

3. The empirical evaluation is impressively broad in terms of tasks (classification, captioning, generation, semi-supervised learning), modalities (vision, text, audio), and architectures (CNNs, Transformers, Mamba, VAEs, diffusion models), demonstrating strong generalization.

4. Ablation studies and comparisons with SOTA methods like InfoBatch convincingly validate the necessity and effectiveness of each component. 

5. Computational overhead is minimal, making RePB highly practical for real-world deployment.

### Weaknesses
1. Despite the diversity of tasks, the scale of some experiments remains limited. For instance, while ImageNet-1K is included, there is no evaluation on larger-scale vision benchmarks (e.g., ImageNet-21K) or billion-parameter language models, which are increasingly standard in efficiency research. The largest dataset used (MJ+ST with 15M samples) is promising, but more large-scale LLM or multimodal experiments would strengthen claims about scalability.

2. The paper assumes sample loss as the sole importance metric. While common, this may not capture all aspects of sample utility (e.g., diversity, influence, or representativeness). Extending LWP to other scoring functions (as hinted in the limitations) would be valuable.

3. The theoretical analysis relies on assumptions like Lipschitz continuity and bounded gradients, which are standard but may not always hold in practice (e.g., with unstable training or aggressive learning rates). A brief discussion of robustness under violation of these assumptions would be helpful. 

4. Missing some data pruning methods: 

Severing Spurious Correlations with Data Pruning

Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models

Data Pruning by Information Maximization

Beyond Efficiency: Molecular Data Pruning for Enhanced Generalization

Pruning-based Data Selection and Network Fusion for Efficient Deep Learning

Data Pruning via Moving-one-Sample-out

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
2
