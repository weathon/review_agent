# From Scaling Law to Sub-Scaling Law: Understanding the Diminishing Returns of Larger Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 1, 6, 6

## Abstract
Traditional scaling laws suggest that performance metrics of language models improve predictably with increases in model or dataset size. However, recent works display sub-scaling growth for large language models, where performance improvements decelerate as the dataset or model size increases. 
This study aims to systematically investigate the sub-scaling law phenomenon through an extensive empirical analysis involving over 400 models, ranging from 20 million to 7 billion parameters, with varying datasets and training strategies.
Our findings indicate that sub-scaling laws arise primarily from high data density and non-optimal training resource allocations. 
Specifically, we observed that both factors contribute more significantly to performance deceleration than previously anticipated. We examine the sub-scaling phenomenon from two perspectives: data density and training strategy. High data density leads to diminishing marginal gains in performance, while optimal resource allocation is crucial for sustaining performance improvements.
Further, we propose a sub-optimal scaling law that generalizes the Chinchilla scaling law to better predict performance and loss in sub-scaling regimes.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors observe that standard scaling laws have poor fit for the very largest LLMs, with them performing worse than would be expected. They propose that this is because of decreasing marginal information gain in the datasets used by these models and propose 'sub-scaling laws' that account for this limitation, finding that it has better fit to what is observed. The implications of this is that more of the compute budget should be allocated to larger models instead of more tokens.

### Strengths
- The paper tries to incorporate the data quality directly into scaling laws, as opposed to past works like Muennighoff et al., which imposes a particular structure (entire repeats of datasets).
- The proposed scaling laws have better fit to observed outcomes.
- The theoretical findings have useful empirical takeaways, namely that existing sota models are likely overtrained.

### Weaknesses
- The paper suffers from a lack of rigorous definitions and justification for why certain formalisms have been chosen. 
- 229 - 231: Where does this definition of data density come from? Why is information gain measured geometrically instead of say, using the notion of V-information or some other framework? 
- The norm that is used to measure distance in embedding space is not explicitly stated. Moreover, LLM embedding spaces tend to be highly anisotropic, yet this metric does not seem to adjust for that at all.
- They decay factor in 3.1 is just another hyperparameter to be tuned; it is not all that surprising that adding another degree of freedom allows the curve to better fit the observed pattern. If the decay factor could be calculated automatically, as a function of the data itself, the result would be much more impressive. In fact, the geometric notion of data density not finding its way into the proposed sub-scaling laws suggests that they are not all that useful as a formalism and are yet another way of some data is lower-quality than other data.

### Questions
- The terms in 251 to 273 need to be defined rigorously: is information gain here the reduction in entropy of the next token? What does performance mean, the perplexity? What is $P_0$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
3

### Summary
The paper proposes a metric to measure “dataset density” (i.e., diversity) in the context of LLM pre-training. It then introduces a scaling law formulation that incorporates such “dataset density” metric. Lastly, the paper introduces another scaling law formulation which accounts for the amount of “overtraining” beyond the number of compute optimal training tokens. The paper includes some experiments aiming to validate the proposed scaling laws.

### Strengths
Better understanding the extent to which traditional scaling laws hold at ever larger compute regimes is an interesting and valuable research direction; and so are novel scaling law formulations which better account for current pre-training practices such as 1) performing multiple epochs on the pre-training data (i.e., data repetition) and 2) “over training” beyond Chinchilla compute-optimal.

### Weaknesses
The paper has severe limitations. I do not think that it makes substantial contributions with respect to prior work. The writing is at times not very clear. Much of the Appendix appears to be LLM generated. Let me first address each of the “key findings” outlined at the end of the introduction. 

> “Sub-Scaling Law Phenomenon: Traditional scaling laws fail to predict performance improvements for very large models and dataset”

Prior work highlights that loss is remarkably predictable, see for instance GPT-4 technical report: “the fitted scaling law predicted GPT-4’s final loss with high accuracy (Figure 1)” or Llama 3 405B technical report “We find this two-step scaling law prediction, which extrapolates over four orders of magnitude, to be quite accurate: it only slightly underestimates the final performance of the flagship Llama 3 model”. Note that performance is even underestimated! This paper does not train models to nearly the scale of GPT-4 or Llama 3 405B, nor does it otherwise present convincing evidence for the purported “sub-scaling” phenomena occurring for “very large models and dataset”. 

Unlike loss, benchmark performance can be notoriously hard to predict (e.g., due to emergence). However, “traditional scaling laws” pertain to loss and not benchmark performance. I therefore find Figure 2 (a) highly misleading — no practitioner would expect MMLU accuracy against compute to admit a log-linear fit. This is not a failure of pushing scaling laws to “very large models and datasets”. Log-linear fits of benchmark performance against compute are generally simply not adequate, as is widely known in the literature, see for instance Wei et al. [1].

> “Impact of Data Density: High data density causes sub-scaling due to diminishing returns from redundant data.”

To substantiate their claims, the authors train two models on two datasets, one with higher “data density” than the other. The one with higher “density” happens to perform better. N=2 is clearly insufficient to convincingly study the relationship between “data density” and performance.

Note that the effect of data repetition has been comprehensively studied by Muennighoff et al. [2]. I do not see what this work adds to that of Muennighoff et al. 	
> Low-density datasets, with more diverse data, align more closely with traditional scaling laws

I believe that the authors aim to substantiate this claim with Figure 8. The experimental set-up here is fundamentally not adequate. Scaling laws do not aim to predict performance throughout training, as is presented in Figure 8. Rather, they aim to predict model performance across different compute scales. Thus, Figure 8 is not informative of the merits of neither “traditional scaling laws” nor the proposed law. Moreover, using intermediate checkpoints for fitting (let alone validating) scaling laws can have severe pitfalls [3, 4].

> “3. Optimal Resource Allocation: Efficient computational resource allocation is crucial to mitigate sub-scaling and sustain performance improvements”

I fail to see how this is a novel insight. Efficient resource allocation is one of the main motivations behind the seminal work of Kaplan et al. [5], which say “For optimal performance all three factors must be scaled up in tandem”. Highly influential subsequent work has discussed better practices for determining such efficient allocation via scaling laws [6]. If the allocation is not optimal, then performance will be worse that it could have been for a given compute scale and no longer follow the scaling law (what in this work is called “sub-scaling”). Figure 9 is highly misleading, since as per Kaplan et al. “Empirical performance has a power-law relationship with each individual factor when not bottlenecked by the other two” — in Figure 9, the larger models are being bottle-necked by not having enough training tokens. Please see Figure 2 of Kaplan et al., such “bottle necks” are not “failures” of “traditional” scaling laws but rather very much to be expected. 

> "4. Sub-Optimal Scaling Law: We proposed a Sub-Optimal Scaling Law that generalizes the Chinchilla scaling law to better predict performance and loss in sub-scaling regimes”

Here I do find interesting the inclusion of the “over-training factor” OTR on the scaling law. Note that an identical factor is considered by the concurrent work of Gadre et al. [7]. However, I find the experiments flawed. The limitations of Figure 9 I have already discussed. The results in Figure 10 are also not informative of the merits of the proposed scaling law, since there is a critical flaw: the law is fitted and tested on different pre-training distributions (one being presumably the Pile) and the other being whatever Llama 3 was trained on. Such distribution mismatch thus confounds the fitting, e.g., to what extent is the loss Llama 3 8B over estimated due to such distribution mismatch?

[1] Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., ... & Fedus, W. (2022). Emergent abilities of large language models. arXiv preprint arXiv:2206.07682.

[2] Muennighoff, N., Rush, A., Barak, B., Le Scao, T., Tazi, N., Piktus, A., ... & Raffel, C. A. (2023). Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36.

[3] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., ... & Sifre, L. (2022). Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

[4] Porian, T., Wortsman, M., Jitsev, J., Schmidt, L., & Carmon, Y. (2024). Resolving discrepancies in compute-optimal scaling of language models. NeurIPS 2024.

[5] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

[6] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., ... & Sifre, L. (2022). Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

[7] Gadre, S. Y., Smyrnis, G., Shankar, V., Gururangan, S., Wortsman, M., Shao, R., ... & Schmidt, L. (2024). Language models scale reliably with over-training and on downstream tasks. arXiv preprint arXiv:2403.08540.

### Questions
See Limitations.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the phenomenon of sub-scaling laws by investigating a total of 400 models, ranging from 2 million to 7 billion in parameter size, motivated by observations that model performance gains decelerate for increasing model sizes. The authors focus on two perspectives for this, training data density (defined through measures reflecting data similarity and redundancy in training corpora) and training strategy (defined through optimality of resource allocation during training).

### Strengths
The paper provides insightful findings around training dynamics of large language models encompassing various sizes. The analysis is overall extensive, and the results contribute to a better understanding of experimental design decisions for the training of LLMs. As such, I believe this paper represents a solid contribution that is of interest to the research community.

### Weaknesses
* I found the paper overall difficult to read, the structure leaves room for improvement and the findings / conclusions could be better depicted and illustrated. The paper focuses heavily on technical derivations and definitions and leaves several experimental details unmentioned in the main manuscript (I formulate these as questions below).
* I would have wished the authors to better explain the implications of the findings provided in this work. How can future research contribute to better understanding the phenomenon of sub-scaling laws? 
* Re. presentation style: Figure 1 could be more clearly explained (e.g., the legend is incomplete), Figure 5 is confusing (the blue and yellow clusters look very similar both w.r.t. their size and the number / distribution of points within them, even though they represent a cluster with high and low similarity / density, respectively), and if citations aren’t used actively, I suggest surrounding them with parentheses for better readability.

### Questions
* Can you provide additional details on Figure 6? How was the data clustered, how was the number of clusters chosen?
* Where does the constant for the definition of $\text{FLOPs}(N, D)$ come from? Can you elaborate on its derivation?
* For OTRs, how is $D_{opt}$ derived? This is unclear in the paper but quite crucial to understand the authors’ argumentation around the training strategy.

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
3

### Summary
These work systematically investigate the sub-scaling law under over-training scenarios, and propose the new formulations with OverTraining Ratio OTR. The observations on data density and data/model size allocations are interesting and convincing. The fit curves seem better than the current widely-used ones, especially under over-training conditions for SLMs.

### Strengths
1. This paper is well-organized and easy to follow.

2. It is an important and interesting topic to explore sub-scaling phenomenon.

3. Extensive experiments and careful analyses demonstrates the effectiveness of the proposed new formulations.

### Weaknesses
1. Some details are missing. For example, in both abstract and introduction, the paper mentions over 400 models, ranging from 20M to 7B, but we can find no details of the used model. Also, many citations are missing. Some claims (e.g., line 280， 763） and existing formulations （e.g., line 307, ) should add citations to support the claims and separate from the proposed formulations.
    
2. When a notation first comes up, there should be clearly description, or at least a clear reference to later sections. For example, R_D and R_N are two very important notations. When I read line 369, I thought R_D might be a hyper-parameter. Then when I read line 456, R_D and R_N seem to be two functions, but the exact definition is just mentioned as in Appendix in line 456 without the clear reference. Finally, I found it in Appendix D.  Additionally, for “optimal training resource allocations”, I fully understood this concept until section 2.2.
    
3. The description should be consistent through the paper. For example, in the relative works, it is said that the difference to current works are they focus on general scaling laws while this work provides under over-training conditions (line 1060-1061). However, main sections seem not convey this message.

### Questions
1. What are these 400 models? 
    
2. What’s the relationship of Eq. (6) and (7)?
    
3. In the appendix D, the final results actually only related to OTR. Why the OTR can both represent data density and data/model size allocations?

### Soundness
3

### Presentation
3

### Contribution
3
