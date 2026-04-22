# WRING Out The Bias: A Rotation-Based Alternative To Projection Debiasing

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Vision-Language models (VLMs), including CLIP, are known to encode biases such as learning spurious correlations that falsely associate background attributes with particular labels. Debiasing approaches typically aim to isolate and remove subspaces corresponding to a target concept via projecting the embedding away from the concept. This strategy succeeds in debiasing VLM embeddings with respect to the concepts considered but can amplify biased shortcuts in unconsidered concepts. In practice, it is impossible to enumerate all possible biases, meaning that an increase in bias can go unobserved during evaluation. We propose a debiasing approach for a set of known concepts such that the relation to the remaining, unconsidered, concepts is minimally changed. We achieve this by rotating the VLM's embeddings within only a relevant subspace, rather than removing these subspaces, which mitigates unintended bias amplification.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a modified debiasing mechanisms for vectorized representations, applied to vision language models (VLMs).  Its goal is to reduce the biased induced by a simple linear projection operation.  It does this by calculating some additional correlation that may happen with other identified concepts, and adjusting the vector representations to remove that additional correlation.

### Strengths
These are neat observations, and a clean mathematically justified approach on how best to adjust it.  The empirical results may (* see note below) show meaningful improvement, for a few simple cases in VLMs.  

The approach and geometric modeling is nice and as far as I know new.  It is a subtle change to who debiasing uses projection that might (not sure) be the "right" way to do it to avoid unwanted bias in the process.

### Weaknesses
- the Bolukbasi etal 2016 paper, while introducing the idea to projection to remove bias, actually advocated for alternative approaches (e.g. Hard Debiasing and Soft Debiasing) which included other operations to try to correct for the affect of the projection on other identified concepts.  While the setting is different here (they considered word vectors), I think this paper should try to compare to the additional correction operations Bolukbasi proposed, and how they relate to the Wring operation.  

  - Related to above, I believe the following paper is the one that actually first proposed just using the projection operation (without other corrections):  https://arxiv.org/abs/1901.07656

 - This is only applied to VLMs.  This is an important modern setting, but I wonder if it is applicable to text only or vision only models?  The reason to ask is that the text-focused debiasing does not seem to work as well in the experiments.  

  - For most vectors v and concepts c, in high-dimensions the extra bias induced in projection should be small.  How targeted does this need to be to observe a meaningful difference?  For instance, in Sec 5.4 and Table 2 where you measure "Worst Group Accuracy" how many groups are there?  In general, I do not understand exactly what this is evaluating.  

 - The main experiments are in Figure 3 & 5.  But I do not fully understand the x- and y-axis, which are labeled "Bias for C_debias" and "%Change in Bias for C_uncon."  These labels are not fully explained (or I missed it).  Yes a "bias" is defined in equations (1) and (3), but this takes in a vector v and two concepts d1, d2.  It is not clear (to me) how this equation is transformed into these the numbers reported on these axes.  


Overall, I like the idea in this paper.  But because of the numerous questions above, especially in how to interpret the experimental results, I cannot yet stand fully behind it.

### Questions
See all questions in the weaknesses above.  The most critical ones are on understanding the precise things that were measured in the experiments.  I like aspects of this paper, but that needs to be clarified.  

Second is understanding how this relates to methods actually discussed in Bolukbasi paper.  I do not think they had the same analysis as here, but the relation should be clarified.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces WRING, a method for debiasing the pre-trained representations of text queries from VLMs such as CLIP.  For a query embedding v and sensitive attribute embeddings $c_1$ and $c_2$, the authors define the bias of v with respect to $c_1$ and $c_2$ as: 

bias(v, $c_1$, $c_2$) = cosine\_sim(v, $c_1$) - cosine\_sim(v, $c2$)

The method rotates the query embedding such that it becomes unbiased with respect to the considered set of attributes - bias(v, $c_1$, $c_2$) = 0. The authors theoretically show that WRING is less likely to amplify the bias (as per the author’s definition) of the query with respect to unconsidered sensitive attributes, than the classical projection-based debiasing approach.

### Strengths
1. The paper addresses a relevant topic - mitigating a set of known biases without amplifying unknown ones. The theoretical analysis seems correct.
2. The proposed procedure is not computationally expensive
3. The method is validated in diverse settings (multiple datasets and targets for debiasing), keeping in mind both the debiasing objective and the retention of performance for downstream tasks.

### Weaknesses
1. (minor) Limited scope \- the method is only applicable to CLIP-like models  
2. The bias metric mentioned above seems to be particular to this work. While intuitive, the relevance of the metric itself is not supported through any arguments. The theoretical analysis on the change in bias is based on the author’s own metric for bias, and it is not argued that the same behaviour translates to established fairness metrics. Also, for the considered set of attributes there are multiple solutions for reducing this bias metric to 0 (e.g. Projection and WRING), yet they are unlikely to attain the same fairness performance, as measured through commonly accepted metrics.   
3. The change in bias ($\\delta$DP) is the metric that highlights the differences between WRING in Projection debiasing \- I find them to perform reasonably similar with regards to the other metrics (Tables 2 and 3). However, $\\delta$DP does not reflect whether the change in bias was an increase or decrease. I think that not using the absolute value of the change in bias would better reflect this. If the change in bias brought by Projection is actually an improvement, that would oppose the claim of the first contribution and greatly reduce the value of the proposed method.

Initial recommendation: weak reject

Motivation: the article addresses a relevant problem, however, the first claim that Projection amplifies biases with regards to unconsidered attributes is insufficiently backed by the empirical results provided.

Additional feedback:

1. Reducing the bias usually comes at the cost of a drop in average performance. I would thus encourage the authors to also report the Average Accuracy in Table 2, or in the Appendix if there is not enough room for an additional column.   
2. Tables 4-6 should also report the Baseline DP for unconsidered attributes to better put the absolute value for the change in bias into perspective.

Minor comments:

- The first 5-6 pages only refer to the bias metric introduced by the authors, but the metrics reported in all subsequent figures and tables are based on Demographic Parity (DP) instead. I would suggest adjusting Section 5.2 to make this aspect more clear.  
- Line 247 \- in the definition of $\\delta\_w$, $d\_i$ and $d\_j$ appear instead of $d\_1$ and $d\_2$  
- Line 250 \- bias(v, $d\_1$, \*\*$d\_1$\*\*)   
- Appendix C.4 \- an explicit description of what the red color and the bold font represent would improve the readability of the tables.  
- The figures on pages 22-23 do not seem to be properly introduced in the article. It is hard to tell at a glance what dataset was used and what is the setting 

Things that could convince me to change my decision

- Addressing weakness 3 would be the most important and straightforward thing to do \- it just requires changing the evaluation metric and repeating the experiments for Projection. The change in bias for WRING is sufficiently low already.   
- Having the Baseline DP for unconsidered attributes in Tables  4-6 would also help in better judging the impact of Projection-based debiasing on the unconsidered attributes.

Regarding weakness 2, using a surrogate metric for bias to motivate the approach is not necessarily an issue, and the analysis seems properly done. However, it would have been appreciated if the authors also motivated its use through some empirical evidence, e.g. higher bias translating into worse fairness, as judged by another metric.

### Questions
Can the authors address the following?

- Addressing weakness 3 would be the most important and straightforward thing to do \- it just requires changing the evaluation metric and repeating the experiments for Projection.
- Having the Baseline DP for unconsidered attributes in Tables  4-6 would also help in better judging the impact of Projection-based debiasing on the unconsidered attributes.

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
3

### Summary
This paper addresses the problem in VLMs that when debiasing for one concept via a projection, this can (often) amplify biases for other concepts. It starts by giving a geometric explanation of this phenomenon. Then it proposes an alternative debiasing technique, called WRING, that performs a rotation in the subspace of the to-be-debiased concept, instead of a projection. It gives a theoretical argument of why WRING suffers less from the above mentioned problem. Finally, it shows empirically that this is indeed the case: for various VLMs and datasets, it shows that WRING debiases for the concept-to-be-debiased, but has a less strong effect on the bias of other concepts that are not considered.

### Strengths
* WRING is, to the best of my knowledge, a novel approach in the field of post-hoc debiasing in VLMs. The novelty is in the fact that a rotation is being used. Since the problem the paper addresses is a relevant problem, a novel approach to tackling this problem is interesting and relevant for the scientific community.
* the paper gives a theoretical explanation of why the problem of bias amplification occurs, from a geometric point of view. This is insightful and also helps to understand the proposed method WRING.
* the experimental setting are mostly solid and the results are convincing.

### Weaknesses
* the theoretical analysis of the problem, in section 3, hints to a different solution than WRING. The bias amplification in Equation (1) is a consequence of the fact that an orthogonal projection reduces the length of the embedding vector. This can be addressed in different ways. One can simply rescale the projected embedding vector to its original length. Or one can use an oblique projection, where the range of the projection is chosen such that the length is invariant. The theoretical argument of why projection leads to bias amplification then disappears. It therefore would be very interesting and relevant to include a rescaling-after-orthogonal-projection method into the paper and see if WRING also outperforms such a method. If it does, it would also be very interesting to understand why. I very much would like to see such an analysis added to the paper. In the way the paper is written now, WRING does not seem the most natural or least impactful solution to the problem of bias amplification.
* there seems to be a mistake in the derivation of WRING. The condition in line 740 is not equivalent to the condition of unbiasedness for concept vectors $c_i$ and $c_j$ in line 227, if the length of those two vectors are unequal. I kindly ask the authors to clarify this and to check whether their WRING method is really doing what they intend it to do.
* all experiments are conducted on models that were not fine-tuned for their specific task. However, in practice, for many problems fine-tuning is a standard practice nowadays. It would therefore be highly relevant to also include experiments with fine-tuned models and to see if WRING then also outperform projection-based techniques.
* the analyses on pages 4 and 5 of the paper are assuming that the bias, defined in line 168, is always positive. But this is not true. Simply swapping the role of $c_1$ and $c_2$ flips the sign. I believe the text/analyses on pages 4 and 5 should be rewritten to make it consistent with this.
* the results in Table 2 would benefit from uncertainty estimates.

Some more minor weaknesses and typo's:
* there are quite a few language typo's. Please use some software to detect and correct them,
* line 247: indices $i$ and $j$?
* line 247: wrong symbol for length of vector (double lines).
* line 254: inequality sign wrong?
* line 257: "opposite". Is this always true? Why?
* line 269: $b$ -> $d$

### Questions
* in linear concept-removal techniques for classification models in the presence of spurious correlations, the concept-to-be-removed is often a binary label and is represented by a single direction in the embedding space. An example is the background in the Waterbirds dataset (land vs water). This seems to differ from VLMs, where the groups "land" and "water" would have their own vector in embedding space. What is the explanation of this difference? And suppose I would like treat the background as a single direction in the embedding space, how would WRING work in that case?
* how does the modality gap play a role in your analysis? It is not mentioned in the paper, although in the experiments there is a distinction between projections based on text-directions and image-directions.

### Soundness
3

### Presentation
3

### Contribution
3
