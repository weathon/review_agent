# Adaptive Hopfield Network: Rethinking Similarities in Associative Memory

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Associative memory models are content-addressable memory systems fundamental to biological intelligence and are notable for their high interpretability.
However, existing models evaluate the quality of retrieval based on proximity, which cannot guarantee that the retrieved pattern has the strongest association with the query, failing correctness.
We reframe this problem by proposing that a query is a generative variant of a stored memory pattern, and define a variant distribution to model this subtle context-dependent generative process.
Consequently, correct retrieval should return the memory pattern with the maximum a posteriori probability of being the query's origin.
This perspective reveals that an ideal similarity measure should approximate the likelihood of each stored pattern generating the query in accordance with variant distribution, which is impossible for fixed and pre-defined similarities used by existing associative memories.
To this end, we develop adaptive similarity, a novel mechanism that learns to approximate this insightful but unknown likelihood from samples drawn from context, aiming for correct retrieval.
We theoretically prove that our proposed adaptive similarity achieves optimal correct retrieval under three canonical and widely applicable types of variants: noisy, masked, and biased.
We integrate this mechanism into a novel adaptive Hopfield network (`A-Hop`), and empirical results show that it achieves state-of-the-art performance across diverse tasks, including memory retrieval, tabular classification, image classification, and multiple instance learning.
Our code is publicly available at https://github.com/shurongwang/Adaptive-Hopfield-Network.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces an Adaptive Hopfield network (A-Hop) that learns a context-dependent similarity measure for content-addressable memory. 
By reframing retrieval as a maximum a posteriori problem under a variant distribution, the authors derive an adaptive similarity function that approximates the likelihood of a stored pattern generating the query. 
They prove that, with appropriate learned weights, 
A-Hop achieves optimal correct retrieval for three common variant types (noisy, masked, biased). 
Empirically, integrating this adaptive similarity into Hopfield networks yields state-of-the-art performance on synthetic memory retrieval tasks and on downstream benchmarks (tabular data, image classification, multiple-instance learning), outperforming prior Hopfield variants.

### Strengths
- Tackles a fundamental issue (retrieval correctness vs. proximity) with a clear probabilistic framework.

- Offers theoretical guarantees for optimal retrieval under well-defined generative scenarios (noise, masking, bias).  I skimmed through the proofs. Looks reasonable to me, but I didn't check them line-by-line.

- Extensive experiments across diverse tasks, showing consistent improvements over strong baselines. Code provided. I skimmed through them. Looks reasonable to me. I didn't check line-by-line nor run them at my end.

### Weaknesses
1. [major] Incremental novelty: Core idea is essentially learning a weighted distance (metric learning) for Hopfield retrieval similar to Wu & Hu et al 2024. It is a relatively straightforward extension of known concepts.

2. [minor] Theory vs. implementation gap: Proofs assume per-dimension weighting (unsorted footprint) for optimal retrieval, but the actual method uses sorted feature footprints without a formal optimality guarantee for that sorting. Please correct me if I am wrong.

3. [minor] Sorting overhead: Computing the similarity footprint requires sorting features ($O(d \log d)$ or $O(n \log n)$). This adds computational cost. The paper does not discuss runtime impact or scalability to very high-dimensional data.

4. [very minor] Reliance on supervision: The adaptive similarity weights are learned using ground-truth associations (either known memory-query pairs or class labels). The method may not apply directly in fully unsupervised retrieval settings without a way to obtain such training signals.

5. [very very minor, almost personal opinions] I kind of like this paper. It shows some solid efforts on theory and numerical validations. This might already be a good paper, but I really hope it is better, even great. But the clarity is not good enough in general. I feel it can be more precise and concise in many places. It will be great if the authors can polish more.

### Questions
1. where did you validate the design choices in the adaptive similarity? if there is no, please add

2. any runtime comparison (or complexity analysis)? please Report the computational overhead of A-Hop. should be at least provided to show how the $O(d \log d)$ footprint computation scales with the dimensionality and number of memories. 

LLM disclaimer: I used LLM to polish my language.

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
3

### Summary
This paper proposes an adaptive similarity metric to augment Hopfield Network-based storage and retrieval. It is a common observation that since the task is memory retrieval, if the query deviates from the stored patterns, then the retrieved pattern may not necessarily be 'semantically' close to the query. The authors analyze three particular query variants, namely, noisy variants of patterns, masked variants of patterns, and biased variants of patterns. The basic problem is to predict the most likely stored pattern given a query. The approach taken is to consider the query as a generative variant of the stored patterns and ask the question,  what is the maximum a posteriori probability that a given patttern is likely given the query. To estimate this, they propose an approach in which the similarity between query and stored pattern vectors are computed at the individual dimension level and the vectors are sorted. A combination of Euclidean and cosine distance are used and the softmax operator is applied to select the final stored pattern.

### Strengths
The beginning of the paper was on the right track in terms of addressing a problem since the current limitation of Hopfield networks was their ability to handle conceptual variants of the stored patterns. Modeling the query as a generative variant of the store patterns is also reasonable. Experimental results are indicating that the technique works under the modeled transformations of stored patterns. The proofs of the theorems are provided in the appendix which adds to some of the clarifications.

### Weaknesses
While the approach was well-motivated, the method presented was a bit under-whelming. Why would sorting along each dimensions help in finding similarity. It is also known that such operations can bring dis-similar vectors close together as well and create distractions. The tabular results are too abstract to interpret and not enough time is devoted in the paper. Having an illustration of the method through visual examples in case of image retrieval would strengthen the understanding of the  method. 

The definitions and theorems are stated but not proved in the text which is fine but no reference to the appendix section is made.

### Questions
It would also be useful to address this problem in the context of cross-modal Hopfield retrieval, particularly using textual queries where it will be easier to show performance against truly conceptual variants. For example, if the stored model is a visual description is that of inhalator, queries such as inhaler should still be able to find them

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
This paper challenges the fixed, proximity-based similarity measures in traditional Hopfield networks, arguing they fail to capture context-dependent associations and cannot guarantee "correct retrieval" of a query's true origin. To solve this, the authors introduce the Adaptive Hopfield Network (A-Hop), which reframes retrieval as a probabilistic problem governed by a "variant distribution". A-Hop replaces the fixed metric with a learnable "adaptive similarity" that is built from a similarity footprint of sorted, dimension-wise similarities. The key advantage is that A-Hop's similarity measure is not fixed. It learns from samples to align with the task's specific context, such as noise or masking.

### Strengths
1. The paper's first strength lies in its novel reframing of the retrieval problem in associative memories through a probabilistic framework, and go beyond traditional epsilon-retrieval to a new concept term as "correct retrieval". The goal becomes finding the memory pattern that is the most probable origin of the query, based on a clear mathematical definition: maximizing the a posteriori probability

2. The authors demonstrate that A-Hop achieves state-of-the-art performance across a diverse set of four distinct tasks.

3. The paper is in general well-written.

### Weaknesses
1. My main concern is whether the proposed similarity footprint and adaptive similarity, $s(\xi,x) = w^{\top}U\tilde{q}$, can approximate the posterior distribution, about which the author spends a whole subsection (3.1) discussing the novelty of framing the retrieval problem in this probabilistic way. Maybe I miss some part of the content, but I feel like it lacks a strong theoretical justification for why this specific basis (a linear combination of cumulative sums of sorted dimension-wise similarities) is a universal approximator for an arbitrary likelihood function $p_{\nu}(x|\xi)$.

2. The authors write a detailed and insightful discussion section in the appendix (A.3), but this crucial context is never linked to or integrated into the main text. This discussion addresses fundamental questions (e.g., the trade-off between the provably optimal and the learnable formulation) that should be visible to the reader. It would be great if the authors could move the most critical parts of the discussion section into the main text, likely condensing and replacing some of the dense definition-heavy material in section 3.1.

### Questions
1. Does the author aim to release their code? The current link provided in the paper leads to a folder containing only a readme file.

2. Want to clarify the complexity of the proposed method compared to the modern Hopfield network. The proposed method requires a sort operation ($\mathcal{O}(d \log d)$) for each of the $N$ memory patterns during the similarity calculation. A modern Hopfield network (M-Hop) performs this in $\mathcal{O}(Nd)$ ($d$ for dot product and need to repeat for N memory patterns), but A-Hop increases this to at least $\mathcal{O}(Nd \log d)$, is my understanding correct?

### Soundness
3

### Presentation
3

### Contribution
3
