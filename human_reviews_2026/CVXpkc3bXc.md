# Kronecker Factorization Improves Efficiency and Interpretability of Sparse Autoencoders

- Avg Score: 5.20
- Decision: Reject
- Scores: 2, 2, 6, 10, 6

## Abstract
Sparse Autoencoders (SAEs) have demonstrated significant promise in interpreting the hidden states of language models by decomposing them into interpretable latent directions. However, training and interpreting SAEs at scale remains challenging, especially when large dictionary sizes are used. While decoders can leverage sparse-aware kernels for efficiency, encoders still require computationally intensive linear operations with large output dimensions. To address this, we propose **KronSAE** – a novel architecture that factorizes the latent representation via Kronecker product decomposition, drastically reducing memory and computational overhead. Furthermore, we introduce mAND, a differentiable activation function approximating the binary AND operation, which improves interpretability and performance in our factorized framework.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a new variant of dictionary learning for language model interpretability. Motivated by increasing computational efficiency, the Kroneker encoder trains a 2-level hierarchy, composing multiple encoders that each operate on distinct subspaces of the embedding space.

### Strengths
- The architecture is substantially more parameter efficient that existing SAE architectures.
- The toy model of correlation section provides evidence of improved learning of correlations compared to TopK SAEs.

### Weaknesses
1a. In my view, the main novelty of this work is imposing a prior on which structures to learn: a 2-level hierarchy, for subspaces of the residual stream space. While I agree this architecture improves over compute efficiency, the paper needs a better motivation for the structural prior. Why do we expect language models to learn this specific hierarchical structure? What kind of features is a TopK SAE not able to learn, while the Kron SAE is?

1b. The analysis section provides a qualitative discussion of examples of learned feature hierarchies. Table 2 exemplifies a polysemantic base component that extends to seemingly unrelated concepts: comparative words and directional words and spiritual words. Quantifying the extent to which ground truth concept hierarches in natural language are identified but the KronSAE. Overall, the hierarchical nature of learned features remains underexplored. The Feature Absorbtion evaluation and SAEBench results provide a useful signal to compare performance of existing saes, but does not directly evaluate the recovery of feature hierarchies. 

2. The hierarchical prior of KronSAEs is related to the prior of Matryoshka SAEs. I'd like to see a baseline of Matryoshka SAE scores on all evaluations.

### Questions
Why are heads only operating on distinct subspaces of the residual streams. What happens to LM features that are an element of the union over input spaces of multiple heads?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new sparse autoencoder (SAE) architecture, KronSAE, where the SAE's encoder consists of multiple Kronecker-factored blocks. They present iso-FLOP comparisons between KronSAEs and TopK SAEs in terms of reconstruction, feature absorption, and interpretability.

### Strengths
1. The KronSAE architecture is clearly described.
2. In some sections, the authors sweep over certain key hyperparameters (F and m) to understand their effect.
3. I found the discussion of the relationship between pre- and post-latent interpretations interesting.

### Weaknesses
Overall I would like to see more systematic reporting of results:
1. Multiple models. KronSAEs are trained on three models, but most results are only reported for one model. This makes it difficult to tell if results are cherry-picked.
2. Consistent values of m and F. The reconstruction performance results are shown for multiple values of m, with m=1 being the best. But later, only m=4 is shown for interpretability results. Since, in general, we should expect there to be a trade-off between reconstruction and sparsity (which is typically correlated with interpretability), I worry that there is a tradeoff to KronSAEs which is not being shown here.
3. Multiple sparsities. Figures 3 and 4 sweep over multiple sparsities; ideally all of the plots would do the same.
Overall, I would find it much easier to understand the results if the plots in this paper were replaced with line plots where the x-axis was sparsity, and there were multiple lines corresponding to different values of m. There should be one such plot for each model (though some of them can be reported in the appendix).

Other notes:
1. KronSAEs are only compared to TopK SAEs. As the authors note, Matryoshka SAEs are an idea in a somewhat similar vein. So it would be better if KronSAEs were also compared at least against Matryoshka SAEs as well.
2. Reconstruction results are best when m=1, but this is also the case when KronSAEs are most similar to the standard architecture; the reconstruction hit is more substantial for m=4,8.

If these concerned are addressed, I could see myself raising the score to as high as 6.

### Questions
1. Is there a reason we would want to localize correlated features to the same head? The paper writes about this as if it's a desirable property, but it's not clear to me why it matters. 
2. This paper makes the choice to study KronSAEs in a resource-constrained setting, i.e. where SAEs are trained for <=1B tokens. I'm curious if you have any sense of what the results look like when training for longer.

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
This paper introduces a new SAE variant with a structured encoder, called KronSAE. Instead of using a since encoder matrix, this work uses a sum of smaller Kronecker factored matrices. While it doesn't improve training speed or final reconstruction, it improves over the baseline in two domains:
- Lower parameter count and all the subsequent advantages.
- A clustering-based (or AND-based) prior for feature extraction, reducing feature splitting.

This leads to extracted features that are more useful for interpretation over the baseline.

### Strengths
Introducing structured priors into SAEs is useful. This offers both regularization for the training process but also helps interpretability of latents. I think this paper hence makes meaningful progress in an important domain.

Despite my later comments nit-picking sections, the overall presentation is sound. The text is clearly structured and the story makes sense.

The presented experiments are thorough and most my questions were answered immediately.

### Weaknesses
--- **Weaknesses** ---

The correlation experiment seems heavily favoured towards your approach since this is precisely what the Kronecker structure relies on for extraction. It's nice to see the KronSAE succeeds but I'm fairly certain there's an equally contrived experiment where TopK will find ground truth structure much better than KronSAEs. Actually, looking at Figure 9, it seems that this correlation plot is basically the same regardless of the original patterns, which further undermines this qualitative experiment.

Not sure why mAND is featured so prominently, if the simpler setup of simply doing $u*v$ is only 1% worse, why not just use that? If I'm not mistaken, this is extremely akin to other efficient matrix factorizations like Butterfly/Monarch matrices. The use of these efficient parameterizations is not new in deep learning [1] and a more thorough literature discussion mentioning this would be useful.

[1]: https://arxiv.org/abs/2204.00595

--- **Improvements** ---


The notation was also a bit confusing to me because bold characters are often used to indicate something being a vector of the same matrix (especially if indexed), e.g. $p^k$ is row $k$ of $P$. While it's the text makes it clear, I recommend changing notation a bit. On a related note, why does table 1 use $u$ and $v$ rather than $p$ and $q$, they're the same thing right?

While satisfactory, the explanation in Section 3 could be easily improved by adding a diagram. I saw there is one in appendix F but it uses strange notation (and doesn't include the top-k). The diagram should be (in einops notation), einsum(x, x, Q, P, "... xtop, ... xbot, xtop q h, xbot p h -> (p q h)"). The Kronecker product is implicit.

Figure is quite hard to read. perhaps there's a nicer way to present this? Also, since you introduced a new metric it seem useful to walk the reader through what they're seeing here. It took some time to figure that out myself.

In general, many of the figures are too cluttered to read easily. I suggest extracting the important bits and moving the full figure to the appendix or something.

### Questions
*"We had not observed any notable differences in feature geometry between TopK and our SAEs"*\
How did you measure or observe this?

If mAND just the RELU variant but with a square root? What's the reason this was preferred outside the slightly higher reconstruction scores?

You mention the KronSAE is unstable w.r.t n, m and h. What happens qualitatively to the features? Any idea why it fails for specific setups? Answering these questions would provide some insight into the representation structure.

Do you intend to share the code?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper introduces the KronSAE Sparse AutoEncoder (SAE) architecture, which decomposes latent spaces using Kronecker-factorization across multiple heads. Theoretically, KronSAE improves computational efficiency relative to prior SAE approaches; and empirically, authors show across a wide range of experiments and ablations that KronSAE substantially reduces feature absorption, better captures compositional structure in latent spaces, and improves interpretability of learned features.

### Strengths
The work introduces an innovative SAE architecture, KronSAE, that makes substantive improvements over prior SAEs in several dimensions (efficiency, feature absorption, compositionality, and interpretability), and authors clearly demonstrates these improvements empirically across comprehensive experiments. Each of these improvements presents real value to the community in their own right; and taken together, KronSAE represents a clear and significant contribution to the SAE literature.

### Weaknesses
I do not see any particularly substantive weaknesses in terms of the technical contributions and empirical work presented in the paper.

My one concern is that the paper fails to cite any works from the very closely-related research area of tensor product representation (TPR). TPR, first introduced in 1990 [1], has long studied how to encode compositional representations in dense embedding vectors via tensor products (a generalization of the Kronecker product), with more recent works leveraging TPR to interpret compositional representations in LLMs [2-5].
- Note that I do not believe this significantly erodes the novelty of the paper -- to my knowledge, there is no work applying TPR directly to SAEs, which I understand as being the primary contribution in this work -- but it is important to acknowledge this large body of highly relevant work and compare it with KronSAE. For instance, [3-4] also leverage dictionary learning with TPR to interpret neural representations; but KronSAE requires less advance knowledge of specific role/filler features to look for, and is more general in terms of where it can be applied to interpret model representations (e.g., [4] requires approximating the mapping from input token embeddings (fillers) all the way to the layer whose activations are being reconstructed; whereas SAEs, including KronSAE, can be easily applied to any layer without having to approximate the full model up to that layer).

[1] Smolensky, P. (1990). Tensor product variable binding and the representation of symbolic structures in connectionist systems. Artificial intelligence, 46(1-2), 159-216.            
[2] Smolensky, P., McCoy, R., Fernandez, R., Goldrick, M., & Gao, J. (2022). Neurocompositional computing: From the central paradox of cognition to a new generation of ai systems. AI Magazine, 43(3), 308-322.                 
[3] McCoy, R. T., Linzen, T., Dunbar, E., & Smolensky, P. (2019). RNNs implicitly implement tensor-product representations. In International Conference on Learning Representations.                                
[4] Soulos, P., McCoy, R. T., Linzen, T., & Smolensky, P. (2020, November). Discovering the compositional structure of vector representations with role learning networks. In Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP (pp. 238-254).                                
[5] Smolensky, P., Fernandez, R., Zhou, Z. H., Opper, M., & Gao, J. (2024). Mechanisms of symbol processing for in-context learning in transformer networks. arXiv preprint arXiv:2410.17498.

### Questions
The authors explain KronSAE's improvements on reducing feature absorption in sec 4.2 as follows:
- Per "Smooth mAND activation", you state that "we introduce a differentiable AND gate [(mAND)] that prevents a broadly polysemantic primitive from entirely subsuming a more specific one." *It is not clear to me why this would be the case. Can you explain this rationale in greater detail?* (That is: I understand that, empirically, KronSAE demonstrably reduces absorption per the SAEBench tests -- which is impressive and significant -- and the second explanation of "Head-wise Cartesian decomposition" also seems intuitive; I just find the "Smooth mAND activation" explanation to be quite opaque. I think the second explanation is already sufficient to make your point, but if you can better clarify the first point as well, that would be helpful.)

One additional note for the authors is: I personally find the improvements in feature absorption (per sec 4.2) and interpretability/specificity/compositionality (per sec 5.3) to be much more significant contributions than what seem like comparatively modest efficiency gains. As such, I feel that motivating this work primarily from the perspective of efficiency (per the abstract and introduction) "undersells" the contribution.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces what the authors call KronSAE, an SAE architecture with an idea to address the scalability and interpretability challenges associated with large dictionary sizes in traditional SAEs. The primary optimisation involves factorising the latent representation using Kronecker product decomposition, which aims to reduce the computational and memory overhead of the dense encoder projection. This was identified as one of the key bottlenecks. The architecture also uses mAND, a differentiable activation that approximates the logical AND operation, encouraging compositional structure and improving latent quality.

### Strengths
- Applying Kronecker product decomposition to SAE seems new.  The tensor factorisation in model compression have been considered before (e.g. Edalati et al. 2021) but the specific application to SAE latent spaces with head-wise decomposition as far as the reviewer can see is novel.

- Aiming to address both computational efficiency (encoder bottleneck) as well as interpretability (compositional structure) is  a more ambitious task than usual optimisation-first approaches.

- The authors use a reasonably comprehensive exprimental setup involving 3 LLMs (Qwen, Pythia, Gemma), 3 dictionary sizes  (32k-131k), multiple token budgets.

- The paper is overall written with clarity and nicely presented figures, although some of the labels are too small (most figures).

### Weaknesses
- The paper mentions that Kronecker factorisation induces compositional/hierarchical features. But the mechanism seems to be underspecified and not rigorously justified/described

- The mAND activation (Eq. 3) is one of the important elements of the method but the paper seems to lack principled justification beyond empirical performance. Why square root? How closely does mAND approximate true binary AND? (Fig. 10)  How much the smoothness introduces unwanted activations (e.g. false positives in logical sense)?

- As considered by Bussmann at al. 2025, “Matryoshka SAEs” impose feature hierarchy by nested training. Is there a direct comparison in expriments? How similar/different is KronSAE’s hierarchy from Matryoshka?

### Questions
- Can you provide a more formal theoretical justification as for why Kronecker factorisation induces hierarchical/compositional features? What mathematical properties of the Kronecker product lead to semantic compositionality?

- Could you provide a more direct comparison to Matryoshka SAEs (Busmann et al, 2025), Switch SAEs (2025) and Gated SAEs (Rajamonoharan et al. 2024)? Since you briefly discuss these methods improve efficiency, but how does KronSAE compare quantitatively? Can you provide a more direct comparison under perhaps equal FLOP/paraemter budget?

### Soundness
2

### Presentation
3

### Contribution
2
