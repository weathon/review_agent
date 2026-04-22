# Universal Set Transformer: A Scalable and Interpretable Set/Multiset Architecture

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The advent of the set transformer (ST) brought about a new method of permutation equivariant modeling by leveraging cross-element interactions. However, ST is still subject to the fundamental challenge of transformers: scaling efficiently with large input sizes. Mini-batch consistent (MBC) methods were developed to address this problem by maintaining permutation equivariance while alleviating context fragmentation when processing partitioned sets. However, current MBC methods limit expressiveness and render the models incapable of producing element-wise contextualized representations and attention scores for prediction explainability. Therefore, the choice between ST or MBC methods results in a tradeoff between expressiveness and large set processing. To reconcile this tradeoff we propose the Universal Set Transformer (UST), a generalization of ST which is mini-batch consistent without sacrificing expressiveness. Additionally, we introduce multiset attention which leverages the MBC property to significantly reduce the computational cost of processing multisets while maintaining mathematical equivalence with standard multi-head attention. We show that UST is competitive with ST's performance while using less memory and outperforms other MBC methods in various benchmark tasks. Finally, we show that UST is capable of producing both whole-set and element-wise representations and demonstrate prediction explainability via attention scores.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents QUANNs, a learnable aggregation function, for (deep) set models. The learned aggregation is favorable over fixed aggregators in terms of performance.

### Strengths
The formulation of the learnable aggregation function via Neural Kolmogorov Mean is theoretically interesting. Combined with the improved performance and the insights of set function approximation (derivation of derivative and invertability) and theoretical analyses and discussion of the benefits, makes the paper very interesting.

### Weaknesses
The ablation is not clear, a discussion of various (left out / alternative) components and their impact on the performance is lacking.

In Table 3, it is not clear whether the numbers show that the method is capable of the task. The numbers vary greatly across tasks, which makes it hard to assess what the meaning of 'good' performance is here.

In related work, it is not clear to me how the methods from 3.2 are different from 3.1 - I know they are different, but a short explanation would make this section stronger. Regarding slot attention, a reference is missing, Unlocking Slot Attention by Changing Optimal Transport Costs [1]. Because they also address set prediction tasks, it's interesting to see how the proposed method compares to this reference. 

In the experiments, it does not say whether there were tasks with sets of varying sizes within one task. Dealing with variable-sized sets would make the experiments more interesting.

There is a discussion but it lacks a critical view of the learnable aggregator, when it may not be beneficial, etc.

### Questions
Can you discuss the various contributions of components that are reported in the ablation, why they matter, and their impact on the performance?

Can you explain the difference between the methods from 3.1 and 3.2?

Do some of the experiments contain a task with variable-sized sets?

What do the numbers in Table 3 tell us, given that they are so broad in their range?

Are there cases where it is not beneficial to learn the aggregator, but to fix it with a well-chosen function?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a minibatch consistent permutation invariant model based on the set transformer. The model can produce elementwise representations of set while previous methods needed to use an invariant pooling mechanism which condenses the elementwise representations into a single vector or set of vectors.

### Strengths
- The proposed method builds upon the weaknesses of prior works which depend on a pooled set representation.

- The proposed method makes use of a flash-attention-like algorithm in the attention implementation to perform full attention while keeping the MBC property.

### Weaknesses
- L236 states: "We also found the unbiased gradient correction technique proposed by Willette et al. (2023) ineffective in our models." --> The referenced unbiased gradient technique was specifically derived for a model which has only elementwise transformations before an invariant pooling layer. This is the case with DeepSets, MBC, and UMBC. The proposed model uses a completely different architecture with transformer layers, so it is misleading to call the cited method ineffective, as it was not designed for the proposed setup.

-  L257: How can a minibatch containing only a few points perform worse? Is it only because the gradient contains a few points? If this is the case, it should only affect training and not inference, right?

### Questions
- Was UMBC in figure 1 (right) and table 3 trained with the full unbiased approxiamtion scheme?

Other than the Guassian clustering experiment which used DBSCAN to identify multisets by clustering close points, what other algorithms are used to identify multisets? I would be interested to see a latency tradeoff of scanning for multisets and performing the multiset attention vs. just doing regular attention without scanning for multisets. 

For instance, if the set size is very large and there is only on repeat, then the scan would probably take longer than just processing the set normally with repeats. At what rate of repeat elements, does multiset processing show lower latency?

---

My main concern regards whether or not the full unbiased gradient approximation was used in the training of the UMBC baseline as well as the ablation study of the multiset latency.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Universal Set Transformer (UST) which is a variant of the Transformer architecture that is specific for sets. The primary characteristics of the model are: 1. it's mini-batch consistent (MBC), meaning that different partitions of the set can be processed sequentially as "mini-batches" to reduce the peak memory requirements, and 2. it's efficient for multisets by removing repetitions and adding a multiplicity to each element in. On MNIST point could classification UST matches previous methods in the full set setting and outperforms in the mini-batch setting. Furthermore, UST shows favorable performance on both a synthetic clustering and bioinformatics taxonomy task.

### Strengths
- the UST is a clear improvement over previous set based neural network architectures for large input sets and multisets
- supporting multisets by adding the multiplicity is a neat way of supporting repeated elements in MSA

### Weaknesses
- one of the main points that the paper focuses is the memory footprint for large sets for (cross-)attention with O(kn). but papers like "Self-attention Does Not Need O(n^2) Memory" by Rabe and Staats show that this can be avoided by a more efficient implementation. 
- the multiplicity is computed only for the input multiset. in later layers elements can still become more similar and collapse to "effectively equal" elements, hence the proposed architecture does not handle multisets in all generality

### Questions
1. It would be helpful to contextualize the work within the broader literature of efficient Transformer implementations. Does something like Rabe and Staats, 2021, eliminate the need for set-specific architecture designs for memory efficiency?
2. How do you compute multiplicity for similar but not equal elements? Or do elements count as repitions only when they are exactly equal?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- The Universal Set Transformer (UST) is a new architecture that enables scalable and interpretable transformer-based modeling of sets and multisets.
- It introduces a mathematically consistent mini-batch processing framework (MBC) that ensures identical results whether a set is processed all at once or in shards.
- UST adds Multiset Attention (MSA), which efficiently handles duplicate elements and reduces computational cost while maintaining full expressivity.
- Experiments show that UST matches or exceeds prior models' accuracy while using less memory and scaling effectively to very large sets.

### Strengths
- The paper is well-written, clearly structured, and effectively explains complex concepts like mini-batch consistency and multiset attention.
- The paper introduces the first transformer architecture that achieves true mini-batch consistency while preserving full self-attention expressivity.
- The architecture maintains attention-based interpretability while achieving strong accuracy and scalability across diverse tasks.

### Weaknesses
- UST still requires storing a full set instance before pooling, which limits true constant-memory scalability.
- The experimental evaluation is mainly limited to controlled benchmark and bioinformatics datasets, making it unclear how the model would perform on large-scale, real-world applications.
- The paper lacks a detailed analysis of interpretability, providing limited evidence on how attention scores meaningfully explain model predictions.
- Comparisons with more recent efficient transformer variants or non-attention-based set models are missing.

### Questions
- Are the gradients in UST theoretically equivalent to those obtained from full-set training, or only approximately consistent?
- Is there any quantitative or qualitative evidence that attention scores in UST provide meaningful interpretability?
- How does UST perform when applied to variable-sized sets with very different cardinalities across samples?
- Can the proposed framework be combined with sparse attention methods to further reduce computational cost?
- Some citations look strange to me. For example, "Deep Sets (Zaheer et al. (2017)) and FSPool (Zhang et al. (2020))" should be "Deep Sets (Zaheer et al., 2017) and FSPool (Zhang et al., 2020)."
- In Line 256, please check grammar for "Mini-batches containing few points cause these models suffer from context fragmentation"

### Soundness
3

### Presentation
3

### Contribution
3
