# Householder-Diagonalized Linear Attention (HDLA): Utilizing Enhanced Decay Mechanism for Efficient Sequence Modeling

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4, 6

## Abstract
Linear attention mechanisms have emerged as efficient alternatives to Softmax attention, exhibiting steady improvements in language modeling capabilities driven by increasingly sophisticated designs for decay matrices—though their structural complexity has typically been limited to the Diagonal-Plus-Rank-1 level. To further advance the understanding and capabilities of linear attention via more complex decay structures, this work makes two primary contributions: (1)  We propose the HDLA linear attention mechanism, which utilizes efficient matrix decomposition to achieve a Diagonal-Plus-Rank-2 structure, thereby extending the decay matrix to a broader, more expressive, rank-enhanced and structured class. (2) We propose a more general chunk-wise parallel algorithm that accommodates both diagonal-plus-rank-$r_{ab}$ decay structure and key-value outer products of rank $r_{kv}$, thus providing a versatile foundation for future research. Comprehensive experiments demonstrate that, compared to linear attention baselines, HDLA sets new SOTA results on language modeling and retrieval tasks at 2.8B parameter scale, delivers at most 80\% and 58.2\% performance gains over baselines on retrieval-based MQAR and RULER tasks, and achieves an average score improvement of 4.39–7.66 on the synthetic MAD benchmark, respectively. Our proposed HDLA model, as well as the rank-generalized chunk-wise parallel algorithm, together provide a versatile algorithmic foundation and promising research prospects for the design of rank-enhanced, structured linear attention mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a diagonal-plus-rank-2 decay / state-transition matrix for linear attention/SSM by using products of householder matrices and diagonal matrices. They also provide a chunk-wise parallel algorithm for this sequential recurrence as a generalization of previous versions with rank 1 constraint. In addition, the authors show that the proposed HDLA method can be viewed as performing online update to the recurrence hidden state, which is a form of “test-time training”. The proposed method achieves good performance on MAD synthetics, MQAR recall tasks, language modeling perplexity and downstream reasoning tasks, retrieval-based S-NIAH tasks, and image classifications.

### Strengths
- The proposed method combines lots of prior efforts (DeltaProduct, Gated Delta Net, etc) to continuously improve the linear attention family of models. This is a good contribution to this line of work.
- The derivation of the chunking algorithm is quite involved but definitely useful for the community.
- Empirical experiments are extensive

### Weaknesses
It’s not immediately obvious how the newly proposed decay matrix is a form of diagonal-plus-rank-2 update. It’s better to lay this out more clearly instead of leaving it to the reader. For general machine learning audience, this is not that obvious so it should be reflected more in the writing.

### Questions
- It’s good to impose new structured assumption in the decay matrix P_t for balancing between efficiency and expressivity, and the chunking algorithm also alleviate some of the computational overhead. However, the proposed HDLA includes a lot more matrix multiplications. In fact, any structured matrix which is formed by a composition of permutation, block-diagonal assumption, summations etc can be viewed as a deep linear transformations which causes more I/O due to sequential matrix loading, computation, and writing between HBM and SRAM. How do you optimize the I/O efficiency here?
- By using a diagonal-plus-2-rank update, you allocate more compute for the decay matrix. But the underlying hidden state matrix S_t is still finite and will still suffer from retrieval failures under long sequence input. What if you allocate that compute not to the decay matrix P_t but to have basically a bit more S_t, i.e. perhaps allocating a few smaller S_t matrix across the sequence dimension. What do you think would be superior here?

### Soundness
3

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
3

### Summary
The paper proposes HDLA which is a linear attention variant that parameterizes a decay matrix as a diagonal matrix with two structured factors computed through generalized Householder matrices. This HDLA attention mechanism performs relative well at a small scale when compared to softmax attention and better than appropriate baselines.

### Strengths
* I find sound and comprehensive the efficiency constraints laid out in section 3.1.1.
* The empirical results compared to different variants of the Gated DeltaProduct are strong and comprehensive.
* I appreciate the connections of HDLA from the TTT perspective.

### Weaknesses
* The community's experience with subquadratic alternatives to softmax attention is that you can find efficiency benefits
for small models (around 7B params) but these benefits fade away as we increase the scale. 
Does your paper advocate to try HDLA for LLMs? Or are you content with the results at the < 3B scale?
I understand that as a researchers we might not posses the access to compute to train such large models,
but is there any indication that HDLA would not suffer diminishing returns at larger scales?

* It is unclear to me how can I incorporate HDLA into a transformer right now. The equations are presented for each
element in the sequence, which is great for the analysis, but then it is unclear how to incorporate this 
method in practice and what implementation details need to be considered.

* There are no runtime comparisons which makes it harder to understand the overhead of HDLA against baselines like DeltaProduct.

### Questions
* Why does criteria P2 presents $H_t$ as an invertible transformation instead of a orthonormal basis? Indeed, $H_t$ is
invertible when it is an orthonormal basis but it is a stronger result from the spectral theorem that you can find such
an $H_t$ for a real symmetric matrix. Moreover, the construction in equation 5 requires orthonormality. Please help me
see if I'm missing a detail.

* I didn't follow the three-step optimization process in equations 19-21 (lines 309-318). I only see an optimization
problem in Eq. 19, the next two lines appear to be just updates. Am I missing something?
Is there any reason as to why you didn't shared code through an anonymized link? I'm wondering how you implemented HDLA
and its chunk-wise parallel version. Especially as you emphasize the I/O cost which to optimize it would require custom
kernels.

* In Table 3, why is Mamba considered a linear attention baseline? Does Mamba not scale as $\mathcal{O}(N \log N)$?, where
$N$ is the sequence length.

* Could you provide runtime comparisons of HDLA for training and inference? Some examples suffice to get an estimate.

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
This paper proposes HDLA, a linear attention mechanism that parameterizes the decay matrix with a structured Diagonal-plus-Rank-2 form. This design aims to improve expressivity over the common Diagonal-plus-Rank-1 family while maintaining parameter, memory, and computational efficiency. The authors also derive a rank-generalized chunk-wise parallel algorithm that supports arbitrary Diagonal-plus-Rank-$r_{ab}$ decay and rank-$r_{kv}$ key–value updates, using a WY-style representation and custom block-triangular operators to enable efficient intra-/inter-chunk computations. Empirically, HDLA outperforms strong linear-attention baselines on language modeling (0.4B/1.45B/2.8B), synthetic benchmarks (MAD, MQAR), and retrieval tasks (RULER), with particularly large gains on MQAR and RULER. The paper also provides a Test-Time Training interpretation of the update rule and reports competitive ImageNet classification performance. Limitations are discussed (recency bias, need for state expansion and potential non-linear updates).

### Strengths
- **Principled architectural design**: The authors motivate and provide a principled, structured Diagonal-plus-Rank-2 decay that extends rank-1 methods while preserving parameter efficiency (O(dk)).
- **Generalized chunk-wise parallel algorithm**: The authors also present a unifying "WY-"style framework accommodating Diagonal-plus-Rank-$r_{ab}$ decay and rank-$r_{kv}$ KV updates, which subsumes several prior works as special cases.
- **Strong and diverse empirical results**: The architecture shows consistent improvements over strong linear-attention baselines (GLA, Mamba2/HGRN2, DeltaNet/Gated DeltaNet, GDP2/3) across LM perplexity at multiple scales, synthetics such as MQAR and MAD, and longer context RULER tasks.

### Weaknesses
- **Presentation**: Some instances could be improved with a bit more detail. 
  - For example, the "WY Representation" should be defined or at least alluded to, e.g., with a reference to [1]. 
  - Separately, while I appreciate the authors' acknowledged comparison to softmax attention, it would be good to see the delta between HDLA and softmax attention on the retrieval tasks such as NIAH or MQAR (i missed these in Table 6 and Figure 2). 
  - To present the advantage of the more expressive Rank-2 structure, could the authors also present a results table (or expand Table 1) that puts side-by-side the model, it's expressivity, and the average empirical performance binned by evaluation (ppl, retrieval acc, LM-eval) ? 
  - Please also fix the first citation: "Sanjeev Arora, Bowen Yang, Selim Eyuboglu, Aditya Narayan, Alexis Hojel, Imanol Trummer, and Christopher Re." fwiw in a random sample of a couple other citations they all seemed to correspond to real author lists. 

- **Wall-clock / "Real World" efficiency**: While I appreciated the complexity analysis and inclusion of efficient algorithms (e.g., chunk-wise parallel), and the claims mostly center on presenting a new decay structure + evaluating it's modeling quality, in being a work on linear attention it would be good to validate how the efficiency translates to real world efficiency boosts, e.g., by measuring wall-clock inference time or generation throughput. 

[1]  Christian Bischof and Charles Van Loan. The WY Representation for Products of Householder Matrices. 1985.

### Questions
Please see Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present an extension of DeltaNet by proposing a more sophisticated decay mechanism. DeltaNet's state update is S_t = (I − βkk^T)S_{t−1} + βkv^T (some dependencies on t suppressed for clarity), which involves a Householder-like transformation. The net effect is to update the state by partially 'erasing' the old target value from the memory and partially 'adding' the new target to the memory. 

In this paper, a new sort of update mechanism is proposed: S_t = (I − βkk^T)Λ(I − βkk^T)S_{t−1}+ kv^T, which the authors show is equivalent to a diagonal-plus-rank-2 decay mechanism. This is more expressive than the decay proposed in DeltaNet, which is a diagonal-plus-rank-1 mechanism. They derive the matrix equations necessary to implement chunkwise parallel training with the more complex update. Their experiments show improvements on an array of synthetic and real-world benchmarks, including Mechanistic
Architecture Design, QA tasks, retrieval, perplexity on Wikitext, etc.

### Strengths
Investigating different decay mechanisms for linear attention is an interesting research direction, and the extension presented in this paper is a very natural one. The proposed method does appear to be novel, contemporaneous work notwithstanding.

### Weaknesses
The results on the non-synthetic benchmarks (Table 7) are very modest. For example, the average common-sense reasoning accuracy improvement with HDLA versus Gated DeltaNet at the 1.45B and 2.8B model sizes are 0.26% and 0.42% respectively. Of course, the rank-2 HDLA should generally outperform the rank-1 methods because it's more expressive and computationally expensive, but the gains are very marginal given the increase in complexity. While there is some discussion of the number of operations per recurrent step in Table 2 of HDLA vs GDP2 and GDP3, it doesn't address the increase in cost versus rank-1 methods. 

For papers that are concerned with efficient methods (as all linear attention papers are), there needs to be some discussion of the tradeoff between model throughput and model performance. The paper is incomplete without some sense of what we are paying in HDLA's added computational cost in exchange for the accuracy/perplexity improvements that it provides. This is not an unusual analysis; I would note that other linear attention papers have addressed this directly (e.g., Yang et al, 2024 compares the training tokens/sec for their DeltaNet implementation against Gated Linear Attention, Transformer++, and Mamba.)

Also, the number of tokens used for LLM training seems to have been limited to 10B tokens (with the exception of the 1.45B models for the comparison between HDLA and Gated DeltaNet, which used 50B), which is quite small in comparison with published work. For example, Yang et al (2024) used 100B tokens from FineWebEdu for its 1.3B models, and Siems et al (2025) used 35B tokens in its experiments. The small training corpus makes it difficult to accept the surprising claim that HDLA outperforms the standard softmax attention mechanism in the Llama model (L407-408) in both perplexity and real-world tasks. Indeed, that finding might not hold up in more realistic training scenarios.

The results are not significantly better than the simpler alternatives on real-world benchmarks, the implementation appears to be significantly more complicated than the rank-1 approaches, and it isn't clear what the throughput penalty is in practice.

### Questions
I hope that the authors will consider including empirical results regarding scaling (as the sequence length increases) and throughput for HDLA versus other methods, and find the tweaks needed to strengthen the results for retrieval tasks in Table 7.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose Householder-Diagonalized Linear Attention (HDLA), a new linear attention formulation that generalizes prior diagonal-plus-rank-1 decay structures to a more expressive diagonal-plus-rank-2 form. Their approach uses efficient matrix decomposition via generalized Householder transformations to achieve increased expressivity without significant computational or memory overhead. The authors also propose a rank-generalized chunk-wise parallel algorithm that extends parallel computation to arbitrary diagonal-plus-rank-r decays and rank-r key-value updates. Their empirical results show that HDLA outperforms different baselines across synthetic, retrieval, and language modeling tasks. It also performs on-par or outperforms in some case, the default softmax attention approach.

### Strengths
- Clear conceptual derivation of HDLA; the two problems it solved are well motivated and the parameterization choices are clearly explained. 
- Thorough comparison to baselines: the comparison to different baselines is captured first through Table 2 in terms of computation costs then through experiments that span different capabilities, datasets, and model sizes. 
- The proposed approach achieves increased expressivity and significantly better results in some settings without significant overhead, making it a good alternative to the default softmax attention. 
- The connection to Test-Time Training formulations in section 3.3 add theoretical depth and understanding of the proposed approach.

### Weaknesses
- The number of tokens used for training seems to follow what's chinchilla optimal for the smallest model (0.4B parameters, about a factor of 20), but it is quite low for larger models: 1.45B and 2.8B (table 7). Under-training these models severely could have a big impact on the results, i.e., results with the correct token budget may look different. 
- The limitations are not discussed thoroughly, e.g., impact of the recency bias (see question below). 
- Intuitive/qualitative analysis for how this approach differs from existing baselines is limited. 

Very minor notes: 
- Need a second pass for typos: eg, in lines 363-364: "interleaved wth arbitrary noisy tokens" -> "interleaved with arbitrary noisy tokens"
- Need to double check the references, eg the first reference is: "Sanjeev Arora, Bowen Yang, Selim Eyuboglu, Aditya Narayan, Alexis Hojel, Imanol Trummer, and Christopher Re. Language models enable simple systems for generating structured views of ´ heterogeneous data lakes. arXiv preprint arXiv:2304.09433, 2023a. URL https://arxiv.org/abs/2304.09433." -> you got the wrong first author name.

### Questions
- You showed that HDLA underperforms softmax attention on Fuzzy In-Context Recall because it requires accurate value prediction from keys interleaved with arbitrary noisy tokens. Can you expand on the consequences of this recency bias limitation? Does this mean that for instance for time series data that might have some higher level structure (e.g., seasonality or periodicity), your approach would not be suitable?
- What is the wall-time comparison between the different approaches? Is HDLA faster for a fixed token budget? 

I am open to raising my score if the questions/weaknesses are properly addressed.

### Soundness
3

### Presentation
3

### Contribution
3
