# Memory-Efficient LLM Pretraining via Minimalist Optimizer Design

- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Training large language models (LLMs) typically relies on adaptive optimizers such as Adam, which introduce extra operations and require significant more memory to maintain first- and second-order moments than SGD. While recent works such as GaLore, Fira and APOLLO have proposed state-compressed variants to reduce memory consumption, a fundamental question remains: *What are the minimum modifications to plain SGD needed to match state-of-the-art pretraining performance?* We systematically investigate this question using a bottom-up approach, and identify two simple yet highly (memory- and compute-) efficient techniques: (1) column-wise gradient normalization (normalizing the gradient along the output dimension), which boosts SGD performance without momentum; and (2) applying first-order momentum only to the output layer, where gradient variance is highest.  Combining these two techniques lead to SCALE (Stochastic Column-normAlized Last-layer momEntum), a simple optimizer for memory efficient pretraining. Across multiple LLaMA models (60M–1B), SCALE matches or exceeds the performance of Adam while using only 35–45\% of the total memory. It also consistently outperforms memory-efficient optimizers such as GaLore, Fira and APOLLO, making it a strong candidate for large-scale pretraining under memory constraints. For LLaMA 7B model, SCALE outperforms the state-of-the-art memory-efficient methods APOLLO and Muon, in terms of both perplexity and memory consumption.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles the problem of reducing GPU memory footprint in LLM pretraining by simplifying the optimizer. Standard adaptive optimizers like Adam require storing first and second moment statistics for each parameter which triples the memory requirement for "persistent" (i.e. non activation or gradient) tensors. The authors pose a central question: what are the minimum modifications to SGD needed to achieve SOTA pretraining performance? They pursue a bottom-up, minimalist design, systematically testing fundamental components (gradient normalization and momentum) to bridge the gap between SGD and Adam with minimal memory cost.

the paper identifies two key techniques that dramatically improve SGD: 
1. Column-wise gradient normalization (normalize gradients along the output dimension of each layer), which boosts training effectiveness with a simple closed-form scaling and no extra memory use
2. First-order momentum applied only to the last layer, where gradient variance is highest, to stabilize and accelerate learning with negligible memory overhead. Combining these yields a new optimizer called SCALE, which uses roughly the same memory as vanilla SGD. Notably, SCALE achieves performance on par with Adam and other SOTA optimizers while using a fraction of the memory. For example, on a 1B-parameter LLaMA model, SCALE reaches similar final perplexity to Adam (and Muon) while using only 35–52% of the memory that Adam/Muon require. Compared to other recent memory-efficient optimizers (e.g. GaLore, Fira, Apollo), SCALE attains better perplexity with only about 59% of their memory cost on a 1B model.

In summary, the paper’s primary contributions are:
1. Defining a minimalist optimizer design approach for LLM training under memory constraints, and executing a principled study to find the smallest necessary improvements to SGD
2. Introducing SCALE, a simple two-component optimizer (column-normalized gradients + last-layer momentum) that is memory-efficient and performant
3. Empirical validation across multiple LLM scales (60M, 130M, 350M, 1B, and a partial 7B) showing SCALE matches or exceeds Adam’s performance while drastically reducing memory usage
4. Theoretical insight into why these minimal modifications suffice: the authors prove that momentum yields the most benefit on layers with high gradient variance, justifying concentrating momentum in the last layer. They also analyze different normalization schemes to explain why column-wise normalization is especially effective for stabilizing training

### Strengths
1. Thorough experimental approach: the authors didn’t jump straight to proposing an algorithm. Instead, they earned the design of SCALE through systematic exploration. They ran extensive ablations to dissect what matters: for example, testing four different normalization strategies across multiple model sizes (60M -> 350M) and demonstrating that while all improve upon vanilla SGD, only column-wise or singular-value normalization come close to bridging the gap. Similarly, they evaluated momentum placement by trying momentum in the last layer and showing it dramatically boosts performance, especially for larger models where _"both singular-value and column-wise normalization + last-layer momentum are matching Adam’s performance"_. This methodical approach builds confidence that the chosen techniques are indeed the critical ones. The paper effectively rules out alternatives (e.g., it shows why row-wise normalization fails by tracing it to unstable gradient distributions), which strengthens the validity of their conclusions.
2. The resulting SCALE optimizer is simple by design. It requires only minimal modifications to existing SGD/Adam implementations to achieves SOTA results. Practitioners can easily adopt SCALE without complex infrastructure changes or hyper-parameter gymnastics. The elegance of using one normalization and one localized momentum is that it’s easy to maintain and understand. The authors also connect this simplicity to existing ideas (showing how it relates to prior works but with fewer components) so the novelty doesn’t come at the cost of obscurity. 
3. The empirical results demonstrate that SCALE offers outstanding memory-perplexity trade-offs at scale. The strength here is the practical efficiency gain: large savings in memory without compromise in model quality. Also the method’s benefits increase with model size. The paper notes that as model size grows, SCALE matches or even exceeds the performance of Adam and others. For example, by 350M parameters, “column-wise + last-momentum” outperforms the tuned Adam (Stable-SPAM) baseline in perplexity. At 1B, SCALE ties with the best baseline, and on a 7B partial run, SCALE achieved a lower perplexity than both Muon and Apollo-Mini under the same training length.
4. Despite introducing an extra normalization step each iteration, SCALE’s design keeps compute costs low. Appendix Table 7 shows that SCALE’s training speed in tokens/sec is essentially the same as Adam and even slightly higher in their setup. This is a significant strength because it means the memory savings do not come at the cost of slower training, which is a common pitfall for some compressed optimizers that spend extra cycles on state transformations.
5. Beyond raw performance, the work provides insights that strengthen our understanding of optimization. The identification of the last layer’s gradient variance as a key issue is backed by both an empirical plot (variance curves) and a theoretical argument (Theorem 2.1). This realistically explans why momentum should applied only to the last layer. It suggests the approach is built on solid principles (variance reduction and convergence analysis) rather than just trial-and-error.

### Weaknesses
While this paper is very strong, there are a few aspects that could be seen as weaknesses or areas for improvement:
1. The optimizer does not introduce fundamentally new primitives beyond what’s known. One could argue that the contribution is more in the clever combination and insight rather than a brand-new algorithmic concept as it still builds upon gradient normalization and momentum.
2. The experiments focus exclusively on transformer-based LLM pretraining (on C4 dataset). It remains unclear how well it generalizes to other domains or tasks such as vision models and post-training. This isn’t exactly a weakness of what’s done (the paper already has an impressive array of experiments), but it leaves an open question about generality.

### Questions
1. The study identifies the last layer as having the highest gradient variance and thus focuses momentum there. Did the authors consider or experiment with applying momentum to the first (embedding) layer as well, since Figure 4a shows the embedding layer had the next-largest variance, while at a significantly smaller scale?
2. By design, SCALE foregoes second-order moment. While the paper’s results suggest this isn’t needed for the tested scenario, are there cases where neglecting second-order adaptivity might hurt?
3. Echoing 2nd point in Weaknesses section, while the results for language model pretraining are strong, do the authors have any preliminary observations on how SCALE performs in other settings?
4. It's mentioned in appendix that SCALE in practice uses Adam (or full adaptivity) for some "vector" parts of the model that are small (and possibly critical like embeddings?). Could the authors elaborate on this choice? For example, did they notice any degradation if even those vector parameters were trained with the simplified optimizer? And are the embedding word vectors treated as “matrix” (since they are large) or “vector” in this context?
5. (Minor) the authors might consider releasing pseudocode or a snippet illustrating the few lines of change needed to implement SCALE in a standard training loop to encourage adoption

### Soundness
4

### Presentation
4

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
This paper studies modifications of gradient descent to enable faster training for large language models. The analyze the importance of gradient normalizations and momentums, and how training is affected. Thereby, they introduce this new training scheme called SCALE, which includes column-wise gradient descent normalization, along with first-order momentum only to the final layer of the LLM. Using this method, it has been empirically shown that SCALE performs well, often better and more memory efficient than Adam optimizer on models with a large number of parameters.

### Strengths
- The paper is easy to understand, the results are neatly written and presented.
- There is an extensive comparison of several previously known optimization techniques along with the proposed ones.
- The algorithm for SCALE is quite simple, yet it outperforms known optimization techniques.
- The analysis of the ideas are also displayed well.

### Weaknesses
- From my understanding, the ideas are not very novel as most of the techniques have been used before.
- I think the proposed method is not very general. They chose to include momentum only for the last LLM layer since observed variance is high in the last layer. However, this might not always be the case, and a complete optimization scheme needs to be more adaptive to general architectures.
- It will look good to write the final term for updating \theta_t after equation 5.
- Please define the norms ||.||_{a -> b}.
- The notation is somewhat confusing, for instance, in line 383 of algorithm 1, g^t_l is computed and it's hard to tell where it's used since it doesn't appear anywhere else in the algorithm.
- The proof of theorem 2.1 is hard to follow since it contains a series of equations. It will be helpful to give a larger overview of the proof and highlighting the steps to prove them, and the purpose of the lemmas. 
- Also, please mention an overview of the proofs of the lemmas, and how the inequalities follow from each other, since they are quite long and tedious to verify.

### Questions
- Is variance of stochastic gradient computed over batches or individual training data?
- Could you discuss the weaknesses above, especially giving proof intuitions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes SCALE (Stochastic Column-normalized Last-layer Momentum) — a minimalist, memory-efficient optimizer for LLM pretraining. Instead of modifying Adam or introducing compression subspaces like GaLore, Fira, or APOLLO, the authors take a bottom-up approach to identify the minimal ingredients necessary for high-performance pretraining under tight memory budgets. They find that (i) column-wise gradient normalization and (ii) applying first-order momentum only to the last layer are sufficient to achieve Adam-level performance while consuming only ~35–45% of the memory.
Extensive experiments on LLaMA models (60M–7B) show that SCALE matches or surpasses Adam and state-of-the-art memory-efficient optimizers (APOLLO, Muon, SWAN) in perplexity–memory trade-offs.

### Strengths
* **Clarity and thoroughness**： The methodology is clearly articulated，and easy to understand.
* **Memory saving**: SCALE is lightweight, simple to implement, and achieves Adam-like performance at one-third the memory cost, even beat the SOTA APOLLO.
* **Originality**: The paper clearly present how to ablate normalization and momentum mechanisms to find the essential ingredients that make adaptive optimizers effective. This is good since whether AdamW is good for high-dim LLM optimziation should be challenged.

### Weaknesses
I have one major concern, which is the generality of the techniques in the paper:
* **Potential overfitting to a single model family.**
The main design choices (column-wise normalization and last-layer momentum) are validated only on LLaMA-style architectures. It remains unclear whether these techniques generalize to other architectures. This is important as the paper's contribution is building a minimal optimizer via ablation; how the conclusion can be extended would be the main concern or flaw of this paper. This is my main reason to give a negative score now, but I am happy to see more results to show the generality of the techniques.

### Questions
* It remains unclear whether these techniques generalize to other architectures.
* Lack of fine-tuning or SFT experiments. The results focus exclusively on pretraining. Adding SFT experioments would be more convincing.

### Soundness
3

### Presentation
4

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
This paper investigates the minimal modifications required to make SGD competitive with adaptive optimizers like Adam for large-scale LLM pretraining. Through a systematic ablation, the authors identify two key components: (i) column-wise gradient normalization and (ii) first-order momentum applied only to the last layer, and then integrate them into a new optimizer, SCALE (Stochastic Column-normalized Last-layer momEntum). SCALE matches or exceeds Adam-level performance while using only 35–45% of the memory across LLaMA models ranging from 60M to 7B parameters.

### Strengths
- The paper addresses a well-defined and practically important question: identifying the minimal components required for memory-efficient yet high-performing LLM training.

- The set of baselines covered is comprehensive, and the systematic investigation of essential design elements is solid and convincing.

- SCALE demonstrates an outstanding memory–perplexity trade-off, effectively establishing a new Pareto frontier in optimizer efficiency.

### Weaknesses
Since the proposed method mainly combines known techniques (normalization and partial momentum) rather than introducing new algorithmic concepts, it would strengthen the paper to provide a deeper analysis of the normalization component. For instance, the poor performance of row-wise normalization appears to stem from the LM head layer. If the last layer is excluded, it would be interesting to investigate whether different layers exhibit specific preferences for normalization granularity, such as normalizing along the smaller or larger dimension of the gradient tensor. Additionally, exploring block-wise normalization (e.g., aligned with attention heads?) or developing a variance-based, principled normalization criterion could yield more general insights. Such a deeper analysis would make the work more convincing and theoretically insightful.

I am still somewhat unclear about why the combination of SGD, normalization, and partial momentum can achieve performance comparable to Adam. Could the authors provide deeper insights or theoretical intuition into this behavior? Is the observed effectiveness related to properties of the data distribution or gradient statistics?

### Questions
Please refer to weakness

### Soundness
3

### Presentation
3

### Contribution
3
