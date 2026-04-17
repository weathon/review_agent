# Context-Aware Contrastive Surrogates for Scaling-up Expensive Multiobjective Optimization

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Addressing expensive multiobjective optimization problems (EMOPs) poses a significant challenge due to the high cost of objective evaluations. We propose FSMOEA, a scalable and efficient framework that enhances surrogate-assisted multiobjective evolutionary algorithms (SMOEAs) by introducing foresighted surrogate models. FSMOEA captures population-level context to improve surrogate prediction accuracy, leverages a low-dimensional latent space to accelerate evolutionary search, and employs lightweight models to reduce computational overhead. Designed for plug-and-play integration, the foresight model can be embedded into existing contrastive (i.e., classification- and relation-based) SMOEAs, improving performance on scaling-up EMOPs. We provide theoretical analysis that formalizes the benefits of population-aware representation and latent-space optimization. Extensive experiments on 107 benchmarks show that FSMOEA consistently outperforms state-of-the-art methods in both convergence speed and optimization quality. Source code is attached and will be available at Linkxxx.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FSMOEA, a framework for expensive multiobjective optimization problems that enhances surrogate-assisted multiobjective evolutionary algorithms with foresighted surrogate models. FSMOEA uses an MLP-based autoencoder to capture population-level context (via a low-dimensional latent space), accelerates evolutionary search in this latent space, and employs lightweight models to reduce computational overhead. It is compatible with existing contrastive SMOEAs (e.g., CSEA, REMO) and evaluated on 107 benchmarks. Experimental results show FSMOEA outperforms state-of-the-art methods in convergence speed and optimization quality.

### Strengths
1. By replacing context-free surrogates with a population-aware autoencoder, FSMOEA addresses a key limitation of existing contrastive SMOEAs, i.e., biased predictions due to population drift. This design is theoretically justified and empirically validated in high-dimensional benchmarks.
2. FSMOEA’s foresight module can be seamlessly embedded into existing classification- or relation-based SMOEAs.
3. Experiments cover diverse EMOP scenarios (multi/many-objective, high-dimensional, real-world), with ablation studies verifying core components.

### Weaknesses
1. The paper mentions PCA-based dimensionality reduction but fails to deeply contrast FSMOEA’s autoencoder with other latent-space methods (e.g., VAE, contrastive learning). It does not clarify why autoencoders better capture population context than these alternatives.
2. FSMOEA succeeds on the constrained TREE domain but lacks explicit constraint-handling mechanisms. It is not compared to constraint-aware SMOEAs, and the paper does not explain why unconstrained design works for TREE.
3. FSMOEA’s autoencoder trains on the current population, but the paper does not evaluate performance when initial populations are low-quality or low-diversity. It is unknown if poor early encoding biases subsequent search.
4. The paper claims “lightweight” models but provides no quantitative comparison of training/inference time with other surrogates (e.g., Kriging, radial basis functions). For EMOPs with ultra-tight budgets (\(FE_{max}<100\)), it is unclear if autoencoder training overhead is acceptable.
5. Theoretical analysis relies on Lipschitz continuity of objectives and small population movement. However, the paper does not verify if these assumptions hold for the tested real-world benchmarks.

### Questions
1. Can authors provide a quantitative comparison between FSMOEA’s autoencoder and alternative dimensionality reduction methods (e.g., PCA, VAE) in terms of population context capture and surrogate prediction accuracy?
2. What is the explicit rationale for choosing k=10 as the default latent dimension? Have you explored adaptive strategies to adjust k dynamically based on reconstruction loss or problem complexity?
3. Since FSMOEA lacks explicit constraint-handling mechanisms, why does it succeed on the TREE domain? Can you compare its performance to constraint-aware SMOEAs on this task?
4. Can authors provide quantitative data (e.g., training time per generation, inference latency) comparing FSMOEA’s autoencoder-based surrogate to other common surrogates (e.g., Kriging, support vector regression) for EMOPs with tight evaluation budgets?
5. How does FSMOEA perform when initial populations are low-quality or low-diversity? Do authors have measures to mitigate biases from poor early-stage autoencoder training?

### Soundness
2

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
4

### Summary
The manuscript presents FSMOEA (Foresighted Surrogate-Assisted Multiobjective Evolutionary Algorithm), a framework designed to enhance existing SMOEAs for expensive multiobjective optimization problems. FSMOEA captures population-level context to improve surrogate prediction accuracy, leverages a low-dimensional latent space to accelerate evolutionary search, and employs lightweight models to reduce computational overhead. FSMOEA demonstrates consistent superiority over state-of-the-art algorithms in convergence speed and solution quality across 107 benchmarks, supported by both theoretical analysis and extensive empirical evidence.  FSMOEA’s scalable, efficient, and plug-and-play design represents an effective solution for surrogate-assisted optimization.

### Strengths
1.	The paper is well-written and easy to follow, with detailed explanations of its methodology. 
2.	The experimental evaluation is comprehensive, confirming that the proposed plug-and-play FSMOEA consistently outperforms state-of-the-art methods in both convergence speed and optimization quality. 
3.	The paper provides the theoretical analysis of FSMOEA’s key components.

### Weaknesses
1.	In terms of methodology, this paper leverages a low-dimensional latent space to accelerate evolutionary search, but this idea is not entirely new. 
2.	The paper does not provide a discussion or comparison with other dimensionality reduction techniques.  
3.	Literature reviews about existing methods (especially the latest methods) need to be summarized and compared with your work. In addition, sufficient comparisons with the latest methods are essential to enhance the validity of the results. 
4.	The evaluation should be supplemented with weakly Pareto-compliant indicators (e.g., IGD+) and complemented by a Friedman test to derive a statistical performance ranking.

### Questions
1.	FSMOEA claims that it embeds population-aware context into the dimensionality reduction process. What does "population-aware context" specifically refer to?
2.	FSMOEA employs lightweight models to reduce computational overhead, but the manuscript lacks discussion on computational complexity.
3.	Please go through the entire manuscript to double check the grammar, language usage and reference. For example, the errors in some references (the capitalization of journal titles is inconsistent); Figure 6 is a table，not a figure. 
4.	See more on Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a surrogate-assisted multi-objective evolutionary algorithm for expensive and high-dimensional optimization problems. The key idea is to introduce a population-aware latent encoder that learns low-dimensional embeddings of the current population, allowing the surrogate to be trained and searched in this latent space.

### Strengths
1) The idea of integrating a population-aware encoder with surrogate modeling is interesting and has potential to improve scalability in high-dimensional settings.

2) The proposed method is evaluated on an extensive set of synthetic and real-world benchmarks.

### Weaknesses
1) Calling a context-aware model in multi-objective optimization sounds a bit strange to me. Most optimization methods in the multi-objecitve context considers distribution of solutions (e.g., by an indicator like hypervolume ) as they aim to find a good representation of the Pareto front.

2) There are some theoretical results, but it is unclear how closely they relate to the proposed method. 

3) The paper states that MOBO is closely related, but no BO method is included in the comparison study. Some important and representative works such as [3–5] should be included in the discussion and experimental comparison.

4) In line 57, the author(s) state that "regression-based surrogates often suffer from modeling inaccuracies in high-dimensional or sparse-data regimes and contrastive SMOEAs bypass the need to predict exact objective values and is often more robust under data scarcity.'' However, several recent studies [1,2] demonstrate that regression-based surrogates can perform effectively on high-dimensional expensive multi-objective problems. 

[1] Horaguchi, Y. and Nakata, M., 2025, July. High-Dimensional Expensive Multiobjective Optimization Using a Surrogate-Assisted Multifactorial Evolutionary Algorithm. In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 572-580).

[2] Horaguchi, Y., Nishihara, K. and Nakata, M., 2024. Evolutionary multiobjective optimization assisted by scalarization function approximation for high-dimensional expensive problems. Swarm and Evolutionary Computation.

[3] Daulton, S., Eriksson, D., Balandat, M. and Bakshy, E., 2022, August. Multi-objective bayesian optimization over high-dimensional search spaces. In Uncertainty in Artificial Intelligence (pp. 507-517). PMLR.

[4] Zhao, Y., Wang, L., Yang, K., Zhang, T., Guo, T. and Tian, Y., 2021. Multi-objective optimization by learning space partitions. arXiv preprint arXiv:2110.03173.

[5] Rashidi, B., Johnstonbaugh, K. and Gao, C., 2024, April. Cylindrical Thompson sampling for high-dimensional Bayesian optimization. In International Conference on Artificial Intelligence and Statistics (pp. 3502-3510). PMLR.

### Questions
Could you please respond to the comments in the section of weakness?

What does population drift mean?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on expensive multiobjective optimization problems (EMOPs) where high objective evaluation costs limit traditional multiobjective evolutionary algorithms (MOEAs). It proposes FSMOEA, a scalable framework enhancing surrogate-assisted MOEAs (SMOEAs) via foresighted surrogate models. Key innovations include: 1) A context-aware autoencoder-based head capturing population-level embeddings to reduce bias from population drift; 2) Low-dimensional latent space learning accelerating search and improving generalization; 3) A lightweight architecture ensuring efficiency across problem scales.

### Strengths
1. The proposal uses an autoencoder to learn population-level embeddings , capturing "context-awareness" that addresses the failure of traditional contrastive surrogates to consider population distribution.

2. The algorithm performs evolutionary operations in the learned low-dimensional latent space k instead of the original high-dimensional space n , which is a direct and effective strategy for tackling the high-dimensional  bottleneck.

3. The "plug-and-play" design of the FSMOEA framework allows the Foresight Model to be embedded as a modular component into existing contrastive SMOEAs.

4. The use of an autoencoder provides a dual advantage: the encoder supplies a context-aware representation for the surrogate model , while the latent space itself offers a more efficient, compact subspace for the evolutionary search.

### Weaknesses
1. The AE is trained each generation using only the current population with pop size 50. For a high-dimensional space training an AE on 50 samples to capture "context" is highly prone to underfitting.
2. If the population changes little, frequent retraining might introduce unnecessary computational cost and representation drift. Was a threshold considered?
3. The AE is trained to minimize reconstruction loss, which has no direct relationship with the optimization objective (finding the Pareto front). Does this cause a distribution shift problem? A latent space that is good for reconstruction is not necessarily good for evolutionary search.
4. The paper uses latent dimension k=10  as a default and shows its robustness. However, shouldn't the choice of k be dependent on the original dimension n  or the objective dimension m?
5.  About the bounded reconstruction error ϵ. In the extreme sparse data case of N=50,n=1000, is this ϵ realistically small? Does the theory still hold if ϵ is large.
6. Why is the AE trained only on the current population Pt  instead of the complete archive of all evaluated solutions? Would using the full archive produce a more stable latent space?
7. The autoencoder (Et,Dt) is retrained on a new population every generation , the "meaning" of the latent space k drifts. Does this representation drift interfere with the effectiveness of evolutionary operators, as good codes learned in generation t may not represent good traits in generation t+1?
8. The core of FSMOEA is context-aware *non-linear* reduction (AE). To clearly isolate its benefits, was a comparison made against a simpler baseline, such as applying *linear* reduction (like PCA) to Pt each generation and searching in that PCA space?

### Questions
The questions and suggestions are detailed in the "Weaknesses" section.

### Soundness
3

### Presentation
2

### Contribution
2
