# Efficient Spectral Graph Diffusion based on Symmetric Normalized Laplacian

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Graph generative modeling has seen rapid progress, yet existing approaches often trade off between fidelity, scalability, and stability. Continuous and discrete diffusion models capture complementary aspects but remain hampered by either structural distortion or heavy computational costs. We introduce Efficient Spectral Graph Diffusion (ESGD), a lightweight framework that performs diffusion in the compressed eigenvalue space of the Symmetric Normalized Laplacian (SNL). This spectral compression guarantees bounded eigenvalues, provable stability, and faster convergence while eliminating hub-node dominance. A novel degree-matrix recovery algorithm enables exact graph reconstruction from the spectral representation. ESGD achieves state-of-the-art generation quality with one of the smallest parameter counts, converging up to 100× faster in training and requiring 6–10× fewer sampling steps with up to 2000× less computational cost. Our findings suggest that progress in graph generation may come less from heavier engineering, and more from principled reformulations that unlock both efficiency and fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes Efficient Spectral Graph Diffusion (ESGD), a graph generative modeling framework built on the symmetric normalized Laplacian (SNL). Its key idea is to perform diffusion in the bounded SNL spectral space, which improves stability and convergence. To reconstruct graphs from spectral representations, the authors design a degree-matrix recovery algorithm. The framework also introduces an ego-subgraph decomposition strategy to make large-graph training computationally feasible.

### Strengths
- The authors took a great care of making their approach theoretically grounded and evaluate on a variety of settings.

- Presentation is good and Figure 1.a) illustrates greatly what kind of trade-off the authors aim to achieve with their method.

### Weaknesses
- The method description requires clarification : you mention l132 that you keep $U$ fixed,  and then mention $\hat{U}$ as the recovered eigenvectors l 148 : what is $\hat{U}$ ? Such a crucial element of the reconstruction process should be clearly explained.

- Section 3.2 lists properties and theorems without giving any intuition and explanations on them. For example, it's not clear for me at all why Remark 3.6 makes sense.

- Table 1 lists a lot of outdated methods but more recent, major ones are missing, such as DiGress, DisCo, Cometh, DeFoG etc.

- In Table 2, the Valid, Unique and Novel (VUN) metric is missing. Therefore, your evaluation do not assess the ability of the model to respect the structural constraints of the datasets.

- QM9 and Zinc have reach saturation for years. For a method that specifically targets efficiency I would have expected evaluation on large scale datasets such as Moses or GuacaMol.

- No errors bars are provided, even though multiple works have demonstrated how MMD metrics can exhibits high variance.

### Questions
- See first weakness, how do we get U to reconstruct samples ?

- It is not clear to me if learning on large networks like Cora is meaningful or not. You claim that your ego-based approach allows to enhance generalization, but you train and test on the same graph. In the end, it seems that overfitting the training graph will yield the best results. By the way, how do you compute the MMD metrics for those large networks. Do you extract k-hop ego subgraphs from the training graph ?

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
the paper presents 1) an incremental improvement over the (cited) GSDM model, replacing the use of adjacency matrix with the symmetric normalized laplacian, and  2) a study of  the theoretical implications of the change, to explain the observed empirical improvements, which consist of

- improved conditions numbers and eigengaps, yielding faster convergence of the sampling process
    
- improved performance on a number of metrics evaluated on community-small,enzymes,grid,ego-small,QM9,Zinc250k,Citeseer,Cora,Pubmed,panar,sbm and tree graphs…
    
- ,,,while allowing for parameter reduction, which coupled with the
    
    improved training convergence allowing much  more fficient training (as well as sampling)

### Strengths
- well motivated, sensible
    
- theoretical analysis which seems correct
    
- clear efficiency gains compared to baselines

hitting the dimensions explicitly

1. originality: incremental improvement over GDSM
2. quality: some nits about the evaluation and comparison, else no flaws
3.  clarity: clearly written and proofs legible
4. significance: clear improvement in convergence speed, decent incremental advance for this

### Weaknesses
- inconsistent/varying comparison set of baselines  => while its good to do many evals, that makes cherry picking possible, needs justification (or pick one and stay consistent with it)
- unfair comparison without isomorpism/VUN check on larger datasets: digress edge etc. generate from scratch, GSDM and present store eigenvectors of training data set => should run an ablation generating the eigenvectors as well, as in GGSD
- needs multiple seeds of the method/multiple sampling rounds and CI intervals, same values are quite close
- would be good to report wallclock time/flop estimate (since e.g. the decoding might add wall clock at low flops/steps)
- compute isomorphisms with dataset on generated graph vs baselines => are we just memorizing due to keeping the Eigenvectors? (this is a flaw inherited from GDSM )
- try guacamol/moses (larger graphs) to see if things hold up there or the differences are washed out

### Questions
See weaknesses.

The most important elements to address are using multiple models/evaluations for reporting CIs on the metrics, trying larger datasets and performing an experiment applying the method to GGSD,then checking the graph isomoprhism rate to the training set and reworking the presentations etc. are extras

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a generative diffusion model that operates in the spectral domain of the symmetric normalized Laplacian.
Specifically, instead of performing diffusion on the adjacency matrix, the method works only on the eigenvalues of the operator $S = -D^{-1/2} A D^{-1/2}$, while keeping the eigenvectors U fixed.

### Strengths
As stated by the authors, performing diffusion only on the eigenvalues is advantageous in terms of computational efficiency.

### Weaknesses
- The results reported in Table 4 for Q9 should be discussed more thoroughly in relation to Table 12, in particular regarding the very low novelty value.
- I’m not fully convinced in terms of novelty, as the proposed approach appears similar to prior spectral diffusion methods such as SPECTRE. The paper would benefit from a clearer discussion of how ESGD differs from this existing model.

Minor:
- In Figure 1, unless I missed something, acronyms are not defined from the beginning.

### Questions
A major limitation concerns the assumption of a fixed spectral basis. The authors state that graph reconstruction is achieved by combining the generated eigenvalues with a fixed eigenbasis U, but it is not clearly explained where U is obtained (from the training set?) or how the model could generate graphs with different topologies if the eigenbasis cannot change. This point should be clarified and discussed in greater depth, as it is a key aspect of the proposed method; if convincingly addressed, it could positively affect my score.

The results in Tables 1 and 2 are promising, yet it is unclear why Table 3 includes fewer competitor methods. Why is GGSD not reported in the first two tables?

### Soundness
3

### Presentation
3

### Contribution
2
