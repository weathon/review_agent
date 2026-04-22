# Latent Space Codon Optimization Maximizes Protein Expression

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Codon optimization, the process of selecting synonymous codons to enhance mRNA translation efficiency and protein expression, is crucial for therapeutic protein production and mRNA-based vaccines. However, it faces two major challenges: navigating a vast, discrete combinatorial space that precludes gradient-based methods, and relying on heuristic proxies like Codon Adaptation Index or GC content balancing, which often fail to capture true expression dynamics. To address these, we introduce the Latent-Space Codon Optimizer (LSCO), which reformulates the problem in a continuous latent space derived from a pretrained mRNA language model, enabling efficient gradient-based optimization. Next, LSCO incorporates a data-driven expression objective trained on mRNA-protein expression data, regularized by a Minimum Free Energy for structural stability, and employs constrained decoding to ensure mRNA-protein fidelity. Evaluated on two mRNA-protein expression dataset, LSCO outperforms baselines such as frequency-based methods and recent naturalness-driven learned codon optimizers in predicted expression yields, while maintaining structural stability and host-appropriate GC content. Our results underscore LSCO's potential in advancing codon optimization, delivering mRNA sequences that excel in expression while ensuring thermodynamic stability and organism-specific compatibility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Latent-Space Codon Optimizer (LSCO) to tackle codon optimization’s two core challenges, an enormous discrete search space and the inadequacy of proxy objectives like CAI/GC to capture true expression, by reframing the problem as continuous, gradient-based optimization guided by learned objectives. LSCO operates in a pretrained mRNA language model’s latent space and uses a differentiable expression predictor regularized by a learned Minimum Free Energy (MFE) model to promote structural stability and avoid proxy-gaming. Decoding is constrained with positionwise masks so the final sequences strictly preserve the target amino-acid sequence, with temperature annealing balancing exploration and commitment. Evaluated on antibody datasets (Ab1/Ab2), the authors shw that LSCO achieves the highest predicted expression while maintaining stability and host-appropriate GC ranges, outperforming frequency-based methods, ICOR, and CodonTransformer; ablations show the MFE term acts as a crucial biophysical regularizer. They also do note limitations around data scarcity, compute demands, and the absence of wet-lab validation, which leaves improvements empirical-prediction–based rather than experimentally confirmed.

Overall, the paper is not well-suited for an AI conference. I like the paper overall as an application work, but it doesn't suit ICLR. The work would require extensive wet lab validation for it to be a publishable paper at any venue. I don't think the authors will have time for that during the rebutal period.

### Strengths
1. I like that the paper converts codon optimization from intractable discrete search with weak proxies into a continuous, gradient-based optimization in latent space, directly addressing the two core challenges.
2. Experiments indicate that LSCO improves predicted expression while maintaining stability (via MFE), which is practical on the reported datasets. However, this is pretty meaningless with experimental data.

### Weaknesses
1. Optimizing discrete sequences via latent/soft representations is a well-established paradigm. I am inclined to see this contribution as an application rather than a new optimization framework. Thus, it may be more suitable for a journal (say Nature Machine Intelligence or Nature Computational Science)
2.  Although the paper frames the task as continuous multi-objective optimization, it does not compare against strong discrete multi-objective methods (e.g., (i) https://arxiv.org/abs/2412.17780 (ii) https://arxiv.org/abs/2505.07086 (iii) https://arxiv.org/abs/2510.00352), which makes it unclear to mewhether mapping to a continuous domain is necessary or superior.
3. The authors focus on antibody datasets; but broader organisms, cell types, and sequence classes were not tested. It's not the authors fault, per say, but the acknowledged scarcity of public expression datasets definitely limits both generality and head-to-head comparisons.
4. The evaluation uses only 21 holdouts per dataset (Ab1/Ab2), which makes me a bit concerned about reliability, variance, and potential bias of the reported gains.

### Questions
1. What is the symbol `E` referenced in line 166? Is this a notation typo that should be $\Phi$?
2. Beyond linear scalarization, the authors shoukld try Tchebycheff or ϵ-constraint methods. Since this is a multi-objective problem, I'd like to see how well does your method cover the Pareto front (e.g., hypervolume/coverage metrics, front shape)?
3. Why did the authors rely on grid search for $\lambda_1$ and $\lambda_2$ instead of sampling weights on the probability simplex (for example. a $\Delta$-lattice or Dirichlet over $\lambda$) to encourage broader Pareto coverage? It's been established that grid search can become expensive and brittle when additional property constraints are introduced.

### Soundness
2

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
4

### Summary
The paper proposes LSCO, which reformulates codon optimization, a discrete combinatorial search over synonymous mRNA sequences, as a continuous gradient-based optimization problem in the latent space of a pretrained mRNA language model.

### Strengths
-The challenge in codon optimization is the discrete, combinatorial search space. LSCO's primary strength is converting this into a continuous latent space using a language model. 

-The paper includes an ablation study (LSCO No MFE) that removes the MFE regularization. The results show that without MFE, the model produces unstable sequences with bad GC content.

### Weaknesses
-Only two antibody datasets (Ab1, Ab2) are used, with <50 total holdout sequences. No cross-species or variable-length test cases are shown.insufficient diversity for claims of “general therapeutic applicability.”

-the expression predictor reuses the encoder E from the pretrained LM trained on the same distribution, making representation leakage likely between training and evaluation.

-The gradient step in latent space assumes local smoothness — that small Δz corresponds to small Δsequence and Δexpression. But codon choices are often non-locally coupled (due to RNA secondary structure). There’s no proof that the latent manifold preserves this continuity.

-it seems that no source code provided, private validation dataset...

-would like to improve my score if above concerns can be well addressed

### Questions
-How robust are the optimized sequences when re-evaluated by an independent expression predictor not used in optimization?

-Why not comparing with LinearDesign? (previous SOTA

-why  using real experimental data to train the expression predictor ($\Psi$), but then use that same predictor, and not real experimental data, to evaluate the performance of the final method (LSCO)? This doesn't seem to make sense.

### Soundness
3

### Presentation
2

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
This paper introduces Latent-Space Codon Optimization that optimizes mRNA codon sequences in a continuous latent space obtained from a pretrained mRNA BERT models. The optimization is guided by the gradients of estimated mRNA protein expression, further regularized with a Minimum Free Energy for structure stability. They also leveraged constrained decoding to ensure the decoded mRNA sequence codes the original protein. In empirical experiments, they showed LSCO outperforms other baselines on optimizing protein expression while also maintaining solid metrics on MFE and GC, striking a better balance between protein expression optimization, structural stability, and GC levels than the other existing methods.

### Strengths
1. The paper is well organized and flows naturally. I appreciate the comprehensive introduction to the mRNA codon optimization task.
2. On empirical validation, they demonstrated that the proposed method LSCO is able to achieve much better protein expression values and is also structurally stable and maintains good GC levels compared to other baselines.
3. The temperature annealing for the constrained decoding during optimization is clever.

### Weaknesses
1. I am concerned that the same expression predictor is used both to optimize LSCO and to evaluate performance. This poses a risk of circularity or overfitting to the predictor model rather than truly improving expression.

2. Evaluation is limited to 21 holdout sequences each from Ab1 and Ab2. This is a very small and single source test set; it is unclear if this generalizes to other proteins or settings.

3. The ablation study could be more interesting by studying more components of the proposed network in addition to studying with and without MFE. For example, in the paper, it’s stated that the authors adapted one design choice over the other such as annealing the temperature vs. setting a fixed one, optimizing in latent space vs. soft sequence space.

4. Lack of qualitative analysis. I appreciate the analysis of the relation between MFE and GC levels for the quantitative analysis but there is no qualitative analysis. For example, it would be interesting to see the latent space used in optimization. 

5. Novelty-wise, while latent-space optimization for biological sequences is not entirely new, its application to codon optimization is interesting. However, the paper could more clearly distinguish itself from prior latent-space molecular design works.

### Questions
I am curious what’s the success rate is for constraining/preserving the decoded sequences, since it’s not discussed or reported in the paper. 

While I find the proposed latent-space approach to codon optimization well-motivated and well-presented, I am concerned that the evaluation does not convincingly demonstrate biological improvement. The same expression predictor is used both for optimization and for evaluation, raising concerns of circularity. Moreover, the experiments dataset is small and coming from a single source. As a result, I am not confident in the strength or generality of the reported gains. I therefore lean toward weak reject. I believe the paper has potential and could be strengthened with independent evaluation and more experimental validation.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces LSCO, which frames codon optimization as a continuous optimization problem in the latent space of a pretrained mRNA language model. It uses a trained expression predictor and a minimum free energy (MFE) predictor to jointly improve mRNA expression and stability.

### Strengths
- Clearly defines the codon optimization problem, an underexplored but important problem.
- Presents a efficient framework for codon optimization that avoids exploring combinatorial space.

### Weaknesses
The evaluation setting is fundamentally flawed. The model is directly trained to optimize a predictor that is exactly same checkpoint used in evaluation. There are several problems associated with it.

1. All comparisons (baselines) are unfair since none of them are using this trained predictor that LSCO used. It is impossible to conclude latent-space optimization is a useful method without having a fair baseline. Example of fair baseline would be mRNA language model alignment or fine-tuning using the predictor that LSCO used.
2. Also, from the related works (Line 115) RNop seems reasonable and strong baseline, why is this excluded from comparison?
3. The validation metrics that authors provide to claim that the model is not hacking the predictor is MFE and GC content, which I found not convincing enough. Authors mention multiple metrics in line 119 (CAI, GC content, mRNA folding energy, and codon pair bias), why only report GC content? MFE is better since authors used trained predictor at train time and ViennaRNA MFE at evaluation time, but can you discuss how ViennaRNA computes MFE? That will help readers to understand how easy/hard to hack the metric.

Benchmark is very weak. While the problem of codon optimization is general, this method is evaluated in antibody dataset only, and the evaluation set has only 21 sequences.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2
