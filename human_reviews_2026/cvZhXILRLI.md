# Bayesian Neural Networks for Functional ANOVA Model

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
With the increasing demand for interpretability in machine learning, functional ANOVA decomposition has gained renewed attention as a principled tool for breaking down high-dimensional function into low-dimensional components that reveal the contributions of different variable groups.
Recently, Tensor Product Neural Network (TPNN) has been developed and applied as basis functions in the functional ANOVA model, referred to as ANOVA-TPNN.
A disadvantage of ANOVA-TPNN, however, is that the components to be estimated must be specified in advance, which makes it difficult to incorporate higher-order TPNNs into the functional ANOVA model due to computational and memory constraints.
In this work, we propose Bayesian-TPNN, a Bayesian inference procedure for the functional ANOVA model with TPNN basis functions, enabling the detection of higher-order components with reduced computational cost compared to ANOVA-TPNN.
We develop an efficient MCMC algorithm and demonstrate that Bayesian-TPNN performs well by analyzing multiple benchmark datasets.
Theoretically, we prove that the posterior of Bayesian-TPNN is consistent.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates functional ANOVA decomposition, a technique for decomposing a high dimensional complex function into lower dimensional components to enhance model interpretability. Existing methods typically require pre specifying the maximum order of interaction in the low dimensional functions and exhaustively considering all feature combinations up to that order. In practice, this often limits analysis to pairwise interaction terms. To address this limitation, the authors propose Bayesian TPNN, a Bayesian framework that adaptively selects the most relevant features for inclusion in the low dimensional components without pre-specifying the maximum order. The method is evaluated across diverse domains, including real world tabular datasets, simulated data, and toy image datasets.

### Strengths
This paper is well written and easy to follow. Using a Bayesian approach to select features for functional ANOVA decomposition provides a theoretically principled framework that could be of significant interest to the interpretability community and potentially a broader audience.

### Weaknesses
My major concern is its evaluation and experiments. Most experiments are conducted on toy or tabular datasets, where the application of deep learning is arguably less compelling. Please refer to my detailed questions below.

### Questions
1. In Table 1, the authors compare the proposed Bayesian TPNN approach with existing methods in terms of prediction accuracy. However, the proposed method does not appear to outperform existing baselines significantly, particularly when taking the standard errors into account.

2. In Table 2, the authors evaluate uncertainty quantification against existing methods. It would strengthen the comparison to include deep ensembles, which are widely recognized as a standard baseline for uncertainty estimation.

3. Most experiments are conducted on simulated or tabular datasets, with the exception of Section 4.4, which uses CelebA HQ and Catdog datasets. However, in these cases, the approach relies on another CBM model to generate interpretable concepts. Could the authors apply Bayesian TPNN directly to these datasets and produce interpretations for the low dimensional functions?

Overall, it is challenging to identify practical scenarios where the proposed method would be preferred. I recommend extending the evaluation to additional domains, such as the genomics datasets used in Martens & Yau (2020), to better demonstrate applicability.

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
The authors propose a Bayesian method for estimating a Tensor Product Neural Network, a method for estimating a functional ANOVA with a neural net proposed at ICML this year.
Authors develop an MCMC algo using Langevin dynamics for continuous parameters and bespoke proposals for the discrete ones.

### Strengths
The article is logically organized and easy to follow overall. It's clear what the goals are. From a structural perspective this piece is very well written.

The contributions of the article are clear, and the proposed methodology is thoroughly investigated.

The numerical experiments seem adequate to me; they use several datasets and compare against a reasonable suite of alternative methods.
For instance, BART is a tough competitor, and matching/beating it while providing for high interpretability is notable.

Though primarily a computation/applied article, the authors prove a basic asymptotic result of their method.

### Weaknesses
The biggest weakness of this article is that this work is fundamentally incremental: it is a straightforward Bayesian version of an existing method, for which the Bayesian inference is straightforward.

Some of the limitations of Bayesian inference in the neural setting, notably the lack of scalability due to the nonexistence of a stochaastic version of the MH algorithm, are not addressed.
Discussion of how to mitigate this would have improved the article.

Needs spellchecking and grammar correction.

### Questions
I think this work was very clearly presented and the motivation for the method itself is clear, so I don't have any questions, but I invite the authors to nevertheless answer:
1) What did I get wrong in my review?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Bayesian Tensor Product Neural Networks (Bayesian-TPNN), a Bayesian neural network architecture for the functional ANOVA model. The key innovation is incorporating Bayesian inference over both network parameters and architectures (i.e., the subsets of input variables forming each component). This allows efficient detection of higher-order interactions without predefining component structures, addressing the scalability issue of ANOVA-TPNN (Park et al., 2025), whose number of sub-networks grows exponentially with interaction order.
The authors design a specialized MCMC algorithm that alternates between growing/pruning architectures and updating continuous parameters via Langevin proposals. Theoretically, the paper establishes posterior consistency for both the overall regression function and its individual ANOVA components.
Empirically, Bayesian-TPNN is evaluated on eight real tabular datasets, synthetic benchmarks, and concept bottleneck image tasks. It delivers comparable or superior predictive accuracy to both interpretable and black-box baselines (NAM, BART, XGB, mBNN) and significantly better uncertainty quantification and component selection, particularly for higher-order interactions.

### Strengths
- Clear motivation and significance: the paper tackles the long-standing scalability bottleneck of functional ANOVA neural models and provides a principled Bayesian alternative capable of learning higher-order terms without combinatorial explosion.

- Novel integration of architecture learning into ANOVA-structured modes: treating subsets of input variables as random variables and exploring them via reversible-jump-style MCMC is a clever idea. The stepwise proposal mechanism guided by feature importance is intuitive and empirically validated.

- The proof of posterior consistency for both the global function and each ANOVA component is nontrivial and strengthens the paper’s soundness.

### Weaknesses
- The paper spends many pages detailing MCMC updates and proofs but offers limited high-level intuition on why the proposed priors or proposal mechanisms work. As I'm not an expert of XAI and ANOVA, more discussion on background and a small synthetic visual example illustrating architecture evolution would help accessibility.

- While results are comprehensive, the gains in predictive accuracy are modest compared to ANOVA-TPNN, and the improvements in uncertainty quantification, though consistent, are small in magnitude. It would strengthen the empirical narrative to include a more demanding high-dimensional or noisy regime where the Bayesian approach truly shines.

Minors:
- Some figures (e.g., component plots) and tables are crowded or use small legends.

- The exposition occasionally repeats prior work descriptions or defers too much to appendices.

- The paper would benefit from clearer separation between methodological explanation and algorithmic detail.

### Questions
Could the proposed method scale to higher-order input, like p=4 or 5?

### Soundness
3

### Presentation
2

### Contribution
3
