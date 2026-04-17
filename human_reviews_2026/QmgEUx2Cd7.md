# LORE: Jointly Learning The Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Learning the intrinsic dimensionality of subjective perceptual spaces such as taste, smell, or aesthetics from ordinal data is a challenging problem. We introduce LORE (Low Rank Ordinal Embedding), a scalable framework that jointly learns both the intrinsic dimensionality and an ordinal embedding from noisy triplet comparisons of the form, "Is A more similar to B than C?". Unlike existing methods that require the embedding dimension to be set apriori, LORE regularizes the solution using the nonconvex Schatten-$p$ quasi norm, enabling automatic joint recovery of both the ordinal embedding and its dimensionality. We optimize this joint objective via an iteratively reweighted algorithm and establish convergence guarantees. Extensive experiments on synthetic datasets, simulated perceptual spaces, and real world crowdsourced ordinal judgements show that LORE learns compact, interpretable and highly accurate low dimensional embeddings that recover the latent geometry of subjective percepts. By simultaneously inferring both the intrinsic dimensionality and ordinal embeddings, LORE enables more interpretable and data efficient perceptual modeling in psychophysics and opens new directions for scalable discovery of low dimensional structure from ordinal data in machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents LORE, a framework that jointly learns ordinal embeddings and their intrinsic dimensionality using a nonconvex Schatten-p regularization and an iteratively reweighted optimization algorithm. Experiments on synthetic and real perceptual datasets show that LORE can automatically recover low-rank, interpretable embeddings with competitive triplet accuracy.

### Strengths
1. The paper presents an interesting framework (LORE) that aims to jointly learn ordinal embeddings and their intrinsic dimensionality, addressing a recognized limitation of existing approaches.

2. The proposed optimization procedure is clearly described and includes a convergence argument, suggesting technical soundness.

3. Experimental results across synthetic and real perceptual datasets provide encouraging evidence that LORE can recover compact and interpretable embeddings.

### Weaknesses
1. The paper lacks a theoretical analysis explaining under what conditions the Schatten-p regularization can correctly recover the intrinsic rank, which limits the strength of its main claim.

2. The iteratively reweighted optimization is presented formally but lacks practical insight; for example, the paper does not show convergence curves, runtime comparisons, or how initialization influences final embeddings.

3. The figures lack sufficient information for interpretation. Several plots (e.g., Figure 2 and Figure 3) omit axis labels or error ranges, and some results do not specify experimental settings or data sources, reducing the clarity and comparability of the findings.

4. The study provides limited discussion of hyperparameter sensitivity, particularly the effects of λ and p on performance and rank estimation.

### Questions
1. Under what data or noise conditions can the Schatten-p regularization reliably recover the intrinsic rank?

2. How sensitive is LORE to the choices of \lambda and p?

3. Could the authors show more evidence of the optimization’s empirical behavior, such as convergence stability or runtime?

4. All experiments are based on psychophysical or human-judgment data (food, music, cars). Have the authors tested, or plan to test, the method on non-perceptual ordinal datasets (e.g., image or text similarity)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LORE (LOw Rank Embedding), a novel and scalable framework designed to jointly learn the intrinsic dimensionality ($d$) and the optimal relative structure of perceptual spaces from noisy ordinal data (triplet comparisons of the form "A is more similar to B than C"). Addressing a fundamental limitation of existing Ordinal Embedding (OE) methods that rely on pre-defined or estimated dimensions, LORE leverages a low-rank constraint on the embedding matrix $Z$ and employs a highly non-convex Schatten quasi-norm as a regularizer to promote the discovery of the true intrinsic dimensionality. The optimization is handled by an effective iteratively reweighted algorithm with provable convergence guarantees. Extensive experiments on synthetic data, simulated perceptual spaces, and real-world crowdsourced datasets demonstrate that LORE successfully recovers the true intrinsic rank, achieves competitive triplet accuracy, and yields semantically interpretable embedding axes.

### Strengths
1. The paper addresses the critical and underexplored problem of jointly discovering the intrinsic dimensionality and relative structure in perceptual spaces, which is a key limitation of prior Ordinal Embedding (OE) methods. The introduction of the low-rank constraint via the non-convex Schatten quasi-norm is highly novel within the OE literature.

2. Unlike many empirical approaches, LORE provides a convergence theorem (Theorem 1, page 5) for its optimization objective and the proposed iterative reweighted algorithm. This rigorous theoretical foundation significantly strengthens the paper's contribution.

3. The experiments are thorough and persuasive. LORE successfully recovers the true intrinsic rank in synthetic and simulated LLM perceptual spaces (Figure 4) where other baselines fail. On real crowdsourced data (Food-100, Musicians, Cars), it maintains high triplet accuracy while achieving significantly lower rank embeddings compared to SOTA OE methods (Table 5).

4. The learned embedding axes (Figure 5) are shown to be semantically interpretable (e.g., "Sweet to Savory," "Learned Axis 1"), offering valuable insights into the underlying perceptual characteristics of the data, which is highly beneficial for discovery tasks.

### Weaknesses
1. While the paper provides a convergence theorem, the optimization objective $\min \Psi(Z)$ remains highly non-convex. The analysis primarily focuses on convergence to a stationary point, which may not always be the globally optimal solution. A more in-depth discussion on the practical robustness to initialization and the likelihood of escaping poor local minima would be beneficial.

2. The LORE objective function includes several regularization parameters ($\lambda, \tau, \mu$). Although Figure 2 demonstrates stability across a range of $\lambda$ values for a fixed $\tau$, a full exploration of the joint sensitivity of $\lambda$ and $\tau$ is absent. These parameters are crucial for balancing triplet accuracy and rank recovery, and their interplay needs more detailed investigation.

3. The paper should explicitly discuss the cases where the intrinsic rank $d$ may not be an integer (e.g., fractional dimensionality in complex manifold structures) and whether LORE's reliance on a rank constraint limits its ability to fully capture these more intricate data structures.

### Questions
1. The optimization relies on the non-convex Schatten quasi-norm. Could the authors provide a more detailed analysis or empirical evidence (e.g., through multiple restarts with different random initializations) showing the consistency and quality of the stationary points reached by the algorithm?

2. The paper uses a simulated perceptual space derived from a large language model (LLM) embedding. Can the authors provide more intuition or validation for why the LLM's embedding space represents a "true perceptual $d$-dimensional space" that LORE is attempting to recover, and how the inherent noise was modeled in this specific experiment?

3. Regarding the runtime complexity, how does the convergence speed (number of iterations for Algorithm 1) of LORE change as the total number of triplets ($T$) and the true intrinsic dimension ($d$) scale?

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
2

### Summary
The paper proposes LORE (Low Rank Ordinal Embedding), an ordinal-embedding framework that jointly learns (i) an embedding that satisfies triplet comparisons and (ii) the intrinsic dimensionality (rank) of the latent perceptual space. The key idea is to regularize the embedding matrix with a nonconvex Schatten-p quasi-norm (with smoothing of the triplet loss), optimized via an iteratively reweighted scheme that is shown to converge to a stationary point. Experiments indicate that LORE achieves comparable triplet accuracy while discovering substantially lower-rank solutions.

### Strengths
1 Tackles a long-standing limitation of ordinal embedding, i.e., choosing the dimensionality, by jointly inferring rank and coordinates, rather than grid-searching over dimensions. 

2 Uses Schatten-p regularization (p∈(0,1)) to promote low rank, with a softplus-smoothed triplet loss and an iteratively reweighted algorithm; provides a convergence-to-stationary-point guarantee and implementation details. 

3 Experiments indicate that LORE achieves comparable triplet accuracy while discovering substantially lower-rank solutions.

### Weaknesses
1 The theory ensures convergence to a stationary point, but not global minima or exact rank identification; this is acknowledged as a limitation.

2 The paper argues that LORE uncovers the intrinsic dimensionality without under- or over-estimating it. As stated, this reads as a subjective claim. Please provide stronger evidence to demonstrate that the method does not “mask” latent structure or inflate rank.

3 This paper claims that Künstle et al. (2022) require specifying plausible dimensionalities, risking misspecification and loss of power if the true rank lies outside those bounds. Do you have experiments showing this failure mode and quantifying how often it occurs under realistic sampling/noise? 

4 The paper states that training separate embeddings per hypothesized rank (as in Künstle et al., 2022) is computationally prohibitive. Please report the total cost to reach the same triplet accuracy for both methods.

5 Missing direct comparison to Künstle et al. (2022).

6 Literature coverage is dated. The citations lean heavily on pre-2022 works and omit several recent, directly relevant works.

[1] Künstle D E. Machine Learning for Psychophysical Scaling with Ordinal Comparisons[D]. Eberhard Karls Universität Tübingen, 2024.
[2] Huber L S, Künstle D E, Reuter K. Tracing truth through conceptual scaling: Mapping people’s understanding of abstract concepts[J]. 2024.
[3] Sauer Y, Künstle D E, Wichmann F A, et al. An objective measurement approach to quantify the perceived distortions of spectacle lenses[J]. Scientific Reports, 2024, 14(1): 3967.
[4] Huber L S, Künstle D E, Reuter K. Tracing truth through conceptual scaling[J]. Cognition, 2026, 266: 106321.

### Questions
1 Can you provide calibration evidence to show that the method neither hides structure nor inflates rank?

2 Do you have experiments where the true rank lies outside the candidate set used by Künstle et al. (2022)? How often does this occur, and what is the performance degradation?

3 For equal target triplet accuracy, what is the cost for LORE vs. training multiple embeddings as in Künstle et al. (2022)? 

4 Why is there no quantitative comparison to Künstle et al. (2022)? 

5 How does LORE differ conceptually and empirically from recent works?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper claims that all existing OE approaches are based on pre-specified embedding dimensions, which may lead to some problems. Meanwhile, the paper emphasizes the advantages of low intrinsic dimensional embedding--easier to interpret, less computationally intensive, while existing OE approaches based on pre-specified embedding dimensions often result in high-dimensional embeddings. 
Based on this, the paper introduces LORE (Low Rank Ordinal Embedding), a novel method for ordinal embedding that jointly learns both the low-dimensional embedding and **its intrinsic dimensionality** from noisy triplet comparisons.
Furthermore, the paper establishes an efficient optimization strategy based on iteratively reweighted minimization and provides a scalable algorithm suitable for large-scale perceptual similarity data.
And the paper validates the effectiveness and efficiency of LOPE through an extensive evaluation

### Strengths
1. The explanation of the background and significance of the problem is very clear, and the problem to be solved is very meaningful.
2. LORE is effective in reliably overlooking the intrinsic dimensionality and demonstrates the interpretability of low dimensional representations in semantics, which is helpful for solving problems in psychology, neuroscience, and social science.
3. The theoretical explanation is very rigorous.

### Weaknesses
1. More new methods should be compared, and more datasets should be compared, especially considering that SOE and t-STE are both methods from 2014. This may lead to doubts about the performance of LORE.
2. On the accuracy metric, which may be the most important metric, LOPE is not always optimal or even suboptimal.
3. A low rank does not necessarily mean an improvement in method performance, so more explanation is need.
4. For the metric of computational efficiency(time), low dimensional embedding is not the only solution(eg: FORTE vs. LORE), so the advantages of LORE should be further explained.
5. Following 3, if the embeddings of other OE methods are not interpretable, the differences between other methods and LORE should be compared in Figure 5.

### Questions
1. If the embeddings of other OE methods are not interpretable, the differences between other methods and LORE should be compared in Figure 5.
2. The factors that affect computational efficiency (time metric) may need to be explained in order for readers to understand why the time differences of methods such as SOE/t-STE can be so significant at the same rank.

### Soundness
3

### Presentation
3

### Contribution
3
