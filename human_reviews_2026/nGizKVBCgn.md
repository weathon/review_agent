# PolyGraph Discrepancy: a classifier-based metric for graph generation

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Existing methods for evaluating graph generative models primarily rely on Maximum Mean Discrepancy (MMD) metrics based on graph descriptors. While these metrics can rank generative models, they do not provide an absolute measure of performance. Their values are also highly sensitive to extrinsic parameters, namely kernel and descriptor parametrization, making them incomparable across different graph descriptors.
We introduce PolyGraphScore (PGS), a new evaluation framework that addresses these limitations. It approximates the Jensen-Shannon (JS) distance of graph distributions by fitting binary classifiers to distinguish between real and generated graphs, featurized by these descriptors. The data log-likelihood of these classifiers approximates a variational lower bound on the JS distance between the two distributions. Resulting scores are constrained to the unit interval $[0,1]$ and are comparable across different graph descriptors. We further derive a theoretically grounded summary score that combines these individual metrics to provide a maximally tight lower bound on the distance for the given descriptors. Thorough experiments demonstrate that PGS provides a more robust and insightful evaluation compared to MMD metrics. A reference implementation of PGD is available at https://github.com/BorgwardtLab/polygraph-benchmark

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces PGS: a method for evaluating graph generative models. The idea behind the paper is to use the JS divergence to measure the similarity between the true graph distribution (reference)  and that of the generated distribution. Similar to the literature employing MMD instead of using the graph distribution it uses the distribution of certain descriptors derived from statistics of the graph (like degrees, orbit counts, etc.) The benefit of using JSD over MMD is that the former is bounded. JSD has been used in GANs as the objective function. Hence, this work approximates JSD using a binary classifier that tries to distinguish the original graph from the generated ones. To take into account multiple predictors it takes the maximum divergence among all descriptors as the divergence between the reference and generated graphs.

### Strengths
* The proposed method for evaluating GGMs addresses some of the limitations of MMD: Critically, its unboundedness and the lack of a way to combine predictors
* It introduces a new axis of evaluating how suitable a given measure (or metric) is: How predictable it is with respect to the training steps of well-known models. Intuitively, the distance should decrease as the training steps increase.

### Weaknesses
* On the other hand, it follows the existing paradigm of MMD -- using a set of predictors to approximate the graph distance with the distance of predictors, without offering a new angle that would revitalize the literature in GGM evaluation.
* The proposed way to aggregate multiple predictors (by taking the maximum distance) is potentially susceptible to outliers & does not take complementary signals from predictors into account.
* The scale issue is remediated by using a bounded in [0,1] distance. However, certain predictors  (or classifiers applied to those predictors) might be more sensitive to perturbations than others, hence the max aggregation scheme could favor them.

### Questions
Q1: Have the authors considered other aggregations than max, or was it some certain predictor that shows higher sensitivity to perturbations and thus most frequently dominates?
Q2: How fast is the training of TabPFN?
Q3: Taking into account that for real graphs (as opposed to synethetic ones) the number of reference graphs could be limited, do the authors see a limitation in a method that also requires to split the dataset into train/test sets and hence a small number of graphs could be available for either training or testing? How many samples does usually TabPFN require?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces PGS, an alternative to MMD for measuring the dissimilarity between sets of graphs. The introduced measure is based on the Jensen-Shannon divergence. The authors make use of the property that this divergence is bounded by the best possible binary classifier between these distributions. To obtain an estimate of this bound, they train classifiers based on graph representations and estimate the JS divergence. 

The authors perform several experiments that highlight defects of MMD and which demonstrate that their JS divergence overcomes these defects.

### Strengths
The information-theoretical setup of the JS divergence is very elegant. 

While the idea is simple, executing it properly requires some amount of engineering. From what I can see, the authors did a good job at this.

Their approach finds the graph descriptor that is best at discriminating the two graph samples (among some selection of descriptors). It is nice that this gives insight into the aspect in which the samples differ.

The paper is well written and I enjoyed reading it.

### Weaknesses
From the paper, it is not entirely clear to me how efficiently PGS can be computed. I'm assuming that it is considerably heavier than computing MMDs based on the same descriptors. In line 231, it says that it is based on TabPFN, which is "fast", though it does not elaborate on this claim.

There's several popular and interesting graph kernels that are missing from the comparison. Like the Weisfeiler-Lehman (WL) kernel, the PyramidMatch (PM, [1]) kernel and the Shortest-Paths (SP) kernel. The SP kernel, in particular, seems to fit the descriptor framework well and it was shown to be very powerful in distinguishing graphs from different distributions [2].

The experiments are somewhat hard to follow and some of the figures are somewhat messy. For example, Figure 3 could be replaced by box-plots for clarity and Figure 5 is quite dense.

The authors mention that MMD lacks an intrinsic scale, but most kernels satisfy some Cauchy-Schwarz-like bound $k(x,y)\le\sqrt{k(x,x)k(y,y)}$, which can be used to normalize the kernels like $k(x,y)/\sqrt{k(x,x)k(y,y)}$. After doing so, the MMD is in the interval [0,2].

Table 1 describes "Multi-Descriptor Aggregation" as one of the advantages of PGS over MMD. I can't find where in the text this is explained. MMD can also be used to combine different descriptors. For example, we can concatenate the descriptors (before computing kernels) or take the mean of several kernels. I do agree that PGS utilizes multiple descriptors more efficiently than MMD, but I would call this "selection" rather than "aggregation".

In the introduction, the authors characterize MMD as a way to compare graph samples using descriptors. I always interpreted MMD as a way to aggregate graph kernels to a distance between sets of graphs. These kernels are usually based on comparing graph descriptors, but this is not a necessity. For example, for the PM or WL kernel, it is cumbersome to think of them in terms of descriptors.

In Appendix M, I see that you generate planar graphs from Delauney triangulations. Note that this does not lead to a uniform sample among planar graph. To sample uniformly from this sample, one could use Boltzmann samplers [3]. I found this implementation that seems to do this https://github.com/towink/boltzmann-planar-graph (disclaimer: I did not test this implementation). Moreover, it appears that in the SBM generation, the connection probabilities are kept constant while the community sizes and number of communities are varied. Perhaps it is better to change these probabilities with the sizes to ensure each vertex has a constant (expected) number of neighbors inside and outside its communities. With the current probabilities, the SBM with $2\times20$ nodes will only have have two inter-community edges in expectation.

[1] Borgwardt, Karsten M., and Hans-Peter Kriegel. "Shortest-path kernels on graphs." *Fifth IEEE international conference on data mining (ICDM'05)*. IEEE, 2005.

[2] Gösgens, Martijn, Alexey Tikhonov, and Liudmila Prokhorenkova. "Evaluating Graph Generative Models with Graph Kernels: What Structural Characteristics Are Captured?." *Transactions on Machine Learning Research*.

[3] Fusy, Éric. "Uniform random sampling of planar graphs in linear time." *Random Structures & Algorithms* 35.4 (2009): 464-522.

### Questions
Can you comment on the running time and scalability of PGS?

PGS uses descriptors as input rather than kernels. Could it be modified so that it works with graph kernels? I could imagine that some k-NN classifier could work, where we take nearest neighbors in the training set and use the validation set to estimate performance. That is, for a graph $G$ in the validation set, we find the $k$ graphs in the training set that best resemble it according to the kernel, and then estimate $D(G)$ as the fraction of these training graphs of each class.

Section 5.2 shows experiments with "Data perturbations" akin to those from O'Bray and Thompson. Instead of perturbing a given distribution, we could interpolate between pairs of graph distributions, as is done in [2]. In particular, it would be interesting to see how well PGS performs on interpolations that are quite subtle, like between geometric graphs with 1D and 2D latent geometries.

Table 2 uses Pearson correlation rather than Spearman correlation, which is used in Figure 3. Why are we interested in *linear* rather than *rank* correlation in the "Trianing Iterations" experiment?

If I understand Figure 2 correctly, we evaluate the MMD between two graph samples from the same distribution. This experiment is used to demonstrate the bias of MMD. However, since the true MMD should be zero and MMD is always nonnegative, it seems to me that any estimate with positive variance would be biased. Am I missing something here?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- The paper proposes PolyGraphScore (PGS), a new metric for evaluating graph generative models by measuring how well a classifier can distinguish real from generated graphs.
- PGS estimates the Jensen–Shannon divergence between the two distributions using probabilistic classification instead of kernel-based distances like MMD.
- It uses graph descriptors (e.g., degree, spectrum, GIN embeddings) as tabular features and applies a TabPFN classifier to compute an interpretable score between 0 and 1.
- Experiments show that PGS provides stable, bounded, and interpretable evaluations that correlate strongly with model quality and training progress, outperforming traditional MMD metrics.

### Strengths
- The paper introduces a principled, classifier-based metric that provides an interpretable and bounded measure of distance between real and generated graphs.
- The method is model-agnostic and descriptor-agnostic, allowing fair comparison across graph generative models without retraining or fine-tuning.
- It demonstrates strong empirical correlations with true model quality and training progress, showing that the score behaves predictably and robustly.

### Weaknesses
- The choice of TabPFN as the discriminator is only weakly justified, with limited empirical evidence beyond a single logistic-regression ablation.
- The method's dependence on precomputed graph descriptors introduces information loss and may underrepresent complex graph structures.
- The paper lacks error bars or confidence intervals, which limits the reader's ability to assess the statistical reliability and variability of the reported results.

### Questions
- Could you elaborate on why you chose the Jensen–Shannon divergence as the foundation for PolyGraphScore, instead of other f-divergences (e.g., KL, Hellinger)?
- Does the PolyGraphScore framework extend naturally to weighted, directed, or attributed graphs, or are there fundamental challenges?
- The paper uses TabPFN as the discriminator; could you provide more detailed empirical evidence comparing it to other probabilistic classifiers (e.g., random forest, XGBoost, small MLP)?
- Since TabPFN supports only limited feature dimensions, how would PGS scale to very high-dimensional descriptors (e.g., graph embeddings >1000 dimensions)?

### Soundness
3

### Presentation
3

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
This paper proposes PolyGraphScore (PGS), a novel evaluation framework for graph generative models (GGMs) that addresses key weaknesses of the current standard Maximum Mean Discrepancy (MMD)-based metrics. Instead of computing kernel distances between graph descriptors, PGS trains a probabilistic discriminator (TabPFN) to distinguish real and generated graphs, using standard descriptors such as degree histograms, Laplacian spectra, or GIN embeddings. The classifier’s log-likelihood provides a variational lower bound on the Jensen–Shannon (JS) distance, yielding a unit-scaled score in [0, 1] that is interpretable and comparable across descriptors.
Comprehensive experiments demonstrate that PGS tracks synthetic perturbations, correlates strongly with model validity and training progress, and provides consistent model rankings across datasets and descriptors.

### Strengths
The paper is well written and easy to follow. The problem statement is clearly defined and well-motivated. The proposed evaluation framework is both theoretically sound and empirically well validated, demonstrating strong performance across diverse experiments.

### Weaknesses
1. Although TabPFN is relatively fast, training separate discriminators for each descriptor introduces additional computational overhead compared to a single MMD computation.

2. Regarding empirical validation, while the paper provides benchmarks on both real and synthetic datasets, expanding the experiments to include more specialized synthetic datasets could further demonstrate the generality of the proposed approach. Specifically, [1] notes that for Grid datasets — which consist of graphs with highly regular local structures — correlations between evaluation metrics are not close to 1, unlike for other datasets. Similarly, [2] reports different performance patterns of GGMs when generating grid-like structures. Comparing the proposed evaluation method with more expressive approaches, such as [1], on datasets where neither statistic-based MMD methods nor GNN-based methods clearly dominate, could further highlight the potential advantages of PGS.

3. The stability analysis of PGS under varying sample sizes (Figures 23–25) indicates that, similar to MMD-based approaches, PGS suffers from relatively high variance, particularly for graph sets containing fewer than 256 samples.


[1]. Shirzad, Hamed, Kaveh Hassani, and Danica J. Sutherland. "Evaluating graph generative models with contrastively learned features." NeurIPS 2022.

[2]. Zahirnia, Kiarash, et al. "Neural graph generation from graph statistics." NeurIPS 2023.

### Questions
How sensitive is PolyGraphScore (PGS) to the choice of graph descriptors? Have the authors explored combining multiple descriptors instead of selecting the best-performing one through max-reduction?

Could the authors clarify how PGS behaves when using different classifier architectures beyond TabPFN (e.g., GNNs), and whether this affects the tightness of the JS lower bound?

### Soundness
3

### Presentation
3

### Contribution
3
