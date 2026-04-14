# NRGBoost: Energy-Based Generative Boosted Trees

- Decision: Accept (Poster)
- Scores: 10, 6, 6, 6

## Abstract
Despite the rise to dominance of deep learning in unstructured data domains, tree-based methods such as Random Forests (RF) and Gradient Boosted Decision Trees (GBDT) are still the workhorses for handling discriminative tasks on tabular data. We explore generative extensions of these popular algorithms with a focus on explicitly modeling the data density (up to a normalization constant), thus enabling other applications besides sampling. 
As our main contribution we propose an energy-based generative boosting algorithm that is analogous to the second-order boosting implemented in popular libraries like XGBoost. We show that, despite producing a generative model capable of handling inference tasks over any input variable, our proposed algorithm can achieve similar discriminative performance to GBDT on a number of real world tabular datasets, outperforming alternative generative approaches. At the same time, we show that it is also competitive with neural-network-based models for sampling.
Code is available at https://github.com/ajoo/nrgboost.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
NRGBoost is an energy-based method for tabular data density estimation and generation using gradient boosted decision trees (GBDTs), in contrast to recent diffusion-based generative methods which cannot give density estimates.

### Strengths
1. The method is original and well-motivated, and the approach for incorporating rejection sampling within the boosting process to improve efficiency was an excellent contribution towards making energy-based tabular modeling practical.

2. The paper is well-written, with the method and experiments clearly explained. Showing how the model progressively models the dataset over boosting iterations in Figure 2 gave helpful insight into how the method works.

3. The experiments on downsampled MNIST data were very nice to see, and convincing that NRGBoost goes beyond current SotA (ForestFlow) [2] in generation in certain settings, besides also offering density estimation.

### Weaknesses
1. [UPDATE: fully addressed in rebuttal] For single-variable inference in Section 6.1, using R^2 as an evaluation metric seems strange, compared to using CRPS, RMSE, and MAE, as employed in [3]. Furthermore, the choice of baseline methods seems incomplete. For example, because Forest-Flow supports conditioning on covariates, one could also generate (say) 100 Forest-Flow conditional samples, then compute the mean and evaluate this.

2. [UPDATE: mostly addressed in rebuttal] The tabular datasets used for evaluating NRGBoost in Section 6.2 were nonstandard. Most recent papers either evaluate on the TabCSDI [1] benchmark of 6 datasets or the ForestFlow [2] benchmark of 27 datasets. Because using one's own choice of datasets does open the possibility for cherry-picking, one should either use a preexisting benchmark or justify one's selection of datasets. 

3. [UPDATE: fully addressed in rebuttal] Some relevant works are missing from related work (and perhaps the experiments). It would be good to discuss how this compares against a previous energy-based modeling method, TabEBM [4], albeit one which uses TabPFN instead of GBDTs. And there is UnmaskingTrees [5] which uses GBDTs to perform both generation and also density estimation via autoregression and recursive partitioning.
  
[1] Zheng, S., & Charoenphakdee, N. (2022). Diffusion models for missing value imputation in tabular data. arXiv preprint arXiv:2210.17128.

[2] Jolicoeur-Martineau, A., Fatras, K., & Kachman, T. (2023). Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees. arXiv preprint arXiv:2309.09968.

[3] Beltran-Velez, N., Grande, A. A., Nazaret, A., Kucukelbir, A., & Blei, D. (2024). Treeffuser: Probabilistic Predictions via Conditional Diffusions with Gradient-Boosted Trees. arXiv preprint arXiv:2406.07658.

[4] Margeloiu, A., Jiang, X., Simidjievski, N., & Jamnik, M. (2024). TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models. arXiv preprint arXiv:2409.16118.

[5] McCarter, C. (2024). Unmasking Trees for Tabular Data. arXiv preprint arXiv:2407.05593.

### Questions
Questions

- The methods section and experiments devoted to DEF seems out of place, given that it is seemingly unrelated to NRGBoost and given that it performs worse. It is nice to tell more people about DETs and how they can be applied and improved, but I suggest perhaps omitting it entirely and submitting it as an ICLR Blogpost or similar venue.

- I realize that NGBoost does not support nonparametric probabilistic prediction, but I think seeing its results in Table 1 would still be helpful. 

Minor suggestions for improved readability:

Replace: We therefore estimate these quantities, with recourse to empirical data for P(X), and to samples approximately drawn from the model with MCMC.
With: We therefore estimate these quantities, with recourse to empirical data for P(X) and to samples approximately drawn from the model with MCMC.

Replace: Because even if the input space is not partially discrete, f is still discontinuous and constant almost everywhere we can’t use gradient based samplers and therefore rely on Gibbs sampling instead.
With: Because, even if the input space is not partially discrete, f is still discontinuous and constant almost everywhere, we can’t use gradient based samplers; we therefore rely on Gibbs sampling instead.

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a tree-based approach to model tabular data density which they claim outperforms traditional generative approaches.

### Strengths
The paper is confusing and hard to follow. I was unable to find any strength after investing a long time.

### Weaknesses
The paper is poorly written, incoherent and uses inconsistent experiments.

### Questions
1. The paper is hard to follow and haphazardly written. It uses some vague terms (mid-range consumer CPU, Line 88 and the best generative models, Line 89). Minor details Line 41: please avoid abbreviation like don't, can't.

2. The paper proposes generative model for tabular data in the abstract, but conducts experiments with vision data like MNIST. I am not sure how downsampling makes MNISt relevant to their setting. I did not understand what is going on in Figure 2. There is no description of the data generation process or distribution. 

3. The equations are incoherent and sometimes most probably wrong. For example, I do not see the purpose of Equation 2. It was never used later. The math is incoherent and written in such a way that it is hard for me to evaluate their correctness. 

4. Table 1: the authors say they bold the best performing algorithms, but they kept their algorithm bold always although it did not perform the best.

5. The paper has bad structure. Related works comes at Section 5 near the end of the paper. There is no flow diagram or pseudocode for their algorithm. 

Overall, I think the paper is far from being a coherent and legit study.

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
This manuscript presents a framework for using boosting to model unnormalised densities.
The proposed framework is used to build a model using decision trees as weak learners.
To accelerate sampling from the model (and thus training), a combination of Gibbs and rejection sampling is used.
Experiments on a down-sampled MNIST and UCI datasets demonstrate competitive performance to other tree-based generative models.

### Strengths
- (clarity) Very well-written paper that is easy to follow.
   I also appreciate the clear presentation and discussion of limitations of the proposed approach.
 - (originality) The proposed boosting framework seems to be novel and provides some refreshing insights.
 - (originality/quality) The overview over existing approaches to use decision trees for generative modelling is extensive and contextualises the work well.
 - (quality) The derivations are explained well and the appendix provides useful additional details.
 - (quality) The experiments seem to have been set up properly with error bars and competitive baselines.

### Weaknesses
- (significance) The abstract mentions a comparison with neural network models, but I could not find these in the main paper.
   Furthermore, the authors claim that tree-based generative models are to be preferred over DL-based models
   because discriminative DL methods do not provide competitive performance on tabular data.
   However, no DL methods have been included as baselines in the experiments to confirm this.
   As a result there is no evidence that DL methods would not be able to outperform tree-based generative models.
   PS: DL methods based on normalizing flows have explicit (tractable) densities.
 - (significance) It is not entirely clear how the proposed model can/should be used.
   The discussion mentions the overhead due to sampling, but this overhead is never quantified.
   This makes it hard to get a feeling for whether the (sometimes minor) increase in performance is "worth it".
   Also, it is not entirely clear why shallower trees are preferrable, as implied in the discussion.
 - (significance) The authors put a lot of emphasis on the tractability of computing densities.
   However, there are no clear experiments that illustrate the importance of this feature.
   The conclusion hints at applications enabled by density models,
   but this seems to indicate that there are no real use cases for this model.
   Especially since the results seem to indicate that there are other competitive methods out there.

### Questions
1. How does the performance compare to DL-based generative models?
2. Can the tradeoffs of NRGBoost vs e.g. DET/DEF be quantified somehow?
3. Is there a concrete use case where NRGBoost is the only/best model?

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
3

### Summary
This paper explores generative extensions of tree-based models explicitly molding the data density for structured tabular data. Specifically, an energy-based generative boosting algorithm NRGBoost is proposed which is trained to maximize a local second order approximation to the likelihood at each boosting iteration. To save the major training budget caused by approximately estimating the event probability of input data with sampling, the authors further propose an amortized sampling approach by maintaining a fixed-size sample pool with rejection sampling at each round to reduce training time. Apart from designing NRGBoost, the authors also explore bagged ensembles of DET as a generative counterpart to Random Forest. Comprehensive experiments on tabular discriminative and generative (data synthetic efficiency & Discriminator Measure) tasks show NRGBoost can be comparable to the prevailing discriminative GBDTs on prediction tasks as well as has competitive generation quality compared to recent tabular generative neural models, all achieved in one gradient boosting algorithm.

### Strengths
I am not expert in the fields of generative models or tree-based models, therefore I think I am not qualified to comment on the paper novelty, while there do exist certain aspects making me impressive.

**Solid formulation, proof & demonstration**: Based on the theory foundation of GBDT, the extension to its generative version NRGBoost is natural and its formulation derivation is clear and convincing. The proposed amortized sampling approach is reasonably designed utilizing the properties of the boosting algorithm. The experiments on tabular discriminative & generative tasks are well conducted with repeated trails, statistical tests and visualization analysis. Besides, complete theoretical and technical Appendix is given, including the computational time comparison for training each baseline.

**Wide application scenarios**: As a generative boosting algorithm, NRGBoost is able to be applied to both discriminative and generative tasks, which distinguishes it from other neural-based generative baselines.

**Reference-worthy exploration on tree-based models**: This paper explores the generative extensions in both Boosting- and Bagging-based tree models, giving the reference for the further research in the community.

### Weaknesses
Several weaknesses from application perspective.

(1) From Table 1, in discriminative tasks the performance gap between NRGBoost and traditional discriminative GBDTs (e.g., XGBoost) seems to be relatively more significant as the data scale increases, there is still room for further improvement before application in large-scale discriminative classification tasks.

(2) From the computational effort analysis during training, in Fig. 4, it seems NRGBoost is not computation-economical in the large datasets compared to other neural-network-based generative models, the data scalability of NRGBoost is not sufficiently discussed.

(3) From evaluated dataset information in Table 5, there lacks large regression datasets, which help us to further realize NRGBoost on large-scale regression tasks.

### Questions
I would like to increase my score according to the comments from other reviewers and the response.

(1) What about the inference time comparison between NRGBoost and other baselines?

(2) Why there is often a performance gap between discriminative GBDTs (i.e., XGBoost here) and NRGBoost? Is it possible to outperform these discriminative baselines from generative modeling paradigm?

(3) How about the discriminative performance of NRGBoost on large regression datasets (e.g., Microsoft, Yahoo dataset in [1]), large-feature-amount one (e.g., Epsilon in [1]) and large-class-number one (e.g., ALOI in [1])?

**Reference**

[1] Revisiting Deep Learning Models for Tabular Data, NeruIPS 2021.

### Soundness
3

### Presentation
3

### Contribution
3
