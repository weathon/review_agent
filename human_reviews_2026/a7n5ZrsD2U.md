# Generalization Error Bound via Embedding Dimension and Network Lipschitz Constant

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
Modern deep networks generalize well even in heavily over-parameterized regimes, where traditional parameter-based bounds become vacuous. We propose a representation-centric view of generalization, showing that the generalization error is controlled jointly by: (i) the intrinsic dimension of learned embeddings, which reflects how much the data distribution is compressed and determines how quickly the empirical distribution of embeddings converges to the population distribution in Wasserstein distance, and (ii) the sensitivity of the downstream mapping from embeddings to predictions, quantified by Lipschitz constants. 
Together these factors yield a new generalization error bound that explicitly links embedding dimension with network architecture. At the final embedding layer, architectural sensitivity vanishes, and the bound is driven more strongly by embedding dimension, explaining why final-layer dimensionality is often a strong empirical predictor of generalization. Experiments across datasets, architectures and controlled interventions validate the theoretical predictions and demonstrate the practical value of embedding-based diagnostics. Overall, this work shifts the focus of generalization analysis from parameter to representation geometry, offering both theoretical insight and actionable tools for deep learning practice.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper advances a representation-centric view of generalization, deriving a bound that depends on (i) the intrinsic dimension of learned embeddings and (ii) the network’s Lipschitz sensitivity from embeddings to outputs. Lower-dimensional embeddings yield faster empirical-to-population convergence in Wasserstein distance (scaling $\approx n^{-1/d}$), while smaller Lipschitz constants limit error amplification. At the final layer, architectural sensitivity vanishes, so the bound is driven primarily by embedding dimension—explaining why final-layer dimensionality often tracks generalization. Experiments support the theory: an MNIST autoencoder matches the predicted scaling; across ResNet-18/34/50/101/152 on CIFAR-10/100, final-layer dimension and train–test Wasserstein distance correlate with the generalization gap. A width-reduction intervention lowers embedding dimension but raises Lipschitz sensitivity, yielding no consistent gain—confirming the joint role of both factors. Overall, the work links representation geometry with architecture, explains the prominence of final-layer dimension, and offers actionable diagnostics (e.g., monitoring embedding dimension) for practice.

### Strengths
- Novel framework linking representation geometry and sensitivity: The paper derives a representation-centric generalization bound that depends on (i) the intrinsic dimension of embeddings and (ii) the network’s Lipschitz sensitivity. This bridges model architecture and data distribution, offering a more interpretable alternative to parameter-based capacity measures. Building on sharp Wasserstein convergence rates, it provides a concrete theoretical link between lower-dimensional learned features and improved generalization, aligning with recent empirical observations.

 - Insight into final-layer representations and generalization: A key insight from the theory is that at the final embedding layer, architectural sensitivity drops out of the bound, leaving intrinsic dimension as the dominant driver of generalization. This explains why last-layer dimensionality reliably predicts performance, and the authors corroborate it empirically (ResNets on CIFAR): lower final-layer dimension → smaller generalization gap. The result offers an intuitive geometric diagnostic that often outperforms parameter-based measures.

 - Comprehensive empirical validation.
The experiments substantiate the theory on three fronts: (1) Scaling law: an MNIST autoencoder shows Wasserstein distance grows with embedding dimension and shrinks with sample size, matching the $\approx n^{-1/d} trend; (2) Cross-architecture correlation: on CIFAR-10/100 (class-wise), final-layer intrinsic dimension and train–test embedding divergence strongly correlate with the generalization gap across ResNet-18/34/50/101/152; (3) Causal probe: narrowing an intermediate MLP layer lowers embedding dimension but increases Lipschitz sensitivity (via spectral-norm proxy), yielding no consistent generalization gains, confirming both factors jointly matter. Results are consistent across datasets and architectures, indicating the proposed representation-based metrics are robust and broadly relevant.

### Weaknesses
- Strong assumptions: The derivation of the bounds involves certain non-trivial assumptions that may limit the scope of the theory. In particular, the analysis assumes that the Bayes-optimal predictor is Lipschitz continuous with respect to the learned embedding space. This assumption is necessary to ensure a well-behaved relationship between changes in the embedding and changes in the target, essentially bounding the “irreducible” part of the problem. While this makes the mathematics tractable, it might be unrealistic in some practical tasks that the true data generating process might not be Lipschitz, especially in high-dimensional raw input space or if classes have very complex decision boundaries. Thus, the necessity of this condition somewhat narrows the situations where the bound is theoretically guaranteed. It would be interesting to know how sensitive the conclusions are to this assumption or if it can be relaxed. 

- Loose bounds and unclear tightness: A potential limitation of the proposed generalization bound is that it comes with unspecified or potentially loose constant factors. The theory introduces constants (e.g. $C_k, D_k, \varepsilon$ in the bound) and relies on order-of-magnitude terms (like $n^{-1/(d+\varepsilon)}$), which means the numerical value of the bound might be quite loose or vacuous for practical network sizes. The authors themselves acknowledge that “our bound contains constants that may be loose”. As a result, while the bound is qualitatively insightful (e.g. predicting which direction generalization will move when intrinsic dimension changes), it might not be quantitatively tight enough to certify performance or compare different models in absolute terms. Indeed, the evidence provided is mostly correlational rather than showing the bound as an exact predictive formula.

 - Practical measurement of complexity terms: From a practical perspective, one challenge is that the two key quantities in the bound, the intrinsic dimension $d_k$ of embeddings and the network’s Lipschitz constant $L_k$, are not trivial to measure or control in real-world settings. Estimating intrinsic dimension of embeddings was done via a k-NN based Maximum Likelihood Estimator in the experiments, which introduces some estimation error, though the authors did cross-validate this with multiple methods. More problematic is the Lipschitz constant of a deep network, which is difficult to compute exactly. The paper turns to an upper bound as an approximation in experiments, but this can be a very loose over-estimate of the true Lipschitz constant, especially for networks with many layers or nonlinear activations. This means the absolute value of the sensitivity term is hard to pin down, limiting the ability to use the bound quantitatively. The authors explicitly note that “estimating the sensitivity term $L_k$ remains challenging in practice, and developing reliable estimators is necessary for broader applicability”. 

 - Relationship between factors: The theory treats the intrinsic dimension and the Lipschitz sensitivity as if they were independent factors, but in practice they can be intertwined. For instance, architectural changes can simultaneously affect both, for example, a narrower layer lowered the dimension but raised the Lipschitz constant. The authors acknowledge this as well, stating that in the current analysis they treat dimension and sensitivity independently, whereas “in reality, these quantities may be correlated,” and understanding their interplay is an important avenue for future work. This interplay makes it non-trivial to directly apply the bound for model improvements: one cannot simply minimize one factor (e.g. enforce a small embedding dimension) without considering potential adverse effects on the other (increased sensitivity). The paper stops short of providing guidance on how to jointly manage these factors.

 - Limitation in empirical validations: It’s worth noting that the empirical evaluation, though extensive within the scope of the paper, is concentrated on vision classification tasks with relatively moderate-sized datasets (MNIST, CIFAR) and standard architectures. It remains an open question how well these representation-based generalization insights carry over to much larger-scale settings (e.g. ImageNet-scale vision models or NLP tasks) or more complex scenarios, which is a relatively important empirical exploration.

### Questions
1. The theoretical analysis assumes the Bayes-optimal predictor is Lipschitz continuous with respect to the embedding space. How realistic is this assumption in practice, and what intuition can the authors provide about when it would hold? For example, are there arguments or evidence that, at least for learned final-layer features, the class label function is approximately Lipschitz? If this assumption were violated, how would it affect the applicability of the bound? It would be helpful if the authors could elaborate on the conditions under which this Lipschitz assumption is justified or provide potential idea on relaxing it.

2. The proposed bound involves unspecified (potentially loose) constants (e.g., $C_{k},D_{k},M_{F},M_{F^{*}},L_{k}(F),L_{k}(F^{∗})$) and an asymptotic rate $n^{−1/(d+\varepsilon)}$, raising concerns about quantitative usefulness. For at least one setting, could you instantiate the bound numerically to show whether it is non-vacuous? How sensitive are conclusions to $\varepsilon$, and can it be data-driven (via local covering behavior) rather than a free slack? Are there identifiable regimes (ranges of 
$d_{k},L_{k},n$) where the bound is guaranteed to be below observed gaps, or a calibration strategy that makes the bound quantitatively predictive across architectures rather than merely correlational?

3. Could the authors clarify how they define the intrinsic dimension of learned embeddings in formal terms, and discuss the reliability of its estimation? The paper uses an MLE method to estimate $d_k$. How sensitive are the results to the choice of dimension estimation technique or to the sample size used for estimation? For high-dimensional embeddings or limited data, do the authors observe any issues with accurately measuring intrinsic dimension? Similar questions for the Lipschitz constant.

4. The width intervention experiment nicely illustrates that simply forcing a smaller embedding dimension can be counterproductive due to the rise in Lipschitz constant. This suggests a trade-off scenario. Could the authors provide more insight into this interplay? For example, do certain architectural choices or layer types inherently produce lower-dimensional embeddings without as large an increase in sensitivity (perhaps residual connections, or using autoencoder-like bottlenecks with regularization)? The paper mentions that treating dimension and sensitivity as independent is a simplification. How might one theoretically approach the scenario where $d_k$ and $L_k$ are correlated? Is there a version of the bound or an analysis that accounts for their joint distribution rather than a simple product of separate terms? Any thoughts on promising directions to address this coupling would be appreciated, as it seems key to turning these insights into concrete design principles for neural networks.

5. Have the authors considered evaluating their representation-based generalization metrics on other domains or larger-scale tasks? How do the authors envision applying their analysis to the next scale of problems, and do they anticipate any new phenomena or difficulties might arise there? For instance, would measures like embedding intrinsic dimension also correlate with generalization in NLP models or on ImageNet-scale vision models? The current results use CIFAR and relatively standard networks; exploring whether the same trends hold for, for example, transformer models in NLP or very deep networks on large datasets would strengthen confidence that these insights are universal. Any preliminary observations or thoughts on potential challenges (e.g. intrinsic dimension estimation in very large networks, or the effect of tasks with many classes) would be interesting.

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
4

### Summary
This paper proposes new generalization bounds for neural networks. The proposed bounds apply to every hypothesis in the parameter space and are based on the intrinsic dimension of an intermediate layer (which can be the last) and on the Lipschitz constant of the subsequent layers. Given a parameter vector and a dataset of iid data points, the main technical idea is analyse the Wasserstein distance between the pushforward of both the data distribution and the empirical distribution of the dataset by the first $k$ layers of the network. Then, Kantorovitch-Rubinstein duality is applied by exploiting the regularity of the subsequent layers. This proof technique is based on state of the art results for the estimation of the Wasserstein distance between a probability distribution and its empirical counterpart. This introduces a notion of intrinsic dimension that the authors relate to the neural network architecture to obtain their final bounds, based on embedding dimensions. The theory is supported by several experiments.

### Strengths
- The paper finely exploits the regularity of the neural network architecture, which is a factor that is important to take into account in generalization bounds.
- The proposed bounds are computable. In particular, the new notion of intrinsic dimension is interesting and the experiments are conducted in various settings.
- The application of the proposed theory at the last layer is particularly interesting, as the regularity conditions are easily obtained. This supports the empirical correlation between the dimension of the last layer and the generalization error.

### Weaknesses
*Main weaknesses:*
 - The main bound (Thm 4.1) holds in high probability for a fixed predictor (in particular, F seems to be independent of the data). As it is assumed that the loss is bounded, in this setting a simple application of Hoeffding inequality gives a bound of order $\sqrt{\log(1 / \delta) / n}$ on $\mathrm{gen}(F)$, which seems to have a better rate than you bound unless the intrinsic dimension is strictly smaller than 2. I believe this requires some additional comments.
 - In practice, the predictor is not fixed, but it is trained based the training dataset. If such a data-dependent predictor is used, then it is not true anymore that the push forward of the empirical distribution at layer $k$ corresponds to the empirical distribution associated with the embedding distribution (because the points are not independent anymore). Therefore, the proposed proof technique would not apply. It seems to be an important issue as obtaining a bound for fixed predictor can be done with classical concentration inequalities, a new high probability generalization bound should be able to apply to data-dependent predictors.
 - If think the paper "Instance-Dependent Generalization Bounds via Optimal Transport" (Hou et al., 2022) should be added to the discussion of related works, as it also uses Wasserstein distances and intrinsic dimensions.


*Other issues:*
 - In definition 7, the notion of effective dimension does not seem to have been defined before.
 - the notation $\mathrm{gen}$ is also used before it is defined.
 - in Theorem 3.1, $d_p^\star$ has not been defined, which makes this theorem hard to read.

### Questions
- Can you explain what is the exact role of the neighbourhood $U_k$ in Assumption 3?
- In what cases can the intrinsic dimension of the embedding distribution be much smaller than the number of neurons in the layer.
- Can you produce the same plot as Fig. 3, but using the number of neurons in the intermediate layer instead of the intrinsic dimension estimator?
- Can you comment on why the reported intrinsic dimensions are quite small on Fig. 3?
- In addition to Figure 3, could you compute correlation coefficients (eg, Kendall) to get a more precise understanding of the observed correlation?

### Soundness
2

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
4

### Summary
This paper presents an exciting argument that the generalization error of deep learning models can be bounded by a combination of the intrinsic dimension of the data at a given layer of the model and its Lipschitz constant at that layer. The authors present a bound from a 2019 paper by Weed & Bach, and point out that the last layer creates a simplification with respect to the Lipshiptz constant. Additionally, the authors present a bound that depends instead on the intrinsic dimension d_k at a given layer of the network. The authors present experiments that attempt to verify that the bounds dependent on the intrinsic dimension are applicable to neural network embeddings, resulting in a series of plots that demonstrate a correlation between intrinsic dimension and a Wasserstein distance (between a empirical and population distribution of embeddings), and Wasserstein distance and generalization gap (difference between test and train error). I was really excited to read this paper, but many of the details of the argument were not clear in the main text leading to my current reservations. I would love to be convinced.

### Strengths
- The paper presents a very important line of research for the community: how is the distribution of embeddings related to the generalization performance of a model?
- The approach of breaking a model into an “encoder” and a “tail” network at any given layer is interesting and widely applicable.
- The relationship between Wasserstein distance of an embedding and the generalization gap is interesting.

### Weaknesses
- The authors are missing a large body of literature related to predicting generalization performance of models. Authors should address why the 2020 NeurIPS challenge on Predicting Generalization in Deep Learning is not applicable, or unsuitable for this study. Some of the same models are included, but importantly at various levels of training (and test) performance. 
- The terminology used in the paper is not consistent. The generalization gap can be approximated as the difference between the train and test error. However, the authors seem to refer to this as the generalization error in the start of the paper. The generalization error can be measured using test accuracy or test loss, i.e. samples that are not in a limited training set. I believe this is an important distinction, because if a model is poorly trained, the gap between train and test may be 0, which is not a very useful. This assumption should at minimum be specified. 
- In 5.1, it appears the experiment is only able to demonstrate log-linear correlation between intrinsic dimension and the Wasserstein distance between two independent samples of embeddings, with other terms ignored.
- In 5.2, the Wasserstein distance is instead computed between train and test embedding distributions. Moreover, I believe there is a methodological flaw in the evaluation shown in Figure 3. Fitting curves and R^2 values should be computed for each model independently.
- The main result in the paper seems to based on replacing d in the bound from Weed and Bach with d_k, the intrinsic dimension computed by MLE. I could be wrong but I’m not finding adequate justification for this in the main body of the paper. It is also not clear why delta (1-delta representing a probability) is introduced in Theorem 4.1.

### Questions
- Why are two independent draws of embeddings at a given layer suitable for representing the “empirical” and “population” distribution of embeddings?
- How are the distributions of embeddings at a given layer learned or summarized? They are almost surely not Gaussian and cannot be summarized with a multi-variate mean and std alone, so I am a bit surprised.
- Most of the results seem to be focused on the last layer, what about results on the interior layers?
- How is the Bayes predictor used in the empirical experiments, if at all?
- In Line 219, what is meant exactly by “[the Bayes predictor] returns the best possible prediction at each embedding” ?
- Provide a reference for the spectral norm of Linear+ReLU layers bounding the layer’s Lipshitz constant.
- Line 236, McDiarmid’s inequality? 

Possible improvements:
- One of the interesting arguments in the paper is that the intrinsic dimension of embeddings at any layer, together with the Lipschitz constant, can be used to bound the generalization gap of the model. Empirical results evaluating this behavior as a function of layer would be interesting and improve the experimental validation portion of the paper. 
- The ability to predict the test performance (e.g. given the train performance) of a model, such as in the setup of the the aforementioned PGDL challenge, would increase the impact of this work. I suggest the authors to review the metrics and criteria used in similar work to validate their cited bounds.
- A larger portion of the paper is dedicated to definitions and assumptions, but there is practically no explanation or proof sketch for Theorem 4.1. It would be very useful to explain how Eqn 1 is meaningfully different from the eqn on Line 191 if the leading term indeed dominates the behavior.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel, representation-centric generalization error bound for neural nets. Unlike traditional parameter-based bounds, the authors link generalization to the geometric properties of the learned embeddings.

The main theoretical claim is that the generalization error at any given layer k is controlled by two key factors:
1. the intrinsic Dimension of the embedding distribution at layer k.
2. Lipschitz Constants of the downstream network (from layer k to the final output). These measure how much perturbations in the embedding space are amplified in the loss.

A key insight is that this bound simplifies at the final layer. Here, the downstream map is the identity (i.e. Lipschitz constant 1) and the generalization error hence is driven by the intrinsic dimension of the final-layer embeddings and a data-dependent notion of smoothness (via a certain Bayes predictor) .

The authors also provide empirical support for their theory.

### Strengths
- This paper provides an interesting theoretical bridge between representation geometry (via intrinsic dimension) and generalization. 
- The central idea of combining Wasserstein convergence rates (Weed & Bach, 2019) together with a perturbation analysis based on the network-dependent Lipschitz constants appears novel.
- Overall, the paper is well-written, with one exception (see Weaknesses).

### Weaknesses
- It is not clear what is formally meant by “intrinsic dimension”, which is defined via “effective dimension” of $P_k^Z$, but never formally discussed. Initially, I assumed that you mean the cardinality of the support, put it appears that your notion of dimension can have fractional values. Since this quantity is a critical component of the analysis, it should be crystal clear to the reader what $d_k$ actually is. Likewise, it should be clear what are the limitations of practically estimating the $d_k$’s. 
- I do not quite understand why the correlation between the bound and the generalization gap is not shown alongside the empirical results. The presented results show that certain components of the bound correlate with the generalization gap, but this is ultimately not sufficient to claim that the bound provides an compelling explanatory theory of generalization.
- Practical Computability of the Bound: as with most work in this area, the derived bound is not practically computable. It contains several terms ($L_k(F)$, $L_k(F^*)$, $D_k$, etc.) that are either data-dependent and unobservable (the Bayes predictor's Lipschitz constant) or computationally hard to estimate for complex, modern architectures. The value of the bound is therefore mostly conceptual.
- Adding to this, For intermediate layers, $\bar L_k$ is estimated via spectral-norm products proxy; this is an upper bound and can severely overestimate sensitivity.

### Questions
- What is the definition of the intrinsic dimension? Are there well-known practical difficulties to estimating it?
- Does the actual bound itself meaningfully correlate with the generalization gap?
- It seems that across your experiments, you sometimes obtain intrinsic dimensions $>2$ (even 5-6 in case of MNIST), which corresponds to your generalization bounds decaying more slowly than $\sim 1/\sqrt{n}$. How can this be reconciled with classical VC theory which states that learning is at most as slow as $1/\sqrt{n}$ (as soon as the number of samples exceeds the VC dimension)?
- For CIFAR-10/CIFAR-100, when you average the intrinsic dimension across classes in your experiments, do you obtain something greater or smaller than $2$?
- Regarding sec 5.4.: did you also compute the correlation of the whole bound with the generalization error when varying the width? It seems that your figure 4 cannot reveal whether the generalization bound admits the correct tendency under changing the width.

### Soundness
2

### Presentation
2

### Contribution
2
