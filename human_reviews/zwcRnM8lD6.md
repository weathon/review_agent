# MILE: Mutual Information LogDet Estimator

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 3, 3

## Abstract
Mutual information (MI) estimation plays an important role in representational learning. However, accurately estimating mutual information is challenging, especially for high-dimensional variables with limited batch data. In this work, we approach the mutual information estimation problem via the logdet function of data covariance. To extend the logdet function for entropy estimation of non-Gaussian variables, we assume that the data can be approximated well by a Gaussian mixture distribution and introduce a lower and upper bound for the entropy of such distributions. To deal with high dimensionality, we introduce ``ridge'' term in the logdet function to stabilize the estimation. Consequently, the mutual information can be estimated by the entropy decomposition. Our method MILE significant outperforms conventional neural network-based MI estimators in obtaining low bias and low variance MI estimation. Besides, it well pass the challenging self-consistency tests. Simulation studies also show that, beyond a better MI estimator, MILE can simultaneously gain competitive performance with SOTA MI based loss in self-supervised learning.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The article offers a Gaussian Mixture-based differential entropy/mutual entropy estimation approach. Furthermore, it provides numerical experiments to test the expected behavior of the estimator and its application to self-supervised learning.

### Strengths
The article addresses an important problem of mutual information estimation. It provides relevant numerical experiments to test the validity of the proposed approach.

### Weaknesses
- The main approach proposed by the authors seem to be already appeared in the literature in some references not cited by the authors (please see the questions part).

- There seems to be a major issue about the expressions provided for the proposed approach (please see the questions part).

- The presentation requires improvement.

### Questions
### I. INTRODUCTION 

**3rd paragraph:** 

- "identify matrix":  identity matrix?

- "The mutual information can be consequently estimated by the entropy decomposition.": This sentence follows identity matrix addition sentence. I guess it might be better to clarify causality here. At this point, it is not clear what is meant by "entropy decomposition", whether it is a trivial procedure and what enables it (mixture of Gaussians modelling?).

### 2.1 BACKGROUND

**Paragraph before (4)**

- After equation (1): instead of "for a multi-variable Gaussian variable" use Gaussian (random) vector ?

- In the notation $$X=[x_1,x_2, \ldots x_n]$$ $x_i$'s appear as column vectors, however, they are actuallly row vectors as $X\in\mathbb{R}^{n\times d}$

- (5) should be

$$\mathbf{H}_D(X)=\sum_{i=1}^k \frac{1}{2} \log \left(\lambda_i+\beta\right)+(d-k)\log(\beta)+C_d$$

- After (5): "Therefore, LogDet can estimate the entropy of multivariate Gaussian variables by approximating the differential entropy.". This is not a surprise/or contribution as the authors  simply defined (5) using (2) by replacing the true covariance with $\beta I$ perturbed sample correlation (covariance?) matrix. This is sort of obvious. 

### 2.1.1 LOGDET ENTROPY ESTIMATOR FOR NON-GAUSSIAN VARIABLE

- Title : ... NON-GAUSSIAN VECTOR

- Replace variable->vector

- There already exists GMM based entropy/mutual information approximation based works such as 

[a]. Lan T, Erdogmus D, Ozertem U, Huang Y. Estimating mutual information using gaussian mixture model for feature ranking and selection. InThe 2006 IEEE international joint conference on neural network proceedings 2006 Jul 16 (pp. 5034-5039). IEEE.

[b]. Huber MF, Bailey T, Durrant-Whyte H, Hanebeck UD. On entropy approximation for Gaussian mixture random vectors. In2008 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems 2008 Aug 20 (pp. 181-188). IEEE.

You need to refer to existing literature and clearly state what is novel in your approach relative to them.


- Theorem 2 and Theorem 3 of [b] above already covers the lower and upper bounds of mixture of Gaussians. It looks like they are same as what is provided in this section. 

- There seems to be a major issue about the upper bound expression. The first expression for the upper bound (at the bottom of page 3), contains covariances ($\Sigma_i$'s ) obtained from the GMM fitting algorithm, whereas the second line contains the overall sample covariance of actual data, instead of conditional covariance estimates. How do you equate these lines? The second line in fact equals to

$$\frac{1}{2} \log \operatorname{det}\left(\frac{X^T X}{n}\right)+\sum_{i=1}^K \pi_i \cdot\left(-\log \pi_i+C_d\right)$$

as $\frac{1}{2} \log \operatorname{det}\left(\frac{X^T X}{n}\right)$ is independent of the summation index $i$. This does not make sense as you disregard covariance parameters of the GMM. 

- How do you make the upper bound objective co

### 2.2 THE ISSUE OF MODEL SELECTION

- Title: Model Selection is to generic for the discussion in this section. "The Issue of Model Order Selection" could be a better title.




### 3. APPLICATION IN SELF-SUPERVISED LEARNING

The logdet-mutual information based SSL appears to be proposed in the following reference:

[c]. Ozsoy S, Hamdan S, Arik S, Yuret D, Erdogan A. Self-supervised learning with an information maximization criterion. Advances in Neural Information Processing Systems. 2022 Dec 6;35:35240-53.

The authors should also clarify the relative novelty relative to [c]. Especially, the impact of GMM order selection as the approach in [c] appears to be for $K=1$. There is also claim in [c] that the use of $K=1$  defines correlative information maximizing which targets a linear (identity in their modified setting) between the representations of augmented versions of inputs. For $K>1$ does  maximizing mutual information between augmentation representation lead to nonlinear mappings between them? Is such organization of representation space desirable for classification tasks, for example?

Or are you just using (18) with order $1$, which seems to be just the approach in [c]. 

### 4. RELATED WORKS & 5 SIMULATION STUDIES

All the references we mentioned above and the relevant references that cite them should be included in this discussion, and simulation results 

- 5.2 : ofBelghazi...-> of Belghazi
- Figure 2: Two small figures and caption could be more informative.
- 5.4 SSL: What is K for EMP-MILE? Is upper bound employed in EMP-MILE?  what if you directly use MILE?
How is backprop used in coordination with the GMM algorithm? As GMM parameters are algorithmically obtained from network output, how does backprop do backward mapping from probabilities $\pi_i$'s (and there should be covariance estimates $\hat{\Sigma}_i$'s, as discussed above)

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a new approach to estimating the mutual information between a pair of random vectors, by extending the closed-form expression that is available to Gaussian variables to non-Gaussian variables. This is done by estimating Gaussian mixture approximations of the involved densities and then using bounds on the differential entropy of Gaussian mixtures.

### Strengths
Estimating mutual information between high-dimensional non-Gaussian variables is an important problem with many applications. The proposed method extends Gaussian (which the authors refer to log-det) estimators to be applicable beyond Gaussian variables via the use of Gaussian mixture approximations, coupled with bounds on the differential entropy of mixtures.

### Weaknesses
Unfortunately. the paper contains several critical flaws, namely a quite sloppy notation, that lead me to recommend its rejection. 

The authors mixture, in a very confusing way, random variables and data matrices, typically using the same notation for both, $X$. For example, in Equations (1), (2), and (10), $X$ is a $d$-dimensional random variable, whereas in Equation (4), $X \in \mathbb{R}^{n\times d}$ is a data matrix. Even worse, in the final equation of page 3, the two different definitions are used together and it is not even clear where the second equality means; it is simply wrong because $X^T X/n$ does not coincide with $\Sigma_i$.

Unlike what the authors claim, Equation (5) is not equivalent to Equation (5); the two differ by $\frac{d-k}{2}\log \beta$.  

Adding a matrix proportional to identity ($\beta I$ in the paper) to the sample covariance was not proposed in a 2021 paper. It is a very classical method that can be found in any classical text on covariance matrix estimation, many decades ago.

The inequality in Equation (8) was not shown by Zhouyin and Liu in 2021. It is a classical result of information theory, that can be found, for example, in the famous Cover and Thomas book. By the way, the citation to this book is wrong in the paper; one of the authors (J. Thomas) is missing. 

The two bounds for the differential entropy of mixtures that the authors claim to have introduced are in fact not new. The upper bound is in fact a well-known corollary of the log sum inequality (see the Cover and Thomas book). The lower bound was proved in 2008 by Huber et al. at https://doi.org/10.1109/MFI.2008.4648062

### Questions
I have no questions.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents a mutual information (MI) estimator called MILE (LE=logdet estimator) which uses 
the log det closed form formula of the entropy of Gaussians.

To accomodate MI to arbitrary densities, a Gaussian mixture model (GMM) is first fit to data and lower/upper bounds on the entropy of GMM is used to define MILE formula Eq 15. 

Then MILE is benchmarked with other MI  estimators and MILE can be used in loss functions in semi-supervised learning in experiments.

### Strengths
- Simple MI estimator method based on  

Zhanghao Zhouyin and Ding Liu. Understanding neural networks with logarithm determinant entropy estimator. arXiv preprint arXiv:2105.03705, 2021

(cited in the paper)

- Very good experiments and comparisons with other MI estimators

- Source codes provided in supplemental information  for reproducible research

### Weaknesses
-The paper is sloppy in its writing, and one problem is to determine the number of components k of the GMM which
 loosen the lower upper bounds on the entropy. 

- Another problem is to deal with near singularity (det close to zero) by introducing a regularization term \beta.

- Give definition of MI and link with copulas, e.g.,
Ma, Jian, and Zengqi Sun. "Mutual information is copula entropy." Tsinghua Science & Technology 16.1 (2011): 51-54.
This will relate to Eq. 8 as well.

- Because MI estimation is an important and well-studied topic, I suggest to put Section 4 on related works after the introduction to that the contributions are better explained.

- The lower/upper bounded of entropy of GMMs are not tight. There is a rich litterature which also compares the tightness of the various bounds.

Huber, Marco F., et al. "On entropy approximation for Gaussian mixture random vectors." 2008 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems. IEEE, 2008.

Even in 1D:
Nielsen, Frank, and Ke Sun. "Guaranteed bounds on the Kullback–Leibler divergence of univariate mixtures." IEEE Signal Processing Letters 23.11 (2016): 1543-1546.

- Notice that some distributions do not admit densities (some elliptical distributions for example)



- Mention MI properties (i.e., tensorization) which defines the self-consistency test of estimators


- small remarks:
* data covariance = scatter matrix
* after (3), define $\Sigma_x$ as scatter matrix?
*  page 3, first sentence need to be rephrased
* some typos: 
page 7  hyperparamter -> hyperparameter
page 9 self-supervied -> self-supervised    competitve -> competitive

### Questions
- Would using PCA beforehand be more appropriate in the case of near singularity?

- Can we tackle robustness/variance with f-MI?

Moon, Kevin, and Alfred Hero. "Multivariate f-divergence estimation with confidence." Advances in neural information processing systems 27 (2014).
Esposito, Amedeo Roberto, Michael Gastpar, and Ibrahim Issa. "Robust Generalization via f− Mutual Information." 2020 IEEE International Symposium on Information Theory (ISIT). IEEE, 2020.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
this paper proposes to use the logdet function for the estimation of mutual information. 
two bounds are proposed for this purpose. the results show improvement in comparison 
to the editing methods. the proposed function itself is "the Coding Length Function".

### Strengths
simple method with good results.

### Weaknesses
In my opinion this paper reinvents "Coding Length Function".  "...the difference is we put a scaling hyperparameter β on the identity matrix I.." - that is not a difference. both affects SNR. The latter can be affected either way: by multiplying the noise covariance or by division of the data covariance. I do agree that the results are interesting, but the novelty is quite limited due the the above. 

please elaborate on the limitations.

### Questions
"So, we recommend β = 1e−3 in the following simulation studies" why not beta=zero? 
Figure 1.b shows that beta=zero correctly estimates the true MI. 
That raises a question why do you need beta > 0?

How do you define $\pi_c$ in e.g., Eq17?

Both bounds are loose. How can you explain that such loose bounds lead to very small variance in MI?

Do you calculate MILE in batches?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes uses bounds on the entropy and mutual information for a mixture of Gaussian random variables based on the log determinant calculations used in calculating the entropy for a single Gaussian. In the context of self-supervised learning, the Gaussian mixture is assumed to known based on the augmentation. In other cases the number of mixture components has to be selected. Empirical results are reported on a synthetic benchmark of correlated Gaussians with and without non-linear transformations. Results of self-consistency measures are reported on CIFAR10.

### Strengths
The paper is a logical motivation. Differential entropy is easy to calculate for Gaussian distributions, and mixture of Gaussians are universal approximations given enough data, so why not use GMM for mutual information estimation. The insight of using the augmentations as defining the GMM is a useful, simplifying assumption.

### Weaknesses
One main weakness is the lack of extensive comparisons of using this method for self-supervised learning versus other. The one example in the main body (Table 1) shows that at 300 epochs the method is better than some other methods but is inferior to EMP-SSL. At 1000 epochs the other methods outperform the listed, but no results for 1000 epochs are reported. 

The second main weakness is the paper does not give a complete description of the method. The paper is lacking in clarity with some key point unaddressed. The notation is confusing since the random variables (Z,Z') are denoted the same as Z_c, which may be a data point in the empirical sample. There should more clarity on random variables as compared to  sample sets, starting back before equation 4. The confusion carries to last paragraph of Section 4 where $\mathbf{X}$ is defined but then $X$ is used in the definition. 

The use of one instance for one cluster is not clear to me upon reading it
"This is because we treat the augmented data from one instance as a cluster, and this data
augmentation strategy automatically clusters the data." This should be re written.

 In equation 17 it is not clear how $\zeta_c$ captures all instances in the batch. It has only a single $i$ index. Perhaps the $\zeta_c$ should concatenate them all. In section 3.2, $\zeta_c$ is a set which indexes the whole match, which makes more sense, but it should be a matrix not a set. In any case, how is the $H(Z)$ term estimated in section 3.1? By keeping $Z_c$ fixed and only augmenting the second the one covariance matrix will be rank-1 (before ridge). 

It doesn't sound like the experiments for the 5.2 are run fairly " our MILE estimator does not require extra training," In this problem the point is that the MI could be changing at each data instance. Thus, other methods do not use access to the change points. MILE should have to be run (which involves performing the GMM since there are no self-clusters as in SSL) at each point. Running an expectation maximization is as much or more training than the updates of network.  	

In the SSL, the trade-off parameter having to be searched in the grid  [0.01,0.1,1.0,2.0] doesn't seem to be efficient compared to EMP-SSL. 
 
In terms of unsubstantiated claims, the method is clearly biased (not only by the choice of number of components) but also on the non-linear transform cases. It is not clear how well the mutual information estimation would actually work on more complicated data. Thus, even if it is useful for self-supervised learning is not necessarily a more accurate estimate of differential entropy. 

**Minor:**
There are a number of typographical mistakes that are distracting.

I don't understand what this means
"often dwarfing traditional parametric and non-parametric approaches in statistics"

" base on the " -> "based on the " 

I'm not familiar with this phrasing "When X subjects to a Gaussian" 

"a ‘noise’ $\hat{X}$ " -> "a noisy $\hat{X}$" 

The paragraph before equation (4) are not clear. " an expanding factor" is not defined nor is it clear what is meant by "enlarging the original covariance matrix".

Extra $=$ on equation 14.

"trading each" -> "treating each" ? 

" ground true data" 

"SMILE: moothed" -> "SMILE: smoothed" 

It should be a parenthetical reference for You et al. (2017) fo LARS optimizer.

### Questions
How is the $H(Z)$ term estimated in section 3.1? Is it also based on augmented data?

In the SSL, the trade-off parameter having to be searched in the grid  [0.01,0.1,1.0,2.0] doesn't seem to be efficient compared to EMP-SSL. Are there hyper-parameters for EMP-SSL?  

Why in Table 1 is 1000 epochs not tested?

Is the GMM method run at each time point in Figure 2?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
