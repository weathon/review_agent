# On PAC-Bayes Bounds for Linear Autoencoders

- Decision: Reject
- Scores: 6, 6, 6

## Abstract
Linear Autoencoders (LAEs) have shown strong performance in state-of-the-art recommender systems. Some LAE models, like EASE, can be viewed as multivariate (multiple-output) linear regression models with a zero-diagonal constraint. However, these impressive results are mainly based on experiments, with little theoretical support. This paper investigates the generalizability -- a theoretical measure of model performance in statistical machine learning -- of multivariate linear regression and LAEs. We first propose a PAC-Bayes bound for multivariate linear regression, which is generalized from an earlier PAC-Bayes bound for single-output linear regression by Shalaeva et al., and outline sufficient conditions that ensure its theoretical convergence. We then apply this bound to EASE, a classic LAE model in recommender systems, and develop a practical method for minimizing the bound, addressing the calculation challenges posed by the zero-diagonal constraint. Experimental results show that our bound for EASE is non-vacuous on real-world datasets, demonstrating its practical utility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a PAC-Bayesian analysis of Linear Autoencoders (LAE), which have shown remarkable effectiveness in recommendation systems. The main contributions include:

* A thorough theoretical revision of convergence analysis from previous work (AAAI 2020), addressing and correcting significant mathematical inconsistencies in the derivations.
* Introduction of two novel theoretical bounds for LAE performance: Gaussian data distributions & data with Gaussian parameters. Both accompanied by computationally efficient implementation methods.
* Empirical validation on major recommendation datasets (MovieLens 20M, Netflix, Yelp2018, and MSD), demonstrating notably tight bounds that fall within twice the actual test error - a significant improvement over typical theoretical bounds in the field.

### Strengths
**Originality:**

*   Novel Application of PAC-Bayes Bounds: The paper's most original contribution lies in its application of PAC-Bayes bounds to analyze the generalizability of LAE models. While PAC-Bayes bounds have been used in other machine learning domains, their use in the specific context of LAE recommender systems is novel. The authors successfully bridge this gap by adapting existing PAC-Bayes frameworks, specifically Shalaeva's bound, to the challenges posed by LAEs. Specifically they had to extend  the analysis for single linear regression problems, to accommodate the multi-regression nature of LAEs. In LAEs, each row of the data matrix is treated as a separate regression problem, requiring a generalization of the bound to handle this specific structure. Here, I think it may exists similar approches in the bandit literature for the analysis of LinUCB like algorithms. 


**Quality:**

*   Rigorous Theoretical Analysis: The paper exhibits a high level of quality through its meticulous theoretical derivations and proofs. The authors provide detailed steps for each proof, ensuring the mathematical soundness of their work. This rigorous approach strengthens the credibility of the proposed bounds and enhances the overall quality of the paper.
*   Development of a practical method for calculating the PAC-Bayes bound based on the bounded data and Gaussian parameter assumptions. This practical contribution bridges the gap between theory and practice, making the PAC-Bayes bound a useful tool for evaluating LAEs in real-world settings. The authors' adaptation of this method to the specific constraints of the EASE model.
*   Empirical Validation: The authors take a commendable step to empirically validate their theoretical findings by conducting experiments on four real-world datasets.  Their choice of datasets, including MovieLens 20M, Netflix, Yelp2018, and MSD, represents a diverse range of recommendation scenarios. The observed tightness of the bound, being within twice the test error is interesting (but the non vaccous claiming is an overclaim in my opinion)

**Clarity:**

*  Paper is well written and provide careful definition of the concepts and notation. 
*  The authors honestly acknowledge the limitations of their work, particularly regarding the applicability of their bounds to more complex recommendation scenarios that use evaluation metrics such as Recall@k and NDCG@k. 

**Significance:**

* **Identification and Resolution of Convergence Issues:** The paper demonstrates significance in its critical examination of Shalaeva's bound. In my point of view the errors made by Shalaeva's work are unacceptable (limit and integral inversion without any check and omitting a distribution when computing an E). They should at least lead to the withdrawal of the 2020 paper.  
* Yet another linear analysis

### Weaknesses
1. The paper's positioning relative to previous work in recommender systems is inadequate and imprecise. For instance, citing Rendle 2022 for Matrix Factorization/ALS is an unusual choice, as Rendle is primarily known for his 2010 work on Factorization Machines. This suggests a need for more thorough engagement with the historical development of these methods.

2. The practical applicability of these findings to real-world recommender systems is unclear. While the experimental section successfully demonstrates the theoretical bounds, it falls short of the standards expected in recommender systems research, lacking comprehensive evaluation on metrics and scenarios that matter in practice.

3. The analysis primarily focuses on the non-regularized model, which has limited practical relevance as real-world systems invariably use regularization. Though the appendix addresses the regularized case, the analysis remains incomplete and doesn't provide meaningful insights into why this form of regularization is effective in practice.

### Questions
1. Since the initial error appears in a paper published at AAAI, why not contact the Program Committee from that edition? I believe AAAI should take responsibility for errors made during their review process.

2. I'm curious about how similar or different this is to the analysis of random projections and LinUCB. While we're working in a Bayesian framework, the mathematical tools used are quite similar, and some research has already bridged the gap between these approaches (such as the work in https://www.jmlr.org/papers/volume17/14-087/14-087.pdf).

3. I wonder if your bound could be helpful for optimistic exploration strategies when the user (row) is sampled from a uniform distribution and the item is selected by the algorithm. This might be challenging since independence is broken, and you'd likely need additional assumptions (such as rank-1 matrix only) - which, while commonly used in state-of-the-art approaches, may be overly simplistic.

4. In all EASE-based system implementations I'm aware of, the zero-diagonal constraint is used. While the paper studies this to some extent, can your analysis provide new insights beyond confirming it as a useful bias?

5. You mention producing 'non-vacuous' bounds. Could you elaborate on what constitutes such a bound? Specifically, if your definition implies practical usefulness, could you explain how these bounds could be used to improve recommender systems?

6. I suspect the relative tightness shown in the final table is due to M, which bounds the error on the test set. While getting a concentration-like inequality is expected, what additional insights does your analysis provide? From a practical perspective, M significantly reduces the bound. Could you discuss how M is computed (is it updated after each sample or calculated once using all data, including test data)? Also, a graph showing how the bound and test error vary with the number of observations would be more informative than a table that doesn't indicate how these results compare to the state of the art.

### Soundness
3

### Presentation
3

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
===========================================Post Rebuttal comments=====================

There were very serious problems/errors in the original version of the paper. The authors have worked extremely hard on the rebuttal with spectacular results. The paper has been substantially rewritten and improved based on my comments during the rebuttal period, making **huge progress**.
Therefore, I believe they have achieved the unlikely feat of pushing the paper above the borderline. However, I cannot fully vouch for correctness as I couldn't check all the details of the new proofs, though the direction makes sense and what is written in the rebuttal makes sense. 


Summary of original issues and changes/resolutions: 


**Problem 1** (solved)                                                        

Severe error in Theorem 7 (trivial and worse than existing results due to inconsistent understanding of sampling procedure)


**Solution 1**

Removed Theorem 7


**Problem 2** (More or less solved)

The sampling regime and learning setting didn't make any sense because of the lack of a train test split within each user. This means the results originally only made sense for multivariate regression (though it wasn't proved for that case) and meaningless for RecSys.


**Solution 2**  

The authors have modified the whole learning setting and corrected all the theorems and reran the experiments. They have also proved the results for multivariate regression in general.  Solving this issue was one of the key deciding factors in my decision to raise my score. 

Nevertheless, it is worth noting that the bounds do depend on quantities such as $\Sigma_{xy}$, which means that although they can be evaluated with empirical estimation, they are difficult to interpret in terms of sample complexity without data dependent assumptions. However, this may well be a general feature of PAC Bayesian bounds in general. Still, in this particular example, it makes the bounds very qualitatively different from other non Bayesian PAC bounds, so the differences could be further discussed. However, I am still ok with the current resolution at this point. 


**Problem 3** (solved)

Bounds don't apply naturally to Recsys because of the use of the square loss, which means the bounds cannot really apply to the implicit feedback setting with any reasonable metric such as Recall despite the fact the authors frequently show illustrations of the bounds from the implicit feedback setting. 


**Solution 3**

The authors have added many more details and caveats downgrading the originally over the top claims. **Please keep the caveats there in the final version.**


**Problem 4** (solved)

The original literature review was extremely sparse, citing only one work generalization bounds for RecSys/matrix completion. 


**Solution 4**

The authors have added a much more detailed literature review based on my comments. 


==========================Changes for the individual scores=====================

Main rating: from 3 to 6


Soundness: from 2 to 3

Contribution: from 2 to 3

Presentation: from 2 to 3


============================================Summary===========================

This paper proves generalization bounds for linear auto encoders based on a famous theorem of Alquier [PB2] which extend/transfer those of [PB1] (which apply to the linear regression setting) to the setting of linear auto encoders such as EASE [LA1].  The authors also fix some errors in the convergence analysis of [PB1]: they show that contrary to the claims in [PB1], even with a Gaussian prior, bounds of the family presented in both [PB1]  and the present paper diverge due to the $\psi$ term in Acquire’s bound involving terms of the form $\mathbb{E}(\exp(X^4))$ for a Gaussian $X$. Theorems 1 proves a bound for the Gaussian case, whilst theorems 3 and 4 show the bound for the case where the observations are bounded. Theorems 2 and 5 show that the bounds for theorems 1 and 4 converge to zero as the number of samples tends to infinity, respectively. Theorem 4 is much more general than theorem 1 as it applies to an arbitrary sampling distribution with the property that the observations are bounded. Later in Theorem 6, the authors show how to calculate the bounds from Theorems 1 or 3 more precisely by computing the optimal posterior to minimize the bound, with similar arguments to the calculation of the analytic solution to the optimization problem in [EASE]. Section 4.4 goes further by also calculating the KL divergence between the prior and posterior explicitly, and shows how to estimate the whole bound in practice: unfortunately, the $\Phi$ term in both theorems 1 and 3 is hard to compute in practice, and the authors use a trick from [PB3] to create a coarser upper bound which may not converge as $m\rightarrow \infty$, but has the more favorable property of being easier to calculate. However, it still involves the population expectation of the norm of the row vector of observations for one user, which the authors estimate empirically with the whole dataset. Experiments on real life datasets show that the bounds are not so far from the true generalization gap, which the authors argue makes the bounds non vacuous whilst existing bounds are vacuous.  Further, Theorem 3 shows another bound based on some standard covering number argument approach as in [AR1].

### Strengths
1. The paper is reasonably well-written, especially the introduction (though I have grave concerns about the content).
2. The paper corrects a mistake in the informal convergence analysis in [PB1] with a more rigorous analysis of the convergence. 
3. The proofs are long and a nice exercise in dealing with various Gaussians and turning them into quadratic forms. The explicit calculation of the posterior in Theorem 6 is also worthy of interest, and most of the proofs I had time to look at seem correct (except theorem 7).

### Weaknesses
Note: at ICLR, revisions of the paper can be submitted. I am quite willing to increase my score if the next revision better puts the results in perspective, or provides some clarification on the points I raised. 

Before I delve into each weakness individually, here is a summary of the issues:

1. (Fatal) The bounds are **meaningless in a Recommender Systems** scenario for many inter-related reasons, only some least serious of which the authors admit to:
    1. (Partially admitted) The bounds don’t apply to any reasonable loss function or training scenario from Recommender Systems: they don’t work for implicit feedback with measures such as the recall or precision or AUC, and they don’t work for explicit feedback by withholding a part of the interactions either. The loss function is merely the reconstruction error of a new sample (i.e. a new user, with all of its interactions being fed to the model for evaluation) in terms of the square loss. This says absolutely nothing about generalization performance. Perfect performance is achieved by the identity function. In particular, the bounds without the diagonal constraint are meaningless.  The authors do say in the conclusion that “the problem in recommender systems is more complicated….potential for further research”, but this doesn’t do proper justice to exactly how weak the results in the present submission are. 
    2. (Not explicitly admitted) Going further in the direction of the point above the first point above, the bounds do not involve any function class restriction apart from Frobenius norm and by the authors’ own admission  the qualitative behaviour of the bounds doesn’t change much with or without the diagonal constraints: there is no non trivial description of the dependence on the number of items $n$, and the rate of decay of in the number of samples $m$ is as weak as $1/m^{1/n}$, in comparison with a more typical $1/\sqrt{m}$.  
2. (Serious) Theorem 7 is **always vacuous**: the authors use a covering number argument based on counting the **number of parameters/dimensions** in a space with $m$ dimensions, where $m$ **is the number of samples**. This is despite the fact that the authors claim this result is superior to the result in [AR1], which cannot be the case. 
3. (Serious) The **related works** on generalization bounds for recommender systems only includes [AR1], when there are plenty of seminal works in similar directions to [AR1]. Perhaps the authors are trying to dismiss the matrix completion literature because the loss function in this branch of the literature concerns explicit feedback rather than implicit feedback. However, because of the weaknesses above, it would be absurd to claim that the loss function in the present paper is somehow better suited to the recommendation task. Furthermore, [AR1] is not exception to that, so there is no reason not to include more modern works in that direction. 
4. (Serious) The authors claim that the bound in [AR1] is vacuous, which I do not believe. They also claim that their own bound is non-vacuous, which I do not believe is a fair statement either. As explained above, if we do not have diagonal constraints, then taking the function class which contains only the identity function gives a vanishingly small bound which is non vacuous in the same sense as the authors’ bound. If we do have diagonal constraints but no specific assumptions on the data, it is not clear how the bounds presented here can take this into account. 
5. (minor) Some of the theorem statements somewhat lack clarity/rigor in their presentation. 
6. (arguable) Whilst I completely agree with the authors that the analysis in [PB1] is wrong, using words like “error”, however pertinently, when describing other works is dangerous. Certainly, one should be more careful doing it than what the authors are doing when they state (cf. line 114 page 3) “Here the convergence analysis from [PB1]”. Where in the reference is it? After checking, I can see that the authors mean the argument on page 3 after the main theorem. I understand that this analysis indeed constitutes one of the main claims of the paper. However, as a courtesy to the authors, it might be better to rewrite the statements in such a way that it appears as if this is a minor component of the paper. Indeed, to the best of my understanding, there is **no error in [PB1] which is in a Theorem environment**. Only the (admittedly important) description below the theorem is wrong. That is something to capitalize on when crafting a more tactful correction. 






***Details:*** 

On weaknesses 1.1 and 1.2 : the loss function is $\\|r^{\top} -r^{\top} W\\|\_{F}^2$   where $r\in\mathbb{R}^n$ is a vector of interactions for a new user. The generalization bounds in the present paper state that if one is able to reconstruct the training samples well, then one is also able to successfully recover a test set sample. This is assuming the whole sample is fed to the model, so that statement doesn’t contain any information about recommendation performance. It is not clear whether the authors are trying to claim that their model shows generalization bounds for the implicit feedback prediction task (predict which interactions will happen in the unseen test set) or the explicit feedback prediction task (predict the ratings on unseen (user, item) combinations). In the beginning of section 3.4, the authors mention ratings  typically being in the range of [0,5] (line 248 page 5), which hints at the explicit feedback case, but on line 69 in the introduction they hint at the implicit feedback case. At the end, they solve neither: I can understand that solving the implicit feedback case would be challenging, and that the loss function needs to be user by user in an autoencoder setting, but at the absolute minimum, for the sampling strategy to make any sense as a proxy to the real recommendation task, the authors should **split each test user into two parts item-wise**: one to be fed to the model and one to be used at evaluation. For instance, the test error could be defined as $\|r_{test}-r_{train}W\|\_{F}^2$ where $\[r_{train}\]\_j=1$ if $j$ is interacted by the user AND $j$ is in a predetermined “training” subset of items and $\[r_{train}\]\_j=0$ if either of those conditions is not statisfied (similarly, $r_{test}$ should be defined over the complementary, “test” set of item). It is acceptable for the training set to vary randomly for each user, but they must be distinct. The bounds in the present paper will certainly not mean anything in this more rigorous setting: indeed, providing any theoretical insight into why EASE works is a very challenging task which the authors haven’t really attempted: it requires understanding what function class restriction is implicit in the diagonal constraint.  It would probably be easier to prove bounds for EASE like models which introduce a low rank condition (cf. [ELSA]).

(More minor) Furthermore, the approximation the authors use for the practical bounds vaguely appeals to the law of large numbers as a justification. This means that the quantity evaluated by the authors isn’t a bound which they have proved. Why not incorporate the quantitative argument here? Instead of using the whole dataset to estimate $\mathbb{E}(r^\top r)$, the authors should use only the training set and independently prove that this quantity approaches the true value, propagating the errors through the bounds, resulting in a new and similar result which they can evaluate. 

On weakness 2: 

Ignoring constants,  if $\epsilon $\leq 1$, the bound can be processed this way: 

$$4 \left( \frac{8M}{\epsilon} + 1 \right)^{2m} \exp\left( -\frac{m \epsilon^2}{32 M^2} \right) \gtrsim  4\exp(2\log(1+8M) m- \frac{m\epsilon^2}{32M^2})$$ 

This doesn't converge to zero when $\epsilon$ is less than the constant $8M\sqrt{\log(1+8M)}$.


(This certainly  has to happen given the vacuous argument on page 23, line 1197, which covers a 2m dimensional ball where m is the number of training samples. )

 I think I understand how the authors got confused: there is indeed a covering number over the rows and columns of the matrix in [AR1], which is acceptable in this case because the individual samples are entries in the matrix rather than entire users, which means that the number of observations can be larger than the number of users. 


On weakness 3: As explained above, the present paper doesn’t prove meaningful bounds for either implicit or explicit feedback. It appears that the authors are trying to position themselves as the first to have proved meaningful bounds for implicit feedback, or for LAEs, neither of which is true. Thus, it is not clear what branch of the literature should be included. However, if we accept results on explicit feedback, then the whole matrix completion literature should be mentioned. It is worth noting that the only work cited, [AR1], also concerns matrix completion (in a binary classification context), so even if the authors deliberately didn’t’ include the exact recovery literature [MC1,MC2,MC3,IMC] due to the exact observation requirement (or the literature on side information [IMCAR1,2,3,IMC] due to the slightly different setting), it is unclear why the followup works [AR2,AR3,AR4,AR5,AR6,AR7,MAX1,MAX2] were not included, despite treating similar learning settings as [AR1]. Likewise, the recent branch of the literature on the low-noise setting  ([PR1] with explicit rank restriction and [PR2] with nuclear norm regularizers) provides spectacular results in terms of the simultaneous dependence on the noise and the ground truth rank, albeit in the uniform sampling setting only.  

On weakness 4: the bound in [AR1], like those of the follow up works, generally scales like $[m+n]r$ in sample complexity where $r$ is the rank. This means the required number of samples for each user is roughly proportional to the rank, up to some constants and log terms. The constants and log terms in [AR1] are not large at all, I cannot believe the bound is vacuous for rank 2 (which achieves reasonably competitive RMSE already).


On weakness 5 (minor) : some theorems are hard to read due to somewhat vague descriptions of the assumptions. For instance, in Theorem 7, the statement “Suppose there exists  $M > 0$ such that $ \|R_i - R_i W\|_F^2 \in [0, M]$  for any $R_i$”  is vague because it appears to make a statement about the training set when what is required is for the inequality to hold with probability one over the test distribution. 
Similarly, the definition of $\beta$ in Theorem 4 should be a maximum over the support of the distribution rather than the distribution itself. Similarly, theorem 3 is really a prelude to Theorem 4 more than anything (perhaps it could be a proposition). Further, in Theorem 4, the notation $\eta_1(R)$ to mean the top eigenvalue of the matrix $R$ is used, but it is only introduced in the proof in the supplementary (line 847). It would certainly not hurt to make a table of notations and simple consequences (for instance, explicitly mentioning somewhere that $Q’{Q’}^\top=\Sigma_W$ would make the proof of Theorem 1 more readable. Adding a citation for the inequality on line 682 would also be good form. 



***References***


On Pac Bayes

[PB1] V Shalaeva, AF Esfahani, P Germain, M Petreczky, “Improved PAC-Bayesian bounds for linear regression”. AAAI 2020

[PB2] Pierre Alquier. “User-friendly introduction to PAC-Bayes bounds”, Foundations and Trends in Machine Learning, 2021.
 
[PB3] P Germain, F Bach, A Lacoste, S Lacoste-Julien, “PAC-Bayesian Theory Meets Bayesian Inference”, NeurIPS 2016.

On Linear auto encoders

[EASE] Harold Steck, “Embarrassingly Shallow Autoencoders for Sparse Data”, WWW 2019.

[ELSA] V Vančura, R Alves, P Kasalický, P Kordík,  “Scalable Linear Shallow Autoencoder for Collaborative Filtering”, RecSys 2022


On exact matrix completion (uniform or near uniform sampling)

[MC1] Emmanuel Candes and Terence Tao, “The Power of Convex Relaxation: Near-Optimal Matrix Completion  “, TIT 2009. 

[MC2] Benjamin Recht , “A Simpler Approach to Matrix Completion, JMLR 2011”

[MC3] Yudong Chen, Srinadh Bhojanapalli, Sujay Sanghavi, Rachel Ward, “Coherent Matrix Completion”, ICML 2014

On Matrix Completion with noise (including several rank-proxys) under non uniform distributions

[AR1] Nathan Srebro, Noga Alon, and Tommi Jaakkola. “Generalization error bounds for collaborative prediction with low-rank matrices” NeurIPS 2004

[AR2] Nathan Srebro, Russ R. Salakhutdinov. “Collaborative Filtering in a Non-Uniform World: Learning with the Weighted Trace Norm” NeurIPS 2010

[AR3] Nathan Srebro and Adi Shraibman, “Rank, Trace-Norm and Max-Norm”, COLT 2005

[AR4] Rina Foygel, Ruslan Salakhutdinov, Ohad Shamir, Nathan Srebro, “Learning with the Weighted Trace-norm under Arbitrary Sampling Distributions. NeurIPS 2011

[AR5] Ohad Shamir, Shai Shalev-Shwartz,“Collaborative Filtering with the Trace Norm: Learning, Bounding, and Transducing”, COLT 2011

[AR6]Ohad Shamir, Shai Shalev-Shwartz, “Matrix Completion with the Trace Norm: Learning, Bounding, and Transducing”, JMLR 2014 

[AR7] Antoine Ledent and Rodrigo Alves, “Generalization Analysis of Deep Non-linear Matrix Completion”, ICML 2024

[MAX1] Rina Foygel, Nathan Srebro, Ruslan Salakhutdinov, “Matrix reconstruction with the local max norm”, NeurIPS 2012

[MAX2] T. Tony Cai, Wen-Xin Zhou,  “Matrix Completion via Max-Norm Constrained Optimization”, Electronic Journal of Statistics 2016. 

On matrix completion with side information

[IMC] Miao Xu, Rong Jin, Zhi-Hua Zhou, “Speedup Matrix Completion with Side Information: Application to Multi-Label Learning “, NeurIPS 2013


On matrix completion with side information and noise

[IMCAR1] Kai-Yang Chiang, Cho-Jui Hsieh, Inderjit S. Dhillon, “Matrix Completion with Noisy Side Information”, NeurIPS 2015

[IMCAR2] Kai-Yang Chiang, Cho-Jui Hsieh, Inderjit S. Dhillon, “Using Side Information to Reliably Learn Low-Rank Matrices from Missing and Corrupted Observations”, JMLR 2018

[IMCAR3] Antoine Ledent, Rodrigo Alves, Yunwen Lei and Marius Kloft, “Fine-grained generalization analysis of inductive matrix completion”, NeurIPS 2021


On nearly exact matrix completion with low noise

[PR1] Yuxin Chen, Yuejie Chi, Jianqing Fan, Cong Ma, “Spectral Methods for Data Science: A Statistical Perspective” 

[PR2] Yuxin Chen, Yuejie Chi, Jianqing Fan, Cong Ma, Yuling Yan, “Noisy Matrix Completion: Understanding Statistical Guarantees for Convex Relaxation via Nonconvex Optimization”, SIAM J. Opt 2020









***Typos/grammar*** (Minor, non exhaustive)



332: extra capital letter at “We”

Line 669: the sentence is not finished. 

Line 247: “ The values…is bounded”…

662: “since… is of multivariate Gaussian…” 

Line 626: “the probability that $I-W$ being…” (that> of)

### Questions
1. Could you rewrite the paper as suggested below? 
2. (minor) could you explain the argument on the second to last line of equation (24)? Since $W^*$ depends on $R^{emp}$, I don’t think it follows from the equation on lines 1155-1156 at face value (though I don’t significantly doubt the big picture of this particular part of the proof). 
3. Why do you only ever look at the case $\lambda=m^{1/n}$ instead of $\lambda=\sqrt{m}$? Does the bound break down in that case? It would be better to express the bounds in terms of sample complexity and study how it depends on $n$ as well. 


Actionable items to fix the paper at this round or the next (I may increase my score if all of the points below are performed in a revised version which is uploaded) : 

1. Unless strong arguments are presented in the rebuttal, it seems clear to me that this paper doesn’t explain the generalization abilities of LAE or any recommender systems method. There is nothing special about the techniques which applies to the reconstruction loss specifically. It would be better to repeat the analysis in the more general context of **multi-output linear regression**, which is what this paper is really about, and to completely change the narrative of the paper to steer clear of any claim of having “the first bounds for EASE” or anything similar. A casual mention of the potential for applications to LAE and why the presented results do not apply as of yet can be left till the end of the paper. If the authors manage to improve the results to cover a different objective as explained in weakness 1, then the relationship with LAE can be reintroduced. 
2. Study or at least mention the dependence on $n$
3. Remove theorem 7 altogether 
4. Incorporate the approximation from the experiments into a rigorous bound 
5. Rerun the experiments on some multi output linear regression model, and present any remaining experiments on RecSys datasets as merely synthetic datasets where the task is different from the recommendation task.

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
1

### Summary
This paper investigates PAC-Bayes bounds for linear autoencoders and presents two distinct bounds. The first bound is developed under the assumption of a Gaussian data distribution, while the second bound is based on bounded data distributions and Gaussian parameter assumptions. To make these theoretical results more accessible for practical applications, the authors introduce a simplified upper bound for the second case. They then adapt this upper bound to the EASE (Efficient and Accurate Sampling-based Estimation) recommender system, demonstrating its effectiveness through experimental validation on standard recommender datasets.

### Strengths
- The paper provides a rigorous and detailed analysis of PAC-Bayes bounds, complete with comprehensive proofs (though the correctness of these proofs has not been independently verified). The theoretical framework is well-developed, contributing meaningfully to the field of statistical learning theory.
- By extending the theoretical bounds to the EASE model, the authors bridge the gap between theory and practice, making the results applicable to real-world problems. The evaluation on recommender system datasets demonstrates the practical relevance and potential impact of their approach.
- The paper is well-organized, with clear explanations and a strong motivation for the research. The structure allows readers to follow the logic and understand the significance of the contributions.

### Weaknesses
-  Given that there may be space available, consider moving the related work section from the appendix (if present) to the main text. Furthermore, a more in-depth discussion on how the proposed bounds compare to existing work would make the paper more comprehensive and inclusive.
- It is unclear whether the proposed PAC-Bayes bounds can be generalized to all linear autoencoder models. The paper should clarify whether these bounds are specific to the conditions outlined or if they have broader applicability across different linear autoencoder architectures.
- The theoretical results are intriguing, but the paper could benefit from a more detailed discussion of what these PAC-Bayes bounds imply for practical recommender system applications. Specifically, how do these bounds inform model selection, regularization strategies, or error expectations in practice?

### Questions
- One of the main questions is whether the practical bounds derived in this work can be applied to all linear autoencoder models. For example, is it possible to calculate a general PAC-Bayes bound for any linear autoencoder using Theorem 6? If there are limitations or constraints, it would be valuable to elaborate on them to clarify the scope of the theoretical results.
- The methodology used to calculate the PAC-Bayes bound in Table 2 requires further elaboration. Did you use a fixed \epsilon in your calculations? If so, what was the reasoning behind this choice, and how might different \epsilon values affect the results? Additionally, how sensitive are the practical training and test errors to hyperparameter choices? If there are significant fluctuations, a discussion on how to interpret these variations (e.g., how fluctuations influence conclusion about 2x or 4x test error) would be very helpful.
- Given that the practical performance of recommender systems is often influenced by hyperparameter settings, how should readers interpret the PAC-Bayes bounds in light of these variations? If hyperparameter tuning introduces substantial variability, how do these fluctuations affect the practical utility of the derived bounds?

### Soundness
3

### Presentation
3

### Contribution
3
