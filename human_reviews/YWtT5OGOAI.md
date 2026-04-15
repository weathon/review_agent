# Practical Epistemic Uncertainty Quantification for View Synthesis

- Decision: Reject
- Scores: 6, 6, 6, 6, 6

## Abstract
View synthesis using Neural Radiance Fields (NeRF) and Gaussian Splatting (GS) has demonstrated impressive fidelity in rendering real-world scenarios. However, practical methods for accurate and efficient epistemic Uncertainty Quantification (UQ) in view synthesis are lacking. Existing approaches for NeRF either introduce significant computational overhead (e.g., "10x increase in training time" or "10x repeated training") or are limited to specific uncertainty conditions or models. Notably, GS models lack any systematic approach for comprehensive epistemic UQ. This capability is crucial for improving the robustness and scalability of neural view synthesis, enabling active model updates, error estimation, and scalable ensemble modeling based on uncertainty. In this paper, we revisit NeRF and GS-based methods from a function approximation perspective, identifying key differences and connections in 3D representation learning. Building on these insights, we introduce PH-Dropout, the first real-time and accurate method for epistemic uncertainty estimation that operates directly on pre-trained NeRF and GS models. Extensive evaluations validate our theoretical findings and demonstrate the effectiveness of PH-Dropout.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose using post-hoc dropout as a tool for epistemic uncertainty quantification in novel view synthesis frameworks like NeRF and Gaussian Splatting. Starting from a model with trained parameters, they uniformly increase the dropout rate of every parameter as long as the training loss is unharmed. Sampling from the distribution defined by this dropout rate can be used to compute spatially-varying variances in each novel view, which is used as a proxy for uncertainty, which is validated experimentally.

### Strengths
I congratulate the authors for their submission. The algorithm is simple, elegant and efficient. Previous attempts at using Dropout in novel view synthesis have failed: the authors have correctly identified the promise in the recent theoretically developed dropout injection methods and ported it to the Computer Vision community and the novel view synthesis problem in particular. Regardless of how exactly it compares against prior work, it surely advances the state of the art in uncertainty quantification for Vision and will inspire future work by the community (it has inspired *me* just by reading it!).

### Weaknesses
The main weakness of the current manuscript is the quality of the technical exposition. Given that what is proposed is a fairly simple extension of the Ledda et al. 2023 work, this poor quality of technical exposition is not enough to worry me about the soundness of the methods, but nonetheless the authors should strive to improve it in a revision, and this (together with the evaluation concerns below) is the reason why my score is not higher (I will gladly increase it if these questions are addressed). For example:

- The pseudocode in Page 3 has several mistakes. Line 4 should be before Line 1, and Lines 1-5 should be indented. The line starting with “Ensure” is redundant.
- The wording around \sigma_max  (L164-168) is confusing. See questions below
- The theorems and proofs in section 3.2 and 4 are incredibly vague, to the extent that I would be more comfortable if they were described as “intuitions” rather than “proofs” or “sketch of proofs”. A proof sketch is an abbreviated version of a proof for which one can be confident a motivated student reader can fill out the details. This is very much not that, as evidenced by my questions below. I would propose the authors rewrite these sections, abandon the pretence of mathematical correctness, explain their intuition and refer the reader to experiments for validation.
- After reading Lemma 4.1 and its proof 10 times, I still do not understand what it means. See questions below.

I would have also appreciated some visual examples of the uncertainty predicted in novel views by this method and its competitors that go beyond the numerical evaluations. By looking at numbers on a table, it is hard to understand what makes this method more precise than others. It claims to be “modeling all epistemic sources of uncertainty”, but this could be said of, e.g., the Goli et al. work as well, which is philosophically very similar (“how much can one modify network weights without harming the loss?”). In which image regions is this method doing a better work? Is this method better in floater removal? The text claims “Bayes Rays fails to correlate depth uncertainty with high prediction error on the LF dataset”, which seems to contradict the results in Goli et al.’s work. I would appreciate a deeper dive into this. This method is theoretically interesting and fast enough that I do not think its acceptance hinges on the evaluation being flattering, but I would expect a scientific paper to be more transparent in this regard and include a more exhaustive evaluation of where its results stand in relation to previous work, as well as its flaws and the flaws of previous works.

The manuscript would also benefit from some English proofreading (e.g., “continuosity” should be “continuity”), even if this has no bearing on my recommendation.

### Questions
I would love if the authors answered these questions in their revision:

- Why does the dropout mask in this method not need scaling the amplitude of the rest of the weights? I cannot think of a theoretical reason why one would do this, am I missing something? If not, is this due to an empirically observed advantage? I would be very curious to see this explored further, since it appears to go against the common wisdom of Dropout methods.
- About sigma_max (L164-168). Is the average being done over the training set only, or also the novel test view(s)? If the first one, isn’t the whole point of the algorithm that the training rendered RGB should not change after dropout injection, in which case sigma_max should be zero (or epsilon)? If the second one, how is something “quantifying the uncertainty of a model” if it depends on the specific view?
- What happens if the network is not perfectly overfit, as happens in almost every case? If the training is ended before the loss goes to zero, the network does not have sufficient resolution power, or it does but it is stuck in a local minima? The theoretical intuition seems to very deeply rely on this fact for the existence of redundancies: how much do the experiments suffer?
- The argument in Theorem 3.1 hinges on the conjecture that redundancy in the function space (e.g., one could modify the low power/high frequency Fourier components of this network without affecting the output much) translates into redundancy in the network parameters (i.e., one could turn network weights on and off without affecting the output much). I cannot immediately say that this conjecture is true (though it may very well be). Do the authors have any broader intuition or justification for this?
- On Lemma 4.1 and its proof: What is this D_KL in the space of weights? What does “p(a)” and “p(b)” mean? Are these conditioned on anything? I similarly do not understand the proof. Is all this lemma is doing saying that “if two weight sets are close, their Bayesian likelihoods are similar”? If so, this seems to me at least contestable. An argument about continuity may be made if the weights are changed only slightly, but if the observed redundancy is 20-30%, how can one state to be in a similar space in the Bayesian weight distribution?
- See “weakness” for questions on evaluation.

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
This paper proposes a Post Hoc Epistemic UQ scheme, referred as PH-DROPOUT, to make uncertainty estimation directly on pre-trained Multi view Scene Reconstruction models (NeRF and Gaussian Splatting in particular). The algorithm estimates the variance of a well-trained model by introducing binary dropout mask into model parameters and greedily select drop out ratio to perturb the output within tolerance, and validate over test views. Authors explore the usage of the measurement from their PH-DROPOUT algorithm by conducting thorough analysis over different scenarios (active learning, correlation analysis, uncertainty driven model ensembles).

### Strengths
Overall it is a good discovery and reflection on how to effectively and efficiently justify and analyze the performance of NeRF or GS model quantitatively. Authors justify their measurements is sound by extensive experiments and they indicate that most NeRF or GS models tend to have more redundant parameters as training views increase. This is a bold yet very inspiring claim. It implicitly conveys an intuitive thought: the information attained from more input views shall reduce the dimension of the model itself.  

Mostly the paper context is well written and well organized, and the paper itself answers most of my concerns while I was reading it.

### Weaknesses
The math claim in this paper can be improved. There are too many "colloquial" proof rather than a mathematical modeling of phenomena. These claims seems to be redundant with respect to the integrity of the paper. For instance, to represent the significant redundancy in model parameters, authors state the following in line 190:
 
$$ \exists 0\ll r<1\rightarrow \forall x\in \mathcal{D}_{train}, \lvert F(x;\theta)- F(x;D(\theta,r))\rvert<\epsilon. $$

Firstly, greater than $\ll$ is not a rigorous notation. Secondly, $\mathcal{D}_{train}$ is not defined. Is it a point set or a multi-view image set? Lastly, the proof of line 190, i.e. line 192-210, is all text and there is no anchor point to refer to appendix for a comprehensive mathematical proof, and I failed to find the corresponding complete proof of theorem 3.1 in appendix. One cannot call statement in line 192-210 a proof of theorem. My suggestion is to make it as a conjecture or rewrite this theorem. Same question for lemma 4.1, thm 4.2, thm 4.3 and thm 4.4.

### Questions
There are some clarification questions I wish to hear from authors:

*    In line 141, the model function $F(x;\theta)$ is introduced. Is $x$ a 3D point? This confuses me when I checked line 190.
*    In Algorithm 1 (line 143-159), does $M\cdot \theta$ refer the element-wise product? Where does subscript $i,j$ come from?
*    In line 173-175, it claims dropout only happens in one of the middle layes in NeRF-based models? Any ablation study over this choice of selective layer dropout?
*    To show performance in active learning, the paper evaluates the performance starting from sparse view training (as low as 2 views). Nevertheless, PH-Dropout has an assumption that the pre-trained model needs to be well trained or over-fitted. I understand that some NeRF-based alrogithm can achieve sparse-view reconstruction. Is there any quantitative assessment on judging whether or not PH-Dropout is applicable in these sparse-view training results?
*    What do AUSE RMSE, AUSE MSE and AUSE MAE, mean in Table 1 (line 443-450). Besides, layout of Table 1 and Figure 6 is bad as title and floatings are not properly aligned.

### Soundness
2

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
4

### Summary
The paper proposes a simple yet effective and extremely fast ad-hoc method for epistemic uncertainty estimation that operates directly on pre-trained NeRF and GS models in the task of novel view synthesis. At its core, the method proposes to use dropout at test time with the maximal drop rate that maintains an epsilon fit to the training views. The method is validated through proxy metrics like Spearman’s correlation of the uncertainty estimates with the RMSE from the GT, and the trend as a function of the number of training views. In addition, the authors demonstrate improved ensembling based on optimal ensemble member selection using their uncertainty estimates.

### Strengths
* The authors considered multiple quantitative proxy metrics to benchmark their method, showcasing extensive evaluations.  
* The method is very simple to implement and adopt, yet it is extremely effective and efficient in terms of runtime.  
* Overall the paper is structured well and is easy to follow.  
* Related work is acknowledged, and the contribution of this paper is put in proper context.

### Weaknesses
* Perhaps the major weakness of this paper is the relatively incremental contribution compared to Ledda et al 2023, which practically proposed the same technique up to the differences mentioned by the authors in L169-175. While I sincerely appreciate the honest citing of this related work by the authors, I find the contribution of PH-DROPOUT to be relatively small.    
* The English writing of the paper can be significantly improved in certain spots (see small list below), although for the most part it is not hard to understand the authors intention based on the context.  
* The proofs in the paper need to be more rigorous. For example, L192-207 are very hard to follow. Similarly, L235-243 seem to have mathematical inaccuracies such as calculating the KL divergence between two parameter instances instead of between two distributions. This overall proof sketch is not very clear and requires significant editing. The same applies to Theorem 4.2 and its proof sketch. More rigorous mathematical definitions and notations are needed. In the full proof from the appendix L777-782 seem to have a mistake in the variance formula. Where did $N\_F$ come from?

My rating is mainly due to the relatively limited contribution compared to Ledda et al. and the lack of mathematical rigor in the proofs. Nonetheless, I’m willing to increase my score if the authors' rebuttal can alleviate my concerns with respect to these two issues, as I still think this work does have the potential for practical value in quantifying the uncertainty of novel view synthesis. 

A few caught (minor) typos/english corrections:

* L103 \- hence hard \-\> hence **it is** hard  
* L104 \- find \-\> found  
* L128 \- in **ad** network \-\> in **a** network  
* L129 \- inject \-\> inject**ing**  
* L130 \- **non-**trivial \-\> **not** trivial  
* L149 \- step 3 in the algorithm is better split into 2 lines/formatted differently  
* L162 \- need \-\> need**ed**  
* L184 \- a common features \-\> common features/a common feature  
* L185 \- method \-\> method**s**  
* L198 \- signal \-\> **the** signal  
* L201 \- what does a “function with nearly discrete pattern” mean?  
* L208 of \-\> of **a**  
* L225 \- spa**c**ial \-\> spa**t**ial  
* L229 \- same structur**al** \-\> same **structure**  
* L229 \- number of parameter \-\> number of parameter**s**  
* L229 \- have similar distribution of parameter \-\> have **a** similar distribution of parameter**s**  
* L235 \- continuou**sity** \-\> continuity  
* ….

### Questions
* What is the number of dropout samples N in your experiments?  
* Did you check the calibration of your uncertainty estimates (e.g. using metrics like ECE)?   
* Did you check the trend of the RMSE from the GT compared to thresholding out pixels with increasing uncertainty levels? Is this expected to behave similarly to a NeRF model predicting both the mean and the standard deviation of the RGB value along each ray?  
* In L262-264 you mention that your uncertainty estimate is biased. Is this bias not important? How does this affect your uncertainty calibration?  
* When you write down $\\rho\_p$ in the results tables you are referring to $\\rho\_{PE}$?  
* What does the acronym “AUSE” refer to in Table 1?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces PH-DROPOUT, an efficient method for epistemic uncertainty quantification in view synthesis, applicable to pre-trained NeRF and GS models. It leverages model redundancy to estimate uncertainty without retraining, showing strong performance across datasets. Despite its efficiency, it faces challenges with hash encoding and sparse scenarios.

### Strengths
* The introduction of PH-DROPOUT as a post hoc epistemic uncertainty quantification (UQ) method is novel and addresses a critical gap in view synthesis research.
* The authors present a strong theoretical foundation, including proofs (e.g., Theorem 3.1) that justify the redundancy in NeRF and GS models, enabling effective dropout-based UQ.
* PH-DROPOUT achieves real-time performance, outperforming prior methods in computational efficiency.
* The method shows strong correlations between UQ metrics and prediction errors (e.g., RMSE), validating its reliability.

### Weaknesses
* The method struggles with hash encoding-based NeRF models, which restricts its applicability to sparse or complex scenes where hash collisions are common.
* The results on 2DGS show limitations in few-view scenarios. 
* The method is specifically designed for view synthesis tasks and may not generalize to other domains.

### Questions
* Can the authors provide a workaround or mitigation strategy for handling hash collisions in NeRF models? This would address a key limitation of the method.
* The inclusion of more real-world datasets, particularly in unbounded scenarios, could strengthen the empirical evaluation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a fast post-hoc uncertainty quantification method for both NeRF and Gaussian Splatting Models. The method is based on test-time drop out of neurons/Gaussians in the model and estimating per-pixel uncertainty based on the error caused in the test views. The proposed method is orders of magnitude faster than other baselines and has competitive performance to them.

### Strengths
- Explores how drop-out as a main UQ approach in ML, fits into the radiance field framework.
- It is applicable to both NeRF-based and 3DGS-based models.
- The proposed method is very fast, and is done post-hoc which makes it a useful method for downstream applications.

### Weaknesses
Main:
- The rendering function in test time (the pre-trained radiance field) is not perfect. I do not see how comparing the rederings from the drop-out version of the model to the renderings of the model can specify uncertainty. Moreover, how are the test-views selected? Is there an assumption on the distribution of the cameras? If you select your camera far enough from the training ddistribution, the dropout rate would drop to zero , as the error from that view would be high anyway?
- The  volume rendering function itself (through its integration) hides the ambiguity and uncertainty in depth, so using the rendered image error as a source for identifying uncertainty can be less robust than a spatial method, unless queried densly from many test views.
- The authors claim this method is not architecture-dependant, a discussion about how drop-out in MLP-based NeRFs vs voxel-based or K-Plane ones results in the same metric would be useful. My confusion comes from the fact that some of these representations have more geometric meaning (like voxel or K-Planes or 3DGS) and in these cases dropping out a cell would directly affect a point in space while dropping out a node/layer from MLP might affect different places simultaneously.
- Qualitative results on uncertainty primarily reflect color uncertainty, as seen in flat regions like the train body in figure 4, where low-opacity, cloudy Gaussians exhibit low uncertainty. However, the paper should clarify whether the reported uncertainty correlates more with color error, depth error, or both, and include qualitative evidence to support this claim. This helps readers understand what use cases this uncertainty is useful for.
Minor:
- What are the PSNRs reported in Table 1? Are they PSNR after noise removal the same way explored in the baseline, if so the coverage % should also be reported alongside this metric.
- A few results on NGP-style NeRFs which are the main NeRF models used, and most results on MLP NeRFs.
- The proof sketches in the main paper can be more detailed.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
3
