# Reduced-Rank Online Gaussian Process Modeling With Uncertain Inputs

- Decision: Reject
- Scores: 3, 6, 5, 5

## Abstract
Gaussian Process (GP) is an increasingly popular modeling approach. In its classical formulation, the inputs are supposed to be perfectly known. However, in some use cases, this assumption is not true: the inputs as well as the outputs can be corrupted by noise. Some methods already insert these uncertainties in GP modeling but the only currently existing online algorithm (i.e.  that incrementally updates the model each time a measure is acquired) still lacks in robustness and precision. In this article we propose a novel online Gaussian Process (GP) modeling approach for vector field mapping with uncertain inputs. They are included into the GP through a complete second-order Taylor approximation with a better estimation of variances. Our experiments prove that our algorithm is more accurate and robust than the previous online method for a shorter computing time. Moreover, for high input uncertainties, our method achieves better performance than both online and offline state of the art methods on simulated data. This algorithm can also be applied to diverse real scenarios which require precise estimation of unknown functions from a small set of corrupted datapoints, as we show in the challenging problem of indoor localization, mapping magnetic fields.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an approach to online (multi-output) Gaussian process inference for uncertain inputs using a reduced-rank approximation of the covariance matrix for efficiency. The approach is based on approximating the latent function evaluated on a corrupted input via a second-order Taylor approximation. The proposed method is evaluated experimentally on a synthetic problem and on an indoor localization task based on measurements of the ambient magnetic field.

### Strengths
The paper proposes a simple idea to handle uncertain inputs by Taylor expanding the GP evaluated on noisy inputs and combines this formulation with a low rank formulation based on a truncated series of the covariance function. This formulation promises computational speedups and initial results on a synthetic dataset that the authors show seem promising.

### Weaknesses
While the paper combines existing ideas to address the challenges of online GP regression with uncertain inputs, there are significant weaknesses in how the results are presented and evaluated:

## Originality
The paper relies on existing ideas for low-rank Gaussian processes (Solin & Särkkä, 2020) for scalability and builds on approaches on how to incorporate noisy data via a Taylor approximation. At times it is difficult to separate what this paper adds from what is known. In particular, how the Taylor approximation differs from the one considered by Bijl et al, 2017. The fact that GPs can be updated in an online fashion through (tractable) sequential conditioning is also well known (e.g. via updating a Cholesky factorization). 

## Method and Theoretical Results
The method is explained in a lengthy fashion but at crucial points leaves out details. For example, as above, what is new about the use of the Taylor approximation? Also how do the update equations (20) - (23) arise? Are these just based on the standard sequential view of GP inference? If yes, why is that a novel contribution? If no, how are they motivated / derived?
The online formulation of the noisy-input GP is the main contribution, so this seems like a crucial point.

## Experimental Evaluation
The experimental evaluation is unfortunately very limited. In only compares to other online approaches on a synthetic dataset using an RBF kernel. There is no consideration of how to choose the rank of the approximation or how to optimize hyperparameters.  It was also unclear to me whether the baselines were also implemented in C++ as the author's approach or in Matlab, which potentially can lead to significant runtime differences.

### Simulated Data
For the simulated data a more thorough comparison on different kernels in my opinion is necessary, in particular because low-rank approximation will work especially well on the chosen exponential kernel due to its fast decaying spectrum. While results in Table 1 seem to be promising, the differences in performance are within one standard deviation, making it hard to judge whether they are systematic or an artefact. Also there is no systematic variation of hyperparameters: It is unclear how the choices of "other hyperparameters" were made.

### Magnetic Map Creation
The only other experiment, the magnetic map creation problem, does not compare RRONIG to another noisy-input method as far as I can tell, which I do not understand. This does not seem an informative comparison to me. If the reduced-rank approach for noisy-inputs is the main contribution, it would seem a comparison to SONIG is required here, no matter its robustness. In fact, if SONIG turns out to be non-robust on real data that would seem to be a benefit of RRONIG to be showcased. Further, error bars in Table 3 are missing, making it impossible to judge how large the performance difference is across different runs.

### Suggested experiments
To improve the paper I would recommend some systematic ablation experiments on (synthetic) data. For example, choosing Matern kernels of increasing smoothness to judge the effectiveness of a low-rank approximation and the required rank for good performance. An experiment to evaluate the impact of the choice of rank or a strategy to choose the rank. Finally, an ablation that considers model mismatch, i.e. how RRONIG performs under slightly misspecified hyperparameters, which will be relevant in practice, since as far as i can tell no model selection strategy is presented.

## Quality of Presentation / Clarity
The quality of presentation and clarity of the paper could be much improved. There is a lot of atypical language usage which makes the paper hard to understand in parts and can lead to confusion with existing mathematical terminology.

Examples:
- "modelize" => model
- "measure" (measurement?) versus (probability) measure 
- the "measure is ignored" => measurement is ignored?
- "truth function" => true / latent function 
- "conditionned to" => conditioned 

Further, the paper could be simplified to make space for better visual illustrations of the results and the above ablation experiments. For example eqn 4 and 23 give the identical formula of the exponentiated quadratic kernel, which is well-known.

## Minor Comments Not Affecting Score
- Paper uses the ICLR 2023 template, not the 2024 template.
- Code comments still state this paper is being submitted to NeurIPS 2023.

### Questions
- How do the update equations (20) - (22) arise? There is no explanation or derivation for them as far as I can tell.
- How do you choose the number of eigenfunctions for the low-rank approximation? Is there an automatic way or is it fixed by hand?
- How do you estimate kernel hyperparameters with your method? In the synthetic experiments you seem to use NIGP to do so.
- What do you mean by "more complete version of the Taylor approximation" compared to Bijl et al. (2017)? What are the exact differences to the approach by Bijl et al (2017)?
- Why are you only comparing your (online) noisy input approach to approaches which are not designed for noisy inputs (such as SONIG) on the magnetic map creation dataset?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a novel online Gaussian Process modeling for vector field mapping with inputs that are corrupted by noise.
The approach is based on a second order Taylor approximation of the data modeling. The author provide the formulas of the posterior distribution of the GP in the "static" setting (with all the datapoints simultaneously) and online setting (one data point at a time).
Numerical experiments show that the model is robust, fast and accurate compared to the most recent online GP with uncertain inputs.

### Strengths
Overall I have to say that it was pleasant to read the paper.
Strengths of the paper:
    - The paper is well written. It is easy to read as it is clear from the beginning to the end. The motivations are also very clear and sound. 
    - While first order Taylor approximation has already been used in such a context, the authors propose to push this to second order combined with truncated approximation of covariance with smooth eigenfunctions. The derivation is well done when all the data are considered simultaneously.
    - the numerical experiments are relevant and show the benefit of the author's approach

### Weaknesses
The paper has a couple of weaknesses.
   - In my opinion, the literature review is not complete. There are many approaches that deal with online regression and sparse approximations (see [1,2]).
   - The paper seems to combine different known facts about uncertain input GPs. Second-order Taylor approximation and smooth basis functions are two tools that have been used already in this context. In other words, I have the feeling that the paper considers the approach of [3] with second-order approximation and usual basis function approximations. Am I right? I am not saying that the contribution is low because of this. However, I think this should be somehow emphasized. 
   -  One of the main weakness of the paper is the explanation around equations (18)-(22). This is not straightforward to go from equations in Section 4.1 to Section 4.2. This would have deserved more explanations. Could the authors provide some more in-depth details about equations  (18)-(22) and their links with equations from Section 4.1?
   - If I have not missed it, the paper makes no mention of hyperparameter optimization. This is a challenging a question in an online context. How would the authors incorporate this crucial step in their current approach? 
   - It would have been interesting to see some results on different simulated benchmarks datasets (heteroscedastic noise variance for instance)


[1] TD Bui, C Nguyen, RE Turner, Streaming Sparse Gaussian Process Approximations. Advances in Neural Information Processing Systems, 2017
[2] A Gijsberts and G Metta, Real-time model learning using incremental sparse spectrum gaussian process regression. Neural networks, 2013.
[3] A Mchutchon and C Rasmussen Gaussian Process Training with Input Noise. Advances in Neural Information Processing Systems 2011.

### Questions
See the weaknesses above.
- Is the code publicly available?

Depending on author responses, I would increase my score.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose RRONIG, a new online learning approach for GPs in the presence of uncertain inputs. The method is based on a second-order Taylor approximation which results in a reduced rank formulation for the model. The main claims are better accuracy, robustness and shorter computing times when compared to other methods.

### Strengths
The work tackles an important problem with practical relevance. Although it has been addressed before for GPs, the better performance with much faster computing time is impressive.

Despite being a bit difficult to follow due to the heavy math notation, the derivations in the main text and in the provided appendix are comprehensive.

### Weaknesses
The main weakness of the work is in the experimental section. For the artificial dataset, the proposed RRONIG is compared only against one online method (SONIG, from 2017) and one batch method (NIGP, from 2011). For the real data, only a single model is used for comparison (besides visual-inertial odometry). I believe the experimental results would be more complete with the inclusion of more recent GP methods for uncertain inputs or other online GP methods.

In Section 5.1, the authors did not optimize the kernel hyperparameters in the the RRONIG runs. Is that also true for the experiments in Section 5.2? If so, it seems to be a significant drawback in the overall evaluation, since it is an important aspect of the GP training.

### Questions
Below I list a few more comments and questions:

- One should always specify when using the term "robust". Is the proposal robust to model misspecification? To outliers? To numerical instabilities? From the experimental section, it seemed that the robustness is related to better numerical stability, but this should be more clear from the beginning of the paper.

- In Section 2, the title "Modifying GP equations" is too generic. Almost all contributions in the GP literature require changing standard GP equations. Maybe "Input noise modeling" is enough?

- Table 1 should include the log predictive density to better evaluate the models' predicted uncertainties.

- Instead of reporting the average of the predicted variances and the "mean ratio", it would be more direct to simply report the mean log predictive density, which already balances the quality of the predicted means and variances.

- Page 7: "The variance represents the amplitude of the errors of the model that should be expected: it should be equal to the mean error." - Since the variance is in squared units, is the reported error also squared?

- The "reduced-rank GP" competitor considered in the SLAM experiment is the one from Solin et al. (2015)? In the Conclusion, it is claimed "an improvement over the state of the art on this context". Was the proposal by Solin et al. (2015) the state of the art so far for SLAM problems? I believe more recent strategies (even without GPs) should be considered.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents an approach to apply Gaussian processes in an online setting when input locations are uncertain. It brings together under one framework online Gaussian processes and uncertain inputs.

### Strengths
I think the problem tackled is interesting but I find the contribution of the paper difficult to identify. The writing needs improvement to convey the contributions of the work.

### Weaknesses
The simulated data does not seem to demonstrate online operation. 

The algorithm should be presented in a way to clearly indicate the steps performed. 

One main claim is that the new method is more stable than a previous contribution, SONIG. This should be demonstrated via a sensitivity analysis or more extensive experiments. For example would multiple restarts still result in instability?

The writing needs improvement. A few examples of issues include:
1. Some terms are inappropriate such as 'modelization' in the introduction and 'modelize' in section 3.1
2. The term 'measure' is used instead of measurement in section 3.1. This is confusing as measure has a precise technical meaning.
3. In section 3.2 it is not clear what LE stands for in $B_{LE}$.
4. The use of the word law is imprecise. What does 'mutual law of B and B_LE' mean?
5. In section 5.2 the line 'As in Equation 4.1, ...' referes to a section and not an equation

### Questions
What levels of input uncertainty are reasonble in practice? $\sigma_x =0.3$ appears high for an input domain $[-5,5]$

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
