## Human Reviewer 1

### Summary
This paper introduces a novel differentiable causal discovery method for latent hierarchical causal models (LHCMs) and derives identifiability conditions of LHCMs in non-linear cases with relaxed assumptions (i.e., no requirement of invertible functions). In the experimental evaluation, the authors show promising results outperforming existing methods on synthetic and image data.

### Strengths
- paper is written clearly while keeping a formal discussion of assumptions and theorems
- the authors derive and prove their identifiability conditions. The proofs look correct after careful checking.
- a novel differentiable DAG learner for LHCMs is introduced, allowing differentiable structure learners to be applied in latent variable settings
- the experimental section introduces an interesting experiment on image data that demonstrates that the proposed algorithm can be seamlessly integrated into the autoencoder framework, thus allowing learning of LHCMs on complex and unstructured data such as images

### Weaknesses
**Section 2**
- the authors discuss "differentiable causal discovery" in the related work. However, most (if not all) works referenced here do not perform causal discovery. This has been shown by several works, e.g., [1], [2]

**Section 4**
While the theorems and proofs in Sec. 4 are correct, it is unclear to me whether the identifiable model still allows for a causal interpretation if variable permutations are allowed (Theorem 3). It would be good to clarify which permutations are allowed and why the permutations do not change the causal structure (and thus $d$-separation statements). To illustrate what I mean, consider a LHCM (where observed $X$ are dropped for the sake of simplicity) $Z_2 \leftarrow Z_1 \rightarrow Z_3$. If (any) permutation is allowed, Theorem 3 would also allow for $Z_1 \leftarrow Z_3 \rightarrow Z_2$. However, this model entails different $d$-separation statements and thus has different causal semantics. Hence the causal model would not be identifiable.

**Section 5**
It is unclear to me how the acyclicity and overall model structure from Condition (1) (ii) is ensured/reflected in the objective (if at all reflected) (Eq. 10). Based on this, it is not easy to see why the proposed method is not just a structure learner, but a causal discovery method. Could the authors please provide more details on how this is achieved?

**Section 6**
- Tab. 1: Why do the baselines perform so badly? Is there any specific explanation for that?
- Synthetic experiment: How were the ground truth structures chosen? By hand or randomly? If by hand, could the authors explain why and why these?
- image experiments: There is the work on causalVAEs [3], why did you not choose this as a baseline? Since it is more related to the overall problem setup of this work than standard VAEs, this baseline would make much sense.

# References
[1] Reisach et al. Beware of the simulated dag! causal discovery benchmarks may be easy to game. NeurIPS 2021.

[2] Seng et al. Learning Large DAGs is Harder Than You Think. ICLR 2024.

[3] Yang et al. CausalVAE: Structured Causal Disentanglement in Variational Autoencoder. 2020.

### Questions
see weaknesses

# Additonal Notes
Note that I decided on a score of 6 as there is no option 7. If the authors address the points in the weaknesses section accordingly, I'm inclined to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
Differentiable causal discovery has been a key focus of the causality community in the past years. Despite the advance of representation learning and deep learning, differentiable hierarchical causal discovery with latent variables has been a challenging subfield with at least empirical limited results and limited impact despite the need and call for these methods from practical applications. 

The paper proposes a new method and investigates some of the conditions for identifiability. 

While the paper has some very interesting and promising components, I overall can not recommend it for acceptance in its current form.

### Strengths
I really like that the evaluation is not just done with respect to a causal metric but wrt to "a regression classifier trained on the learned representation". If the causality field would move towards the standard evaluation practices of deep learning progress would be faster and this paper is one of the few which actually does perform this evaluation! 
However, when reading the paper in more detail e.g. Table 1 is then again evaluated wrt to discovery metrics only table 2 is evaluated with a learned classifier and arguably table 2 provides only a very limited setting and very limited evaluation. Especially given that these are deep learning approaches, the performance should not even reported in a table but as plots where the x-axis is training time and the y-axis performance. This would account for complexity and cost of training and really allow for a fair comparison of the approaches. 

While it is argued that causal representations lead to better generalizations and transfers this is so far actually not shown in the literature. DomainBed and or [1] clearly state the need for better evaluation and clearer demonstrations of the benefits beyond deriving identifiability results. I am thus really encouraging the authors to significantly extend the ablations and plot train vs performance curves and the performance of the classifier at different stages of training in a larger scale setting and across significantly more datasets. 

[1] Saengkyongam, Sorawit, et al. "Identifying representations for intervention extrapolation." arXiv preprint arXiv:2310.04295 (2023).

### Weaknesses
The key claimed advantage for better identifiability results comes from the fact that instead it is assumed that "not yet account for structures where measured variables have children"

There is some exchangeability of these assumptions and in that sense I agree that the current assumption is a more practical one but it is not a novel one or a clear contribution until a clear relation between the assumptions is shown. 

The evaluation is really lacking wrt to datasets and shown clear benefits across different settings. As mentioned I think the authors already take a very valuable step for the community by not only evaluating wrt to discovery metrics (see strengths) but adopting the established evaluation frameworks in deep learning of training a classifier on top of a learned representation. However that evaluation is unfortunately severely limited.

### Questions
It seems that the baselines are chosen from one lab only i.e. Xie et al, Kong et al and Huang et al which are used to sell the method are all from one lab. 

Given the number of baselines available for the task that seems a bit strange. Can you please clarify?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper shows that are particular class of causal graphs with  hierarchical latent variables are identifiable by leveraging properties of the Jacobian of the conditional exception function between subsets of observed variables. They then present an efficient algorithm for inferring the hierarchical graph. They present strong empirical results on both synthetic & image based problems.

### Strengths
- I thought this was interesting, original work. The class of graphs that they study is obviously limited but seems practical & the rank condition is intuitive.
- The paper is very well written - both the theory and methods section do a good job of explaining the intuition for why the method works
- The empirical results are strong on the datasets that they tested.

### Weaknesses
* The coloured MNIST results appear very strong (though this is not my area), but not contextualized in the domain generalization literature. I would have at least expected you to report the published numbers from recent work from that setting. Autoencoders & Beta-VAE is not the right baselines?
* I would have liked a more detailed discussion of the learned MNIST graph. I am not sure what to make of figure 4 or table 3 in the appendix? Do those latents make sense? Is there a natural hierarchical structure that we would expect?

### Questions
See weakness above.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
The main theoretical contribution of the paper is showing identifiability of nonlinear latent hierarchical causal models. Building on this theory, the authors propose a practical differentiable latent causal discovery approach. Experiments are performed on synthetic data as well as the coloured MNIST dataset to demonstrate efficacy of the approach.

### Strengths
1. The paper is, to the best of my knowledge, the first to provide identifiability results for nonlinear latent hierarchical causal models. The proof technique seems correct to me, though I did not check it thoroughly (for example, the appendix).

2. Estimating equation 9 using Donsker-Varadhan representation is novel.

### Weaknesses
1. **Experimental limitations**:

    a. **Synthetic experiments**: Instead of experimenting on just 4 structures given in figure 3, I would encourage authors to randomly generate DAGs and run experiments on these structures. For the synthetic experiments, the analysis would be stronger if the authors also try nonlinear activations for eq 1, instead of piecewise linear activation such as LeakyRELU.

   b. **Real experiments**: The baselines for the experiments on CMNIST are VAE and $\beta$-VAE -- both of which do not learn a structure over latent variables -- when  better baselines exist [1-3]. Applications to real world data is also limited, and even in the colored MNIST setting, only 2 digits seem to be used. 

2. **Missing/weak motivation**: It is also unclear why such models are useful in the real world: motivation for why one needs such models would make the paper more strong. In the introduction, causal discovery is motivated but the there is no true causal structure for the CMNIST data. Given this, what is the purpose for obtaining a hierarchical structure as in Fig 2b? For what tasks, is such a hierarchical representation useful?

3. L447 - 453 mentions interventions but key details are missing regarding interventional data generation (single node or multi node interventions, soft vs hard intervention, and intervention values).

4. **Related work**: The task of causal discovery over latent variable hierarchical models is closely related to causal representation learning but this has not been discussed and  works in the space have not been cited [1, 2]. 

---

[1] Brehmer, Johann, et al. "Weakly supervised causal representation learning." Advances in Neural Information Processing Systems 35 (2022): 38319-38331.

[2] Subramanian, J. et al. Learning latent structural causal models. arXiv preprint arXiv:2210.13583 (2022)

[3] He, J. et al. Variational autoencoders with jointly optimized latent dependency structure. In International Conference on Learning Representations, 2019.

### Questions
1. What is the implication of condition 3? 

2. There is a typo in equation 8, the number of small norms and large norms do not match.
 
3. From eq 6, we see that $|| M_{i, :} \odot \pi (1 - M_{j, :})||_1 \geq 2$. 

However in the subject to constraint in eq 8, $||M_{i,:}||_1$ times the above entity is enforced to be $\geq 2$. This is a bit unclear -- can the authors clarify?

4. Caption for figure 2c is unclear.

PS: Score has been increased post rebuttal.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3