# Learning Chaos In A Linear Way

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 5

## Abstract
Learning long-term behaviors in chaotic dynamical systems, such as turbulent flows and climate modelling, is challenging due to their inherent instability and unpredictability. These systems exhibit positive Lyapunov exponents, which significantly hinder accurate long-term forecasting. As a result, understanding long-term statistical behavior is far more valuable than focusing on short-term accuracy. While autoregressive deep sequence models have been applied to capture long-term behavior, they often lead to exponentially increasing errors in learned dynamics. To address this, we shift the focus from simple prediction errors to preserving an invariant measure in dissipative chaotic systems. These systems have attractors, where trajectories settle, and the invariant measure is the probability distribution on attractors that remains unchanged under dynamics. Existing methods generate long trajectories of dissipative chaotic systems by aligning invariant measures, but it is not always possible to obtain invariant measures for arbitrary datasets. We propose the Poincaré Flow Neural Network (PFNN), a novel operator learning framework designed to capture behaviors of chaotic systems without any explicit knowledge of the invariant measure. PFNN employs an auto-encoder to map the chaotic system to a finite-dimensional feature space, effectively linearizing the chaotic evolution.  It then learns the linear evolution operators to match the physical dynamics by addressing two critical properties in dissipative chaotic systems: (1) contraction, the system’s convergence toward its attractors, and (2) measure invariance, trajectories on the attractors following a probability distribution invariant to the dynamics. 
Our experiments on a variety of chaotic systems, including Lorenz systems, Kuramoto-Sivashinsky equation and Navier–Stokes equation, demonstrate that PFNN has more accurate predictions and physical statistics compared to competitive baselines including the Fourier Neural Operator and the Markov Neural Operator.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, the authors address learning chaotic dynamical systems by introducing a model / learning paradigm that incorporates the physical properties of dissipative chaotic systems, namely that the volume of transient states contracts towards an attractive set on which measure is preserved by the dynamics. Across a variety of well-studied systems, the authors demonstrate the improvement from using their proposed PFNN method.

### Strengths
The problem and proposed methodology are well motivated. The exposition of why the various design decisions are made is rigorously stated and clear.

The experimental section is extensive and the improvements of PFNN over baselines is compelling. The authors provide a comprehensive appendix with all the relevant experimental details.

Figure 1 is very illustrative and a great way to convey the high level idea.

### Weaknesses
Overall, my main criticism is that while the methodology is well-motivated and rigorous, the actual implementation is somewhat difficult to follow. The phases and design decisions seem to be somewhat _ad hoc_, which is reflected in the number of hyperparameters (5 or 6 by my count), and I personally find the implementation difficult to follow. For example, I appreciate that the authors provided code, but looking through the code implementation, I struggle to see how the pseudocode in the algorithm maps to the implementation: in neither the Lorenz96 or NS script do I see the two-phase aspect of the training nor how equations 4, 6, or 8 get consistently mapped to code. I want to be clear, I am not taking the code implementation into account in my review/score, I only bring it up to illustrate that I think this method is less straightforward to implement.

Below I list more specific comments/concerns. The list roughly follows the order of appearance in the manuscript. In my opinion W1, W3, W5, W7/W9/W11, and W8  are the most important among these.

---

### W1: Is PFNN a parameterization or a training algorithm?
At a high level, I think the authors should clarify what PFNN refers to: i.e., is it the parameterization of the learned model (encoder + linear feature evolution) or does it refer to the 2 phase training paradigm with the various objectives proposed?

### W2: Lorenz63 is missing from the list of experimental setups in the abstract.
Not a big deal, but for completeness, I would recommend listing it.

### W3: Characterization of Schiff et al. (2024) is incorrect.
The authors place their work in contrast to Jiang et al. (2024) and Schiff et al. (2024), stating that unlike these works, the current proposal does not make any assumptions about the underlying PDE. While I agree that Jiang et al. (2024) requires knowledge of the system to build their set of invariant properties, this is not the case in Schiff et al. (2024). In fact, Schiff et al. (2024) clearly state that their proposed methodology does not make any assumptions about the underlying system, only that it supports an invariant measure, as in this work. In Schiff et al. (2024), the invariant measure is simply estimated as an empirical distribution from the trajectories in the training/test sets. Indeed, the same way the authors in this manuscript compute KL/MMD as evaluation metrics, Schiff et al. (2024) compute MMD as a regularizer.

I believe this characterization of previous work should be corrected. This does not detract from the fact that the current work, with its parameterization and two phase training, is novel. 

**Additionally, it would be interesting to see how PFNN compares to baselines, say MNO (the strongest baseline), trained in conjunction with the regularized objective from Schiff et al. (2024). I highly recommend that this comparison be added to the experimental section.**

### W4: Introduction of the method is a bit confusing/misleading.
In the first paragraph the authors state that the algorithm / method has 3 steps. But if I understand the methodology correctly, then step 1 (i.e., learning the autoencode) is happening throughout both phase 1 (contractive) and phase 2 (measure preserving) of the training. This should be re-worded / clarified.

### W5: Consistency regularization seems to be dropped after Eq 4 is introduced.
The authors introduced the second term in Eq 4 as a way to enforce bijectivity. However there are a few things I find confusing:
1) After Eq 4, (e.g., in Eq 6 and Eq 8) this term is dropped. Why?
2) The hyperparameter $\gamma$ in Eq 4 is not discussed elsewhere in the manuscript and is not included in ablation studies.
3) The training algorithm in Algorithm 1 indicates that losses are computed by Eq 4 and Eq 6 (Line 1363) / Eq 4 and Eq 8 (Line 1366), but what does this mean in practice? Is the regularization term from Eq 4 included? The first term of Eq 4 is shared in both Eq 6 and Eq 8, is this term double counted? (presumably not)

### W6: Missing citation to “Evolve Smoothly, Fit Consistently: Learning Smooth Latent Dynamics For Advection-Dominated Systems” Wan et al. (2023).
In Wan et al. (2023), the learning dynamics are learned in a latent space and there is consistency regularization to ensure that the autoencoder is invertible. This shares several properties with the proposed methodology in the current manuscript. The authors should include Wan et al. (2023) in their related works and discuss how PFNN differs.

### W7: Isn’t the regularizer in Eq 6 quite expensive to compute?
To enforce the contractive property of $\hat{\mathcal{G}}$, the authors compute eigenvalues. Isn’t this quite expensive: $\mathcal{O}(L^3)$?

### W8: The choice of hyperparameter $k$ seems hard / requires knowledge of the system
The authors claim their method does not require any knowledge of underlying true dynamics, but the cutoff $k$ between contractive and measure preserving phases seems like it would highly depend on the system of interest.

Additionally, I would recommend adding more information (perhaps in the appendix) about how $c_{LSI}$ is derived/computed.

### W9: What is the complexity of computing objective in Eq 8?
$\mathcal{G}^*_m$ is described as the conjugate transpose, but in the provided code implementation it seems like it is computed as the Moore-Penrose pseudoinverse of the parameters of $\mathcal{G}_m$, unless I am misunderstanding something here. Is this not quite expensive to do and backprop through on each step/batch?

### W10: Algorithm should be provided in the main text
While I recognize that space is limited, I highly encourage the authors to find a way to include the training algorithm in the main body of the text. There are several components to keep track of in the methodology, and having them centralized in the main text in the form of the training Algorithm is really important and would aid clarity, as opposed to having to refer to the appendix. 

### W11: Computation complexity of PFNN vs. baselines should be noted and commented on.
Related to my comment in W7/W9, it seems PFNN has several compute intensive overhead components. The complexity (both asymptotic and actual wall clock) of PFNN should be remarked on / included in the results sections.

### W12: Additionally, parameter counts should be included in Table 1
In addition to wall clock, the parameter counts of the various baselines and PFNN should be indicated to ensure methods are fairly compared.

### W13: Adding +/- for different random seeds on MMD/KLD would be very helpful
In my experience, ranking on metrics for the experimental setups explored here can vary significantly across different random seeds. If possible, adding this to Table 1, particularly for the MMD/KLD metrics, would make the results more compelling.

### W14: Why is MMD/KLD omitted for the NS experiment?
In particular, in the appendix, TKE is indicated to be a short-term accuracy metric, and so NS experiments seem to not have any reported “long-term” metrics.

### W15: In the ablation table, why is performance worse for $L = 512$?
I’d expect the trend (better results) to continue as $L$ grows.

### W16: Figure 5a (and other similar ones) is missing legends/labels
I am assuming the top row indicates trajectories and the bottom row shows error, but this needs to be indicated.

### W17: Minor typos / Formatting Issues & Suggestions:
- Defining $\nu$ as the data distribution should be more clear and included in the notation table in the appendix
- $m$ as the system dimension should be defined in paragraph 2 of Section 4. I would recommend also recalling what $m$ is in the caption of Table 1.
- Table 1 seems to have some bold formatting typos, e.g., the standard deviation is bolded for some rows (e.g. Line 401).
- Line 212: $\mathcal{G}_{ij}$ is missing the $L$ subscript.
- Line 221: In equation 4, the expectation should be over $\nu$ not $\mu$?
- Line 259/260: The footnote spacing is off. Should be attached to the previous word; looks like there is an added space.

---

**References:**

Ruoxi Jiang, Peter Y Lu, Elena Orlova, and Rebecca Willett. Training neural operators to preserve
invariant measures of chaotic attractors. Advances in Neural Information Processing Systems, 36,
2024.

Yair Schiff, Zhong Yi Wan, Jeffrey B Parker, Stephan Hoyer, Volodymyr Kuleshov, Fei Sha, and
Leonardo Zepeda-Nunez. Dyslim: Dynamics stable learning by invariant measure for chaotic systems. arXiv preprint arXiv:2402.04467, 2024.

Wan, Zhong Yi, Leonardo Zepeda-Nunez, Anudhyan Boral, and Fei Sha. "Evolve smoothly, fit consistently: Learning smooth latent dynamics for advection-dominated systems." arXiv preprint arXiv:2301.10391 (2023).

### Questions
### Q1: Is Equation 2 obvious?
As someone less familiar with operator theory, is the existence of $\mathcal{G}$ in Eq 2 obvious? The authors provide strong citations for the other theoretical aspects of their work, but I think citations are missing / would be helpful for the section where $\mathcal{G}$ is introduced.


Additionally, does the existence of $\mathcal{G}$ require any assumptions on the features $\phi$?

### Q2: In Lines 192-193, why do the authors state that the invariant measures “varies over time”?
Isn’t the invariant measure consistent over time?

### Q3: Why does $\gamma_2$ need to be between 0 and 1?
Is this a strict requirement or simply what the authors explored in their experiments?

### Q4: In the ablations, did the authors explore training only on the attractor?
In other words, what if we ignore $k$ in the training and assume all initial states are already on the attractor (e.g., by simply time evolving trajectories in the training/test sets and “throwing out the first $k$ steps). How does this affect performance of PFNN and baselines?

### In Figure 5b/c and other similar figures, where do the error bars / shaded regions come from?
How did the +/- get computed here?

### Soundness
4

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
This paper proposes the Poincare Flow Neural Network (PFNN), an architecture for learning in dissipative chaotic systems. PFNN maps the input to a latent space and proposes to learn linear evolution operators in this latent space. PFNN is applied on several systems, including Lorenz 96, Kuramoto Sivashinsky, and Navier-Stokes, and it demonstrates improvements over prior baselines.

### Strengths
The paper is well-written and includes a good exposition of key concepts (e.g., ergodicity, global attractor, etc.) for audiences that may not be as familiar with the mathematical background. The experiments are in-depth, particularly in the appendix. The proposed method also outperforms prior works across a variety of settings and metrics. The detailed proofs in the appendix are also helpful.

### Weaknesses
The authors should explain in more detail the distinction between their framework and prior works. In particular, some details on the distinction between neural operators and the Koopman operator approach are not clear. To my understanding, the proposed approach is based on Koopman operator theory and is learning, in part, both the discretized operator matrix $\mathcal{G}_{ij}$ and the encoder and decoder feature functions. However, the chaotic systems in question are assumed to be defined over a subset of $\mathbb{R}^d$. In contrast, the neural operator (e.g., FNO, DeepONet) setting is one of mapping between function spaces. As such, methods like MNO are most interested in learning the solution operator to a time-evolving function (PDE). To my understanding, the proposed approach cannot do this directly in function space as designed, as it is fixed to a specific data resolution. Is this accurate? To me Koopman and neural operators seem to be different paradigms, but in my opinion the authors do not make sufficient distinction between them. However, my interpretation may be mistaken. Please let me know if I am missing a connection in this case.

I also have some questions regarding the experiments and baselines. These can be found in the “Questions” section below.

### Questions
1. At inference-time for a system, how do you choose $k$? How robust is the proposed model to variations in $k$? 
2. Does this $k$ depend on the input trajectory? One could imagine a trajectory initialized very close or very far from the attractor. How is $k$ determined in that case?
3. How are the operator learning architectures (FNO and MNO) parameterized for the finite-dimensional problems (L63, L96)? Is this different from how they are parameterized for KS and NS?
4. Does constraining the spectrum of the operator $\mathcal{G}$ give contraction in the physical space as well? I suggest adding more details about this in the paper to make it clearer.
5. Is there potential for this method to be applied on function spaces, e.g., to forecast a Navier-Stokes fluid flow in a discretization-agnostic manner?
6. Since the model is dependent on the input discretization, how does the proposed architecture scale with input resolution? For instance, how does convergence and performance compare on Navier-Stokes at 256x256 resolution vs. on 64x64 resolution?
7. Does $\hat G$ in equation 4 correspond to the approximation finite-rank operator?

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
In this paper, the authors proposed a Poincare Flow Neural Network (PFNN) to model dissipative chaotic dynamical systems that improves matching of invariant statistics in forecasting over long horizons. The proposed method uses an autoencoder to reformulate prediction on the state space into prediction on a finite-dimensional learned feature space, on which a linear operator might be able to represent the forward evolution of the system in the feature space theoretically. By estimating the transient time the system needs to converge to the attractor with "relaxation time", the trajectory data sets are separated into a contraction phase and a measure-invariant phase. Using soft constraints formulated as regularization loss of the learned linear operator, the physical properties of the system, namely (1) dissipativity, in other words, volume contraction during the contraction phase (2) measure invariance due to a unitary operator construction, are incorporated into the training loss. Overall, the method leverages a Koopman-like structure to encourage the model to match intrinsic physical behaviors of a dissipative chaotic system by adding soft constraints on the linear operator. The improvement in preserving invariant statistics in the true system of the proposed method is validated in a few numerical experiments compared to recent benchmarks in the literature.

### Strengths
Originality: The proposed method provides a new perspective on incorporating dissipative chaos physical behaviors into a neural network model through simple to compute yet powerful linear operator properties.
Quality: The proposed method has a strong mathematical foundation, which is explained well in the paper. Additionally, the reviewer finds the numerical experiments to be extensive and the ablation study helpful.
Clarity: Overall the paper is written clearly, with minor editorial comments on presentation listed in other sections of the review.
Significance: The paper addresses an important problem in learning chaos, which has extensive practical applications including weather forecast, turbulent flow modeling etc.

### Weaknesses
Overall, the reviewer finds the method innovative and well articulated. However, there are a few technical details and writing concerns that need to be addressed to make sure the paper can be accepted in a form that clarifies its own contributions.

Technical issue:
1. It is very crucial in this approach to obtain a good separation of trajectory data into the contraction and measure-invariance phase, since these two phases use different operators. However, it seems like all the approximation techniques introduced in Section 3.2 can only reduce the estimation to the same order of magnitude, which might be problematic if the transient period is relatively short.
More importantly, the steps for separation chosen in the paper $\frac{1}{c_{LSI} log(||\phi||_2/\epsilon)$ depend explicitly on $\phi$, which is the learned feature function. It is confusing how the steps can be chosen using this formula before the network is trained, however, this is a necessary step to preprocess the trajectory dataset.

2. The auto-encoder does not seem to have a structure to help learn the feature functions that are orthonormal or help construct a good finite-rank operator approximation. The reviewer would appreciate clarification of the link between Theorem 3.1, or more generally the design of autoencoder structure, and the Koopman theory.

Writing issues:
1. To the reviewer's best knowledge, the proposed structure is very similar to the auto-encoder structure inspired by Koopman operator theory proposed by Steven Brunton (which includes a series of papers). Although the authors have cited one of these works, the reviewer suggests that the author to clarify that the auto-encoder structure and Koopman-inspired network design is perhaps not a novel design.
2. It seems like quite a few theorems in the paper e.g., Proposition 3.2 and Theorem 3.3 are standard results either in functional analysis or Koopman literature. The reviewer fails to see the points of rederiving such results, instead, one might be able to cite these well-known results and directly use them.
3. It seems like the theory in Theorem 3.1 and functional analysis in relation to constructing the finite-rank operator which approximates the designed linear operator suggests the feature functions need to be orthonomal or at least somewhat independent/orthogonal. Since the feature functions are learned using auto-encoders, it would be very helpful to provide some insights on what features are learned. It's a bit concerning to the reviewer that the authors make the point that FNO/MNO relies on periodic Fourier modes contradictory to the chaos nature without the presence of their own learned feature map results.
4. This paper clearly has its own contributions in leveraging linear operator theory and incorporating certain properties. In order to keep the literature review fair, the reviewer suggests the authors to consider revising introduction especially comparison to neural operators. To make matters clear, the reviewer is not at any point a co-author of any neural operator work but is familiar with this line of work. It seems like more general forms of neural opeartors (1) do not necessarily use Fourier modes (2) might use nonlinear evolution such as kernel structure. It might not be fair to group this entire line of work into the linear evolution box and claims strict improvement over FNO/MNO (Neural operators with Fourier modes), which is a fraction of them.
5. It seems like the framework assumes a finite-dimensional ODE structure of the dynamical system. Given a discretization grid, this might be true for a broad range of systems, the reviewer suggests the authors to make this limitation clear and restricts the scope to given spatial discretization. There are other papers in the domain that can deal with more general discretization setup such as DeepOnet or Neural operators. Clearly, this paper would not be able to address discretization issues like the other papers.

### Questions
1. When the finite-dimensional features spaces are constructed (L different learned features), wouldn't the "operators" G_c, G_m simply collapse to matrices? If so, please make it clear as matrix transpose notion has already been used in the paper. 
2. What is the main difference between the proposed architecture from the Koopman learning literature? e.g. Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton. "Deep learning for universal linear embeddings of nonlinear dynamics." Nature communications 9.1 (2018): 4950.
3. In Theorem 3.1, it seems that the result does not depend on the choice of the autoencoder. Is this true for every set of feature functions learned by the encoder?
4. Just out of curiosity, does such a linear operator always exist? If so, is there any intrinsic difficulty in approximating such infinite-dimensional operator using finite dimensional feature space for certain systems?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes Poincaré Flow Neural Network (PFNN), which maps chaotic dynamical systems to a latent space where their evolution is approximated by one of two linear operators: a dissipative operator for the initial contraction phase and a unitary operator for the measure-invariant phase on the attractor. PFNN enforces these properties of the learned operators using carefully designed loss functions. The switching time between the two phases / operators is a hyperparameter of the method that must be tuned. The proposed method is evaluated on several chaotic systems including Lorenz 63/96, Kuramoto-Sivashinsky, and Navier-Stokes equations.

### Strengths
1. The proposed method is rigorously grounded in dynamical systems theory, and the design of the loss function provides a nice example of how a theoretical or mathematical understanding of a problem can be incorporated into the learning process.
2. The paper is generally clear and very well presented, although the overall score here is let down by how the proposed method is contextualized with respect to the literature, as discussed below.

### Weaknesses
### Major
1. A major issue with the presentation of the paper is how it deals with Koopman operator learning. In particular, the core of the proposed method consists of a learned embedding followed by learned linear evolution in the latent space. This is Koopman operator learning [1,2,3,4], which is a vast and active area of research. Despite forming the foundation of the present work, Koopman learning only receives a few passing mentions in the text.

   For example, with regard to the proposed learning setup, the authors state "This concept is fundamentally aligned with neural operators, such as the Koopman operator (Bevanda et al., 2021)..." However, rather than simply being "aligned" with Koopman operator learning, the method proposed here _is_ Koopman operator learning. For someone not already familiar with Koopman learning, it would be easy to read this paper and think that the core learning strategy -- linear evolution in a learned latent space -- is a novel contribution. 

   The actual contribution of this paper is in fact to split the time domain into separate contractive and measure invariant phases, with a carefully designed loss function for each. In other words, what we are doing here is a kind of "physics-informed Koopman learning" for dissipative chaotic systems. This is fine, albeit not particularly groundbreaking. Either way, it should be made very clear how the proposed method relates to prior work on Koopman learning.

    Taking all of this into account, I think it is essential that the proposed method is properly and rigorously placed within the context of existing literature on Koopman learning, both theoretically and experimentally. This requires a detailed discussion of key papers on the foundations as well as the state-of-the-art of Koopman learning. It is also necessary to compare empirically to the state-of-the-art in Koopman learning (the authors do include a Koopman operator baseline, but they do not include any details of the exact setup.) Finally, please also include a paragraph clearly stating the marginal contribution of this paper.

    Ultimately, it is for this reason that I have given this paper a 1 for presentation, even though it is generally quite well written.

2. The results presented in Table 1 are inconclusive and could be presented better. Firstly, only the NRMSE is provided with a measure of uncertainty (although I think the standard error would be more useful than standard deviation for judging the significance of the results). For almost every system, there is no significant difference between FNO, MNO, and PFNN for short-term predictions, as measured by NRMSE. For longer-term predictions, as measured by KLD/MMD/TKE, PFNN performs comparably with MNO for every system except KS, where it does appear better, but the lack of uncertainty estimates makes it hard to draw conclusions. Ultimately, the results as they are presented suggest that PFNN offers relatively little advantage over MNO.

3. The authors do not compare to echo state networks, which are extremely effective networks for learning chaotic dynamical systems. In my opinion, they are an essential comparison in a paper like this. See, for example, [5,6], as well as [7] and citations therein.

### Minor
1. The authors are a bit loose with the term “neural operator”, which I understand to mean a family of models for learning on function spaces. These models possess specific properties, such as grid invariance, that are not shared by the proposed method. See, e.g., [8,9].

2. As a shortcoming of MNO, the authors note that “choosing the right size for this absorbing ball is tricky”. However, the proposed method requires choosing the relaxation time $k$ as a hyperparameter, essentially replacing one hard-to-estimate hyperparameter with another. More generally, I have concerns about the dependence of the proposed method on the choice of $k$, since it requires that the dataset at hand can be neatly split in this manner (what if certain initial conditions are already on the attractor?). In addition, I find the ablation study in Table 2 unconvincing, since there is little meaningful difference between $k = 7$ and $k = 9$; I think the range of $k$ needs to be extended to larger values.

### References
[1] Lusch, B., Kutz, J.N. & Brunton, S.L. Deep learning for universal linear embeddings of nonlinear dynamics. Nat Commun 9, 4950 (2018). https://doi.org/10.1038/s41467-018-07210-0

[2] Qianxiao Li, Felix Dietrich, Erik M. Bollt, Ioannis G. Kevrekidis; Extended dynamic mode decomposition with dictionary learning: A data-driven adaptive spectral decomposition of the Koopman operator. Chaos 1 October 2017; 27 (10): 103111. https://doi.org/10.1063/1.4993854

[3] Otto, S. E., & Rowley, C. W. (2019). Linearly Recurrent Autoencoder Networks for Learning Dynamics. SIAM Journal on Applied Dynamical Systems, 18(1), 558-593. https://doi.org/10.1137/18M1177846

[4] E. Yeung, S. Kundu and N. Hodas, "Learning Deep Neural Network Representations for Koopman Operators of Nonlinear Dynamical Systems," 2019 American Control Conference (ACC), Philadelphia, PA, USA, 2019, pp. 4832-4839, doi: 10.23919/ACC.2019.8815339.

[5] Jaideep Pathak, Zhixin Lu, Brian R. Hunt, Michelle Girvan, Edward Ott; Using machine learning to replicate chaotic attractors and calculate Lyapunov exponents from data. Chaos 1 December 2017; 27 (12): 121102. https://doi.org/10.1063/1.5010300 

[6] Pathak, J., Hunt, B., Girvan, M., Lu, Z., & Ott, E. (2018). Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach. Physical Review Letters, 120(2), 024102.

[7] Yan, M., Huang, C., Bienstman, P. et al. Emerging opportunities and challenges for the future of reservoir computing. Nat Commun 15, 2056 (2024). https://doi.org/10.1038/s41467-024-45187-1

[8] Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2023). Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs. Journal of Machine Learning Research, 24(89), 1-97.

[9] Azizzadenesheli, K., Kovachki, N., Li, Z. et al. Neural operators for accelerating scientific simulations and design. Nat Rev Phys 6, 320–328 (2024). https://doi.org/10.1038/s42254-024-00712-5

### Questions
1. How do you understand the relation of the proposed method to Koopman learning?

1. For the Koopman operator baseline, what is the exact network setup? Are there any other Koopman learning architectures you think are relevant to the current problem?

2. Why is the model called Poincare Flow Neural Network? The name seems to be an homage to Poincare rather than a description of what the network actually does.

3. In Table 1, it’s not clear what is meant by TKE. Is this an error in TKE with respect to ground truth?

4. Could you elaborate on this statement? "Notably, while FNO uses FFT to extract features and provides relatively accurate short-term predictions, its reliance on integer Fourier modes tends to produce periodic behavior."

5. Could you elaborate on this statement? "Models like LSTM and Koopman, which do not account for transient behaviors for states lying outside attractors..." 

6. Is there a reason the longer-term experimental results are not reported with uncertainty estimates?

7. Is there a reason you haven't compared with echo state networks?

### Soundness
2

### Presentation
2

### Contribution
2
