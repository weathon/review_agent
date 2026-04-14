# Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport

- Decision: Accept (Oral)
- Scores: 8, 8, 10

## Abstract
Reconstructing dynamics using samples from sparsely time-resolved snapshots is an important problem in both natural sciences and machine learning. Here, we introduce a new deep learning approach for solving regularized unbalanced optimal transport (RUOT) and inferring continuous unbalanced stochastic dynamics from observed snapshots. Based on the RUOT form, our method models these dynamics without requiring prior knowledge of growth and death processes or additional information, allowing them to be learned directly from data.  Theoretically, we explore the connections between the RUOT and Schrödinger bridge problem and discuss the key challenges and potential solutions. The effectiveness of our method is demonstrated with a synthetic gene regulatory network, high-dimensional Gaussian Mixture Model, and single-cell RNA-seq data from blood development. Compared with other methods, our approach accurately identifies growth and transition patterns, eliminates false transitions, and constructs the Waddington developmental landscape. Our code is available at: [https://github.com/zhenyiizhang/DeepRUOT](https://github.com/zhenyiizhang/DeepRUOT).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces DeepRUOT, a method for reconstructing dynamics from snapshots using regularized unbalanced optimal transport. This work tackles both the unbalanced and stochastic settings simultaneously, which few works have attempted thus far. An algorithm is proposed to learn drift, score, and growth functions from snapshot data including both synthetic and single-cell transcriptomics data.

### Strengths
* Provided code provides additional clarity on experimental setting.
* Tackles an important problem in trajectory inference from population level data.

### Weaknesses
* I found it extremely difficult to understand the multistage training procedure for DeepRUOT. There are some stages described in the main text, but then initial log density training also in the appendix? While there are a number of losses and initializations proposed to stabilize training, it is unclear to me what effects these have on final performance and how difficult RUOT is to train in practice. The paper would be much stronger if it was made clear which parts of the training procedure are important practically potentially through an ablation study on different components of the training procedure. 
* For a largely empirical work proposing a new algorithm to tackle this problem, the empirical studies are limited. The DeepRUOT is only benchmarked against balanced transport methods. It would greatly strengthen this paper if it was also compared to other unbalanced transport methods such as Action matching [Neklyudov et al. 2023] and unbalanced diffusion Schrödinger bridge [Pariset et al. 2023]. It would also be very useful to know how the many different weighting parameters affect performance, and how sensitive training is to them.
* It is difficult to tell what parts of the theory are novel and which are minor adaptations from prior work. Theorem 4.1 seems like a subclass of Baradat & Lavenant for the Fischer information case. I don’t believe varying sigma(t) over time changes anything theoretically and as far as I understand it is not used in practice.

### Questions
It is unclear to me exactly what the computational / numerical cost for allowing varying growth rates is in this framework. From my understanding both the deterministic and stochastic setting are extremely efficient, but it has thus far been numerically challenging to incorporate varying growth rates. How scalable is DeepRUOT? Can it be applied to the higher dimensional settings (20,50,100,1000D) in previous works?

I don’t understand the motivation behind the reconstruction loss R_d. What is its purpose and why does it correspond to a reconstruction loss vs. say an OT loss?

Comments:

\Psi(g) is not explicitly defined (although is fairly clear from context).

> So to specify an SDE is equivalent to specify the probability flow
ODE and its score function.

Probably missing an article here. 

### Overall

Overall this work has great potential, however I believe further empirical study would greatly improve its value to the community. I would be eager to reconsider my evaluation given further empirical study relative to existing unbalanced dynamic OT methods and a better empirical understanding of the various components of training and loss weighting.

### Soundness
3

### Presentation
2

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
This paper proposes a new approach for learning dynamics from the observation of distributions at different time steps. The paper proposes to use an unbalanced optimal transport approach that allows modeling particle growth or death over time, which is an important aspect of single cell dynamics modeling. The authors compare their approach on datasets from syntethic gene regulatory networks and single cell trajectories dynamics. The paper shows favorable performance results compared to previous methods.

### Strengths
- Population dynamics from snapshot data is an important problem, especially in biological application such as single cell trajectories inference.
- The proposed method appears to surpass the performance of previous methods.
- The related works section is extensive.

### Weaknesses
- One of the most important weaknesses of this paper at this stage is its clarity. In its current form, the paper is poorly written, and the ideas are poorly motivated. The main limitation from prior works identified is the unabalanced aspect e.g. "However, computational methods for learning high-dimensional unbalanced stochastic dynamics from snapshots without prior knowledge of growth and death or additional information are still lacking.". Yet, the paper only indirectly addresses that issue, which confuses the reader. Despite the favorable experimental results, the current presentation of this paper prevents it from being appreciated by the community.

- Section 5, which is the core of your method, is extremly confusing. For instance, the reconstruction loss is not introduced previously such that it's extremely challenging to follow the development until the final loss in Equation 10. The algorithm 1 does not provide any help either in much needed clarification. The algorithm is too laconic and does not refer to corresponding parts of the text.

- It's not clear to me why the connection with the Schrodinger bridge is necessary. As far as my reading goes, it seems that you only need to leverage the dynamic version of regularized unbalanced OT. As such, it does not seem that the connection with Schrodinger bridges helps the reader understand the method, and you don't leverage that connection later on in your method.

- Theorem 4.1 uses $v$ which is not introduced previously. It's later undertstood that it corrresponds to the vector field of the corresponding ODE but this hinders the clarity of the paper.

- The presentation of the results in Figure 2 raises questions. Why does panel (b), with the true dynamics has ground truth, predicted and trajectories, if it is the ground truth dynamics ? What is the "predicted" in this case ? 

- Because the most important contribution of this paper relies on modeling the growth rate, an ablation study is required where the growth rate part of the method is disabled (e.g. g = 0). This would allow to understand the impact of it on the final performance. 

- Despite the extensive related works section, the experiments only compare with two baselines. Baselines such as Bunne et al. or Yutong et al. could also be considered.

Sha, Yutong, et al. "Reconstructing growth and dynamic trajectories from single-cell transcriptomics data."

### Questions
- Regarding the connection with SB, could you either make the impact of the connection more explicit, or considering removing that part from the paper for clarity ?

- Could you please restructure Section 5 to help the readers follow the reasoning until the final loss and algorithm ? Having a more detailed pseudo-code description would already help significantly.

- Can you clarify Figure 2 - the predictions dots in the ground truth panel ? 

- Could you please add the results of an ablation experiment where you disable the learnable growth rate in your model ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
2

### Summary
The paper introduces DeepRUOT, a new deep-learning approach for solving regularized unbalanced optimal transport (RUOT) and inferring continuous unbalanced stochastic dynamics from observed snapshots. The method models dynamics without requiring prior knowledge of growth and death processes, allowing these to be learned directly from data through a Fisher regularization formulation. The authors explore theoretical connections between RUOT and the Schrödinger bridge problem while reformulating RUOT to transform stochastic differential equations into more computationally tractable ordinary differential equations (a constraint). When tested on both synthetic gene regulatory networks and real single-cell RNA sequencing data, DeepRUOT outperformed existing methods in constructing meaningful developmental landscapes.

### Strengths
- The experimental results seem very convincing
- The paper provides thorough theoretical foundations with formal theorems and proofs connecting RUOT to the Schrödinger bridge problem.

### Weaknesses
- it remains unclear how the loss function weights (cost, reconstruction error and the PINN loss) are set
- Figure 1 and its caption are not that helpful

Typos:
- Theorem 3.1: "probelm" -> "problem"
- page 8: "a increase" -> "an increase"

### Questions
- is not knowing the death/growth rates (what you call prior knowledge) such a big advantage?
- Why did you project the data onto a 2d manifold for the "Real Single-Cell Population Dynamics" experiment? Isn't it possible to be done on a higher dimensional space, e.g. gene expression of PCA space?
- The method involves multiple neural networks (v, g, s) and complex loss functions, which could require substantial computational resources as well as necessary tuning efforts, potentially limiting its practical application in real-time analysis. Would be great to get a comment on this.

### Soundness
3

### Presentation
3

### Contribution
4
