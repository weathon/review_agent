# Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
We study in-context learning (ICL) of linear regression in a deep linear self-attention model, characterizing how performance depends on various computational and statistical resources (width, depth, number of training steps, batch size and data per context). In a joint limit where data dimension, context length, and residual stream width scale proportionally, we analyze the limiting asymptotics for three ICL settings: (1) isotropic covariates and tasks (ISO), (2) fixed and structured covariance (FS), and (3) where covariances are randomly rotated and structured (RRS). For ISO and FS settings, we find that depth only aids ICL performance if context length is limited. Alternatively, in the RRS setting where covariances change across contexts, increasing the depth leads to significant improvements in ICL, even at infinite context length. This provides a new solvable toy model of neural scaling laws which depends on both width and depth of a transformer and predicts an optimal transformer shapes as a function of compute.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper theoretically investigates in-context learning for linear regression using deep linear self-attention models, analyzing performance based on width (N), depth (L), context length (P), pretraining time (t), and data structure. Examining isotropic (ISO), fixed structured (FS), and randomly rotated structured (RRS) covariance settings, the authors find that depth primarily benefits ICL in the ISO and FS settings only when context length is limited; for infinite context length in these settings, increasing depth beyond L=1 offers no advantage. However, in the more complex RRS setting where covariances vary, depth significantly improves performance even at infinite context length. For this RRS case, the paper derives a Chinchilla-like scaling law and predicts a compute-optimal shape scaling, linking optimal architecture to task data properties.

### Strengths
- The paper provides a comprehensive theoretical analysis of multi-layer linear self-attention models for in-context linear regression across three distinct covariate settings (ISO, FS, RRS).
- This paper rigorously characterizes the training dynamics using gradient flow analysis, revealing how the model learns under different data structures and providing an interpretation of the learned estimator as implementing multi-step gradient descent with optimal step sizes.
- The derivation of a Chinchilla-like neural scaling law incorporating time, width, depth, and context length for the RRS setting in the context of linear regression with power-law features is a significant theoretical contribution.
- The application of Dynamical Mean Field Theory (DMFT) to derive a two-point deterministic equivalent for the loss landscape under random rotations represents a novel technical approach for analyzing complex learning dynamics in this asymptotic regime.

### Weaknesses
-The presentation of detailed proofs and derivations within the appendix could be improved for clarity and accessibility, making it challenging to fully verify the technical steps.

### Questions
- Could the authors elaborate on the necessity of employing DMFT to derive the closed-form loss expression in Result 7? Is this approach required because directly analyzing the gradient flow dynamics in (13) is intractable, perhaps due to the lack of a known closed-form solution for the ODE governing $\gamma(t)$ in the randomly rotated setting with finite width N?
- The current analysis focuses heavily on the proportional asymptotic regime. Are the techniques employed amenable to deriving non-asymptotic results that might provide insights into the behavior of the system with sizes?
- The derivation of the two-point deterministic equivalent using DMFT in Appendix A introduces notation that appears distinct from the parameters used in the main text to describe the Transformer model and its dynamics and it's hard to directly match (22) with the main result. Could the authors provide a clearer mapping between the DMFT variables/order parameters and the model parameters/dynamics described earlier in the paper?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies deep linear self-attention trained on the in-context linear regression task, characterizing how performance depends on width, depth, number of training steps, batch size and data per context.

### Strengths
The paper studies an interesting and relevant problem of training a deep linear self-attention model on in-context linear regression tasks, considering different in-context task structures. Extensive prior work has examined one-layer linear self-attention trained on in-context linear regression with isotropic task vectors. Analyzing deep models and non-isotropic task vectors represents meaningful progress.

### Weaknesses
- My main concern is the looped transformer assumption, $W_i^l=W_i^{l'}, i\in \{k,q,v\}$, which means that all layers have the same weights. The expressivity and optimization properties of a deep self-attention model can differ significantly with or without this assumption. Why would the scaling limits derived under this constrain reflect those of a real multi-layer attention model, which typically learns different weights across layers?

- In the reduced model in Equation (4), all weight matrices within a layer appear to be merged into a single trainable matrix $\Gamma$, effectively making the self-attention layer "shallow". Since gradient descent dynamics and the loss landscape are sensitive to such reparameterization, the true loss landscape probably differs from those shown in Figure 2 and Figure 4b. If this is indeed the case, it would be helpful to explicitly highlight this distinction.

- I suggest that the authors perform another round of proofreading and polishing. There are presentation issues that make the paper unnecessarily difficult to read smoothly. I list some below.

  It appears that manual vertical spacing commands have been used in several places of the paper. The formatting on page 14 seems irregular.

  The clarity of Equation (3) could be improved by specifying the dimensionality of weight matrices. 

  The authors use inner-product and transpose notations interchangeably. The expectation notation $\mathbb E(\cdot)$ and the angle brackets notation $\langle \cdot \rangle$ are also used interchangeably. Adopting consistent notations throughout would enhance readability.

  The symbols $\boldsymbol X, \boldsymbol y$ in Equation (4) seem to be undefined.

  The symbol $i$ is used inconsistently: sometimes as an index and sometimes as the imaginary number. In particular, it is undefined in Result 7. Clarifying its meaning in each context would avoid confusion.

  In Equations (13), (15), it appears that a scalar is being added to a matrix, e.g., $(1-L^{-1}\gamma\Lambda), (i\omega+\Psi\Lambda)$. Please check these terms for dimensional consistency. Additionally, the trace operator $\text{tr}$ is used without parentheses, which could be ambiguous.

### Questions
The problem considered in this paper is interesting and potentially important. However, recurring issues with presentation and clarity make it difficult to fully assess the contributions. Improving the clarity would make the results more accessible for proper evaluation.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors tackle the problem of how should we allocate depth, width, context length, and training compute when scaling transformers for in-context learning (ICL) on regression tasks. The paper analyzes a deep loop linear attention transformer trained (via SGD) to do linear regression in context, without finetuning at test time. It studies three task regimes: (i) isotropic data, (ii) fixed but structured covariance, and (iii) randomly rotated structured covariance (task distribution shifts every context), and derives dynamical equations and asymptotic scaling laws linking pretraining time, width, depth, and context length. The contribution lies in a unified, theoretically grounded scaling law that distinguishes the roles of depth, width, context length, and training time in ICL.

### Strengths
1. I appreciate that the paper gives a concrete, theoretically grounded answer to a question that is widely discussed in practice: how should depth vs. width vs. context length scale for in-context learning? Instead of treating “bigger model = better,” it isolates when depth specifically matters, and ties that to properties of the task distribution (shared vs. varying covariance across contexts). Specifically, the model decouples network width $N$ from the problem dimension $D$ when studying the scaling law, and the use of loop transformer ensures that the total number of parameters does not go up with the number of computes---which I believe provides a decoupled and generic test bench that adds to similar work studying scaling with linear model.

2. A demonstration that the usefulness of depth is task-distribution dependent: If all tasks share the same covariance, depth is asymptotically unnecessary (long enough context suffices, as the model weights can "encode" the covariate information). If covariances vary across contexts, depth is fundamentally valuable, even with infinite context (reflecting the philosophy of test-time compute). 

3. A good match between theories and experiments.

### Weaknesses
1. The entire analysis is built around linear regression tasks solved via in-context learning with (mostly) loop linear attention. I didn't find much discussion surrounding the use of loop attention block. One benefit I could imagine is the decoupling between total model weights and the depth. However, the use of loop could potentially restrict the model's expressiveness, where model could possibly implemented higher-order optimization algorithm, e.g., Newton's step rather than gradient descent [1], and the model might demonstrate different scaling behavior, especially in depth. 

2. The key conceptual result is that depth becomes essential when task covariances vary across contexts (“randomly rotated structured,” RRS). But the diversity they model is very specific: random orthogonal rotations of a shared spectrum. That’s mathematically nice but arguably still a stylized shift. Real heterogeneity looks more like mixture of domains, sparsity structure, nonstationary label noise, hierarchical latent factors, etc. It’s not obvious that random rotations is the right stand-in for natural distribution shift.

3. The paper argues its results are relevant to large-scale LLM design, but the experiments cap out at synthetic regression and relatively small controlled transformers. There’s no ablation on modern-scale architectures (residual blocks with MLPs, nonlinear attention heads, long-context finetuning) to show even qualitative alignment. So the significance for frontier models is still somewhat speculative.

[1] Giannou, A., Yang, L., Wang, T., Papailiopoulos, D., & Lee, J. D. (2024). How Well Can Transformers Emulate In-context Newton’s Method? arXiv preprint arXiv:2403.03183.

### Questions
1. Your theory and experiments focus on linear regression tasks with (mostly) linear attention, Gaussian feature distributions, and controlled covariance structure. How confident should we be that the same depth–width scaling conclusions hold for nonlinear Transformers trained on natural language, vision, or multimodal data? 

2. You mostly analyze single-head linear attention plus residual depth. How do you expect multi-head structure and MLP blocks (i.e., actual transformer blocks) to affect the depth vs. width story? For example, could significantly more attention heads (possibly also scales with D) also benefit the learning process?

### Soundness
4

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
This paper studies deep linear self-attention by analyzing the corresponding solvable model. The authors systematically investigate how depth of the model, width, context length, and training steps affect the solution of the model. Specifically, the authors focus on three distinct types of data (termed as ISO, FS, and RRS), where the authors reveal that depth is unnecessary for long contexts on ISO and FS, along with a series of other results that characterize the gradient flow dynamics. Furthermore, the authors derive a separable scaling law for the RSS setting.

### Strengths
1. This paper is well written and well organized. The presentation is very clear and the flow of this paper is consistent, which I think can allow the readers to easily appreciate the core contributions of this work (the summarized theoretical results along with immediate numerical experimental results). 

   The proof is also easy to follow (though I did not check all the details): first transforming the gradient learning dynamics to an equivalent linear model, which has a simpler dynamics, then diving to this new dynamics under different conditions of the covariance. Although this general idea is not new, incorporating the depth to the analysis is novel.

2. The derived theoretical results indeed consider attention in the multi-layer case, which I believe is an improvement over prior works. The authors demonstrate when and why the depth can be necessary. The summarized results are indeed novel and interesting.

### Weaknesses
While the results are interesting by considering the depth, I think the setting is still not significantly novel compared to prior works. In particular:

 - While equation (3) indicates a dependence on the layer depth $l$, the induced parameter $\Gamma$ in fact does not depend on $l$, because the matrices $W_i$'s are treated equally for different layers given one specific $i$. Instead, this dependence on $l$ is replaced by a simple summation over $l$ in $\Gamma$. As a result, the corresponding analysis in fact does not provide significantly novel analysis compared to prior works in this line of research, i.e., studying the gradient flow dynamics of the induced parameter $\Gamma$ (which is still a linear regression) as a proxy of the true weight matrices of the attention model. It remains unclear whether doing so can really capture the effects of depth.

- The RSS is positioned as the most general case, but the randomly rotated and structured across contexts covariance still cannot effectively capture  the essence of task diversity. In addition, the theoretical framework is built on taking a very specific join limit where $P, K, B, D, N \to \infty$ with fixed ratios. While convenient for analytical tractability, this obscures efects that are relevant at finite scales. 

- Due to the aforementioned limitations, the generality of the derived results remains unclear.


Minor: As this paper considers solvable models and scaling laws of attention, I think the related work [1], which also studies a solvable model for attention and its scaling laws, could be discussed a bit in the related work.

[1]. Lyu et al.  A Solvable Attention for Neural Scaling Laws. ICLR 2025.

### Questions
1. As I'm mostly concerned about the role of the depth, which plays an important role in the novelty of this work, can the authors justify the validity of assuming equal weight matrices across layers and the corresponding generality?

2. Furthermore, can the authors discuss the difficulty brought by varying weight matrices w.r.t layers and how the current framework can still be applied in that case?

3. Is taking the joint limit of $P, K, B, D, N$ necessary for the results presented in this paper?

### Soundness
3

### Presentation
4

### Contribution
2
