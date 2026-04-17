# Inter-Task Learning Dynamics in Deep Linear Multi-Task Networks

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Despite significant empirical progress in Multi-Task Learning (MTL), the theoretical understanding of task interactions and their dynamics remains limited. We present a theoretical analysis of how task alignment shapes learning dynamics in linear MTL, providing a theoretical justification for why task importance is inherently dynamic and why loss weighting schemes should adapt during training.
Leveraging the Riccati formulation of gradient flow, we analytically characterize the evolution and interaction of shared and task-specific components in deep linear neural networks. For a broad class of initializations, we show how task alignment and magnitude differences govern the trajectories of task outputs, losses, and neural representations throughout training, as well as the representations at convergence. Our analysis reveals that task alignment impacts learning speed and modulates the relative importance of tasks throughout training, with magnitude differences further amplifying these effects. We further show that these factors determine how the structural relationships of the tasks are encoded at convergence in deep linear networks.
Our framework provides a principled comparison between single-task and multi-task settings, grounded solely in data and task alignment. These results establish a theoretical foundation for understanding task interactions and pave the way toward principled approaches to adaptive loss weighting and task grouping.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a theoretical analysis of learning dynamics in multi-task learning (MTL), focusing on the common hard parameter sharing setting.
The approach of the authors follows a series of works in the single-task setup, where two-layer deep linear networks are analyzed through the use of the Riccati formulation of gradient flow. These results are adapted and employed in the MTL setting.
A detailed analysis in the cases of aligned, conflicting and orthogonal tasks is presented, somewhat mirroring common intuitions in the MTL literature. 
The appendix compares these theoretical results with a small dataset commonly employed in the MTL literature.

### Strengths
The MTL setting is still far from being understood, with a series of works in the area drawing conflicting conclusions for instance concerning the utility of multi-task optimizers. 
This work is definitely a step towards providing a better understanding of MTL learning dynamics, and will hopefully pave the way for practical advances.
Most of the results also appear to be relatively intuitive, and in line with common intuitions in the area.
I appreciate the effort by the authors to provide intuition behind their results, and the extensive appendices.

### Weaknesses
- While the work is definitely interesting and novel, its methodology heavily builds on previous work on STL. This is however appropriately acknowledged by the authors.
- The employed assumptions appear to be extremely strict, and of course do not apply to deep MTL.
- Not a lot of emphasis is placed on real-world data. Multi-MNIST results are delegated to a fairly long appendix, however it would be quite important to more concisely and directly discuss the relationship between the theoretical results and real-world data in the main body of the paper.
- Some of the authors' conclusions are not immediately apparent from the figures. For instance, in Figure 3, it is honestly quite hard to discern any difference between the orthogonal and the conflicting tasks setups. This is fairly surprising, though. I am even more surprised by the fact, when considering the loss function in Figure 4, orthogonal tasks seem to make for a harder learning problem.
- As far as I understand, the provided theory will not easily take into account the effect of network capacity, which however appears to be a very important factor in MTL, especially considering whether any task orthogonality will effectively hinder the learning process.

### Questions
- How can network capacity be factored in the provided theory?
- The practice of Multi-MNIST appears to more or less mirror the provided theory. Would this happen on larger-scale MTL datasets too?
- Could you please share some more intuition as to how your results could influence the practical design of MTL techniques?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper extends the Riccati formulation of gradient flow from single-task to multi-task learning in deep linear networks. The authors characterize how task alignment (measured through SVD overlap) and magnitude differences affect learning dynamics, showing that these factors determine whether tasks learn simultaneously, sequentially, or in a conflicting manner. The main theoretical contribution is an analytical solution for the evolution of shared and task-specific components in linear MTL networks, which provides a motivation for dynamic loss weighting schemes.

### Strengths
**Practical motivation**: The work addresses a real gap - the lack of theoretical understanding of task interactions in MTL. The connection to dynamic loss weighting is a valuable insight, even if not fully developed.

**Clear presentation**: The paper is well-written with effective visualizations. Figure 2 and 3 nicely illustrate the key phenomena of how task alignment affects learning dynamics.

**Systematic analysis**: The categorization of task relationships (aligned/orthogonal/conflicting) and their impact on learning provides a useful framework for thinking about MTL.

### Weaknesses
**Limited theoretical novelty**: The core contribution is essentially applying Braun et al. (2022)'s framework to a multi-task setting. The mathematical machinery remains unchanged - just the dimensions and notation are extended. The proofs follow the same structure with minimal technical innovation.

**Restrictive MSE-only assumption**: The paper assumes all tasks use MSE loss, which severely limits its applicability. Real-world MTL typically involves heterogeneous tasks - classification (cross-entropy), regression (MSE), ranking, etc. For example, in computer vision, depth estimation (MSE) often trains jointly with semantic segmentation (CE). The interaction between different loss types could fundamentally change the dynamics, yet the current framework cannot handle this. This makes the theoretical insights less relevant to practical MTL systems.

**Lack of algorithmic contributions**: Despite providing theoretical motivation for dynamic loss weighting, the paper fails to propose any concrete adaptive weighting algorithms based on their analysis. This is a significant missed opportunity, the theoretical insights about task alignment and learning dynamics could directly inform the design of new weighting strategies.  Without demonstrating how their theory translates into practical algorithms, the paper remains purely descriptive rather than prescriptive. 

**Shallow insights**: While the paper describes *what* happens under different task alignments, it doesn't explain *why* these phenomena occur at a deeper level. For instance, why exactly do conflicting tasks create non-monotonic learning curves? The analysis stays at the level of observing consequences of the analytical solutions rather than providing fundamental understanding.

**Experimental limitations**: 
- All experiments use toy problems with small dimensions
- No comparison with existing dynamic weighting methods (GradNorm, PCGrad, etc.)

**Gap between theory and practice**: The results are confined to linear networks with restrictive assumptions (whitened inputs, zero-balanced weights, MSE loss only). The paper doesn't bridge this gap or provide actionable insights for real MTL systems. How do these results guide the design of actual MTL algorithms?

**Missing key analyses**:
- No discussion of which task alignments are optimal for generalization
- No theoretical analysis of sample complexity in MTL vs STL
- The convergence analysis (Theorem 3) is just stating what the solution converges to, without rates or conditions

### Questions
1. **Can you provide learning dynamics for realistic problem sizes?** The current examples are too small to be convincing. What happens with, say, 100-dimensional inputs and 10 tasks?

2. **How do your theoretical predictions compare with actual dynamic weighting algorithms?** It would strengthen the paper to show that your theory explains when methods like GradNorm succeed or fail.

3. **What happens when assumptions are violated?** Real data isn't whitened, and weights aren't zero-balanced. How robust are your conclusions?

4. **Can you derive optimal task groupings from your framework?** Given task statistics, can you predict which tasks should be trained together vs separately?

5. **The "conflicting" case ($0 < \alpha_i< 1$) seems to cover a huge range of scenarios.** Can you provide more granular analysis? When does partial alignment help vs hurt?

6. **How does the number of tasks NT scale in your analysis?** All examples show $N_T=2$. Does the framework become intractable or reveal new phenomena for many tasks?

7. **What's the computational cost of computing these theoretical predictions?** Could they be used in practice to guide training?

### Soundness
3

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
The work extends a Riccati framework to deep linear multi-task learning and provides an analysis of the training dynamics and task interactions. It analyzes how task alignment and magnitude imbalance shape interaction patterns: orthogonal, conflicting, and aligned.

### Strengths
1. It categorizes tasks into orthogonal, aligned, and conflicting cases and clarifies, in a principled way, when interference grows.
2. It visualizes how alignment and magnitude imbalance lead to learning delays and changing task importance.

### Weaknesses
1. As the authors already note, the analysis is limited to deep linear models, which weakens confidence about transfer to realistic nonlinear systems. Beyond that, the paper offers little guidance on how to operationalize the insights, making scalability and generalizability hard to judge. For example, the computational cost of estimating alignment spectra and applying Riccati-based diagnostics on deep networks with large datasets is unclear. It is also uncertain whether online alignment estimation during multi-task training or optimization is feasible.

2. In practice, across many multi-task settings, relationships between tasks rarely fit neatly into “orthogonal,” “aligned,” or “conflicting.” Different aspects can appear simultaneously to varying degrees. How should we handle such mixed cases in real systems? These aspects are not adequately discussed in the paper.

### Questions
Please refer to the Weaknesses section and provide detailed responses addressing each point.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies theoretically the dynamic of inter-task learning in linear multi-task learning (MTL) networks. To this end, the authors write down the gradient flow equations of the MTL network using the Riccati formulation and derive from it a closed-form solution for the dynamics of the networks at any step. Then, the authors use these equations to simulate the dynamics of the network under different relationships across tasks (aligned, conflicted, and orthogonal) as well as the effect of having tasks with relatively different magnitudes. Interestingly, the authors also use this approach to obtain theoretical results and simulations on the representations obtained by this linear MTL networks and compare them with their single-task counterparts.

### Strengths
- **S1.** The manuscript seem to be theoretically sound and technically involved. I have not looked into the proofs, but I am pleased to see that the authors corroborated that their findings match a linear MTL in the appendix.
- **S2.** While technical, the paper is well written and can be easily followed with a bit of patience.
- **S3.** Experiments are rather extensive given the assumptions on the model, and the conclusions drawn while expected (from my view, these conclusions were relatively known from empirical results over the years), it is great to obtain a theoretical validation.
- **S4.** I find particularly interesting the results regarding the representations learned by the models in the MTL setup.

### Weaknesses
- **W1.** My main concern is the extent to which the assumptions made make the setting interesting. In particular, the assumption that concerns me the most is assumption A3: It is relatively well known in the community that if there is no bottleneck the model will learn all tasks one way or another. The more interesting set-up is when there is a bottleneck and the model needs to learn to use a shared representation across tasks. The linearity of the model also concerns me, but to a lesser extent.
- **W2.** Another concern is the novelty of the theoretical results. In particular, the article constantly references for the results and proofs to Braun et al. (2022). I'd appreciate it if the authors could make clear what is the contribution of this work in relation with the aforementioned work.
- **W3.** I find the plots particularly difficult to read. In particular, I am unable to give meaning to those as in Figure 3. I understand that each line represents the evolution over time of one entry of the matrix $W_iW_s$. However, I cannot interpret these values myself.
- **W4.** Maybe due to the previous point, I do not see how the authors can draw statements regarding the temporal component of the dynamics. For example, in line 395 where is says "Over time, a task that was initially harmless may later interfere with or suppress the learning of others."
- **W5.** It would be nice to have a section on related work on MTL analysis as well as in MTL techniques to alleviate gradient conflict (such as FAMO, PCGrad, GradNorm, RotoGrad, among many others). Moreover, it would be a great plus to see whether any of these techniques helps with the linear/ReLU MTL examples that appear in the appendix.

### Questions
- **Q1.** How does the overlap in definition 2.1 with the cosine similarity between task gradients, which is the de-facto measure used in MTL works?
- **Q2.** Eq. (11) is rather unclear. Are the $i$ and $j$ powers? Is the $0$ supposed to be a $o$?

Other feedback:
- Regarding the motivation in the second paragraph of the intro, I'd like to share with the authors [this recent study](https://arxiv.org/abs/2505.10347) that points out at the reasons of why _sometimes_ simple heuristics work as well as other MTL optimizers.
- Similarly, I'd also like to share [this recently accepted paper](https://web3.arxiv.org/abs/2510.18258) where the study also the effect of tasks in MTL through the lens of neural tangent kernel methods.
- You are sharing the same label across equations, and sometimes point to the appendix instead that to the main content.
- I assume the "+" in line 263 is supposed to be a "-".
- The caption of Fig. 4 needs to be updated.

### Soundness
3

### Presentation
3

### Contribution
3
