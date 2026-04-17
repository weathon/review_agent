# Building Flow Uniqueness in One-step Generative Modeling

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Recent advances in generative modeling frameworks, such as diffusion models and flow matching, have achieved record-breaking performance.
Nevertheless, these approaches involve iterative sampling procedures across many neural network passes, which severely limits their practical deployment, particularly in domains demanding real-time interaction.
Although considerable effort has been devoted to accelerating sampling, achieving high-quality one-step generation remains an open challenge, motivating research into a new era of generative modeling.
Motivated by this, we put forward a novel and effective framework, termed \textit{Flow Uniqueness Models} (\textbf{FUM}).
The core idea of FUM is to construct strictly one-to-one image pairs, thereby enforcing velocity uniqueness along the entire sampling path, which forms as the foundation for few-step sampling.
By leveraging this modeling mechanism, FUM not only achieves remarkable one-step generative performance but also provides the flexibility to balance image quality against the number of sampling steps.
Extensive experiments on three benchmark datasets comprehensively validate the superiority of our proposed FUM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Flow Uniqueness Models (FUM), which splits the overall generative trajectory $P(\epsilon \rightarrow x_0)$ into two sub-paths, $P(\epsilon \rightarrow x_s)$ and $P(x_s \rightarrow x_0)$. The first sub-path is to enforce velocity uniqueness via strictly one-to-one sample paris, while the second is aligned through a flow consistency constraint. To achieve this, this paper introduces two variants of flow consistency: a Shortcut Models-based strategy and a MeanFlow-based strategy. In addition, this paper reports empirical results on three benchmark datasets to demonstrate the performance of FUM.

### Strengths
+ This paper addresses the important direction of one-step generation and reports results on multiple datasets.
+ This paper unifies ideas from Shortcut and MeanFlow models under a unified framework, with reproducible experiment settings.

### Weaknesses
- Unconvincing motivation and weak necessity of sub-path division. Velocity ambiguity in flow matching originates from marginalization over random $(x_0, \epsilon)$ pairs, not from the temporal structure itself. Simply splitting the trajectory into segments cannot reduce or eliminate this ambiguity. 
- Sampling scheme restricts model capacity. As shown in Eq.(3)/(4), the method samples $i \sim U(0,s)$ and $j \sim U(s,1)$, forcing training pairs to be drawn only across the two sub-paths. Unlike MeanFlow or Shortcut, which allow arbitrary $(i,j)\in[0,1]^2$, this design reduces temporal coverage and learning diversity. Moreover, it potentially breaks smoothness of the learned continuous flow. 
- Limited experiments and modest performance. There is no ablation study demonstrating that the sub-path division improves training stability or performance. On ImageNet-64 (Table 2), FUM underperforms several strong baselines (e.g., sCT, ECM).

### Questions
- How is R, the so-called reversible range, quantitatively determined? Please provide a reproducible criterion and sensitivity analysis.
- Why is the sub-path division necessary at all? Show an ablation comparing full-path vs two-path training.
- There are several typos and inconsistencies. For example, in Eq.(4), $v_{\theta}(x_i, 0, i)$ should be $v_{\theta}(x_i, i, s)$.
- For clarity, it may be better to use $u$ to denote average velocity instead of $v$, to distinguish it from instantaneous velocity.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a framework Flow Uniqueness Models (FUM) that obtains strong one-step generative performance by constructing strictly one-to-one image pairs to enforce strong velocity uniqueness along the sampling path. To do this, the paper divides the entire flow path into two sub-paths, where the velocity of the first sub-path preserves strong uniqueness by leveraging strictly one-to-one image pairs, and the velocity of the second sub-path is linked to the first velocity by flow consistency strategies. The paper experiments on CIFAR-10, FFHQ and ImageNet to show the effectiveness of the proposed algorithm.

### Strengths
•	The paper is clearly organized, including both the theoretical analysis on the unique velocity and the empirical experiments.

•	The proposed method is straightforward and has the ability to incorporate multiple flow consistency strategies.

•	The experiments are comprehensive, including comparisons with recent few step generation method and ablation studies.

### Weaknesses
•	The novelty of the paper is limited, the flow consistency strategies are adopted from the well-known works, making the main contribution only focusing on the division of the two phases of the path trajectory.

•	The standard on the determination of the reversible range $R$ is not thoroughly discussed in the paper. This may require additional simulation and hinder the sample efficiency.

•	The paper lacks theoretical discussion on the benefit of introducing the two sub-path division compared to other one step generation methods.

•	Leveraging strictly one-to-one image pairs may affect the diversity of the generated images, which could lead to images with noticeable artifacts.

### Questions
•	How is the reversible range $R$ determined? Does it require additional simulation or pretraining models?

•	Shortcut models and MeanFlow models are the flow consistency strategies used in the paper. Which strategy is better in what kind of tasks?

•	Is the proposed method numerically stable in linking the two sub-paths? Does the prior distribution $\pi$ affect the numerical training difficulty? If yes, what type of prior $\pi$ should we choose?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework called FUM, which construct strictly
one-to-one image pairs in flow matching. FUM enforced strong velocity uniqueness along the
entire sampling path, and improves the generation efficiency in few-step sampling.

### Strengths
- The motivation of FUM is clear and reasonable
- The proposed method is well-supported by the derivation
- Comprehensive experiments on image generation benchmarks prove the effectiveness of the proposed method.

### Weaknesses
The experiments are performed on relatively small datasets, some text-to-image experiments should be included.

### Questions
Since FUM still needs fine-tuning on the pre-trained $v_\theta$ model, what is the training costs of Algorithm 1?

### Soundness
3

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
4

### Summary
The paper proposes Flow Uniqueness Models (FUM), a framework for achieving high-quality one-step generation within flow-matching models. The key idea is to enforce velocity uniqueness across the sampling path by dividing the flow into two sub-paths: (i) a first sub-path trained on strictly one-to-one image pairs to ensure deterministic flow, and (ii) a second sub-path trained with a flow consistency strategy to align its velocity with the first. Two consistency variants (Shortcut-based and MeanFlow-based) are introduced. Experiments on CIFAR-10, FFHQ, and ImageNet demonstrate competitive one-step and few-step generation performance compared to prior diffusion distillation and consistency models.

### Strengths
- The paper tackles an important practical issue—reducing sampling steps in generative models without major loss in quality.

- The notion of “velocity uniqueness” provides an intuitive way to understand and regularize one-step generation.

- FUM shows consistently strong or comparable performance to leading methods (e.g., MeanFlow, Shortcut) across multiple datasets.

### Weaknesses
- Limited novelty: While the idea of enforcing path or velocity consistency is meaningful, the proposed approach mainly combines existing components (pairwise matching + consistency regularization) without introducing a fundamentally new principle.

- FUM relies on pre-trained diffusion models to obtain the one-to-one image pairs, meaning it is not a fully independent or end-to-end training scheme.

- At several places, the authors mention “strong uniqueness”. Could the authors please define to quantify “strong” here?

- The paper contains many typos and inconsistencies (see list below), which makes me doubt the technical correctness of the paper. 

List of potential typos/inconsistencies:
- Line 128: definition of $a_t$ is not correct
- Equation (1): $\epsilon$ is missing in the objective function
- At the beginning, the flow is defined on $t \in [0, 1]$. Later on, the index is shifted to $\{0, …, T\}$. 
- Around equation (2): it is unclear whether $s \in [0, T]$ or $s \in [1, T]$. 
- Equation (3) and (4): How could I sample $j \sim \mathcal U(s, j)$?
- Algorithm 1, line 14: the EMA update is incorrect.
- Algorithm 2, line 6: Do we miss a $\Delta k$ term?
- Proposition 1 only holds under some assumptions of the noise distribution $\pi$. These assumptions have not been made explicit.

### Questions
- How sensitive is the method to the choice of the split points between sub-paths?

- Does the performance degrade significantly if the initial diffusion model is not well-trained?

- Can FUM be trained entirely from scratch without relying on a pre-trained ODE sampler?

- What is the computational overhead compared to MeanFlow or Shortcut models during training?

- Could the velocity uniqueness concept be extended to text-to-image or multimodal setups?

### Soundness
2

### Presentation
1

### Contribution
1
