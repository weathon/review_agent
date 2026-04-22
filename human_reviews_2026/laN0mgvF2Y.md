# Revisiting Mixture Policies in Entropy-Regularized Actor-Critic

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 8, 4, 6

## Abstract
Mixture policies in reinforcement learning offer greater flexibility compared to their base component policies. We demonstrate that this flexibility, in theory, enhances solution quality and improves robustness to the entropy scale. Despite these advantages, mixtures are rarely used in algorithms like Soft Actor-Critic, and the few empirical studies that are available do not show their effectiveness. One possible explanation is that base policies, like Gaussian policies, admit a reparameterization that enables low-variance gradient updates, whereas mixtures do not. To address this, we introduce a marginalized reparameterization (MRP) estimator for mixture policies that has provably lower variance than the standard likelihood-ratio (LR) estimator. We conduct extensive experiments across a large suite of synthetic bandits and environments from classic control, Gym MuJoCo, DeepMind Control Suite, MetaWorld, and MyoSuite. 
Our results show, for the first time, that mixture policies trained with our MRP estimator are more stable than the LR variant and are competitive compared to Gaussian policies across many benchmarks. In addition, our approach shows benefits when the critic surface is multimodal and in tasks with unshaped rewards.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper systematically analyzes the benefits of Mixture Policies over standard Gaussian Policies in the context of entropy-regularized Reinforcement Learning (RL), specifically addressing the major algorithmic barrier to their adoption.

The authors theoretically analyze the advantages of mixture policies over Gaussian policies, demonstrating their improved robustness to the entropy scale. Crucially, they introduce the Marginalized Reparameterization (MRP) estimator for Gaussian Mixture (GM) policies and provide a theoretical proof of its lower variance compared to the standard Likelihood-Ratio (LR) estimator. Finally, the work comprehensively validates the advantages of mixture policies across an extensive suite of benchmarks.

While the application of multimodal policies, including Gaussian Mixture Policies, to enhance RL exploration capabilities and final performance has been explored in prior work, it often lacked systematic theoretical grounding and extensive empirical validation. This paper successfully fills this gap by providing a much-needed comprehensive study.

### Strengths
Clear Structure and Readability: The paper is very well-structured, making the theoretical analysis and empirical results clear and easy to follow. 


Systematic Analysis of Policy Flexibility: The work provides a systematic analysis of the theoretical benefits of mixture policies over unimodal Gaussian policies, especially concerning the non-existence of stationary points and superior objective values under high entropy regularization. 




Comprehensive Experimental Validation: The authors conducted extensive and systematic experiments across a large and diverse set of synthetic bandits, classic control, Gym MuJoCo, DeepMind Control, MetaWorld, and MyoSuite environments.

### Weaknesses
Limited Algorithmic Novelty of Mixture Policies:  The application of mixture policies in RL is a topic that has been explored, even briefly in early versions of Soft Actor-Critic (SAC). The core contribution lies in enabling this class via the MRP estimator, not the policy parameterization itself. 

Insufficient General Performance Gain: Despite the broad and systematic experiments, the performance improvement of mixture policies (SGM-MRP) over the standard Gaussian policy (SG-RP) is generally modest or only competitive on major benchmarks. The choice of "unshaped reward" examples to demonstrate superiority (Section 5.3) could be further strengthened. Consideration should be given to including simpler toy environments designed explicitly with known multi-modal optimal policies to provide a clearer and more persuasive visual demonstration of the policy's multi-modality advantage.

Limited Sensitivity Analysis on Component Number: For Gaussian Mixture Policies, the number of components ($N$) is a crucial hyperparameter that dictates policy complexity. The sensitivity analysis for it is confined to a simple set of classic control environments with unshaped rewards.  A more convincing analysis involving at least one high-dimensional environment is needed to confirm the generality of the finding that $N=5$ (or small $N$) is sufficient.

### Questions
Noticed that the SGM-MRP estimator failed to converge in environments such as disassemble-v2 and stick-pull-v2. Could you provide a plausible explanation for why the MRP estimator, which is proven to be stable and low-variance, exhibits non-convergence or poor performance in these specific environments?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper revisits the use of mixture policies in entropy-regularized actor–critic reinforcement learning and provides both theoretical and empirical evidence for their benefits.
The authors show that mixture policies lead to better or more robust stationary solutions under entropy regularization and propose a Marginalized Reparameterization (MRP) gradient estimator that reduces variance in training.
Experiments across a wide range of continuous-control benchmarks demonstrate consistent, though modest, improvements in performance and stability.

### Strengths
1. The theoretical results are internally consistent and logically organized.
In particular, Proposition 3.3 provides a novel robustness argument showing that Gaussian base policies may lose stationary points when the entropy coefficient $\alpha$ exceeds $\tfrac{3}{2}r_{\max}$, whereas Gaussian mixture (GM) policies continue to maintain valid stationary solutions.
This insight connects entropy regularization and multimodal policy landscapes in a clean way.

2. The optimality results (Propositions 3.1–3.2) extend known properties of entropy-regularized optimization, while the robustness to entropy scaling (Propositions 3.3–3.4) and the variance-reduction guarantees for the MRP estimator (Theorem 4.3, Proposition 4.7) represent genuine theoretical value.
Together, they strengthen our understanding of mixture policies in a principled manner.

3. The experiments cover a wide range of continuous-control benchmarks.
The results convincingly demonstrate that mixture policies improve exploration and stability, especially under high-entropy or multimodal reward settings.

### Weaknesses
Some of the theoretical contributions are mainly extensions of established analysis rather than entirely new formulations.
The results build on well-known principles of entropy-regularized optimization and existing variance-reduction techniques, providing thoughtful refinements rather than foundational changes.
That said, the extensions are clearly presented and meaningfully deepen the understanding of mixture policies in entropy-regularized reinforcement learning.

### Questions
1. How sensitive are the results to the number of mixture components $K$ and to entropy coefficient $\alpha$?
2. Could similar robustness hold for non-Gaussian or discrete mixture families?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper analyzes mixture policies in entropy-regularized actor-critic algorithms, addressing the long-standing issue that mixture policies have not been practically effective due to the lack of efficient reparameterization (RP) gradient estimators. This paper propose a Marginalized Reparameterization (MRP) estimator for mixture policies, which is proven to have lower variance than the standard likelihood-ratio (LR) estimator. Through extensive experiments across synthetic bandits and diverse continuous control benchmarks, the paper demonstrates that mixture policies trained with MRP are stable, competitive with Gaussian policies, and particularly useful in environments with multimodality.

### Strengths
This paper provides rigorous theoretical analysis, including proofs that mixture policies achieve better or comparable objective values than base Gaussian policies and are more robust to high entropy regularization.

This paper covers a wide range of environments, from synthetic bandits to complex robotic control tasks to demonstrate effectiveness.

### Weaknesses
Mixture policies require more parameters (e.g., 5-component policies have 15 outputs vs. 2 for base Gaussian policies) but this paper does not provide a detailed analysis of computational costs

This paper briefly contrasts mixture policies with implicit policies (e.g., diffusion models) but does not include empirical comparisons on benchmarks.

While this paper theoretically proves that the MRP estimator has lower variance than the likelihood-ratio (LR) estimator, it overlooks practical constraints of MRP. For instance, marginalizing over mixture components may implicitly amplify the impact of outlier components (e.g., components with extremely low weights but large parameter deviations), which could introduce hidden instability in long-term training. Additionally, the paper does not test MRP's robustness to hyperparameter variations.

The ablation study on component numbers (2, 5, 8) shows noisy results but does not address the risk of component collapse. This paper does not report whether components retain distinct roles (e.g., specializing in different sub-policies) throughout training or if they degenerate into redundancy. This ambiguity undermines claims about the mixture policy's flexibility in exploring diverse action modes.

### Questions
Could you provide a more detailed analysis of the computational overhead (training/inference time, memory usage) of mixture policies with MRP compared to standard Gaussian policies across different environments?

Given the noisy results on the effect of component numbers, do you have any heuristic or theoretical guidance for selecting the optimal number of components for a given task?

 Why the multimodal exploration of mixture policies is said to be more effective than the exploration of Gaussian policies in such settings? Can you given a more detailed explanation?

The MRP estimator relies on marginalizing over mixture components. Have you observed cases where outlier components (with low weights but extreme parameter values) distort the gradient signal, and if so, how might this be mitigated?

Diffusion policies have shown stronger multimodal modeling capabilities than GMMs in robotic tasks , particularly in position-controlled systems. Could you compare the mixture policy’s performance with diffusion policies on more continuous control tasks?

Component collapse is a known issue in GMMs. Did you track the divergence of component parameters during training? If components converged to similar distributions, how does this affect the mixture policy’s ability to explore diverse actions, and what safeguards can be added?

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
This paper revisits the use of mixture policies in entropy-regularized reinforcement learning (RL), specifically within Soft Actor-Critic (SAC). The authors argue that mixture policies offer greater flexibility than unimodal policies (e.g., Gaussian) but have been underexplored due to the lack of effective reparameterization gradient estimators. The paper makes three main contributions: (1) theoretical analysis showing that mixture policies achieve better optimal stationary points and exhibit greater robustness to entropy regularization as compared to Gaussian policies; (2) proposing Marginalized Reparameterization (MRP) estimator, which marginalizes over mixture weights to provide an unbiased, low-variance gradient estimator; and (3) empirical validation across synthetic bandits, classic control, and large-scale benchmarks. The results demonstrate that mixture policies with the MRP estimator are competitive with or superior to Gaussian policies on standard benchmarks, with significant improvements in environments with unshaped rewards.

### Strengths
Originality
- The theoretical results for the marginalized reparameterization (MRP) estimator provide novel insights into how policy parameterization affects stationary points of the non-convex entropy-regularized objective.
- Propositions 3.1-3.4 rigorously establish that mixture policies achieve at least as good or better stationary points compared to base policies and retain stationary points under higher entropy regularization, where Gaussian policies diverge.

Quality
-  The experimental evaluation is comprehensive and methodologically rigorous, spanning across diverse domains with appropriate statistical reporting, including 95% bootstrap confidence intervals. 

Clarity
- The paper is clearly written with logical progression from motivation through theory to empirical validation, along with a discussion on the limitations.

Significance
-  The finding that mixture policies significantly outperform base policies in unshaped-reward environments provides valuable practical guidance about when mixture policies are helpful.

### Weaknesses
1. Assumptions 4.5 and 4.6 in Proposition 4.7 require specific smoothness properties of the reward function and importance sampling variance relationships that are neither verified empirically nor characterized in terms of when they hold in practice. The variance reduction analysis focuses on multimodal bandits and univariate actions, with only a remark (Remark 4.9) suggesting multivariate extension is possible. The gap between the bandit theory and MDP experiments is substantial, and it remains unclear whether the variance reduction guarantees meaningfully apply to the complex high-dimensional control tasks tested.

2. Figure 4 shows that SGM-MRP (mixture policy) is only marginally better on average across MuJoCo, DMC, MetaWorld, and MyoSuite, with the main benefits concentrated in specific MetaWorld tasks. Given that the experiments use hyperparameters from the SAC paper,  the gains might improve with proper tuning, yet the paper does not investigate this.

3. The paper does not report the computational overhead of the MRP estimator compared to standard RP for Gaussian policies, which might be critical for practical adoption. The choice of five components appears arbitrary, and while Appendix F.3 provides limited ablation, there is no principled guidance on selecting the number of components for a given task. 

4. The paper does not compare against other flexible policy classes like beta policies, heavy-tailed policies, or recent implicit policy methods beyond a brief discussion in the introduction and related work. 

5. The claim that mixture policies enable "mode-directed exploration" is intuitive but not rigorously quantified through metrics such as state coverage or exploration efficiency, for instance, in toy gridworld domains.

### Questions
1. Can the authors empirically validate Assumptions 4.5 and 4.6 on representative tasks from the considered benchmarks? Specifically, is it easy/difficult to verify whether the reward functions satisfy the required smoothness conditions and whether the importance sampling variance relationships hold during training?

2. What is the computational overhead of the MRP estimator compared to the standard RP estimator for Gaussian policies? Please provide wall-clock time comparisons across benchmarks.

3. Can the authors provide principled guidance on selecting the number of mixture components, perhaps based on task characteristics such as action dimensionality, reward structure, and/or state space complexity?

4. Can the authors provide quantitative metrics for exploration efficiency, such as state coverage or diversity of trajectories, to rigorously validate the claim that mixture policies enable better mode-directed exploration?

### Soundness
3

### Presentation
3

### Contribution
2
