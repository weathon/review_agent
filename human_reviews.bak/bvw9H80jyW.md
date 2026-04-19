# ComSD: Balancing Behavioral Quality and Diversity in Unsupervised Skill Discovery

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 5

## Abstract
Learning diverse and qualified behaviors for utilization and adaptation without supervision is a key ability of intelligent creatures. Ideal unsupervised skill discovery methods are able to produce diverse and qualified skills in the absence of extrinsic reward, while the discovered skill set can efficiently adapt to downstream tasks in various ways. Maximizing the Mutual Information (MI) between skills and visited states can achieve ideal skill-conditioned behavior distillation in theory. However, it's difficult for recent advanced methods to well balance behavioral quality (exploration) and diversity (exploitation) in practice, which may be attributed to the unreasonable MI estimation by their rigid intrinsic reward design. In this paper, we propose Contrastive multi-objectives Skill Discovery (ComSD) which tries to mitigate the quality-versus-diversity conflict of discovered behaviors through a more reasonable MI estimation and a dynamically weighted intrinsic reward. ComSD proposes to employ contrastive learning for a more reasonable estimation of skill-conditioned entropy in MI decomposition. In addition, a novel weighting mechanism is proposed to dynamically balance different entropy (in MI decomposition) estimations into a novel multi-objective intrinsic reward, to improve both skill diversity and quality. For challenging robot behavior discovery, ComSD can produce a qualified skill set consisting of diverse behaviors at different activity levels, which recent advanced methods cannot. On numerical evaluations, ComSD exhibits state-of-the-art adaptation performance, significantly outperforming recent advanced skill discovery methods across all skill combination tasks and most skill finetuning tasks. Our code is available at ***.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work introduces a new unsupervised skill discovery method, ComSD. The main idea of ComSD is to decompose mutual information into $I(\tau; z) = H(\tau) - H(\tau|z)$ and to introduce a skill-dependent coefficient $\beta(z)$ to weight the second term. The authors employ a CIC-like contrastive estimator for the second term and a particle-based entropy estimator for the first term. They show that ComSD outperforms previous skill discovery methods (BeCL, CIC, APS, SMM, and DIAYN) in DMC locomotion environments in terms of both fine-tuning and hierarchical RL performances.

### Strengths
- The final objective in Eq. (12) is simple and intuitive.
- The authors evaluate ComSD on both fine-tuning and hierarchical RL settings, where ComSD achieves better performance compared to five previous approaches.

### Weaknesses
- The contribution of the proposed method (ComSD) appears incremental compared to CIC. Especially, [this previous version](https://openreview.net/pdf?id=Z12zA99EFEi) of CIC is almost identical to ComSD, with the same contrastive objective for $H(\tau|z)$ and the same particle-based entropy estimator for $H(\tau)$. The only difference is that ComSD additionally uses a skill-dependent linear coefficient (i.e., using $\beta(z) \alpha H(\tau|z)$ instead of $\alpha H(\tau|z)$), which, in my view, is too incremental to justify an ICLR publication.
- Given the high similarity to CIC, I believe this work requires more thorough comparisons with CIC. Since ComSD uses $\beta(z)$ and $\alpha$ individually tuned for each environment, it should be compared against (the $H(\tau) - \alpha H(\tau|z)$ version of) CIC with individually tuned $\alpha$ to ensure a fair comparison.
- There is a mismatch in CIC performance between this paper and [the original CIC paper](https://arxiv.org/pdf/2202.00161.pdf). Weirdly, their performances are the same in `walker walk` and `walker stand`, but different in `walker run` and `quadruped run`. Could the authors clarify these differences?

Minor issues

- "(Liu & Abbeel, 2021a) first employs the second MI decomposition Eq. 2 for a better MI estimation." in Appendix B: (1) I would recommend using `\citet` for inline citations, and (2) to the best of my knowledge, DADS (Sharma et al., 2020) is the first such method.
- Nitpick: $\propto$ can't be used in Eq. (4) because the right-hand side is not technically proportional to the left-hand side.

### Questions
- As far as I understand, the authors use a fixed latent vector of $z = [0, 0.5, 0.5, \dots, 0.5]$ for fine-tuning experiments. In this case, why don't we just use a fixed coefficient of $\beta = w_{low}$ during training?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a unsupervised skill discovery method named ComSD. With ComSD, the mutual information between states (or pairs of consecutive states) and skill vectors is estimated with the NCE-style contrastive learning loss and particle-based entropy estimation, and the coefficient for weighting the two intrinsic reward terms is designed to differ across skill vectors. They also empirically compare ComSD with baseline methods in two representative skill discovery evaluation settings, skill combination and skill fine-tuning on locomotion simulation tasks including DMControl and provide further quantitative and qualitative analyses.

### Strengths
- The empirical evaluation is done in various settings. Also, compared to the selected set of baselines, the proposed method shows fair performance on the tasks.
- The manuscript basically easy to follow, and Fig. 1 and 2 help readers to understand the proposed approach more quickly.

### Weaknesses
- Skill-based Multi-objective Weighting (SMW), which is the main contribution of this paper in my view, needs rationale for it. Apart from empirical confirmation that it improves the resulting performance, I don't think why the weighting coefficient should be different for different skill vectors and why it needs to be structured like that are not clear. I'm not trying to argue that just a constant coefficient is enough, but I'm rather asking the following questions: Exactly what benefits does that form provide and why? How does having different coefficients for different skills like that affect the overall learning objective?
- I believe the statement about the proposed method's difference from CIC and APS (Sec. 4.1) needs further clarification. If the "contrastive results" employed for the state entropy maximization mean the state representations, CIC uses the state representations from the contrastive learning for estimating the state entropy.
- Some presentation issue: the use of "reasonable" and "unreasonable" for describing the MI estimation methods doesn't seem technical/scientific and is without appropriate backup.

### Questions
Please check out the weakness section.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper points out that one of the major challenges in MI-based skill discovery is balancing two intrinsic reward terms, the state entropy for exploration and the negative conditioned state entropy for exploitation or state-skill alignment. On top of the prior skill discovery method, CIC, this paper proposes Skill-based Multi-objectives Weighting (SMW) to dynamically weight these two reward terms for different skill vectors. The proposed method outperforms prior MI-based skill discovery approaches in both skill composition and skill finetuning experiments on 4 URLB locomotion domains.

### Strengths
* The paper points out the challenge of balancing exploration and exploitation in skill discovery and then provides a simple practical solution, Skill-based Multi-objectives Weighting (SMW).

* The exhaustive experiments demonstrate that the proposed method, ComSD, can discover diverse skills and adapt better to downstream tasks. Especially, ComSD significantly outperforms other methods on most skill combination tasks.

* The paper is well-organized and easy to follow.

### Weaknesses
* In Section 3.3, it is clear that the weighting between two intrinsic reward terms is challenging. However, it is not straightforward to get how the proposed SMW resolves this issue. The choice of the dynamic weighting term in Equation 11 is not justified and explained sufficiently. As this is the main contribution of this paper, it has to be clearly stated and examined in the paper.

* The approach of ComSD seems very similar to CIC except for the dynamic weighting (SWM) and it is unclear how ComSD is different from CIC other than SWM. In Section 4.1, the paper says "For CIC, ComSD follows it in state entropy estimation but first proposes to employ contrastive results for explicit state entropy maximization." but I cannot follow the "explicit state entropy maximization" part. Could the authors elaborate on this more?

* Many MI-based methods show their limited applicability to domains other than simple locomotion environments. Although the strong skill discovery performances on the locomotion tasks are impressive, the proposed approach could overfit to the specific domain. Comparisons on manipulation tasks, as in (Park 2023) would make the claim of this paper much stronger.

* Although the proposed method outperforms prior MI-based skill discovery approaches, recent non-MI-based skill discovery methods (Park 2021, Park 2023) have shown much diverse skill sets. Thus, it is important to compare with these non-MI-based approaches.


---

Although the author response did not address all my concerns, it is clear that ComSD has resolved some issues in CIC implementation, which seems useful for future research. Thus, I increased my rating to borderline accept.

### Questions
Please address the weaknesses mentioned above.


### Minor questions and suggestions

* In Equation 11, $w_{high}$ at the end should be $w_{low}$.

* In Section 4.1, Evaluations, "skill combination in URLB" should be "skill finetuning in URLB".

* Section 4.2 mentions that 6-10 different random seeds are used in skill combination and finetuning experiments. Does this mean a pre-trained skill policy and meta- or finetuned- policy for each random seed? Or, does this mean one pre-trained skill policy and 6-10 different meta- or finetuned- policies? The variances in Figure 3 and Table 1 seem a bit smaller than expected given the high-variance nature of skill discovery methods.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work, the authors present a novel unsupervised skill discovery algorithm called ComSD (Contrastive Multi-Objective Skill Discovery) that uses contrastive learning and entropy estimation to learn skills in an unsupervised fashion in simulated environments. The primary insight from the authors are twofold: using contrastive learning to learn a similarity metric between skill latents z and trajectories tau, and using a coefficient to balance the aspect of quality of the policy vs. the diversity of the explorations.

In this work, the authors first explain their algorithm, which is to first learn an estimated lower bound on the p(tau, z) by using an NCE loss function to find the entropy of a skill, and using a particle based entropy estimator to estimate the entropy of the trajectories. Then, the authors explain their automatically rebalanced multi-objective weighting (SMW) which help the learned skills balance between the quality and the diversity of the learned skills.

The authors then show some experiments on standard unsupervised skill discovery environments from DM-control. Their experiments contain two experiments first for skill finetuning and skill combination using a hierarchical controller. Then, in an ablation experiment they show that both the dynamic weighting and the contrastive encoding are both important for ComSD to perform well. Finally, they show that ComSD also outperforms other comparable algorithms in state diversity metrics.

### Strengths
The work shows a number of positive qualities:

+ The authors motivate their algorithm well; there are only so many ways of doing unsupervised skill discovery, but the authors identify an approach that differs from the baseline approaches and execute on it.
+ The trade-off between diversity and consistency is not explored often enough in the literature, but the authors identify it as an important factor, and consciously optimize for this trade-off across their skills.
+ They evaluate their algorithm on a good number of environments, and against a good number of baselines.
+ The authors also evaluate ComSD on two state diversity metric, which is also quite important for an unsupervised skill discovery metric.

### Weaknesses
The primary flaw with this work is the incomplete comparison with prior state of the art. Unsupervised skill discovery is a crowded field of research, so it is natural that they may miss certain previous works during their literature review. However, a quite relevant work that the authors seem to have missed is [1]. [1]  seems relevant to the authors work because of the (a) information gain expansion used in the work, (b) use of the point-based entropy estimator to compute the reward, and most importantly (c) use balancing coefficient to trade off between diversity and consistency between different skills. Moreover, [1] seem to outperform the standard unsupervised skill discovery algorithms at the time, so it would be good for the authors to add the work as a baseline and/or explain the differences between ComSD and [1] and why they are not compatible for direct comparison.

Apart from that, there are some other issues of the work:
1. The notation r_exploration and r_exploitation look quite similar to each other, and I was thrown off multiple times while reading the explanations. If they could change the notation it may be much easier to read.
2. The SMW objective is not motivated well. It seems like it was pulled out of nowhere. More motivation for this would be apt.
3. The skills are trained for 2M steps, however, there is no clear reasoning for why this number was picked. How does this look in the limit, at say 10M steps?

[1] Shafiullah, Nur Muhammad Mahi, and Lerrel Pinto. "One After Another: Learning Incremental Skills for a Changing World." International Conference on Learning Representations. 2021.

### Questions
See above. Specifically, what is similar vs not about previous works, more motivation for the SMW module, and the behavior in limiting number of environment steps would be good.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
