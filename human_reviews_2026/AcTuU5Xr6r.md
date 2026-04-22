# Evaluating the Role of Great Pre-trained Diffusion Models in Few-shot Phase: Warm-up and Acceleration

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Due to the customized requirements, few-shot diffusion models have attracted much attention. Despite the empirical success, only a few works analyze few-shot models, and they do not involve the fast few-shot optimization process. However, fast optimization is important and necessary in quickly responding to users. In this work, for the first time, we evaluate the role of each operation in the optimization process and prove the convergence guarantee for few-shot diffusion models. A standard operation for the few-shot model is only fine-tuning some key parameters to avoid overfitting the limited target dataset. We first show that this operation is insufficient from empirical and theoretical perspectives. More specifically, we conduct real-world few-shot fine-tuning experiments with underfitting and overfitting bad pre-trained models and show that the few-shot results are heavily influenced by these bad models. Theoretically, we also prove that the few-shot phase can not learn the ground-truth parameters and suffers a small gradient when using a bad pre-trained model. Based on these observations and theoretical guarantees, we highlight the importance of a great pre-trained model by showing it can warm up few-shot models and lead to a strongly convex landscape for few-shot diffusion models. As a result, the few-shot model fast converges to the ground-truth parameters. In contrast, we show that with a bad initialization, the pretraining phase requires large optimization steps to converge. Combined with the above results, we explain why few-shot diffusion models only require a few optimization steps compared with the pretraining phase.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses an important and empirically observed phenomenon: the rapid convergence of few-shot diffusion models like DreamBooth. The authors claim to be the first to provide a theoretical optimization guarantee for this *fast convergence*. The core argument is that a "great" pre-trained model provides a "warm-up" by transforming the few-shot optimization landscape into one that is *strongly convex*.

### Strengths
Idea is intriguing as it aligns well with the empirical success of methods like LoRA, which also involve fine-tuning linear adaptations.

### Weaknesses
1. The paper's theoretical guarantee hinges on an extremely simplified experimental setting: the data is assumed to lie on a linear subspace, and the shared latent distribution is modeled as a simple 2-mode Gaussian Mixture Model (GMM). While this "toy problem" setting certainly makes the theoretical analysis tractable (i.e., allows for a closed-form score function), it is a far from the complex, high-dimensional, and non-linear manifolds of real-world image data. It is highly questionable whether the finding of strong convexity in this trivial setting holds any meaningful implication for the optimization of large-scale models like Stable Diffusion on real images. 

2. The proposed problem can be solved from existing literature: The primary concern is whether the paper's score-matching-based analysis is necessary at all. The objective function for diffusion models, Denoising Score Matching ($\mathcal{L}_{DSM}$), is essentially equivalent to a weighted **Mean Squared Error (MSE)** problem, where the model is trained to predict either the original data or the added noise from a corrupted input.* Therefore, this entire problem can be **re-formulated**: it is not about the convergence of a "Score Matching" function, but about the fine-tuning of a **pre-trained regression model on an MSE loss**.* There is already a vast body of existing literature in transfer learning theory that explains why and how the fine-tuning of pre-trained regression models converges quickly (e.g., good feature extraction, flat minima, etc.).The authors fail to justify **why this existing body of literature is insufficient** to explain their observations. They must clearly articulate what *fundamentally new insight* their complex, score-function-based analysis provides that could not have been derived by applying standard transfer learning theory to the equivalent MSE problem.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on few-shot adaptation of pre-trained diffusion models, studying the theoretical aspect of fast adaptation of diffusion models with only a few optimization steps. The observations they found include: if the pre-trained model is not good, few-shot adaptation cannot find the groundtruth parameters, yielding a bad model but with small gradients. They contrasted their work with several previous related works aiming for few-shot diffusion model guarantees, where they positioned their paper to explain about a few optimization steps, which was not done before. They proved: a few-shot adaptation with a great pre-trained model converges to the groundtruth parameters with convergence guarantee, with few optimization steps.

### Strengths
- The problem of theoretical analysis of the few-shot diffusion fine-tuning they tried to tackle looks interesting.

### Weaknesses
- First, the use of the term "great" in the title sounds too informal. Use a more objective and formal term instead.

- Sec.4 about the failure of few-shot fine-tuning of bad pre-trained (overfit, underfit) models looks too obvious. Do you need to do this?

- Sec. 5 aims to show with bad pre-trained models the fine-tuning cannot find the groundtruth parameters by extending (Yang et al. 2024) from gaussian $q_z$ to a mixture of gaussians.

- There are a bunch of assumptions made that sound overly simplified and unrealistic, such as the mixture of Gaussians with the means exponentially decaying with time and opposite signs; the mixture of two isotropic Gaussians (not shown to be extended to general mixtures), and the particular choice of the one-layer tanh score neural network assumption.

- In Thm 5.5, they show the gradient is bounded, but it all depends on some unknown constants that are unknown. So it is a logical leap to argue that the gradient is small.

- The assumption 3.1 says that the latent is shared between pre-trained data and the target data distributions, and they are linearly related to the latent. Let's say this assumption is true considering that the same assumption was made in (Yang et al. 2024). But the follow-up assumption 5.1 of the mixture of two gaussians with the same isotropic covariances and exponentially decaying means with opposite signs, looks too contrived and simplified. Also they rely on the particular (simple one-hidden tanh layer networks) score neural network defined in the unlabelled equation right below after Eq.(3). All subsequent theorems are based on these assumptions, and considering that the paper is aiming to become a theoretical paper with lack of strong empirical evidence, it is not a solid paper. 

- Sec. 6, the argument about the strong convex few-shot optimization, is also relying on those particular (unrealistic and overly simplified) assumptions. 

- In theorem 6.4, the convergence also depends on the scale of the $\kappa$ value. Eg, if $\kappa >> 1$, then it is hard to conclude that you can attain optimal values in just a few steps. In other words, the practical significance of this theoretical result is doubtful.They did some simulations in Table 2, but it looks overly contrived.

- Overall, reading this paper gives me an impression that they are building a sandcastle in the air on top of unrealistic and overly simplified assumptions. Any empirical evidence, eg, how SD1.4 fits in their assumptions, would make their argument convincing.

### Questions
See questions in the weakness section.

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
2

### Summary
The authors provide theoretical backing and insights to the question why few-shot diffusion models require only a couple of optimisation steps, and how the convergence behaviour as well as resulting performance after adaptation is affected by different ‘levels of quality’ of the pre-training (i.e. initialisation) stage. The presented theoretical insights are additionally motivated and supported through a few experiments to reiterate the well-known empirical findings.

### Strengths
**Originality & Significance:**  
- The authors do a good job in outlining why their analyses matter, and that this work tries to provide theoretical backing for empirically observed insights
- The theoretical analyses and backing are first motivated through some empirical analyses with current popular models, which provides a nice basis to ‘grasp’ what the authors aim to theoretically show later

**Quality:**  
- Their work is placed well within related efforts, and the specific gap the authors tackle is clearly presented and justified
- Combination of the empirical and theoretical angle regarding what a ‘good/great’ and ‘bad’ pretrained model is opens the insights up to a wider audience, especially given the provides visuals

**Clarity:**
- The work is mostly well written and easy to follow, and a good mix of illustrations, text and equations

### Weaknesses
**Major:**  
- Interesting fact/observation regarding the difference of the sampling process that is used within the diffusion model unfortunately remains unexplored, and no comment is made how/whether this could affect the statements/conclusions drawn regarding adaptation capabilities; see questions. 
- Experimental results provide limited new insights: While I understand the intention to (again) shot that badly trained models adapt worse than well-trained ones, the experimental side of the results seems very expected and yields little-to-no surprise: The fact that very overfitted and/or undertrained models don’t adapt well/fast is quite well known across the few-shot learning community, and several works have been tailored to combat this (e.g. avoiding supervision collapse in works like Doersch et al., NeurIPS2020 or Hiller et al., NeurIPS2022, etc.)
- Some remaining questions on how the presented simplified setup relates (and generalises) to the usually more complex, real-world experiments – see questions.

**Minor:**  
- Consistency in formatting of formulas would be improved in places, e.g. eq in l.191 keeps the $\sigma^2$ in the denominator of each term, whereas the closely-related eq in l.200 uses a pre-factor of $1/sigma^2$; Keeping them in the same form would make it easier for the reader so directly see the (almost) identical formulations
- Choice of words could be improved to avoid confusion in some settings, see questions.
- (minor suggestion) Structure of manuscript could be improved, as many sections have only one subsection (e.g. 3.1, 5.1 but no second); I’d recommend to either make it 3.1 and 3.2, or simply keep it all in the main one (might be down to preference though)

### Questions
**Main Questions:**  
- The authors present their ‘interesting observation’ regarding the difference in adaptation quality that results from a deterministic sampling process (in SD3) vs. the more stochastic process in SD1.4. This observation is unfortunately only mentioned in the text with results shown in Table 1, but stays entirely unexplored beyond this: I’d be quite curious whether it’s possible to include this into their analysis and/or provide any theoretical backing/background for this?  
$\rightarrow$ I assume this is due to the additional energy that is introduced to the diffusion process when adding additional noise during the sampling process, which from an empirical standpoint provides more chances for the model to correct initial errors; and in some sense to then ‘cover more ground’ during generation time  
$\rightarrow$ However, given that the authors attempt to provide insights from a theoretical standpoint, I’d like to hear some comments on their thoughts, and how / whether this would influence their current derivations/analyses & conclusions!   
$\rightarrow$ Is there a way to quantify/formulate this ability of apparently being able to compensate for worse pretraining?  
- The authors show the difference in the loss landscape (re. convexity) via the sign of the Eigenvalues, and go on to deduct the convergence properties;  
$\rightarrow$ However, although a problem can be convex and therefore converge eventually, the actual ‘convergence speed’ isn’t necessarily ‘fast’ or ‘in few steps’ when a method like gradient descent is used  
$\rightarrow$ Wouldn’t the *condition number* of the Hessian (e.g. via the ratio of max/min Eigenvalues) also play a big role? I’d like to hear the authors’ thoughts on this, and to what extent this is already factored into their approach, and whether they could provide additional insights.  

- I’m wondering whether the authors could provide some further insights to what extent their theoretical insights based on the simplified 2-modal distribution (GMM) would generalise (or could be generalised) to more complex real-world distributions; I can see some additional information provided in appendices B.2 – B.4, but it seems to mostly discuss what aspects other works have or haven’t covered yet;  I understand that simplified models as well as assumptions are required for theoretical analyses, but it might be interesting for the wider audience to have some insights/comments on how the authors see the remaining gap between their findings and current models used in practice 

**Minor / Recommendations:**  
- The authors often use the expression ‘large optimization steps’ (e.g. abstract l. 029/030); To me, this seems more related to the optimizer needing to ‘move far’ to get to the solution, which I think would be better expressed by “many steps” (if the authors want to use ‘steps’), given that most gradient-decent-like optimisers operate with a fixed step size (albeit rescaled internally);  
- Recommendation: The authors repeatedly state that “the model can not learn the *ground-truth* parameters” (e.g. l.024, l.084/085, etc.). While I understand that for small toy’ish theoretical setups, a ground-truth set of parameters exists (e.g. via close-form solution or other assumptions), this formulation can be quite confusing for the wider ML audience, given that for the actual ‘practical’ deep-learning-based setting that the authors address here, there usually is no ‘one ground-truth’ solution for the parameters.  
$\rightarrow$ I’d hence recommend the authors to adapt the wording around these statements, e.g. to ‘optimal set of parameters’ / 'near-optimal set of parameters', or sth along these lines.

### Soundness
3

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
This paper investigates the optimization process of few-shot diffusion models. It addresses a critical gap in existing literature by focusing on why few-shot models require only a small number of optimization steps to achieve high performance, rather than merely explaining the sufficiency of limited data. This paper combines empirical experiments with theoretical analysis to demonstrate that the quality of pre-trained models is pivotal: great pre-trained models warm up the few-shot phase, leading to a strongly convex landscape and fast convergence, while bad pre-trained models (overfitting or underfitting) cause failures such as memory phenomena or loss gaps. The paper provides the first convergence guarantee for few-shot diffusion models under simplified assumptions.

### Strengths
1. The paper proves that great pre-trained models induce strong convexity in the few-shot objective function, leading to linear convergence rates for gradient descent (Theorem 6.4). This is the first convergence analysis for few-shot diffusion models, offering a solid foundation for future work.
2. This paper provides a new theoretical lens for understanding few-shot diffusion models, moving beyond data efficiency to computational efficiency.
3. This paper conducts experiments using standard datasets and models to validate theoretical claims.

### Weaknesses
1. The theoretical analysis is built on strong simplifications (e.g., linear subspaces, 2-mode GMMs), and the extension to K-mode GMMs in Appendix B remains intuitive rather than rigorous. Nonetheless, the empirical support and conceptual value of this discussion are sufficient to prevent this limitation from affecting my overall score.
2. The "bad" pre-trained models in the paper are created through extreme and artificial operations, such as overfitting Stable Diffusion 3 (SD3) Medium on only 5 dog images for 1,000 steps. This approach does not reflect real-world failure modes, where bad models typically arise from poor data quality, suboptimal architecture choices, or inadequate training strategies, rather than targeted overfitting.
3. The paper should establish evaluation metrics that are independent of fine-tuning outcomes (e.g., representation smoothness or distribution distance), and through large-scale experiments, quantify the correlation between pre-trained model metrics (such as FID) and few-shot performance, thereby providing actionable predictive guidelines.

### Questions
This paper presents a pioneering theoretical contribution by systematically analyzing the optimization process of few-shot diffusion models, demonstrating for the first time that high-quality pre-trained models induce a strongly convex landscape—enabling fast convergence with rigorous guarantees. Although limitations like the absence of quantifiable pre-training evaluation criteria and simplified theoretical assumptions slightly narrow its immediate applicability, these do not undermine the core innovation, as the work successfully bridges theory and practice, offering valuable insights for rapid customization in generative AI and setting a foundation for future research.

### Soundness
3

### Presentation
3

### Contribution
3
