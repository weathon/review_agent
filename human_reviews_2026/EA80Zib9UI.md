# SAFETY-GUIDED FLOW (SGF): A UNIFIED FRAMEWORK FOR NEGATIVE GUIDANCE IN SAFE GENERATION

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 10, 4, 6, 6

## Abstract
Safety mechanisms for diffusion and flow models have recently been developed along two distinct paths. 
In robot planning, control barrier functions are employed to guide generative trajectories away from obstacles at every denoising step by explicitly imposing geometric constraints. 
In parallel, recent data-driven, negative guidance approaches have been shown to suppress harmful content and promote diversity in generated samples. However, they rely on heuristics without clearly stating when safety guidance is actually necessary. 
In this paper, we first introduce a unified probabilistic framework using a Maximum Mean Discrepancy (MMD) potential for image generation tasks that recasts both Shielded Diffusion and Safe Denoiser as instances of our energy-based negative guidance against unsafe data samples. 
Furthermore, we leverage control-barrier functions analysis to justify the existence of a critical time window in which negative guidance must be strong; outside of this window, the guidance should decay to zero to ensure safe and high-quality generation. We evaluate our unified framework on several realistic safe generation scenarios, confirming that negative guidance should be applied in the early stages of the denoising process for successful safe generation.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper proposes Safety-Guided Flow (SGF), a novel framework that uses Maximum Mean Discrepancy (MMD)-based potentials within a negative guidance framework for diffusion and flow models to produce safe samples. The authors make two primary theoretical contributions. First, they formally prove that the gradient of the kernel MMD potential is equivalent to the repulsive fields used in Shielded Diffusion and Safe Denoiser, thereby suggesting a unified probabilistic framework. Second, the paper provides evidence for the effectiveness of imposing negative guidance in the initial denoising stage by adopting the control barrier theorem. The effectiveness of the proposed method is validated through comprehensive experiments against both training-based and training-free baselines. The evaluation assesses performance on safety and diversity metrics, along with the ablation studies that confirm the theoretical findings regarding the time window of the negative guidance. The experiments demonstrate substantial safety improvements while having a minimal impact on image quality on benign datasets. 
This paper shows its algorithmic significance by proving the existence of a critical time window of negative guidance by proposing a unified probabilistic framework for image generation applying Shielded Diffusion and Safe Denoiser, topics which were previously only dealt with empirically without a theoretical approach. Furthermore, the paper is well structured with systematic and clearly described proofs, supported by extensive experiments.

### Strengths
The contributions of the paper are (1) its outstanding theoretical contribution and (2) its novel application of control theory to provide the first formal safety guarantees in this domain. 
Strong theoretical foundation: The paper rigorously connects previously empirical safety methods (Shielded Diffusion, Safe Denoiser) through an elegant MMD-potential framework.
Novel control-theoretic insight: The use of the control barrier theorem to justify time-varying safety guidance represents an innovative cross-disciplinary contribution, providing the first formal safety guarantees for generative flows.
Clarity and structure: Proofs and derivations are logically organized, with intuitive interpretations. The writing effectively bridges probabilistic modeling and control theory.
Comprehensive experiments: Evaluations include both safety (nudity suppression, memorization) and generative diversity metrics, supporting the theoretical claims.
Empirical alignment with theory: Ablation results clearly demonstrate the effectiveness of early-stage negative guidance, validating the proposed “critical window” theory.

### Weaknesses
The theoretical analysis assumes a weak base drift near the boundary layer, which is reasonable for late denoising stages but could be complemented by a short discussion on its general applicability across different model configurations. The computation of the MMD potential may introduce some inference overhead, though it is likely negligible compared to the overall generation time. While current experiments focus on visual safety domains such as nudity suppression and memorization, extending the framework to more abstract or semantic safety concepts (e.g., copyright or bias) would be a valuable direction for future work, and a brief discussion on this point would further strengthen the paper.

### Questions
Regarding the theory, clarification is needed on the following points: 
- In Section 4.4, the proof relies on the strong assumption that the base drift \tilde{f} in the boundary layer has a small effect, which is reasonable in the generative model. While the paper reasonably argues that this drift is weak in the final stages, it does not sufficiently address the strength of the gradient of E. The magnitude of the MMD-based guidance depends on the density of the unsafe set. I wonder whether a certain prompt (especially planned adversarial) or the specific data distribution of the unsafe set could lead to cases where \tilde{f} overpowers the safety guidance, i.e., the gradient of E. An adversarial prompt can intentionally steer \tilde{f} (even if small in the final stage) with high precision toward a sparse region in the data distribution of the unsafe set where the magnitude of the gradient of E would also be minimal. Could there be such cases?
- In Section 4.4, a decreasing weight is defined, and the authors interpret this to mean that earlier safety guidance is better. While this is intuitive for standard diffusion, it would be beneficial to specify the exact conditions for this assumption to hold. Are there any non-standard flow models or schedules where this assumption might be violated?
Regarding the experiments, the following should be addressed:
- It seems SGF may introduce an inference cost at sampling time due to the calculation of the MMD gradient at each step. It would be beneficial for the readers to see how this cost scales with the size of the unsafe set, and a comparison with other baselines in terms of inference cost. 
- In addition to the comment regarding sparse regions in the data distribution of the unsafe set that may lead to a minimal effect from ||gradient of E||, it seems that the performance of SGF may depend on the quality and size of the unsafe set. It would have been better to also show a sensitivity analysis regarding the quality and size of the unsafe set. 
- Does SGF also show strong performance in domains other than nudity? For example, some critical safety concerns in image generation are conceptually abstract and not visually clustered like nudity, such as copyright infringement, or complex harmful concepts. While the MMD potential measures the distance in a feature space and shows effectiveness in capturing negative guidance when dealing with nudity prompts, it may fail to deal with abstract concepts where the unsafe set is difficult to define. Does the MMD potential-based guidance prove effective in preventing this form of conceptual replication?
Minor comments:
1. Page 9: The second paragraph in 5.4 ABLATION STUDIES, there’s a typo showing ‘later times later times’

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a unified framework for safety guidance in generating images by using the gradients of an MMD metric to steer the sampling trajectory away from unsafe images. They then perform experiments showing that this strategy works and perform theoretical analysis to show that this work subsumes some prior works.

### Strengths
* Tackles an important problem
* Section 4.2 and 4.3 shows that the proposed method subsumes prior work.
* Section 4.4 presents an interesting analysis.
* Empirical studies seem appropriate in illustrating the effectiveness of the proposed approach.

### Weaknesses
## Primary concerns
1. How do you pick $s_c$? One of the criticisms of Safe Denoiser is that they pick the interval to apply guidance on heuristically. Is this not the same?
2. What is the compute cost of estimating the MMD and likewise autograd cost for calculating the gradient wrt $\boldsymbol x$? This seems like it could become very expensive once as the size of the unsafe reference dataset grows. How large does $\mathcal D^-$ need to be for the distance to work well, clearly a degenerate singleton distribution should work poorly, right?
3. If we have the $h$ control-barrier function in Section 4.4 why bother with MMD? Wouldn't it be easier to use standard gradient guidance *a la* [1-7] with $h$ instead of calculating the MMD.
4. The **largest** weakness in this paper in my mind is that wouldn't standard gradient guidance with the *control-barrier* function work just as well? I feel like any one of the strategies from [1-7] (there are more papers on this topic but I just listed a few notable ones. For a more complete list of such methods I refer the authors to [2, Figure 5]). If the authors can successfully argue why MMD (or really any probability distance) is more useful for the end goal of safe generation than just using standard gradient guidance with the *control-barrier* function I will raise my score, otherwise I will retain a **reject**.

## Minor comments
* Some notes on Section 3.1. Why $\epsilon_\theta(x_t, t)$ and not $\mathbb [x_1 | x_t]$ to be more in line with the seemingly preferred notation?
* Likewise I would say that the equation in line 143 is more accurately described as an optimal-transport formulation of flow matching with Gaussian source distribution.
* Moreover, in line 50, why follow $s < t$ for Gaussian flow matching? In flow matching literature we commonly set the source distribution at time 0 and the target at time 1. Some clarity on this would be helpful.
* Especially in light of section 4.4 I would just adopt the flow matching conventions for time (which in the reviewer's opinion are far better and less ambiguous)
* For the work on Section 4.4 the authors should mention some other works which look at gradient guidance and show that the impact is greater at earlier times. While not 100% addressing the same topic these works are closely related and I recommend the authors review them, in particular [1 Section 4] and [2, Proposition 5.2]. These are probably the most relevant theoretical results all there are other heuristic observations from people working on general gradient guidance for flow/diffusion models.

### References
[1] Ben-Hamu, Heli, et al. "D-Flow: Differentiating through Flows for Controlled Generation." International Conference on Machine Learning. PMLR, 2024. https://arxiv.org/pdf/2402.14017

[2] Blasingame et al., "Greed is Good: A Unifying Perspective on Guided Generation", NeurIPS 2025, https://openreview.net/pdf?id=s14pdQgoLb

[3] MOUFAD, Badr, et al. "Variational Diffusion Posterior Sampling with Midpoint Guidance." The Thirteenth International Conference on Learning Representations. https://proceedings.iclr.cc/paper_files/paper/2025/file/ed524bb14de1b52c8522b977ded241d3-Paper-Conference.pdf

[4] He, Yutong, et al. "Manifold Preserving Guided Diffusion." The Twelfth International Conference on Learning Representations.

[5] Pan, Jiachun, et al. "AdjointDPM: Adjoint Sensitivity Method for Gradient Backpropagation of Diffusion Probabilistic Models." The Twelfth International Conference on Learning Representations.

[6] Yu, Jiwen, et al. "Freedom: Training-free energy-guided conditional diffusion model." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[7] Chung, Hyungjin, et al. "Diffusion Posterior Sampling for General Noisy Inverse Problems." The Eleventh International Conference on Learning Representations.

### Questions
1. How is Shielded Diffusion described in equation (3) applied to the ODE solver?
2. What does the *Vendi* score measure?
3. What is a control-barrier function, it's not defined well in the paper. Does it have special mathematical properties over some other map $\mathbb R^d \to \mathbb R$
4. In line 313 what is $L$? Is it a map $\mathbb R \to \mathbb R_{\geq 0}$?
5. In equation (5) why does MMD have the hat symbol over it? Is it is because it is empirically estimated?

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
3

### Summary
The paper presents a unified MMD-based formulation of negative guidance for safety-aware generation. The authors show that this new MMD-based framework generalizes previous arts, Shielded Diffusion and Safe Denoiser. Moreover, they provide a control barrier function analysis, finding that the guidance strength must decay so as to remove undesirable nonlocal guidance impact on already safe regions. The proposed method, SGF, is evaluated on several carefully designed experiments, showing that it can effectivively prevent offensive content generation without hurting too much on diversity. Additionally, SGF is also shown to be effective at mitigating the memorization issue.

### Strengths
The theoretical insights are novel. The authors provide a more principled objective for safety-aware negative guidance. Unlike previous methods' formulation based on binary/proximity classification, the proposed SGF views the problem as maximizing a proper divergence metric (MMD) between the undesirable distribution and the generated distribution. The critical time window theory also explains why early stopping is effective.

### Weaknesses
While the theoretical insights are novel, the pragmatic novelty is limited. The paper is mainly focused on 'why it works.' For instance, the MMD-based formulation is sound and novel, but the resulting parametric form of the guidance model itself is effectively identical to SafeDenoiser. The critical time window theory provides why certain stopping parameter is better, but this can be empirically chosen without theory.

### Questions
None.

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
3

### Summary
This work proposes a unified framework for diffusion models with negative guidance for safe generation. The authors apply Maximum Mean Discrepancy (MMD) as a potential function to measure the distance between the current generation and a negative sample dataset, then use its gradient as guidance for the diffusion generation process. By choosing different kernel functions, the framework recovers existing works such as Shielded Diffusion and Safe Latent Diffusion. The paper further investigates the effectiveness of applying guidance at the early stages of the generation process and demonstrates this through empirical studies.

### Strengths
1. Safe generation is critical for real-world applications, making this an important research direction.
2. The proposed unified framework based on MMD guidance effectively covers and connects recent works in the field.
3. The early-stage guidance analysis is insightful and validated through empirical studies.

### Weaknesses
1. Assumption 1(b) needs more justification. While the authors provide an intuitive understanding for the assumption at the final time step, it is unclear how this assumption holds at other time steps and how it should be interpreted more generally. If this is a standard choice in the control barrier function literature, please provide a detailed discussion and relevant citations.
2. Theorem 2 and the ablation study need better alignment and justification. Why does the ASR decrease and then increase across the three time window settings? How does this non-monotonic behavior align with the analysis provided in Theorem 2? In practice, what is the best approach to set the guidance window and budget based on the analysis of Theorem 2?

### Questions
1. How do the negative samples in the potential function calculation impact generation? Specifically, what are the effects of the number and diversity of negative samples on the final output quality and safety?
2. Theoretically, how can we understand the tradeoff between generation quality (distribution approximation error) and safety (constraint satisfaction) under the negative guidance framework?
3. Despite the computational efficiency of the MMD potential function and its ability to cover two recent works, what are the potential drawbacks compared to using other probability measures as the potential function?


I will adjust my score upon clarification of these points.

### Soundness
3

### Presentation
3

### Contribution
2
