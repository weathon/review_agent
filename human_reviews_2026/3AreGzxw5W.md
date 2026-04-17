# Ctrl-Z Sampling: Diffusion Sampling with Controlled Random Zigzag Explorations

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Diffusion models have shown strong performance in conditional generation by progressively denoising Gaussian samples toward a target data distribution. This denoising process can be interpreted as a form of hill climbing in a learned representation space, where the model iteratively refines a sample toward regions of lower noise and higher quality.. However, this learned climbing often converges to local optima with plausible but suboptimal generations due to latent space complexity and suboptimal initialization. While prior efforts often strengthen guidance signals or introduce fixed exploration strategies to address this, they exhibit limited capacity to escape steep local maxima. In contrast, we propose Controlled Random Zigzag Sampling (Ctrl-Z Sampling), a novel sampling strategy that adaptively detects and escapes such traps through controlled exploration. In each diffusion step, we monitor the trajectory of quality-scores of predictions over denoising steps, given a reward model that serves as a surrogate for the underlying sample quality, and identify plateaus as local optima along this trajectory. Upon such detection, we inject noise and revert to a previous, noisier state to escape the current plateau. The reward model then evaluates candidate trajectories, accepting only those that offer improvement, otherwise scheming progressively deeper explorations when nearby alternatives fail. This controlled zigzag process allows dynamic alternation between forward refinement and backward exploration, enhancing both alignment and visual quality in the generated outputs. The proposed method is model-agnostic and also compatible with existing diffusion frameworks. Experimental results show that Ctrl-Z Sampling consistently improves generation quality across different NFE budgets compared to the original sampler.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a technique called "Control Z Sampling," which reframes the sampling process in conditional diffusion models as a form of hill climbing in a reward space. When the sampling process stagnates at a local optimum according to a reward model (e.g., ImageReward), the algorithm injects controlled noise and reverts to a noisier latent state to explore alternative trajectories. This process creates a backward-forward or "zigzag" exploration path intended to escape local optima. The method is presented as a model-agnostic plugin compatible with existing diffusion frameworks, requiring no retraining. Experiments show modest improvements in evaluation metrics compared to baselines. However, these gains come at a significant computational cost, requiring approximately 7.72 times the neural function evaluations (NFEs) on average while also requiring evaluation of a second reward model.

### Strengths
* The paper is well-written and mostly clear in its presentation.

* The idea of conceptualizing conditional diffusion sampling as a hill-climbing problem is interesting.

* The process of combining simulated annealing-like adaptive noise re-injection with a best-first search-like method to explore alternative trajectories is a reasonable blend of explore and exploit techniques.

* The quantitative and qualitative results clearly show that the method is having an effect, with moderate gains in performance metrics.

### Weaknesses
* The most significant weakness is the substantial computational cost, which averages roughly 7.72 times the neural function evaluations (NFEs) of the original process. In my opinion, this massive increase in energy and clock time makes the method practically unusable for what are only modest gains in output quality. Other hyperparameter choices for the algorithm would make this NFE multiplier even greater.

* Following up on the above point, the technique currently feels like a brute-force example or the first step in a research process, where one demonstrates how a search-based method *could* work but is then expected to introduce an efficient implementation to contrast with it. The paper has its heart in the right place but is essentially missing the efficient algorithm that would make the core concept viable.

* The fundamental re-framing of diffusion sampling as "search" or "hill climbing" is interesting but debatable, as diffusion models are fundamentally about sampling from a distribution, not necessarily about moving toward a single global optimum. A method that is too effective at achieving a global maximum could actually cause mode collapse by severely reducing output diversity.

* Related to the above point, the authors seem to be performing search in "reward space," but the structure of the diffusion sampling problem can easily lead a reader to think that the search is being performed in *probability* space. This distinction is not as clear as it could be in the text or Figure 1. Indeed, one comes away with opposite impressions from reading the abstract ("a form of hill climbing ... where the model iteratively refines a sample toward regions of higher **probability**") versus Section 4 ("as a hill-climbing process ... where each denoising step moves the sample toward regions of higher **reward**") [emphasis mine].

* The method primarily uses the ImageReward model, which is trained on clean images, to evaluate states at intermediate steps of the denoising process. This likely places the reward model's input out-of-distribution, which may impact the reliability of the guidance signal.

* The evaluation seems limited to latent space sampling and does not include testing in pixel space for computational reasons. This further limits the practical applicability of the work, as the field is increasingly moving back toward pixel-space diffusion.

* Some relevant work (e.g. Direct Noise Optimization [1], Diffusion Tree Sampling [2]) should be discussed in the context of the authors' method and ideally compared to if possible.

[1] Tang, et al (2024). Inference-Time Alignment of Diffusion Models with Direct Noise Optimization. 

[2] Jain, et al. (2025). Diffusion Tree Sampling: Scalable inference-time alignment of diffusion models.

### Questions
1. The authors use a deterministic sampling process. Given that a significant amount of the benefits (exploring alternative paths, breaking out of local optima) could be inherently achieved more cheaply via stochastic sampling methods (Langevin Monte Carlo, DDPM), why was a comparison to or an evaluation on an established stochastic sampling process not included?

2. I would like to see the authors respond to the issues raised in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Ctrl-Z, a novel inference strategy for diffusion models to address semantic misalignment caused by convergence to local optima during conditional generation. The method leverages a reward-guided, controlled zigzag exploration process, dynamically alternating between forward refinement and backward steps with adaptive noise injection to escape optimization plateaus. Some experiments are provided to validate the effectiveness of the method.

### Strengths
* The approach is model-agnostic and substantially improves semantic alignment on text-to-image benchmarks, demonstrating a practical and efficient balance between exploration and computational cost.
* The writing and results are clear to the readers.
* Thorough ablation studies are conducted to show the robustness of the method.

### Weaknesses
* In the abstract, the authors state that denoising is analogous to climbing a probability hill, and that the process may get stuck at some local maxima. Why do you consider these local maxima suboptimal for generation? Could you visualize the hill and illustrate the imperfect samples corresponding to these local optima? My understanding is that if the score function has learned a distribution with local optimal points, then generating samples around these points is reasonable, since they belong to the true distribution. In that case, I do not quite understand the motivation for manually pushing the denoising process away from these local maxima.
* In Lines 43–46, you state that semantic misalignment and global inconsistency stem from local optima. Could you provide some evidence to support this claim?
* The computational cost increases to 7.72 times that of the original diffusion generation process, imposing a significant computational burden in industrial or commercial settings.
* You use ImageReward as the reward model during inference and also as the evaluation metric. This result is not convincing, since the evaluation metric is exactly what your method optimizes during generation. It is therefore expected that your method performs well on this metric. However, does this optimization degrade performance on other metrics? In addition, you do not report the FID score, which is the most commonly used metric in image generation. Although FID has some known limitations, I did not see any justification from the authors for omitting it.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose Ctrl-Z Sampling, a sampling strategy for diffusion models. The method targets a key limitation of standard denoising processes, which the authors liken to hill climbing: the tendency to become trapped in local optima. This can yield results that are semantically misaligned with the given condition or that lack global consistency. The core idea is an adaptive, reward-guided exploration mechanism. During sampling, a reward model monitors generation quality. If it detects optimization stagnation (suggesting a local trap), the method injects noise to roll back to a previous, noisier state. From that state, the reward model evaluates multiple candidate trajectories and selects a path that shows improvement. The rollback depth can be increased adaptively when shallow exploration fails. The authors report that this controlled “zigzag” exploration improves both generation quality and conditional alignment.

### Strengths
- The paper provides a clear and intuitive characterization of the problem, framing diffusion sampling as "latent space hill-climbing." This analogy effectively explains why samplers get stuck in local optima, leading to semantic mismatches or global inconsistencies. This narrative provides a unified motivational framework that helps readers grasp the necessity of the proposed strategy.
- The primary innovation lies in its feedback-controlled mechanism that determines when to explore and how deep the exploration should be. Unlike methods with fixed perturbation frequencies or amplitudes, Ctrl-Z Sampling uses reward stagnation as a trigger for exploration. It then progressively deepens the rollback (via DDIM-based noise injection) until a superior candidate trajectory is identified. This creates a controlled "forward-backward-forward" zigzag path.

### Weaknesses
- A core component of the method is the use of a reward model to detect stagnation ("Local Maxima Detection"). However, the paper does not propose a new reward model specifically designed or targeted for this task. Furthermore, the paper lacks discussion on several critical aspects of this component: 1) The stability and noise sensitivity of the chosen reward model. 2) The potential impact of reward misjudgments (false positives/negatives for stagnation) on the exploration path. 3) Any systematic biases in generation quality that might be introduced by the specific choice of reward model.
- The paper relies heavily on the "hill-climbing + rollback" intuition but provides no theoretical analysis to support it. Key guarantees are missing, such as convergence analysis, proof of escape from local optima, or bounds on the expected improvement. For instance, the paper does not provide any theoretical analysis of the relationship between the probability of successfully escaping a local optimum and the method's hyperparameters (e.g., the depth budget, the threshold $\delta$, the window $\lambda$).
- Although Ctrl-Z claims "controlled" exploration, its compute remains high: average NFEs are 7.72$\times$ (SD-2.1) and 8.79$\times$ (Hy-DiT) versus SOP’s 9.00$\times$, while quality gains over SOP are modest/inconsistent.

### Questions
Regarding the reward stagnation criterion, have you considered replacing the fixed threshold $\delta$ with a more flexible approach such as a dynamic or schedule-based threshold, a relative improvement test, or an adaptive gate based on reward variance or uncertainty? In regions where the reward values plateau at high scores, a constant $\delta$ might incorrectly interpret small but meaningful improvements as stagnation and trigger unnecessary rollbacks. How do you address this potential issue or prevent such misclassifications?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Ctrl-Z sampling, a sampling method for diffusion models designed to improve output quality by escaping local optima during the denoising process. The approach evaluates the generated state after each denoising step to determine whether it is a local optimum; if so, it injects an adaptive amount of noise to revert the process to a noisier state. This backward exploration mechanism is intended to leads to better text alignment and enhanced visual quality, while costing approximately 7.72× more function evaluations (NFEs) compared to standard diffusion sampling. The effectiveness of the proposed method is demonstrated through experiments on standard text-to-image benchmarks.

### Strengths
* Interesting problem: The paper addresses an interesting and relevant issue in diffusion models — the tendency of sampling trajectories to stagnate or over-converge — which could be impactful. 
* Extensive empirical evaluation: The paper includes a comprehensive set of experiments across multiple text-to-image benchmarks, providing both quantitative metrics and qualitative visualizations.
* Thoroughness and effort: The work is detailed and carefully executed, with significant experimental effort and a long appendix that documents settings, ablations, and implementation details.

### Weaknesses
- The problem studied in this paper is not well explained or sufficiently justified. From the outset, in both the abstract and introduction, the authors describe how the denoising process converges to local optima. For example, lines 43–46 state:

      “Despite their strong generative performance, diffusion models often exhibit semantic misalignment or global inconsistency in conditional generation. These issues arise when the denoising process converges to local optima that prioritize local visual plausibility over semantic relevance or structural coherence.”

    The main concern is that it is unclear what the term *local optima* means in the context of diffusion models, as no explicit optimization occurs during inference. The authors could have more effectively explained and motivated this concept by framing it as follows: the denoising process is an iterative refinement procedure in which samples may occasionally collapse to suboptimal regions of the data manifold (for example, producing blurry or incomplete images). This underlying phenomenon of sampling stagnation or collapse could then be described as becoming stuck in a local optimum.


- The procedure for detecting local optima during sampling is not well justified. The paper lacks an intuitive explanation or illustrative experiment demonstrating why the criterion defined in Equation (5) effectively identifies states corresponding to local optima.

- The experimental results are not particularly convincing. The quantitative improvements over the baselines, especially on the SOP dataset, are marginal and do not clearly demonstrate a significant advantage. While the qualitative examples in Figure 3 more effectively illustrate the benefits of Ctrl-Z sampling, the additional qualitative results provided in the supplementary material appear less compelling and more comparable to benchmarks. 

- Considering the quality of the results, I am concerned that the approximately 7.72× increase in NFEs may not justify the relatively modest gains reported in the experiments.

### Questions
1. The definition of $\Phi$ provided in line 139 and equation 1 conflict as the output of $\Phi$ is $x_0$.
2. Line 153, it's better if "clean image" was replaced with "clean sample" as image was never used up to this point.
3. Line 157: Why is the sample estimate $\hat{x_0}$ a good proxy for the final clean sample $x_0$?
4. Line 161: The concept of "inversion" is not a very popular concept. It would be useful to add an explicit description of inversion by something as short as "...inversion (re-noising)...".
5. Line 181: "forward denoising" is probably incorrect, right? In the forward process, noise is added and in the backward, noise is removed.
6. Is the "Resampling" method discussed in Figure 1 the same as the stochastic SDE sampler?
7. Line 210: "Zigzag Sampling takes a backward step along the direction of un-
conditional generation, alternating between conditional and unconditional denoising to better inject
conditioning signals." --> what does conditioning refer to here? condition on what exactly?
8. Line 223: It would be beneficial to clarify what is meant by "latent trajectory" as it's not a standard term used in the diffusion community.
9. Why does the criteria of equation 5 make sense?

### Soundness
3

### Presentation
2

### Contribution
2
