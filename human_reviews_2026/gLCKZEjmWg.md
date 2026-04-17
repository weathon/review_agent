# Sample by Step, Optimize by Chunk: Chunk-Level GRPO for Text-to-Image Generation

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Group Relative Policy Optimization (GRPO) has shown strong potential for flow-matching-based text-to-image (T2I) generation, but it suffers from two key limitations: inaccurate advantage assignment, and the neglect of temporal dynamics of generation. In this work, we argue that shifting the optimization paradigm from the step level to the chunk level can effectively alleviate these issues. Building on this idea, we propose Chunk-GRPO, the first chunk-level GRPO-based approach for T2I generation. The central insight is to group consecutive steps into coherent 'chunk’s that capture the intrinsic temporal dynamics of flow matching, and to optimize policies at the chunk level. In addition, we introduce an optional weighted sampling strategy to further enhance performance. Extensive experiments show that Chunk-GRPO achieves superior results in both preference alignment and image quality, highlighting the promise of chunk-level optimization for GRPO-based methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Chunk-GRPO, a novel reinforcement learning optimization paradigm for flow-matching-based text-to-image (T2I) generation. The method extends the Group Relative Policy Optimization (GRPO) by addressing two of its core limitations: (1) inaccurate advantage attribution across timesteps and (2) the neglect of temporal dynamics.

Instead of applying optimization at every single generation step, the proposed method groups consecutive timesteps into temporally coherent chunks and performs optimization at the chunk level. The paper also proposes an optional weighted sampling strategy, which biases optimization toward chunks that correspond to higher-noise regions of the generation trajectory.

The authors present both theoretical justification and comprehensive experiments demonstrating improvements in preference alignment and image quality over prior methods such as Dance-GRPO.

### Strengths
1. The paper is well-organized and easy to follow. 
2. Chunk-GRPO is conceptually straightforward, requires only minor modifications to existing GRPO frameworks, and can be easily implemented.
3. The paper includes a clean mathematical analysis showing why chunk-level optimization yields smoother gradients and more accurate updates under imperfect advantage attribution.
4. The analysis of chunk sizes and the discussion of how temporal dynamics guide segmentation provide clear practical guidance for choosing chunk configurations in future applications.
5. The authors perform extensive experiments across multiple datasets and reward models, demonstrating consistent gains in preference alignment and image fidelity.

### Weaknesses
1. Missing significance analysis. Reported metrics are presented without standard deviations. Without these, it is difficult to assess whether the observed improvements are statistically significant.
2. A user study is missing. Since the paper focuses heavily on preference alignment and perceptual image quality, a user study would provide validation of the improvements.
3. The improvement is rather small, especially considering the results on WISE.

### Questions
1. Have you experimented with adaptive chunking, where the chunk boundaries evolve during training?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper argues that existing reinforcement learning methods for text-to-image generation place excessive emphasis on global advantages while neglecting local advantages, which may prevent trajectories from converging to the optimal solution. The authors propose a chunk-GRPO approach that segments the generated trajectory into chunks and computes advantages for each segment to address this issue. In addition, temporal dynamics is incorporated to dynamically adjust chunk sizes. Experimental results demonstrate improved performance on text-to-image generation tasks.

### Strengths
(1)	The authors propose the idea of incorporating both global and local advantages to evaluate the optimality of trajectory sequences, which is an interesting direction worthy of further exploration.
(2)	The proposed temporal dynamics method avoids complex hyperparameter configurations, thereby enhancing the generality of the approach.

### Weaknesses
(1)	More intuitive results: The example provided in Figure 2 of the paper is merely a schematic illustration. The authors are encouraged to present real image cases demonstrating whether, during the early or middle stages of generation, intermediate images exhibit higher quality, yet the final convergence results in an inferior output.
(2)	Motivation concern: Although the authors argue that certain steps in the generation process may possess local advantages, I believe that a well-formed generation trajectory does not—and need not—ensure local optimality at every step. The objective of reinforcement learning should still focus on achieving global optimality.
(3)	Experimental concerns: Based on the experimental results presented in the tables, the performance gains from chunking are minimal. This further calls into question the necessity of the chunking operation.

### Questions
All of my concerns are presented in the Weaknesses section.

### Soundness
2

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
The paper proposes Chunk-Level GRPO, which shifts GRPO optimization from step-level to chunk-level for flow-matching text-to-image models to address inaccurate advantage attribution and the neglect of temporal dynamics during generation.​ Consecutive timesteps are grouped into chunks guided by prompt-invariant relative L1 latent dynamics, and policies are optimized using a chunk-level importance ratio, with an optional weighted sampling strategy that emphasizes high-noise chunks.​ Experiments on FLUX Dev with HPDv2.1 show consistent gains in preference alignment (HPSv3, ImageReward) and competitive WISE benchmark results, alongside ablations on chunk configurations, per-chunk training, and reward-model robustness, plus analysis of a stability trade-off introduced by weighted sampling.

### Strengths
- The motivation claimed in Figure 1 is very interesting and insightful. Additionally, the findings on temporal dynamics are beneficial to the community.

- Chunk boundaries are informed by prompt-invariant temporal dynamics via relative L1 distance, yielding a principled, dynamics-aware segmentation rather than arbitrary chunking.

### Weaknesses
I thank the authors for their efforts in this work. Below are some concerns about this paper.
- This work claims to be the first “chunk‑level” method but does not compare against other GRPO variants like Flow-GRPO, Pref-GRPO, it cited, weakening the contribution boundary beyond a single Dance‑GRPO baseline. Moreover, the proposed method performs only on par with Dance‑GRPO on WISE.

- The chunking implementation is heuristic. Boundaries are precomputed from relative L1 latent dynamics and kept fixed, lacking adaptivity and making performance sensitive to the sampling step $T$, the model, and certain prompts.

- Despite the insight the authors claimed in Figure 1, such a chunk-based design did not show an optimal approach to solving such issues. In some cases, there are still issues, such as $ Chunk_{1}$ has the greater final reward (advantage), its $t=1$ timestep is worse
than that in $Chunk_{2}$.

### Questions
Please see the #Weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method that involves fine-tuning the image generation process at the chunk level instead of at separate timesteps. Furthermore, it introduces a weighted sampling strategy derived from the proposed chunk split design.

### Strengths
1. The core idea of optimizing at the chunk level instead of at separate timesteps is an interesting and novel direction for image generation.
2. The proposed method is explained with clarity.

### Weaknesses
1. Proposition 1 and its corresponding proof raise concerns.
Specifically, Proposition 1 claims that a smaller chunk size leads to better performance. Since sampling through separate timesteps is equivalent to sampling with a chunk size set to $K=1$, the authors should further clarify the optimal limit for "how small is enough."

2. The proof of Proposition 1 also presents issues.
Eq. (35) states that $J_{\text{chunk}} = \frac{1}{T}J_{\text{GRPO}}$, suggesting the objective function for the proposed chunk split is simply a scaled version of the original GRPO objective.
While the optimal parameters $\theta_{\text{chunk}} = \arg\max_{\theta} J_{\text{chunk}}(\theta)$ and $\theta_{\text{GRPO}} = \arg\max_{\theta} J_{\text{GRPO}}(\theta)$ would be mathematically equal due to scaling, the proof's objective should focus on demonstrating how the policy parameters $\theta$ are affected by the chunking scheme, rather than comparing the squared errors in the form $\|\hat{J}(\theta) - J_{\text{GRPO}}(\theta)||^{2} \geq \|\hat{J}(\theta) - J_{\text{chunk}}(\theta)||^{2}$. The change in the optimal parameter $\theta$ should be explicitly shown. 

Minors:
1. Typo errors were noted in some equations (e.g., in **Eq. (18)**, the notation should likely be $T_a \cup T_{ia} = \{1, 2, \cdots, T\}$).
2. The experimental results appear to show **only a marginal improvement** over the current state-of-the-art method, **Dance-GRPO**.

### Questions
1. Could the authors provide a more detailed justification for the selected chunk sizes of $[2, 3, 4, 7]$? (e.g provide the specific details of the $\ell_1(x, t)$ values across all timesteps to validate the chunk split design)
2. Could the authors elaborate on why the final attained policy parameters $\theta$ yield superior results? Is this improvement primarily attributable to enhanced stability in the reinforcement learning training process, or is there another underlying mechanism?

### Soundness
2

### Presentation
2

### Contribution
2
