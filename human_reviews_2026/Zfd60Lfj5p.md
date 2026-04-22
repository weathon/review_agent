# Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Diffusion large language models (dLLMs) are promising alternatives to autoregressive large language models (AR-LLMs), as they potentially allow higher inference throughput. Reinforcement learning (RL) is a crucial component for dLLMs to achieve comparable performance with AR-LLMs on important tasks, such as reasoning. However, RL algorithms that are well-suited for dLLMs' unique characteristics have yet to be developed. This paper proposes **Distribution Matching Policy Optimization (DMPO)**, a principled and theoretically grounded RL fine-tuning method specifically designed to enhance the reasoning capabilities of dLLMs by matching the dLLM policy distribution to the optimal, reward-tilted one through cross-entropy optimization. We identify a key challenge in the implementation with a small training batch size and propose several effective solutions through a novel weight baseline subtraction technique. DMPO exhibits superior performance on multiple reasoning benchmarks without supervised fine-tuning, with an accuracy improvement of up to $54.3\\%$ over previously SOTA baselines and $66.41\\%$ over the base model, underscoring the effectiveness of the distribution matching framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper presents Distribution Matching Policy Optimization (DMPO), a novel reinforcement learning framework for enhancing reasoning in diffusion LLMs (dLLMs). Instead of maximizing rewards, DMPO matches the model's policy distribution to an optimal, reward-tilted target distribution. This is achieved using a Weighted Denoising Cross-Entropy (WDCE) loss, which enables efficient off-policy training by leveraging the dLLM's forward noising process. A novel weight baseline subtraction technique is introduced for training stability. On multiple reasoning benchmarks, DMPO achieves state-of-the-art results, significantly outperforming baselines without supervised fine-tuning, validating its distribution-matching approach.

### Strengths
1. The paper presents clear, well-structured reasoning that connects diffusion LLM characteristics to the choice of forward KL (cross-entropy) objectives, with intuitive illustrations (e.g., mode-seeking vs mass-covering in Figure 2) and algorithmic clarity in Algorithm 1.
2. The empirical results show strong and consistent gains across diverse reasoning tasks (GSM8K, MATH500, Countdown, Sudoku) and multiple generation lengths, with particularly large improvements on planning-style benchmarks (e.g., +18% over d1 on Countdown and +11.8% on Sudoku), supporting the method’s effectiveness.
3. The experimental setup is carefully controlled and transparent, comparing against competitive dLLM baselines, reproducing d1 with recommended settings, and reporting training dynamics (Figure 4) and ablations (Figure 5), which strengthens the credibility of the findings.

### Weaknesses
1. Limited model generality: The study evaluates DMPO exclusively on the LLaDA-8B family, leaving its applicability to other diffusion LLM , which constrains claims of broader generalization.

2. Incomplete efficiency characterization: Although the method emphasizes forward-only, off-policy training as an efficiency advantage, the paper lacks comprehensive, end-to-end comparisons (throughput, latency, memory footprint, and wall-clock cost) against strong baselines like diffu-GRPO or AR-LLM RL, making the practical scalability gains insufficiently quantified.

### Questions
1. Does DMPO generalize beyond LLaDA (to other dLLM architectures/sizes) without extensive retuning?
2. Can you provide end-to-end efficiency metrics (throughput, latency, memory, wall-clock) versus diffu-GRPO and AR-LLM RL?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the DMPO algorithm, an RL algorithm designed for reasoning dLLMs training that treats policy optimization as a distribution matching procedure. The algorithm employs a novel objective function, WDCE, an off-policy forward loss that can leverage the inference acceleration techniques of dLLMs. Experimental results demonstrate that models trained with DMPO achieve SOTA performance on math and puzzle tasks without requiring SFT.

### Strengths
1. The method is not merely a policy optimization algorithm for AR-LLMs. It proposes an RL modeling framework tailored for dLLMs.
2. The proposed loss function is computationally aligned with dLLM inference mechanisms and off-policy RL algorithm.
3. The approach achieves strong results across multiple reasoning tasks.

### Weaknesses
1.The sequence length appears to be somewhat short. E.g. problems in MATH500 may require up to around 2048 tokens. It is unclear why the authors imposed such a limitation on reasoning length; testing with longer sequences may be worth considering.

### Questions
Please consider addressing the issue raised in the weakness section.

### Soundness
3

### Presentation
2

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
This paper is motivated by criticizing the reverse KL objective's mode-seeking behavior, which might result in diversity collapse and reward hacking. They derive a distribution matching objective that learns to match the closed-form optimal distribution (originally derived as the solution to the reverse KL problem). They design a method DMPO to match this distribution using forward KL minimization via weighted cross-entropy with importance sampling. Their importance sampling approach removes the need to be on-policy, enabling replay buffer reuse. They provide multiple design choices for the weight baseline that gets subtracted from the importance weights.

### Strengths
1. This paper provides a principled theoretical framework with clear derivations of the weighted denoising cross-entropy loss through importance sampling. The writing is clear and it is a good attempt to apply distribution matching for policy optimization for LLMs.
2. Their method enables off-policy training because their derived importance weights don't depend on the current policy, allowing replay buffer reuse for potential improved sample efficiency.
3. They shows consistent empirical improvements across multiple reasoning benchmarks.
4. They also identify and address the small batch size training challenge through novel weight baseline subtraction techniques with multiple design choices. The f-divergence and base-weight ablations are novel and insightful.

### Weaknesses
1. The paper criticizes reverse KL for being mode-seeking and causing reward hacking, but it derives the target distribution p* as the solution to a reverse KL problem, then uses forward KL to match this p*. The paper should clarify whether the issue is with reverse KL as optimization dynamics versus as objective definition.
3. Regarding the stability of the proposed method: The partition function is estimated with only N=16 samples per prompt, which might create high variance in importance weights, especially when reward exponentials vary widely. Weight baseline subtraction adds another approximation layer on top of already noisy normalized weights. 
7. It seems this method could also be applied to autoregressive LLMs. Has any prior work in autoregressive LLM policy optimization done similar distribution matching work (this seems to be missing from related works)? And what's the uniqueness of applying this method for diffusion LLMs, given that the marginalization over orders and masking patterns creates more layers of variance on top of the importance ratio approximation?
4. Line 151 says d1 "fully masks all response positions except d" to estimate per-token probabilities. According to d1's Figure 2, it masks all response positions without exception in a single forward pass.
5. One main strength of this paper is that importance weights don't depend on the current policy, enabling replay buffer reuse for sample efficiency. However, the authors don't provide experiments demonstrating this sample efficiency with versus without replay (on-policy methods).
6. This work is motivated by improving diversity using forward KL. Can you share the entropy curves during RL to validate whether forward KL actually achieves its theoretical goal of maintaining diversity.

### Questions
please see weakness.

How sensitives is this method wrt to hyperparameters like buffer sampling frequency F, number of rollouts N. Guidance on which are critical would be helpful.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a new method for post-training/fine-tuning of diffusion based LLMs (dLLMs). The authors argue that existing RLHF methods, which were originally proposed for  autoregressive LLMs, are suboptimal for dLLMs. They argue this because of mismatch between likelihood of generation and that used in RL. The main idea is to shift from reward maximization to the boltzmann distribution, a classic result knwon for entropy-regularized RL objectives. The method employs a reweighted denoising loss, by using importance weighting. Another proposed contribution is addressing a "small batch size" challenge, where they propose different forms of control variates. The paper shows performance gains over prior dLLM RL methods on reasoning benchmarks.

### Strengths
The paper's core premise is timely, as more research is being put into dLLMs. Applying RL to dLLMs, is a challenging and interesting problem. I do not see major theoretical flaws in the core argument. The small batch size problem that was considered is potentially interesting. The overall description is clear enough for someone to try replicating without much issue.

### Weaknesses
Other papers have already identified the 'boltzmann distribution' matching idea. While that theoretical result is atleast a decade old (Ziebert et al), its application to dLLM is also known [1, 2].
This paper proposes KL minimization as objective, compared to the aforementioned works. 
The paper also claims DMPO avoids mode collapse and encourages diversity. While that makes sense from a high-level nature of the optimization objective; this also just assumes that the model cannot fit the data (which is unlikely for LLMs). Additionally, no analysis of these claims is presented. A similar concern arises from the 'off-policy' claim. Finally, even with methods like PPO, off-policy buffers get used regularly, so I do not see the specific possibilities arising from the proposed method. Finally, while the paper claims the PPO/GRPO objective is not principled due to mismatch between generation and used likelihood, that still applies here because of the ELBO. 

A last small issue is that I am not sure of the results table. While the numbers are broadly in line with what is reported in d1, they still differ significantly from reported results. Another missing baseline is [3]

[1] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models
[2] PADRE: Pseudo-Likelihood based Alignment of Diffusion Language Models
[3] DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation

### Questions
I would like to see an ablation with SFT. While that may be important for boosting d1's result, it would be good to evaluate the proposed method with and without SFT.


In line 232, its written that Eq 7 is not amenable to optimization due to the partition function of p*. But that is irrelevant for optimization. Infact that is exactly what happens in Eq 9. 

While I can understand the criteria of variance reduction, I do not think that is not what is happening here. By changing the importance weighting, the whole idea of this being a distribution matching loss becomes untenable. I would also like a more detailed explanation of the small batch phenomena mentioned. It should be irrelevant with infinite data and perfect optimization. So it seems to me that the authors are claiming something else is happening, but I did not understand the explanation.


The final best subtractive weight is perhaps the simplest one. And the results shown there are also for the reward learning problem. I would like to see those baseline control variates in terms of actual results as well. I am also surprised that this weight reduction could not be achieved by clipping weights or reward standardization.

Most of the gains come from sudoku and/or countdown. Is there something specific in these tasks that make the proposed method better than GRPO etc.

### Soundness
2

### Presentation
3

### Contribution
2
