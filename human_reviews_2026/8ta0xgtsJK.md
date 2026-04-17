# VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Inspired by human SYSTEM 2 thinking, LLMs excel at complex reasoning tasks via extended Chain-of-Thought. However, similar test-time scaling for diffusion models to tackle complex reasoning remains largely unexplored. From existing work, two primary challenges emerge in this setting: (i) the dependence on an external verifier indicating a notable gap from intrinsic reasoning of human intelligence without any external feedback, and (ii) the lack of an efficient search algorithm. In this paper, we introduce the Verifier-free Test-time Scalable Diffusion Model (VFScale) to achieve scalable intrinsic reasoning, which equips number-of-sample test-time scaling with the intrinsic energy function of diffusion models as the verifier. Concretely, VFScale comprises two key innovations to address the aforementioned challenges. On the training side,  VFScale consists of a novel MRNCL loss and a KL regularization to improve the energy landscape, ensuring that the learned energy function itself serves as a reliable verifier. On the inference side, VFScale integrates the denoising process with a novel hybrid Monte Carlo Tree Search (hMCTS) to improve search efficiency. On challenging reasoning tasks of Maze and Sudoku, we demonstrate the effectiveness of VFScale's training objective and scalable inference method. In particular, trained with Maze sizes of up to 6×6, our VFScale solves 88\% of Maze problems with much larger sizes of 15×15, while standard diffusion model completely fails. The code can be found at https://github.com/AI4Science-WestlakeU/VFScale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address the lack of test-time scaling capabilities in diffusion models for complex reasoning tasks (e.g., maze solving and Sudoku). The core idea of this work is to use the diffusion model's own intrinsic energy function as a verifier to guide the search.

To achieve this, VFScale incorporates two key training innovations: Monotonic-Regression Negative Contrastive Learning (MRNCL) and KL regularization—to improve the energy landscape so that the model’s learned energy can act as an intrinsic verifier. At inference time, a hybrid Monte Carlo Tree Search (hMCTS) algorithm is proposed. This algorithm uses Best-of-N (BoN) for broad exploration during the early, noisy denoising stages and switches to MCTS for deep exploitation in the later, less-noisy stages.

Experimental results show that VFScale (e.g., trained on 6x6 mazes) can achieve a high success rate on complex tasks far beyond the training distribution (e.g., 15x15 mazes), where baseline models fail completely.

### Strengths
1. **Interesting Problem Setup:**  
   The paper tackles the idea of *intrinsic reasoning*—removing external verifiers for diffusion-based reasoning—which is conceptually appealing and timely. This aligns well with the emerging research trend on *test-time scaling* and reasoning within generative models.

2. **Clear Positioning within Existing Literature:**  
   The paper establishes clear connections to prior work, particularly Du et al. (2022, 2024) on energy-based diffusion, effectively situating the contribution within the broader context of diffusion-driven reasoning research.

3. **Algorithmic Insight of hMCTS:**  
   The hybrid design of hMCTS demonstrates sound algorithmic intuition. Employing Best-of-N (BoN) for parallel exploration during early noisy stages and transitioning to MCTS for fine-grained exploitation in later denoising stages is well-motivated and consistent with the diffusion dynamics.

4. **Strong OOD Generalization Performance:**  
   The experiments show notable improvements in out-of-distribution (OOD) generalization, suggesting that the proposed intrinsic reasoning framework may enhance robustness beyond the training distribution.

### Weaknesses
1. **Marginal Gains from MRNCL:**  
   The improvement attributed to MRNCL appears small and inconsistent across tasks. The MRNCL objective, which enforces a linear energy–distance relationship, feels somewhat ad hoc. Imposing strict linearity between the $L_2$ distance and energy may oversimplify the energy landscape and lacks clear theoretical justification. Empirically, much of the observed gain seems to stem from the KL term rather than MRNCL itself (see Tables 1 and 3, particularly in the Sudoku experiments).

2. **Questionable Inference Design in Continuous Space:**  
   The proposed hMCTS in practice appears to reduce to a Best-of-N sampling strategy with branching Gaussian perturbations. It remains unclear what specific advantages MCTS provides in this continuous setting, as its benefits are typically most pronounced in discrete action spaces.

3. **Unclear Experimental Setup and Visualization:**  
   The implementation details for the Maze task are under-specified. While the paper implies continuous-valued diffusion states, the Maze itself is inherently discrete. It is unclear how intermediate diffusion states are visualized (e.g., Fig. 1) or discretized back into paths. This ambiguity raises questions about whether the claimed “continuous reasoning formulation” genuinely applies to these discrete tasks.

4. **Limited Novelty of Components:**  
   The major components—contrastive energy training, KL regularization, and MCTS-based inference—are largely adapted from existing literature (e.g., Du et al., 2024; Silver et al., 2016). The central contribution, MRNCL, introduces a monotonic regression term that offers modest empirical improvement, making the overall technical novelty somewhat incremental.

5. **Limited Evaluation Scope:**  
   The experiments are confined to synthetic reasoning tasks (Maze and Sudoku). While suitable for controlled evaluation, the absence of results on larger or real-world structured reasoning benchmarks limits the generality and impact of the work.

---

#### Miscellaneous Issue

1. **Inconsistent Notation:**  
   The definition of the energy function is inconsistent across sections. In Du et al. (2024) and line 198, it is defined with both the input and its corresponding solution, whereas in other parts of the paper, the energy function is presented as depending solely on the solution.

### Questions
1. **Computation of $p_{\theta, t}(x)$:**  
   How is $p_{\theta, t}(x)$ computed exactly? Please also clarify its relationship to the predicted energy function.

2. **Representation of Discrete Data:**  
   Please elaborate on how the Maze and Sudoku datasets are represented. Specifically, how are discrete states encoded into continuous vectors for diffusion modeling and energy computation?How are the intermediate Maze visualizations (e.g., Fig. 1) generated? Are they obtained by projecting continuous latent states, or by decoding discrete states from intermediate diffusion steps?

4. **Energy–Distance Relationship:**  
   What is the theoretical motivation for enforcing a *linear* relationship between the predicted energy and the \( L_2 \) distance? Would a monotonic but non-linear mapping (e.g., logistic or exponential) be equally valid or even more flexible?

5. **MCTS vs. Best-of-N Sampling:**  
   Since each MCTS expansion involves multiple Gaussian perturbations, what makes your MCTS procedure fundamentally different from a multi-step Best-of-N or beam search strategy?

---

#### Miscellaneous

1. **Definition of “Guided with Ground Truth” (Table 2):**  
   The term *“guided with ground truth”* in Table 2 is not clearly defined. Please clarify the exact experimental setup and what this guidance entails.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes VFScale, a way to make diffusion models solve reasoning problems better without relying on an external verifier. It treats the model’s own energy as an intrinsic verifier and improves both training and inference: on the training side, a loss called MRNCL enforces that samples farther from the ground truth have monotonically higher energy and adds a KL term to smooth the landscape so lower energy better aligns with correctness; on the inference side, a hybrid search hMCTS uses Best-of-N early (broad exploration) and MCTS later (deep exploitation) to convert extra compute into higher success more efficiently.

### Strengths
- Treats diffusion energy as an intrinsic verifier, removing dependence on external verifiers and focusing on verifier accuracy + search efficiency.
- Training (MRNCL + KL) aligns low energy with correctness; inference (hMCTS) sensibly combines early Best-of-N with later MCTS to turn extra compute into better.

### Weaknesses
- Using diffusion energy as an intrinsic verifier and aligning it via MRNCL + KL is largely heuristic; the paper provides no formal results
- MRNCL relies on how “near/far” negatives are defined and enforces only local monotonicity; the paper does not analyze robustness to the distance metric nor show that local constraints imply a global ranking alignment.

### Questions
1. Have the authors tried using majority voting for the task? If the diffusion model can consistently output a path more often than the other, it can also be used as a verifier.
2. How about pitting MRNCL against margin / listwise ranking losses—and showing how results shift when you swap the “near/far” distance metric for negatives?
3. Any quick Kendall $\tau$ / Spearman $\rho$ between energy and correctness across timesteps to show the intrinsic-verifier story holds up?

### Soundness
3

### Presentation
3

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
This paper introduces VFScale, a framework for enhancing the test-time scalability of diffusion models for reasoning tasks, notably without requiring an external verifier. The core idea is to leverage the model's intrinsic energy function as a verifier. On the training side, the authors propose MRNCL loss and a KL regularization to improve the energy landscape. For inference, the authors adopt a hybrid MCTS for the denoising process to improve search efficiency. VFScale demonstrates good performance on challenging Maze and Sudoku tasks.

### Strengths
Strengths of this work are summarized below:
- The paper tackles a highly relevant and challenging problem: enabling test-time scaling for diffusion models on reasoning tasks without relying on an external verifier. Overall, the research direction is interesting and valuable.
- The test-time scaling capability of VFScale seems strong. For example, Figure 3 clearly illustrates that the proposed method scales significantly better with an increased computational budget (N) compared to all baselines. While other methods see diminishing returns and plateau quickly, the success rate of VFScale continues to climb under both MCTS and BoN.
- This paper is well-written and clear.

### Weaknesses
Weaknesses of this work are summarized below:
- My primary concern stems from the ablation study in Table 1. These results, which test the base model's generalization (N=1), contradict the paper's central claim about the utility of the proposed training losses. Specifically, on Sudoku: The full model, "VFScale tr. (ours)," performs significantly worse than the ablated models. For instance, at D=33, the full model achieves a 0.1953 success rate, whereas both "w/o MRNCL" and "w/o KL" achieve 0.4219. At D=29, the full model (0.0078) is drastically outperformed by "w/o KL" (0.2578). Similarly, on Maze for M=12, the "w/o KL" model achieves a perfect 1.0000 success rate, while the full model only reaches 0.5391. I would like to know the potential reasons behind the results, for example, providing an explanation for why their full model can be the worst performer in this naïve inference setting.

- While the performance on Maze and Sudoku is good, the experiments are limited to grid-world toy problems. The paper's impact would be stronger if the authors demonstrated VFScale's applicability to more complex and practical reasoning or optimization domains where diffusion models are being applied.

### Questions
Please refer to the weaknesses above. I am inclined to raise my score if my concerns are adequately addressed.

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
2

### Summary
The paper introduces **VFScale**, a diffusion-based framework for intrinsic reasoning that scales test-time inference *without any external verifier*. The core idea is to use the model’s own **energy landscape** as a dense verifier, training it to correlate with solution quality. Two key mechanisms enable this:  
1. **Monotonic-Regression Negative Contrastive Learning (MRNCL)**, enforcing *performance–energy consistency* (lower energy ↔ better performance), and  
2. a **KL regularizer**, smoothing the diffusion trajectory.  

At inference, the paper proposes **hybrid MCTS (hMCTS)**: a transition from **Best-of-N** exploration in early noisy steps to **Monte Carlo Tree Search** exploitation as denoising progresses.  

Experiments on **Maze** and **Sudoku** show substantial out-of-distribution scaling — e.g., a model trained on 6×6 mazes solves 15×15 mazes at ~88% success.

### Strengths
- **Innovative framing:** Redefines verifier-free reasoning as intrinsic energy shaping.
- **Clear empirical impact:** Demonstrates significant generalization to harder unseen settings.
- **Methodological synergy:** MRNCL + KL training improves energy consistency; hMCTS improves search efficiency.
- **Alignment with trustworthy AI:** Encourages interpretable internal uncertainty signals instead of opaque verifiers.
- **Reproducibility:** Code, datasets, and checkpoints are to be released; strong commitment to open science.

### Weaknesses
1. **Limited experimental diversity.**  
   Evaluation restricted to Maze and Sudoku; lacks heterogeneous reasoning domains (e.g., SAT solving, graph-based CSPs, symbolic reasoning). Limits generalization claims.

2. **Insufficient theoretical analysis.**  
   MRNCL is intuitively justified but mathematically underdeveloped. No formal calibration or ordering guarantees.

3. **Heuristic inference strategy.**  
   The hMCTS switching mechanism (BoN→MCTS) is empirical and lacks sensitivity or ablation analysis.

4. **Statistical rigor.**  
   No standard deviation, confidence intervals, or multi-seed statistics are reported.

5. **Compute transparency.**  
   Training with MRNCL+KL is more expensive; wall-clock and GPU-hour trade-offs are not quantified.

6. **External verifier comparison.**  
   The claim that intrinsic energy outperforms a 0.99-correlated verifier is interesting but needs deeper error analysis (e.g., top-k misrankings).

### Questions
1. Could you formally define **MRNCL**, provide its gradient, and state any conditions ensuring monotonic energy–performance ordering?  
2. How exactly is the **BoN→MCTS switch** triggered — by diffusion timestep, entropy, or compute budget? Have you tested adaptive criteria?  
3. Why does the **external verifier** underperform despite 0.99 correlation? Can you analyze the ranking errors?  
4. Have you applied VFScale to **non-grid CSPs** (e.g., graph coloring or SAT)? If not, what prevents generalization?  
5. How is the **performance–energy consistency (PEC)** metric computed (e.g., Kendall-τ)? Please include its formal definition.  
6. Could you report **wall-clock cost and memory usage** for the largest setups to evaluate practical scalability?

### Soundness
3

### Presentation
3

### Contribution
3
