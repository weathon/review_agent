# WavefrontDiffusion: Dynamic Decoding Schedule for Improved Reasoning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Diffusion Language Models (DLMs) have shown strong potential for text generation and are becoming a competitive alternative to autoregressive models. 
The denoising strategy plays an important role in determining the quality of their outputs.
Mainstream denoising strategies include Standard Diffusion and BlockDiffusion. 
Standard Diffusion performs global denoising without restricting the update range, often finalizing incomplete context and causing premature end-of-sequence predictions. 
BlockDiffusion updates fixed-size blocks in a preset order, but its rigid structure can break apart coherent semantic units and disrupt reasoning. 
We present WavefrontDiffusion, a dynamic decoding approach that expands a wavefront of active tokens outward from finalized positions. 
This adaptive process follows the natural flow of semantic structure while keeping computational cost equal to block-based methods. 
Across four benchmarks in reasoning and code generation, WavefrontDiffusion achieves state-of-the-art performance while producing outputs with higher semantic fidelity, showing the value of adaptive scheduling for more coherent and efficient generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a wavefront-style dynamic decoding schedule for discrete, mask-based diffusion language models. Instead of global synchronous denoising or fixed-order block decoding, the model maintains a frontier of finalized tokens and iteratively performs scoring, denoising, expansion, and pruning. By capping both the per-step update size $F$ and total steps $T$, it enforces compute parity with block decoding, ensuring that improvements stem from scheduling rather than added computation. The approach is training-free and integrates easily with existing diffusion LMs. However, despite the elegant design, the paper’s experimental analysis remains limited. The evaluation is confined to the LLaDA family (e.g., 8B and 1.5 variants) rather than a broader set of diffusion LMs, and it lacks calibration robustness checks. The MHCO metric shows improved consistency, yet its dependence on the frontier radius $R$ is unexamined; a short theoretical or empirical study would clarify interpretability and stability, and several ablations (e.g., local scoring efficiency, calibration impact) are missing. While the reported gains are consistent, they are modest, and the absence of broader baselines or theoretical grounding weakens the generality of the conclusions.

### Strengths
The work is well organized and clearly presented, with strong empirical grounding and sensible ablations. It introduces a novel scheduling perspective for diffusion decoding that reallocates compute dynamically rather than increasing it, preserving $F×T$ parity. The design is practical and model-agnostic—requiring no retraining and the safeguards (frontier pruning, capped updates) are thoughtfully engineered. Experimental reporting is transparent, and the improvements, while modest, are consistent across settings. The paper’s clarity and practicality make it an appealing addition to the diffusion decoding literature.

### Weaknesses
1. **Dependence on Confidence Calibration**
   The proposed approach relies heavily on token-level confidence scores to determine which tokens are finalized. The approach relies on token-level confidence scores to finalize tokens but does not evaluate robustness under different temperature settings or calibration shifts, although the discussion section acknowledges calibration as a general limitation. While Section 4.5 acknowledges calibration as a general limitation of diffusion-based decoders, there is no empirical analysis showing how WavefrontDiffusion behaves under miscalibrated conditions. If the confidence estimates are unstable, the wavefront expansion could prematurely freeze or miss critical tokens, potentially affecting reliability.

2. **Narrow Baseline Coverage**
   The experiments compare only with Standard Diffusion and BlockDiffusion. More recent dynamic decoding methods—such as remasking-based or truncated-block strategies are not included. Since these methods also dynamically adjust the denoising schedule, evaluating under a matched $F×T$ compute budget would clarify whether the observed gains come from the proposed scheduling design itself or from confidence-based gating.

3. **Limited Theoretical or Consistency Analysis**
   While the “wavefront” intuition is compelling, the paper does not formalize why this adaptive frontier expansion should outperform fixed or global schedules. The MHCO metric shows improved consistency, yet the connection between this measure and denoising optimality remains informal. Moreover, the sensitivity of MHCO to its hyperparameter $R$ (the frontier radius) is not analyzed, leaving unclear whether its interpretability or stability depends on this choice. A brief theoretical or empirical study of MHCO’s dependence on $R$ would improve clarity.

4. **Lack of Analysis on Long-Sequence Behavior**
   Although the paper reports results on reasoning and code benchmarks, it does not visualize or analyze how the frontier evolves for long sequences. For instance, it is unclear whether the wavefront ever converges prematurely or oscillates during extended decoding. A step-wise visualization or “finalization regret” analysis would clarify how the dynamic schedule behaves as sequence length grows.

5. **Efficiency and Scoring Computation**
   In the core iterative process, the model scores all masked positions before selecting the top-$k$ candidates from the current frontier $W_{t−1}$. This full scoring procedure may increase computational cost. The paper does not discuss whether a *localized* scoring strategy—scoring only masked tokens near the current wavefront—has been tested, nor whether it could reduce compute while maintaining performance.

6. **Presentation and Appendix Details**
   Some hyperparameter details and appendix cross-references could be clearer. For instance, the main text does not specify the exact ranges or adaptation rules for $R$ and $F$ across tasks. Additionally, an explicit scope note that this work focuses solely on *discrete* diffusion (rather than continuous-space diffusion) would help prevent confusion.

7. **Evaluation Limited to a Single Model**
   All experiments are conducted on LLaDA-8B. Although the method is described as “model-agnostic,” it is not validated on other diffusion language models with different masking or training policies. Since diffusion LMs vary in their noise formulation, reweighting, and decoding dynamics, results based solely on one model may not fully establish generality. Demonstrating consistent improvements across multiple diffusion models would further strengthen the claim.
   
8. **Missing long-sequence scalability analysis**
   The paper does not analyze wavefront dynamics on long sequences (e.g., premature convergence or oscillation). Visualizing the frontier trajectory and reporting a “finalization regret” diagnostic would clarify stability as sequence length grows.

### Questions
1. **Confidence Stability:**
   Have the authors evaluated how WavefrontDiffusion’s performance changes under temperature scaling or calibration shifts? If confidence scores become over- or under-confident, does the frontier expansion remain stable? Would entropy-based or learnable temperature gating improve robustness?

2. **Baseline Scope:**
   Could the authors include comparisons with other dynamic decoding approaches (e.g., remasking-based or truncated-block methods) under the same $F×T$ compute constraint? This would help isolate the benefits of the scheduling mechanism itself from those of confidence gating.

3. **Long-Sequence Dynamics:**
   For long reasoning or code-generation tasks, how does the frontier expand and finalize over time? Are there cases of premature convergence or late instability? A quantitative or visual analysis would be insightful.

4. **Theoretical Insight:**
   Can the authors provide intuition or approximate analysis explaining why expanding around finalized tokens with radius $R$ improves semantic fidelity? Does this mechanism implicitly encourage smoother denoising trajectories or faster entropy reduction?

5. **Calibration Ablation:**
   Have the authors tested whether applying calibration techniques (e.g., temperature scaling or isotonic regression) changes the MHCO metric or final accuracy? This would clarify the sensitivity of the method to calibration errors.

6. **Model Dependency:**
   The method is evaluated only on LLaDA. Does the approach depend on LLaDA’s specific sequence-level noise or architecture? Would it generalize to other diffusion language models, such as those using token-level noise or trained with reinforcement learning? Any preliminary evidence or reasoning supporting model-agnostic applicability would be valuable.

7. **MHCO Sensitivity:**
   Since MHCO depends on the radius $R$, have the authors examined how different $R$ values affect MHCO scores? Does the metric remain consistent across reasonable $R$ choices, or is it highly sensitive to this hyperparameter?

8. **Localized Scoring Efficiency:**
   In the core iterative process, the model first scores *all* masked positions before selecting tokens from the current wavefront $W_{t−1}$. Have the authors tested a variant where scoring is restricted to $W_{t−1}$ and its neighboring masked positions? Would this local scoring reduce computational cost without hurting performance?
9. **Long-sequence dynamics and stability:** 
   On longer reasoning/generation sequences, does the wavefront exhibit premature freezing or oscillation? Could you report curves/statistics of frontier radius $R$, finalized-token ratio, and “finalization regret” over steps and sequence length?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
WavefrontDiffusion dynamically expands a confidence-guided frontier in diffusion LMs, preserving context, matching block compute, and improving reasoning/code accuracy and semantic fidelity across benchmarks under compute-parity.

### Strengths
1. The method avoids premature EOS and half-baked spans by not locking in locally high confidence tokens too early.
2. It completes semantically “ready” regions first (e.g., function signatures, reasoning steps) and is not hostage to rigid chunk boundaries.

### Weaknesses
1. Only F and R are studied; there is no analysis of the per-step finalize quota k_t, nor strict equal FLOPs / equal token updates controls.
2. The setup is mostly zero-shot with T=1024 and temperature 0; it lacks length/temperature sweeps and multi-seed variance.
3. The very long context engineering story is unclear; the overhead of frontier maintenance and cache policies at extreme lengths is not evidenced.
4. Autoregressive baselines at matched latency are missing; there is no head-to-head against speculative decoding under equal delay/throughput.
5. Early errors can still propagate; once an incorrect span is finalized, downstream reasoning may be constrained by that commitment.

### Questions
1. Add equal FLOPs and equal token updates tables, and ablate k_tallocation strategies.
2. Report mean/σ over multiple seeds, and vary context length, temperature, and prompting (few-shot, CoT).
3. Test entropy/energy or calibrated confidence to reduce dependence on raw max-softmax.
4. Include longer-context reasoning and additional code sets (e.g., MBPP, DS-1000).
5. Provide matched-latency/throughput, end-to-end comparisons vs. speculative decoding to map advantage boundaries.

### Soundness
3

### Presentation
3

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
This paper introduces a dynamic decoding scheduling strategy named WavefrontDiffusion, designed to address the issues of semantic coherence and computational efficiency in text generation with diffusion language models. Whereas traditional decoding strategies like Standard Diffusion and BlockDiffusion have inherent limitations, WavefrontDiffusion dynamically expands a "wavefront" region of active tokens. This allows the denoising process to align with the natural flow of semantic structure while maintaining the same computational cost as block-based methods. Experiments demonstrate that this approach achieves state-of-the-art performance across multiple benchmarks and generates outputs with higher semantic fidelity. This research presents a new paradigm for applying diffusion models to complex reasoning and code generation tasks.

### Strengths
1. The method is intuitive and addresses the limitations of hard boundaries in BlockDiffusion.

2. The research methodology is well-structured, with clear explanations of the wavefront theory, a four-step algorithm, and mathematical definitions. The experimental design covers four benchmark tests and evaluates the method using multiple metrics such as accuracy, BERTScore, and the MHCO indicator.

3. The experimental analysis is thorough and provides insights into parameter selection.

### Weaknesses
1. The method is an incremental improvement over BlockDiffusion; both the methodology and the experimental results are incremental in nature.

2. Regarding the writing, Figure 1 is not sufficiently intuitive and requires further revision.

### Questions
1. Are there any unique application scenarios where this method can demonstrate a more significant advantage over BlockDiffusion?

### Soundness
2

### Presentation
3

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
The paper introduces WavefrontDiffusion, a dynamic decoding strategy for Diffusion Language Models that adaptively expands a “wavefront” of active tokens during generation. This approach preserves semantic coherence and contextual completeness while keeping the same computational cost as block-based methods. Experiments on reasoning and code generation benchmarks show that WavefrontDiffusion consistently improves accuracy and output quality over existing diffusion decoding strategies.

### Strengths
1. WavefrontDiffusion dynamically adjusts the denoising process to follow the evolving semantic structure, preventing premature or fragmented token generation.

2. By expanding from finalized tokens outward, it ensures each token is generated with sufficient context, leading to smoother and more logically consistent outputs.

3. The method matches the computational cost of block-based decoding while delivering higher accuracy and better output quality.

### Weaknesses
1. The paper lacks a baseline for its method. It needs to compare its approach with current decoding methods in the DLM field to demonstrate its advantages.

2. Experiments were only conducted on one model category. Similar experiments need to be performed on Dream for comparison.

### Questions
1. Will this method slow down the inference process? How does its speed compare to other methods?

2. Could you run the results on MBPP?

### Soundness
3

### Presentation
3

### Contribution
3
