# From Collapse to Control: Understanding and Extending Context Length in Emerging Hybrid Models via Universal Position Interpolation

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Hybrid Mamba-Transformer models have emerged as promising alternatives to pure Transformers, offering efficiency and competitive performance. However, they struggle to generalize beyond their training context windows, collapsing on long-context tasks. We provide the first systematic analysis of this failure, showing that it arises from uncontrolled state growth and uneven receptive field contributions across the hybrid architecture. Guided by this understanding, we introduce Universal Position Interpolation (UPI), a closed-form, training-free scaling method that unifies Mamba's cumulative decay with Transformer rotary frequency scaling. UPI selectively stabilizes unstable Mamba dynamics while rescaling Transformer encodings, controlling state growth and enabling reliable long-context generalization, with only a few auxiliary forward passes.  Evaluation shows that UPI extends multiple state-of-the-art hybrid and pure Mamba models from 4K to up to 64K tokens on PG-19 perplexity, LongBench and RULER benchmarks, without sacrificing short-context accuracy. These findings establish the first principled bridge between Transformers and state-space models and open a new direction for training-free context extension methods for emerging hybrid models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses context length extension (CLE) for hybrid Mamba-Transformer models, which combine SSM layers with attention layers for efficiency. The authors identify that these models fail beyond training context lengths due to uncontrolled state growth in certain Mamba heads and uneven receptive field contributions. They propose Universal Position Interpolation (UPI), a training-free method that selectively rescales unstable Mamba heads (top 20% by ERF) while applying YaRN to Transformer layers. The method is evaluated on Bamba, Nemotron-H, and Mamba2 models across PG-19, RULER, and LongBench benchmarks.

### Strengths
1 - First systematic analysis of context length limitations in hybrid Mamba-Transformer models, addressing a practically relevant issue.

2 - The ERF analysis (Section 3.1) and state magnitude explosion analysis (Section 3.2) provide valuable insights into why hybrid models fail on long contexts.

3 - UPI requires no fine-tuning, making it practically appealing with minimal computational overhead (<3 minutes calibration).

### Weaknesses
1.  **Heuristic Head Selection:** The method identifies "unconverged" heads using a proxy: the "top-20% by ERF". While the authors justify this correlation and show its robustness across datasets (Table 6), it remains a heuristic. A more direct, analytical method to identify heads where $E[\overline{a}] \approx 1$ would be more theoretically satisfying.
2.  **The "20%" Threshold:** The 20% threshold is presented as a robust choice, but the ablation (Table 5) is only shown for Bamba-v2. It's unclear if 20% is a "magic number" that generalizes to all models, or if it was re-tuned for Nemotron-H and Mamba2. If it requires per-model tuning, it slightly weakens the "tuning-free" claim (though it's still far cheaper than the baseline).
3.  **Missing Empirical Validation of $\Delta_t$:** The key derivation (Eq 2) relies on the fact that the unstable heads have $\Delta_t \rightarrow 0$. This is a reasonable assumption, but the paper would be strengthened by an empirical validation. A histogram of $\Delta_t$ values for the top-20% ERF heads versus the bottom-80% would provide direct evidence to confirm this crucial assumption.

### Questions
1.  The "top-20% ERF" heuristic is practical and effective. Have you explored a more direct method for identifying the unstable heads during calibration? For example, could one directly measure the average forget gate $\overline{a}$ or the state growth rate per head and set a threshold based on that?
2.  How did authors select the 20% threshold for Nemotron-H and Mamba2-2.7B? Was the 20% value from the Bamba-v2 ablation (Table 5) found to be universally optimal, or did this hyperparameter require tuning for each model?
3.  Can authors provide an empirical analysis (e.g., a histogram or density plot) of the $\Delta_t$ distributions for the selected top-20% heads versus the remaining 80%? This would directly validate the core assumption used in the derivation of the simplified input gate scaling (Eq 2).
4.  In Table 3, LongMamba outperforms UPI on the pure Mamba2 model. Authors plausibly argue this is due to LongMamba's task-specific tuning. Could this result also suggest a potential limitation of UPI's "universal" $\Delta_t / n$ scaling when compared to LongMamba's more complex mechanism, especially in a pure-SSM setting?
5.  The RoPE-Mamba duality is a fascinating conceptual bridge. Does this duality suggest that other, more advanced RoPE-scaling techniques (e.g., NTK-aware scaling) could be "translated" to Mamba gate-scaling? What do authors see as the most promising future work based on this "principled bridge"?

### Soundness
3

### Presentation
3

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
This paper addresses the long-context generalization failure of hybrid Mamba-Transformer models and proposes a training-free context length extension (CLE) method called Universal Position Interpolation (UPI). UPI unifies Mamba’s cumulative decay and Transformer’s rotary positional encoding (RoPE) scaling, enabling hybrid/pure Mamba models to extend context length from 4K/8K to up to 64K tokens without sacrificing short-context accuracy.

### Strengths
1. This paper provides the first systematic investigation of long-context failure in hybrid Mamba-Transformer models, addressing a critical gap, prior work focused exclusively on pure Transformer or Mamba architectures, ignoring the unique challenges of hybrid designs. The identification of "uncontrolled state growth" and "uneven receptive field contributions" as root causes is a foundational insight.  

2. It balances mathematical rigor (e.g., state magnitude explosion derivation, UPI closed-form proof) and empirical validation. ERF analysis and state growth characterization provide concrete evidence for failure modes, while ablation studies (selective head adjustment, Top-K thresholding) validate the method’s components.  


3. Hybrid Mamba-Transformer models are emerging as efficient alternatives to Transformers, but their long-context limitation hinders real-world use (e.g., document understanding, multi-turn dialogue). UPI extends their context window from 4K to 64K tokens without performance loss, making these models viable for long-sequence tasks.

### Weaknesses
1. Subjectivity and Non-Generality of the Top-20% High ERF Head Threshold: UPI relies on the rule of "selecting the Top-20% of high ERF heads for adjustment," but this threshold is determined solely based on experimental experience from benchmarks such as PG-19 and RULER, lacking systematic validation. The selection logic for "20%" is not explained, nor is its adaptability to different model architectures (e.g., models with different ratios of Mamba/Transformer layers) and task types (e.g., code generation, multi-turn dialogue) verified.

2. Insufficiently Comprehensive Baselines: Only "LongMamba + YaRN" was selected as the baseline. However, LongMamba is a non-specific baseline designed for pure Mamba and is not compared with "CLE methods specifically optimized for hybrid models" (e.g., Hymba's hybrid head adjustment logic, Samba's state space co-extension). Furthermore, "fine-tuning CLE methods" (e.g., long context fine-tuning for hybrid models) are not compared, making it impossible to quantify the actual value of UPI's "no-training" advantage (e.g., fine-tuning methods may have higher performance; is UPI's deployment advantage sufficient to compensate for the performance gap?).

### Questions
1. Rationale for the 20% Top-K Threshold for Mamba Heads. 
What is the systematic basis for choosing 20% specifically? Is it derived from theoretical analysis (e.g., the proportion of unstable heads in hybrid models) or just empirical tuning on the tested benchmarks? Does this threshold generalize across different hybrid model architectures (e.g., models with more Transformer layers than Mamba layers) or task types (e.g., code completion vs. multi-document QA)? If the threshold needs adjustment for new models/tasks, what guidelines can be provided?


2. Generalization to Longer Contexts and Unseen Model Configurations
The paper extends context lengths from 2K/4K/8K to 64K, but key gaps remain: Have the authors tested UPI on longer contexts beyond 64K (e.g., 128K or 256K tokens)? If not, does the authors anticipate performance degradation, and what mechanisms might limit scalability?

3. Comparison with More Relevant Baselines for Hybrid Models. 
The baseline comparison uses LongMamba (for pure Mamba) + YaRN (for Transformer), but this is a "component-wise combination" rather than a dedicated hybrid CLE method. Have the authors considered baselines that jointly optimize Transformer and Mamba layers for hybrid models?

4. Applicability to Pure Transformer Models and Adaptive Head Selection. The paper frames UPI as "universal," but its utility for pure Transformer models is unaddressed. Can UPI be adapted to pure Transformer models? Since UPI unifies RoPE scaling and Mamba gate dynamics, would applying its core logic (e.g., selective dimension-wise scaling based on ERF-like metrics) outperform existing Transformer CLE methods (e.g., YaRN, LongRoPE)?

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
- This paper addresses the Context Length Extension (CLE) problem for hybrid Mamba-Transformer models. While these hybrid models are computationally efficient and perform well within their training context window, the authors show they "collapse" catastrophically when presented with sequences longer than what they were trained on (e.g., perplexity spikes when a 4K-trained model sees 8K tokens).  
- The failure is not just due to the Transformer components, but primarily stems from a subset of "unstable" or "unconverged" heads within the Mamba layers. These heads have forget gates consistently close to 1, causing their internal state to grow uncontrollably beyond the training length. This "state magnitude explosion" overwhelms the outputs of other, stable heads, leading to a "feature collapse" where most of the model's neurons become inactive.  
- The authors introduce UPI, a training-free, closed-form method to scale the context window of hybrid models. UPI works by:  
  - Using a small calibration dataset (100 samples, <3 minutes) to find the top 20% of Mamba heads with the largest Effective Receptive Field (ERF), which are the unstable ones.  
  - For these selected heads, it scales their discretization step size $\Delta_t$ by a factor of $\frac{1}{n}$ (where $n$ is the context scaling factor). This effectively "slows down" the state updates, preventing explosive growth.  
  - It seamlessly integrates with existing Transformer CLE methods like YaRN to handle the positional encodings in the Transformer layers.
- UPI is evaluated on models like Bamba and Nemotron-H, extending their effective context from 4K/8K up to 64K tokens. It achieves substantial reductions in perplexity and improves performance on long-context benchmarks (RULER, LongBench) without any fine-tuning, outperforming baselines that combine methods like LongMamba and YaRN.

### Strengths
- The paper is well written and easy to follow.  
- The paper provides the first systematic analysis of why hybrid models fail on long contexts.  
- UPI is a simple, lightweight, and highly effective. Its training-free nature and minimal computational overhead for calibration.  
- The experiments are comprehensive, using multiple benchmarks (PG-19, LongBench, RULER) and models. The results are convincing, showing dramatic improvements in perplexity and task accuracy at extended contexts.

### Weaknesses
- The choice of the "top-20% of heads by ERF" is shown to be effective but is ultimately a heuristic. While the paper shows this selection is robust across datasets, a more theoretical or analytical justification for this specific threshold would strengthen the method.  
- The largest model tested is 9B parameters. While this is a reasonable scale, validating the method on larger hybrid models would be important to confirm its scalability.  
- The results on the pure Mamba2 model show that UPI is outperformed by the LongMamba baseline on the LongBench benchmark.  
- The tables do not include comparison to DeciMamba [1].  

[1] DeciMamba: Exploring the Length Extrapolation Potential of Mamba

### Questions
- How well does this method scale to larger models?  
- Could you please include a comparison with DeciMamba?

### Soundness
3

### Presentation
4

### Contribution
4
