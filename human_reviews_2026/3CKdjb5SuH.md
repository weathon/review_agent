# Once-More: Continuous Self-Correction for Large Language Models via Perplexity-Guided Intervention

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 4

## Abstract
Large Language Models (LLMs) often experience compounding errors during long text generation. Early mistakes can propagate and lead to drift, faulty reasoning, or repetition. While scaling up models improves capabilities, it requires substantial computational resources, and the resulting self-correction behaviour remains unpredictable at inference time. Self-correction is a promising technique for addressing this issue. However, existing approaches have limitations. Supervised training methods can build self-correcting behaviours into models, but require training data collection and lack cross-domain generalizability. Current post-hoc iterative refinement methods operate only at inference time, but must wait for substantial portions of the draft to be generated before providing feedback. This feedback does not guarantee effective guidance, and the same mistake patterns can still reappear. In this paper, we introduce Once-More, a model-agnostic post-hoc self-correction framework that intervenes during generation. Once-More leverages token-level perplexity and feedback from verifiers to provide continuous guided steering of the generation path through a logit redistribution mechanism. This approach essentially helps accumulate "more correct" steps throughout the generation process. Evaluation on multiple benchmarks demonstrates that Once-More achieves state-of-the-art results compared to other self-correction methods. To our knowledge, Once-More is the first post-hoc method to leverage token perplexity and external feedback to perform continuous guided self-correction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a novel algorithm for self-correction at a more fine-grained level of thoughts, sentences, or reasoning steps. The self-correction process is driven by model uncertainty estimation to guide the model toward the correct reasoning path. The authors also develop a procedure for suppressing incorrect token generation during the refinement stage. They demonstrate improvements over several baselines.

### Strengths
In general, I find this work valuable for several reasons:
- the introduction of self-correction on a more fine-grained level is novel, up to date, and an important step;

- the general framework is a great contribution, especially when driven by the uncertainty estimator, which is a powerful tool for analyzing correctness and is believed to be a bottleneck in self-correcting papers;

- interesting decoding strategy with observable gains.

### Weaknesses
Despite the general novelties in the methodology of the paper, I find it very raw in terms of experiments and presentation. I find the ideas really worthy and encourage authors to continue in this direction, but for now the empirical proof of validity is very scarce and unconvincing. Let me elaborate.

**Experimental Procedure**

 The main weakness I see lies in the very weak baselines. For example,
- Self-Refine is known to struggle with small models, and it is one of the first techniques.

- The implementation of CRITIC is absolutely awful: the RAG system is not defined properly, as there is no clear definition of the search query, search engine, or how the context is incorporated. Moreover, taking the top 1000 words from the search is not a proper RAG system; it is simply an incorrect implementation of the method.

- There is also an absence of other valuable baselines. I can name several self-correction techniques that should have been considered: Self-Correction [3] and its generalization STaSC [1], SCoRE [4], and REFINER [2].

- Another issue is the absence of adequate classical methods. It would be very reasonable to include methods with the same compute intensity. Once self-refinement generates multiple times, the authors should include a self-consistency approach leveraging the same amount of compute.

Regarding the existing benchmarks, SC3-MATH results are not reported on AIME24, AIME25, LiveBench, or GPQA.

Despite the benchmarks, there is evidence that the variation of such approaches is quite high, so reporting the standard deviation is a must here.

I’m really concerned about the token usage section. How is token usage calculated? Conditioning a model on 10 tokens is one forward pass, while generating 10 tokens involves 10 forward passes — these are not equivalent, which raises a lot of questions. Moreover, i see no comparison with SC3.

Regarding the experiments, they really lack more in-depth analysis. For example the ablation for PPL-K, why using PPL at all, when there are many other measures of uncertainty, such as entropy, sequence probability, etc.?

Moreover, I doubt the generalization of the uncertainty estimator threshold, such as perplexity. The beginning of the paper states the aim for generalization, but a PPL-based threshold would hardly generalize to other architectures — in fact, it is known to be difficult to achieve [5].

**Presentation**

In general, the paper is difficult to follow. The methods are presented unclearly, and the guided generation process, in particular, remains quite opaque. Many of the technical components could be introduced and explained in a clearer and more structured way.

Moreover, the paper seems somewhat overloaded with various tricks and heuristics. It would be more effective if the authors focused on one specific aspect and demonstrated its effectiveness comprehensively, rather than introducing multiple loosely justified additions. For instance, the idea of fine-grained revision even without the decoding stage is already promising on its own, i really like it. It would be valuable to show that it works well and analyze it thoroughly. I hope that ICLR is still a place for scientific papers, not a pursuit of state-of-the-art leaderboard results. After reading the paper, I gained little new understanding beyond the notion that fine-grained corrections may help, which, however, still lacks solid empirical support.

For example, when the model rolls back to a previous step, is there any analysis of such behavior? How often does it occur? How accurate are these corrections? These are important questions that could open up valuable future research directions.

Finally, several important details are missing, and the accompanying repository does not include the code. This significantly lowers the reproducibility of the work, which is unfortunate.



[1]  Moskvoretskii, Viktor, Chris Biemann, and Irina Nikishina. "Self-Taught Self-Correction for Small Language Models." arXiv preprint arXiv:2503.08681 (2025).

[2] Paul, Debjit, et al. "Refiner: Reasoning feedback on intermediate representations." arXiv preprint arXiv:2304.01904 (2023).

[3] Welleck, Sean, et al. "Generating sequences by learning to self-correct." arXiv preprint arXiv:2211.00053 (2022).

[4] Kumar, Aviral, et al. "Training language models to self-correct via reinforcement learning." arXiv preprint arXiv:2409.12917 (2024).

[5] Moskvoretskii, Viktor, et al. "Adaptive retrieval without self-knowledge? bringing uncertainty back home." arXiv preprint arXiv:2501.12835 (2025).

### Questions
“Struggle with domain generalization, and still suffer from error accumulation (a single failed self-correction will just cascade through subsequent generations).” — there is no proof provided for this claim. It might be true, but as stated, it sounds like an overstatement.

y_i is not defined l196

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Once-More, a perplexity-guided, continuous self-correction framework that intervenes during autoregressive generation by monitoring token/unit perplexity, invoking verifiers when uncertainty is high, and applying a perplexity-driven logit redistribution to guide regeneration. The method is model-agnostic and evaluated on several reasoning and math benchmarks (AIME24/25, LiveBench, GPQA, SVAMP, GSM8K) using Qwen3 models (4B/8B/14B) and comparisons with iterative refinement and supervised baselines. Results show consistent improvements and a token-efficiency advantage over some baselines. The idea is timely and practically relevant for improving reliability of LLM generation.

### Strengths
1.Continuous, unit-level perplexity monitoring combined with verifier feedback and logit redistribution is a sensible and practical inference-time approach to reduce error compounding without retraining.
2.Experiments cover multiple benchmarks and model sizes, showing gains across tasks and reasonable comparisons to both post-hoc and supervised baselines.
3.The paper includes component ablation and token usage comparison, which help motivate the claimed efficiency advantage over some iterative refinement methods.
4.Appendices contain algorithm pseudocode and mathematical derivations; hyperparameters are stated in the main text.

### Weaknesses
1.Several grammatical mistakes and minor typos appear in the manuscript (examples: “these methods often fails” should be “these methods often fail”; “it still suffer from this issue” should be “it still suffers from this issue”; duplicated word ”under under”).
2.Results are averaged over three runs but standard deviations or confidence intervals are not reported.
3.Ablation covers main components (feedback and redistribution) but lacks sensitivity analysis over key hyperparameters.
4.Token counts are reported, but there is no quantitative measurement of inference latency or computational overhead.
5.The conclusion reiterates results but would benefit from an explicit discussion of limitations and planned directions for improvement.

### Questions
1.Could you elaborate on how verifiers are chosen or configured for different tasks (e.g., math reasoning vs. factual QA)? Is there any criterion for when to use LLM-based versus rule-based verifiers?
2.Since Once-More operates on adaptive “units” (clauses, sentences, paragraphs, or code blocks), how is this segmentation determined in practice? Is it based on punctuation heuristics or learned boundaries?
3.Have you observed cases where Once-More fails to correct errors or even amplifies them? For example, when perplexity misjudges a correct but low-probability reasoning step. How often does this happen, and how do you mitigate it?
4.During regeneration, how sensitive is the framework to the quality or consistency of verifier feedback? Would weaker verifiers (e.g., smaller LLMs) substantially degrade performance?

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
This paper introduces Once-More, a model-agnostic framework that enables continuous self-correction during large language model (LLM) generation by combining token-level perplexity monitoring with verifier feedback. Unlike post-hoc refinement methods that only intervene after a full output is produced, Once-More operates incrementally, detecting and correcting high-perplexity segments before errors propagate. Its core mechanism redistributes token logits to steer the model toward more reliable generation paths while preserving its internal beliefs. Experiments on reasoning and mathematical benchmarks show consistent performance gains over baselines such as Self-Refine, CRITIC, and S3c-MATH, with improvements up to 10 points and reduced token overhead. The framework demonstrates that perplexity-guided, fine-grained intervention can make LLM generation more accurate and stable without retraining or modifying the base model.

### Strengths
- The paper introduces a simple yet effective framework for real-time self-correction that can be applied to any language model without retraining or architectural changes.

- The method provides a clear mechanism for detecting and mitigating local generation errors, showing how token-level perplexity can guide correction in an interpretable way.

- Experimental results demonstrate consistent performance gains across diverse reasoning and math benchmarks, confirming the approach's robustness and practicality.

- The framework is computationally lightweight compared to multi-pass refinement methods, since it reuses the model's own activations and verifier feedback rather than generating full alternative outputs.

- The paper contributes new insights into how continuous monitoring of model uncertainty can improve stability and factual accuracy in step-by-step generation.

### Weaknesses
- The paper lacks a clear theoretical justification for why token-level perplexity reliably correlates with factual or reasoning errors, relying mainly on intuition and anecdotal examples (Figure 2).

- The approach introduces additional inference cost through iterative token re-evaluation and verifier calls, but the paper does not report latency, FLOPs, or wall-clock overhead to substantiate its claims of efficiency. The only quantitative evidence concerns token usage, which measures output length rather than actual computational or time efficiency.

- The method's reliance on external verifier feedback limits reproducibility and fairness, since different verifier choices or access levels could yield inconsistent results across users and model settings.

- The verifier component is insufficiently analyzed: the paper does not show how verifier quality, domain alignment, or calibration affect correction accuracy or error propagation.

- The experiments focus narrowly on reasoning and math datasets, leaving it unclear whether the method generalizes to open-ended generation, dialogue, or long-context settings.

### Questions
How does Once-More decide when to trigger a correction during generation, and what threshold or criterion determines whether a high-perplexity segment actually warrants intervention?

### Soundness
3

### Presentation
2

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
This work proposes a model-agnostic framework for continuous self-correction for LLMs at inference time. Unlike contemporary post-hoc iterative refinement methods that wait for complete outputs, authors propose to intervene at a more granular level of a "unit" (sentences, paragraphs or code blocks) using two key mechanisms: (1) perplexity monitoring to detect potential errors (2) fusing logit redistribution with verifier feedback (LLM-based or symbolic) to guide regeneration away from failed attempts.

The work motivates the necessity to intervene early to prevent the propagation of errors in early steps to the final outputs. The continuous intervention approach is conceptually interesting, and to the best of my knowledge, the combination of perplexity signals with external feedback mechanisms is novel. The proposed method achieves strong empirical results on mathematical reasoning (AIME), the reasoning subset of LiveBench, and GPQA compared to Self-Refine and Critic.

However, the work suffers from several practical limitations, including complex hyperparameter setup, several missing details, and evaluation being limited to reasoning tasks.

### Strengths
- Novel Combination of Signals: The proposed method uniquely integrated unit-level perplexity with external verifier feedback for continuous guidance, correcting the wrong steps in the early stages.
- Model-Agnostic and Training Free: The framework shows strong performance despite being an inference-time technique. The setup is model agnostic and thus can be broadly applied.
- Token Efficiency Gains: The proposed method shows gains in generated token efficiency while achieving better performance.
- Scaling Behaviour: The Authors show that the framework benefits more from larger models, suggesting it effectively leverages richer model representations.

### Weaknesses
- The authors provide no ablation for the selection of the "unit". Unit granularity plays a pivotal role in the proposed method; however, it is not clear how the performance changes with the changing definition of the unit.
- The core assumption of the paper is that each unit can be verified without looking at the future unit. A discussion on this is warranted.
- The method introduces numerous hyperparameters (K for top-K, $\alpha$ for suppression strength, $\beta$ for redistribution sharpness, $\sigma$ for diffusion bandwidth, $\tau$ and $\gamma$ for distance decay, and $c$ for truncation radius) with minimal justification. Section 4.1 only provides the value for K, $\alpha$, $\beta$ without sensitivity analysis. The authors provide no ablation experiments to study the effects of various hyperparameters or how they were selected.
- Perplexity threshold is calibrated using empirical quantiles on a held-out set of prompts. However, no further details are provided, such as how many prompts were used for calibration or the value of verification rate $\eta$. Similar to other hyperparameters, no sensitivity analysis is done for this value.
- Authors provide a computational cost analysis in the form of generation token efficiency. However, they do not provide information on wall-clock time. Depending on the selection of units, the wall clock time may exceed that of contemporary methods. I believe an analysis of the number of verifier calls, wall-clock time will make the claims stronger.
- It is not clear if the proposed method works for an open-ended generation task. Authors limit their experiments to reasoning tasks where units can be sufficiently defined. However, for open-ended tasks such as story generation, the definition of a unit and the core assumption (each unit can be verified without access to future steps) may break down.

### Questions
- Verifier Quality Analysis: What happens when a stronger verifier is used in an asymmetric setting? I understand this method proposes to improve the self-correction capabilities; however, seeing that the method generalizes well with scale, it may be worth showing sample results in an asymmetric setting with a stronger verifier.
- Hyperparameter Sensitivity: Please provide sensitivity analysis for all hyperparameters. Are the results robust to these choices, or does performance degrade significantly with the choice of other hyperparameter values? At the very least, please provide values for all hyperparameters that were used for experiments.
- Calibration Data Requirements: How many held-out examples are needed for reliable $P_{th}$ calibration? How does performance degrade if calibration and test distributions differ?
- Computational Overhead: What is the average wall-clock time increase compared to raw generation, and chosen baseline method?
- Perplexity-Error Correlation: What is the actual correlation between $PPL_{unit} > P_{th}$  and the presence of errors in your datasets? What are the false positive/negative rates? How does wrong perplexity calibration affect the downstream performance? 
- Verifier Disagreement: How is verifier disagreement handled in the case of multiple verifiers? Does the framework support weighted voting or consensus mechanisms?
- Complexity of Logit redistribution: Have the authors tried simpler logit penalties/redistribution strategies? What is the comparison with a simple token penalty based on perplexity? Are all the components, such as distance decay, diffusion to prevent overlocalization, important? An ablation of these components would be appreciated.

### Soundness
2

### Presentation
3

### Contribution
3
