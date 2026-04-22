# ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
While most autoregressive LLMs are constrained to one-by-one decoding, diffusion LLMs (dLLMs) have attracted growing interest for their potential to dramatically accelerate inference through parallel decoding. Despite this promise, the conditional independence assumption in dLLMs causes parallel decoding to ignore token dependencies, inevitably degrading generation quality when these dependencies are strong. However, existing works largely overlook these inherent challenges, and evaluations on standard benchmarks (e.g., math and coding) are not sufficient to capture the quality degradation caused by parallel decoding. To address this gap, we first provide an information-theoretic analysis of parallel decoding. We then conduct case studies on analytically tractable synthetic list operations from both data distribution and decoding strategy perspectives, offering quantitative insights that highlight the fundamental limitations of parallel decoding. Building on these insights, we propose ParallelBench, the first benchmark specifically designed for dLLMs, featuring realistic tasks that are trivial for humans and autoregressive LLMs yet exceptionally challenging for dLLMs under parallel decoding. Using ParallelBench, we systematically analyze both dLLMs and autoregressive LLMs, revealing that: (i) dLLMs under parallel decoding can suffer dramatic quality degradation in real-world scenarios, and (ii) current parallel decoding strategies struggle to adapt their degree of parallelism based on task difficulty, thus failing to achieve meaningful speedup without compromising quality. Our findings underscore the pressing need for innovative decoding methods that can overcome the current speed-quality trade-off. We release our benchmark to help accelerate the development of truly efficient dLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presented an information-theoretic analysis on the capabilities of diffusion LLMs using toy tasks to compute their theoretic bounds. The paper performed a distribution based analysis and a decoding strategy based analysis. The paper also presented ParallelBench, composed of the toy tasks and realistic tasks, to evaluate the capabilities of dllms empirically.

### Strengths
- The paper presents a theoretical explanation for the capabilities and limitations of parallel decoding with dLLMs. This provides some intuition for whether to use dLLMs or whether to enable decoding multiple tokens per step depending on the task at hand. The theoretical result is corroborated with empirical results on the same toy tasks.
- The empirical evaluation performed a wide variety of ablations that provide some insights on how different unmasking techniques and models compare.
- The presented benchmark can serve as a baseline for evaluating whether a particular language model/decoding technique can adaptively exploit parallelism, beyond dLLMs.

### Weaknesses
- The empirical evaluation on realistic tasks seem to deviate from the expectations set by the theoretical results. For example LLaDA 1.5 performs better on Latin Square than Sudoku, but Sudoku has C(Y|X) = 0 while Latin Square has C(Y|X) > 0. This weakens the significance of the theoretic analysis. The difference in complexity from the toy tasks to realistic tasks seem dominate the difference in task type.
- The empirical results for 3f-j show accuracy going down as the # tokens per step increase, which is to be expected that the more naive parallelism (as in the technique does not explicitly target/decide when parallelism is appropriate) employed the worse the accuracy would be. Most realistic tasks should fall into these regimes, rather than the extremes of Copy or Replace Index. And it is unlikely to be possible to perform the same information-theoretic analysis on realistic tasks to quantify the theoretic accuracy. 
- The best the analysis provides is an intuition, which in general can be useful. However, the intuition that more dependencies between tokens in the expected response makes parallelism harder is hardly surprising. The difficult task is knowing when parallelism can be enabled, which the work does not provide directions towards any solutions, as the toy examples shown do not generalize/scale to real examples.

### Questions
- How is it that Sudoku having supposedly C(Y|X) = 0 and Latin Square having C(Y|X) > 0, but the empirical results showing that LLaDA 1.5 performs better on Latin square than Sudoku? 
- On Figure 7, what is the arrow indicative of?
- Is there a reason why ParallelBench is for evaluating dLLM specifically and not parallel decoding in different LLMS (e.g. parallel decoding in AR LLMs) in general?

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
The paper introduces ParallelBench, a benchmark and analysis suite for diffusion LLMs (dLLMs) that isolates the speed--quality trade-offs of parallel decoding. It provides an information-theoretic lower bound (via conditional total correlation) showing why parallel decoding degrades when token dependencies are strong, confirms the theory on analytically tractable list operations, and then demonstrates the same failure modes on realistic tasks across three categories (Waiting Line, Text Writing, Puzzles). The study further contrasts static (Top-k) and adaptive (Threshold) unmasking, semi-AR strategies, and "oracle" per-sample thresholds, revealing substantial headroom for adaptive methods.

### Strengths
1. The theory connects nicely to practice: the conditional-correlation argument predicts the “New City” style errors you later see in real tasks. It’s satisfying when the math lines up with the plots.

2. The benchmark fills a gap. We've all seen parallel decoding look great in one demo and fall apart in another; this suite finally gives a way to measure where it cracks and by how much.

3. The oracle threshold curves are especially helpful -- they show there’s real headroom for adaptive, per-sample control rather than one-size-fits-all knobs.

4. The write-up is clear and the figures are readable; I didn’t have to reverse-engineer the setup to follow the argument.

### Weaknesses
1. Coverage feels a bit narrow in the main text. A compact table that pulls model/method results out of the appendix would make the big picture easier to scan.

2. The quality metrics are fine, but a couple more "at a glance" indicators (e.g., BERTScore or constraint-violation counts) would help interpret how quality degrades as parallelism goes up.

3. Most examples are short sequences. One longer-form scenario in the main paper would help readers judge whether the conclusions hold in more realistic lengths.

### Questions
1. You show big gaps between fixed thresholds and the oracle per-sample threshold. How close can a simple learned heuristic (say, a tiny classifier on token stats) get to the oracle without heavy training?

2. The task taxonomy is useful. Could you add a small table of "meta-features" (rough dependency strength, locality vs. global constraints, etc.) so practitioners can guess which decoding policy to use before running heavy tests?

3. Semi-AR helps sometimes and hurts other times. Do you envision a lightweight runtime policy that flips block sizes per sample using quick signals (entropy, margin, "AR-ness")? Any early results there?

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
2

### Summary
his paper investigates the speed–quality trade-off in parallel decoding for Diffusion Language Models (dLLMs). Although dLLMs promise faster inference through parallel decoding, the underlying conditional independence assumption often leads to severe quality degradation in tasks with strong token dependencies. To expose this issue, the authors provide an information-theoretic analysis and introduce ParallelBench, specifically designed for evaluating dLLMs under parallel decoding. It comprises 17 tasks across three categories (Waiting Line, Text Writing, Puzzles) that are easy for humans and autoregressive (AR) models but challenging for dLLMs. Experiments reveal that (1) dLLMs experience substantial quality loss during parallel decoding, and (2) existing decoding strategies (both static and adaptive) fail to balance quality and speed effectively.

### Strengths
1. This paper tackles a core dLLM challenge: the quality impact of parallel decoding, a key advantage over AR models, and highlights the limitations of existing benchmarks in assessing it.
2. Provides theoretical insights.
3. Clearly illustrates token dependency variations across tasks through analysis of synthetic “list operations” such as Copy, Replace Index, Replace Random, and Shuffle.
4. Systematically evaluates multiple dLLMs—including LLaDA, Dream, and the closed-source Mercury—across various decoding strategies such as Top-k, Threshold, and Semi-AR on PARALLELBENCH.

### Weaknesses
1. Need for benchmark comparison: More experiments are needed to show PARALLELBENCH’s added value over existing benchmarks.
2. Missing actual speed measurements: Experiments focus on parallelism vs. quality, but no wall-clock latency or time–quality curves are provided.
3. Limited real-world coverage: The benchmark may not capture all challenges dLLMs face in complex, open-ended tasks.

### Questions
See Weaknesses

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
The paper studies parallel decoding in diffusion-based large language models (dLLMs) in order to address the challenge that parallel decoding often fails to capture inter-token dependencies leading to quality degradation, due to the conditional independence assumption among tokens.

The authors make three main contributions:
	
1. Information-theoretic analysis — They formalize the lower bound of parallel decoding quality loss using conditional total correlation C(Y|X), proving that even ideal models face an inherent speed–quality trade-off.
	
2.	Synthetic case studies — They analyze list operations like Copy, Replace Random, and Shuffle to quantify how dependency strength affects parallel decoding accuracy.
	
3.	PARALLELBENCH — A new benchmark with 17 tasks across 3 categories (Waiting Line, Text Writing, Puzzles) to empirically evaluate dLLMs and AR LLMs. It exposes how parallel decoding degrades quality in realistic settings and shows that current adaptive unmasking methods (e.g., Top-k, Threshold) cannot fully balance speed and accuracy.

The paper concludes that dLLMs suffer from severe quality degradation under parallel decoding, especially in dependency-heavy tasks, and that current decoding strategies fail to adapt parallelism dynamically to task difficulty.

### Strengths
1. The use of conditional total correlation to quantify unavoidable quality loss provides a solid mathematical basis for analyzing the parallel decoding trade-off, making the paper has clear theoretical grounding.

2. The combination of theoretical proofs, synthetic tasks, and realistic benchmarks (including grammar-sensitive and reasoning tasks) provides a well-rounded evaluation.

3. The combination of theoretical proofs, synthetic tasks, and realistic benchmarks (including grammar-sensitive and reasoning tasks) provides a well-rounded evaluation.

4. The writing, presentations accompanied with interpretations are overall clear. Code is publicly released, and the benchmark tasks are well-documented, facilitating future research.

### Weaknesses
1. Although 17 tasks span diverse categories, most involve short outputs or synthetic patterns; results may not generalize to long-context reasoning or dialogue generation.

2. Comparisons largely focus on fixed unmasking or basic threshold schemes, missing newer adaptive scheduling approaches (e.g., dilated scheduling, SlowFast decoding, or hybrid AR-diffusion). 

3. The paper focuses on decoding strategies, not how model pre-training or architecture might affect the issue. 

4. Other decoding/scheduling work that should be covered (and possibly compared): the paper could be strenghthen to include decoding strategies that cover dynamic stage-based scheduling [1], dilated unmasking[2], block decoding [3,5] and revocation/remasking [4]. 



-----
References
1. Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles (Wei et al., 2025) 
2. Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models (Luxembourg, Permuter, Nachmani, 2025) 
3. Fast‑dLLM: Training‑free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding 
4. Wide‑In, Narrow‑Out: Revokable Decoding for Efficient and Effective DLLMs (Hong et al., 2025)
5., Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (Arriola et al., 2025)

### Questions
1. The oracle analysis suggests per-sample thresholds could yield large gains — how feasible is this in practical inference (e.g., latency, calibration)?

2. How do results change with larger blocks or variable block scheduling strategies beyond fixed lengths (with semi-AR)?

### Soundness
3

### Presentation
3

### Contribution
3
