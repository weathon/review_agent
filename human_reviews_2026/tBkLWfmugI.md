# T1: Tool-integrated Verification for Test-time Compute Scaling in Small Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Recent studies have demonstrated that test-time compute scaling effectively improves the performance of small language models (sLMs).
However, prior research has mainly examined test-time compute scaling with an additional larger model as a verifier, leaving verification by sLMs underexplored.
In this work, we investigate whether sLMs can reliably verify the output candidates under test-time scaling.
We find that even with knowledge distillation from larger verifiers, sLMs struggle with verification tasks requiring memorization, such as numerical calculations and fact-checking.
To address this limitation, we propose Tool-integrated verification (T1), a two-stage framework that first filters candidates with external tools and then uses an sLM for final verification, offloading memorization-heavy steps to tools such as a code interpreter.
Within T1 we prove that offloading to external tools reduces the memorization burden on sLMs and improves test-time scaling performance.
Experiments on the MATH benchmark demonstrate that, with T1, a Llama-3.2 1B model under test-time scaling outperforms the significantly larger Llama-3.1 8B model. 
Moreover, T1 improves the verification accuracy of both process reward models (PRMs) and critic models.
Our findings highlight the potential of tool integration to substantially improve the verification abilities of sLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Tools can offload specific skills, such as arithmetic or fact retrieval, from language models to purpose-built systems. This paper extends that idea to verification: it introduces a tool-integrated verifier that allows smaller language models to act as effective verifiers. The method first applies a tool-based binary filter to discard incorrect candidates, and then uses a reward model to score the remaining answers. A simple theoretical model suggests that tool integration reduces the need for memorization, and experiments on GSM8K, MATH500 and MMLU-Pro show improved Best-of-N performance.

### Strengths
* Well written.
* Consistent empirical gains across several model families and sizes.
* Theoretical intuition connects tools to reduced memorization burden and to improved Best-of-N selection via reduced verifier noise.
* The fact that ToolV and GenRM stages are just LoRA adapters of the same model (which is the same as the reasoner) makes the deployment in low-resource scenarios easier.

### Weaknesses
1. I understand that offloading to tools allows smaller models, but the design choice seems odd: why restrict tool use to the verifier? Qualitative examples (e.g., Example G.1) show the verifier merely calls an equation solver that could equally have been used by the generator. Giving tools to the reasoner feels like the more direct and widely studied approach (e.g., Toolformert, PAL, TORA), so the novelty here mainly comes from an asymmetric setup rather than a new tool integration method.

2. It’s unclear if the improvements from Distilled GenRM → ToolV + Distilled GenRM arise from tool use or simply from adding another verification stage. A control experiment where the two-stage verifier is used but with no external tools would clarify whether the tools themselves drive the gains.
3. The proposed approach spends roughly 2× verification compute compared to the baseline, yet this is not accounted for. If Figures 3 and 4 plotted accuracy against the number of generated solutions, the ToolV + Distilled PRM curve should arguably be shifted right to reflect the extra verification cost. This additional cost might erase scaling gains in several cases. Without such normalization, it is difficult to judge compute-efficiency trade-offs fairly. Reporting wall-time, total output tokens, or at least the number of model calls would provide a more balanced comparison than using only the number of generations.

### Questions
Please provide the 2 suggested ablations above, and provide plots showing total cost in the x-axis, not only the number of generated solutions.

### Soundness
2

### Presentation
3

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
The paper proposes Tool-integrated Verification (T1): a two-stage verifier for small LMs (sLMs) that (i) first filters candidate solutions using a tool-based verifier (ToolV) e.g., running generated Python or fact-checking with retrieved documents and (ii) then scores remaining candidates with a reward-model verifier (PRM or generative verifier). The core idea is to shift memorization-heavy verification to external tools so sLMs benefit more from test-time scaling (Best-of-N). The method is trained via multi-LoRA distillation from large teacher verifiers. Experiments on MATH500, GSM8K, and MMLU-Pro report that sLMs with T1 outperform or match much larger models, although some larger-model baselines are run with N = 1 while sLMs use N up to 64, which complicates fairness.

### Strengths
* Clear theoretical decomposition of sub-optimality into OTC and a verifier-driven factor; three-regime picture is intuitive and actionable. (Theorem 3.6; Fig. 1.)
* Precise AiC analysis with explicit coverage violation condition and sub-optimality formulas.
* SMC construction via maximal coupling with matching complexity to SRS; neatly links transport optimality with compute.

### Weaknesses
* Plots lack error bars/CI; seeds aren’t disclosed. Given the sampling-heavy setup, dispersion matters. Report mean ± sd over ≥3 seeds, plus 95% CIs or stratified bootstrap CIs for all main tables/figures.
* The binary-generator and simple addition check help intuition but are far from math word-problem or retrieval settings. The paper should tighten the stated scope and discuss where the guarantees plausibly transfer.
* We don’t see how accuracy moves with # ToolV code completions (k), # retrieved docs, verifier thresholds, or retrieval quality (BM25 vs dense, top-k). These directly test your causal story; please add these sweeps.

### Questions
* Cost/latency is missing for a method that leans on tools. Can you report tokens, wall-clock, and $ per instance for T1 vs PRM/GenRM-only under equal budgets?
* Please show accuracy vs k code completions (e.g., 1–8), vs # retrieved docs (1–5), and vs verifier thresholds. These ablations would isolate where T1 helps most.
* Which components of the theorems do you believe carry over to real math/RAG settings, and which do not? A brief “theory-to-practice” paragraph would calibrate reader expectations.

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
3

### Summary
This paper proposes Tool-integrated Verification (T1), a two-stage framework that uses external tools (e.g., code interpreters) to filter candidate solutions before sLM-based verification, offloading memorization-heavy tasks like numerical calculations. The authors provide theoretical analysis showing tools reduce memorization requirements and prove the two-stage design improves test-time scaling performance. Experiments demonstrate that T1 enables a Llama-3.2 1B model to outperform Llama-3.1 8B on MATH benchmarks, improving verification accuracy for both PRMs and generative verifiers.

### Strengths
The paper enables small language models to be reliable verifiers by combining it with the tools and boosts their performance by test-time scaling. The paper also provides solid theoretical proofs for the method, not only to well explain the method in the paper, but also to show a general analysis foundation for future research. The experiment design is also comprehensive, where the proposed T1 is compared with Majority Voting, Zero-shot GenRM, Distilled GenRM and Themis. The performance improvements are also remarkable on Math500 and GSM8K.

### Weaknesses
- Limited novelty and narrow focus: The core idea of using code interpreters for mathematical verification is well-established (PAL, Toolformer). The contribution is mainly combining existing techniques in the verification stage rather than introducing new methods. The paper focuses almost exclusively on mathematical reasoning (MATH, GSM8K), a narrow domain where code execution naturally provides ground truth verification. Even for MMLU-Pro tasks in the appendix, that retrieval helps verification is unsurprising and well-known from prior RAG literature.
- Missing key ablation: The paper fails to evaluate ToolV alone as a verifier, only ToolV + RM combinations. This makes it hard to assess the individual contribution of each stage or determine if the two-stage design is necessary versus using ToolV alone.
- Missing cost-benefit analysis: No computational overhead reported for generating and executing code or retrieval, unclear if performance gains justify additional inference cost versus using a larger model.

### Questions
1. What’s the exact performance of the proposed method and baselines in the chart? Could you provide some statistics at least for some key points?
2. The MATH500 and GSM8K are fairly easy benchmarks. Could you try the more challenging benchmarks such as AIME 24, AIME 25 (also math domain)? Perhaps the method will shine better for challenging tasks.
3. As mentioned in the weakness section, could you add the ablation and the cost-benefit analysis?
4. Theoretically, the proposed method is not limited to the small LMs. Will the performance improvement vanish if the method is based on larger LMs?

### Soundness
3

### Presentation
3

### Contribution
2
