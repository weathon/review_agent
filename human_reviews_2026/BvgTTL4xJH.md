# Diffusion with Truncated Blocks: Towards Fast and High-Quality Text Generation using Truncated Block Generation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 0, 6

## Abstract
Diffusion-based Large Language Models (dLLMs) are emerging as a powerful alternative to traditional autoregressive models. These models learn to generate text by iteratively denoising masked sequences. In this work, we identify a critical problem in dLLMs that using token-level noise: the model's attention is wastefully expended on uninformative mask tokens, diluting its focus on meaningful context. We term this phenomenon ``attention dilution". We further show that it is an artifact of token-level noising, whereas models with sentence-level noise does not have such phenomenon. To resolve this problem, we introduce Truncated Block Generation, a novel sampling algorithm that not only mitigates attention dilution but also enables faster inference and flexible-length sequence generation. Extensive experiments validate our analysis and demonstrate the marked effectiveness of our proposed method in enhancing both the performance and efficiency of dLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Truncated Block Diffusion, a decoding strategy for token-level diffusion language models (DLMs) that mitigates the attention dilution problem — where excessive masked tokens reduce focus on informative context. The proposed method dynamically truncates and resumes generation in smaller blocks, allowing the model to maintain attention on semantically meaningful regions while supporting flexible-length decoding. The authors present theoretical analysis of how token-level noise leads to attention dilution and propose truncation to maintain contextual density. Experiments on code (HumanEval, MBPP) and math reasoning tasks (GSM8K, MATH) show improved accuracy and efficiency over baselines such as Dream and LLaDA, with results consistent across sequence lengths. While the technical design is well motivated and validated, the paper omits detailed analysis of failure cases or ablations regarding truncation sensitivity. The writing also tends to overemphasize strengths without clarifying limitations in robustness, calibration dependency, or computational trade-offs. Claims of compatibility with other decoding accelerations (e.g., Fast-dLLM) are promising but should be stated more cautiously since direct integration experiments are not reported.

### Strengths
1. The paper identifies and analyzes a concrete limitation of token-level diffusion (attention dilution), offering both theoretical reasoning and empirical validation.
2. The proposed Truncated Block Generation is conceptually simple, implementation-friendly, and does not require retraining, making it broadly applicable to existing DLMs.
3. Extensive experiments on reasoning and code tasks demonstrate consistent performance and speed improvements under fixed compute budgets.
4. The theoretical justification for truncation and the visual attention analysis are clear and insightful, helping to connect the intuition of “context density” with measured decoding quality.

### Weaknesses
1. The approach is primarily validated on structured domains (code and math) using Dream and LLaDA; generalization to open-ended or natural language tasks remains unclear.
2. The model’s reliance on token-level confidence for truncation may lead to instability under poorly calibrated confidence distributions.
3. The paper lacks qualitative or failure-case analysis — e.g., what happens when truncation disrupts a coherent semantic span, leading to invalid continuations.
4. No adaptive or learnable mechanism is explored for determining truncation boundaries, which could improve robustness but is omitted.
5. Broader comparisons to models with weaker calibration or longer contexts are missing, leaving uncertainty about the scalability and universality of the method.
6. The paper provides qualitative claims of acceleration but lacks comprehensive tables comparing latency and total compute under matched budgets.

### Questions
* **Baseline coverage and robustness under weaker confidence calibration:** Since the truncation relies on confidence-based masking, how does the approach perform on DLMs with less stable confidence scores? Would the authors include results on other diffusion LMs to verify robustness?

* **Behavior under flat or uniform confidence distributions:** When token confidences are nearly uniform, how does the model decide truncation or continuation? Can the authors provide visualization or analysis showing this failure mode?


* **Failure case analysis and adaptive truncation:** In some generations, errors may arise when truncation happens across semantically connected regions, breaking coherence. Have the authors analyzed such cases? Could an *adaptive truncation policy*—where the model learns when to truncate—alleviate this issue?

* **Compute and latency analysis:** Could the authors provide explicit comparisons of generation speed, latency, and total step counts versus fixed-length or block-decoding baselines to better quantify the claimed acceleration?

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
Truncated Block Generation combats attention dilution in diffusion LMs with token-level noise by decoding in short blocks and truncating before rollover, improving code/math accuracy and throughput under compute parity.

### Strengths
1. Formalizes “attention dilution” under token-level noise with definitions/claims plus intuitive visual evidence.
2.  Training-free; integrates with existing diffusion LMs without architecture changes.
3. Truncation removes boundary cues that trigger early <eos>, reducing “half-baked” spans.
4. Consistent gains on HumanEval/MBPP (and math sets) alongside higher tokens/sec.
5. Stacks with Fast-dllm; retains more accuracy than using acceleration alone.
6. Threshold-gated continuation makes length flexible instead of being bound to a single fixed [MASK] tail.

### Weaknesses
1.  Benefits concentrate on token-level noise (e.g., Dream); for sequence-level noise (e.g., LLaDA) the lift is muted—generalizability is constrained.
2. Block length, truncation length, and threshold γ all matter; best settings vary by task, raising tuning costs.
3. Focuses on code/math; lacks long-form generation, dialogue coherence, factuality, or human evals.
4. Compared to fixed-length, naive block, and Fast-dllm, but fewer head-to-heads with semi-autoregressive / multi-step remasking and other modern decoding strategies.
5. The “information value” modeling is stylized; real multi-layer attention/copy dynamics may deviate, so theory–practice gaps can appear on new models/data.

### Questions
1. Systematically sweep block length, truncation length, and the threshold γ; report mean ± σ across multiple random seeds and show performance/latency trade-off curves.
2. Replace raw max-softmax triggers with entropy/energy-based or calibrated-confidence signals; auto-tune γ and truncation length online to reduce hand-tuning and improve robustness OOD.
3. Evaluate beyond code/math: include long-context reasoning, dialogue coherence, and factual QA; add broader code sets (e.g., DS-1000) plus long-form generation tasks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes Truncated Block Generation (TBG), a decoding method for diffusion-based LLMs trained with token-level noise (e.g., Dream).
It argues that such models suffer from attention dilution-where long tails of uninformative mask tokens weaken attention focus.
TBG generates text in smaller masked blocks and truncates partial outputs iteratively, claiming to improve efficiency and text quality for long sequences.

However, the approach closely overlaps with semi-autoregressive or blockwise diffusion decoding already established in SSD-LM and Block Diffusion (BD3-LMs). The only new element-truncation-is a heuristic, not a fundamental algorithmic advance. Baselines are incomplete, and empirical support for both speed and “attention dilution” is weak.

### Strengths
Identifies a plausible failure mode (attention dilution) in token-level noise training.
Simple heuristic (TBG) that is easy to implement on top of existing diffusion LLMs. However, the approach is not properly compared agianst prior baselines!
Empirical results show modest improvements on some reasoning and coding benchmarks, still not compared to prior methods!

### Weaknesses
* The paper lack novelty and fails to compare against prior blockwise diffusion baselines such as SSD-LM and BD3-LMs (https://arxiv.org/pdf/2503.09573) in comparisons, which severely undermines the claim of novelty and contribution.

The proposed Truncated Block Generation (TBG) is conceptually almost identical to SSD-LM (Han et al., 2023) and Block Diffusion / BD3-LMs (Arriola et al., 2025), which already generate text in sequential diffusion blocks conditioned on previous outputs. However, such strong 
Both prior works support variable-length text generation, KV caching, and efficient blockwise denoising - the same core benefits claimed here. The only new element is the truncation heuristic, where the generated block is shortened before continuation. However, Looking into appendix this approach comes with heavily tuning the hyper-parameters which questions the practicality of this approach in realworld setting. 


The main baseline is “Dream with full-length mask decoding,” which is known to perform poorly on long outputs and serves as a weak strawman.



* Unconvincing theoretical framing (“attention dilution”)
The “attention dilution” argument - that many uninformative MASK tokens from token-level noise distract the attention distribution - is intuitively reasonable but not experimentally validated.


* The analysis merely restates the known property that softmax weights are normalized over all keys; it does not establish a causal connection between dilution and degraded text quality.

The authors argue that truncating uninformative MASK positions (or reducing context length)  provide improvements.
Since sequence-level noise models (like LLaDA) are unaffected by this problem, a simpler alternative would be to adjust training rather than add decoding heuristics.

* Strong baselines such as SSD-LM, Block Diffusion (BD3-LMs), and LLaDA with standard decoding are not compared, even though they directly address the same limitations. without proper comparison, it is unclear if the approach bring any benefits. It remains unclear whether TBG helps other diffusion LMs, semi-autoregressive LMs, or sequence-level-noise systems that already avoid dilution.


No comparison with autoregressive models on runtime or accuracy, despite the claim of “fast and high-quality generation.”

“Faster inference” is repeatedly claimed but not substantiated: the paper reports no wall-clock time, throughput (tokens/sec), FLOPs, or NFEs (number of function evaluations).

TBG introduces multiple iterative decoding rounds (generate -> truncate -> repeat), each requiring diffusion denoising, which likely increases latency. this is good to clarify this in the paper.


* High hyperparameter sensitivity and tuning overhead

TBG depends on several heuristic hyperparameters (block length, truncation length, threshold), which are tuned per dataset . Looking into appendix it looks like they are heavily tuned. this would question practicality of this method.  In contrast, block diffusion and SSD-LM decoding work robustly across datasets without such per-task adjustments. This extensive tuning contradicts the claim of being a “simple, fast decoding algorithm.”

There is no direct causal experiment showing that truncation specifically restores attention concentration or improves generation quality for the same noise schedule.

### Questions
How is TBG different from SSD-LM or Block Diffusion decoding beyond the truncation heuristic?

Paper needs to compare with SSD-LM and BD3-LM as prior baselines. Could you provide the comparisons?

If attention dilution only arises in token-level noise (Dream), why not just adopt sequence-level noise (LLaDA)?

could you provide wall-clock or NFE comparisons supporting “faster inference”?

How sensitive is TBG to hyperparameter tuning (block length, truncation, threshold)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies and analyzes an “attention dilution” problem that arises in masked/diffusion LLMs under token-level noising, and proposes a simple yet effective sampling strategy called Truncated Block Generation to mitigate this issue—thereby improving generation quality and speeding up inference.

### Strengths
1. It pinpoints and rigorously analyzes a previously overlooked “attention dilution” problem in token-level noising dLLMs, supporting the claim with both theoretical arguments and attention visualizations.

2. It proposes a simple, practical sampling algorithm—Truncated Block Generation—that directly mitigates the dilution issue by generating in short blocks and truncating to keep context informative, making the method easy to integrate into existing dLLM pipelines.

3. The approach is empirically validated: experiments and ablations show consistent quality improvements and inference speedups on code and math benchmarks (e.g., MBPP, HumanEval, GSM8K), and the paper demonstrates robustness to key hyperparameters like truncation length and threshold.

### Weaknesses
1. There needs to be some baseline comparison, such as adding comparisons using methods like remasking during the inference stage. Currently, there are almost no baselines.

2. I want to see how this method performs on some general tasks such as MMLU and GPQA.

### Questions
1. How does truncation interact with long-context reasoning or compositional generation—does repeatedly truncating and regenerating blocks risk losing global coherence or factual consistency over long outputs?

2. I observed that the longer the generated length, the better the performance. What if the generated length is 8k or even 16k? What are the advantages of this method?

### Soundness
3

### Presentation
3

### Contribution
3
