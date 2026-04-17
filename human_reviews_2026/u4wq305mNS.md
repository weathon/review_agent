# Beyond Parameters: Exploring Virtual Logic Depth for Scaling Laws

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Scaling the size of large language models typically involves 3 dimensions: depth, width, and the number of parameters. In this work, we explore a 4th dimension: virtual logical depth (VLD), which allows increasing the effective algorithmic depth without changing the overall parameter count by reusing parameters within the model. While parameter reuse is not new, its role in scaling dynamics has remained underexplored. Unlike currently trending test-time methods, which mainly scale in token-wise, VLD alters the internal computation graph scaling during training, inference, or combination. We carefully design controlled experiments and have the following key insights on VLD scaling: 1. Knowledge capacity vs. parameters. At a fixed parameter count, VLD leaves knowledge capacity nearly unchanged (with only minor variance), while across models knowledge capacity scales with the number of parameters; 2. Reasoning vs. reuse. Properly implemented VLD substantially improves reasoning ability without increasing parameter count, decoupling reasoning from sheer model size. This provides a new possibility for scaling besides the current token-wise test-time scaling used by most reasoning models. 3. Robustness and generality. The trend of improved reasoning persist across architectures and configurations (e.g., different reuse schedules and step counts), indicating that VLD captures a general scaling behavior. These findings not only provide useful insights into the future model scaling strategies, but also introduce an even deeper question: Does super intelligence necessarily require ever-larger models, or could it have some trade-offs by re-using parameters and increasing virtual logic depth? We believe that there are many unknown dynamics within the model scaling that need exploration. Codes are available at https://anonymous.4open.science/r/virtual_logical_depth-8024/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper “Beyond Parameters: Exploring Virtual Logic Depth for Scaling Laws” introduces Virtual Logical Depth (VLD) as a new scaling dimension for large language models, complementing the traditional three — depth, width, and parameter count.
VLD increases a model’s effective algorithmic depth by reusing parameters within the computation graph, thereby extending the model’s reasoning process without adding new parameters.

### Strengths
Conceptual: Introduces Virtual Logical Depth (VLD) as a fourth dimension in LLM scaling laws.

Methodological: Implements parameter and layer reuse to simulate increased logic depth without parameter growth.

Empirical: Demonstrates that VLD improves reasoning performance while keeping knowledge capacity and parameter count fixed.

### Weaknesses
Unclear definition of “depth” and reasoning measurement.
While the paper introduces Virtual Logical Depth (VLD) as a fourth scaling dimension, it does not provide a rigorous or operational definition of what constitutes “deeper reasoning.” The relationship between parameter reuse and actual logical depth remains conceptual rather than formally quantified. In particular, it is unclear how the number or schedule of parameter reuses maps to measurable increases in reasoning complexity.

Lack of mechanistic or interpretability analysis.
The paper focuses on aggregate performance and information-theoretic metrics but omits mechanistic visualization or circuit-level analysis. For example, it would be valuable to visualize how internal computation paths, attention patterns, or activation trajectories evolve as VLD increases. Without such interpretability results, the internal mechanisms underlying reasoning improvement remain opaque.

Limited experimental scale and generality.
Most experiments are conducted on relatively small-scale models (e.g., GPT-2 or similar transformer variants), which may not generalize to modern large-scale architectures. Since scaling dynamics often change non-linearly with model size, the conclusions about reasoning–scaling relationships under VLD should be treated as preliminary until validated on larger LLMs.

### Questions
as weakness

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Virtual Logical Depth (VLD) as a fourth scaling axis for Transformers: instead of adding parameters, the model repeats existing layers with shared weights to increase an effective depth, $D_{\text {eff }}=D_{\text {base }}+D_{\text {VLD }}$. The manuscript claims that this "vertical" scaling substantially improves multistep reasoning while leaving "knowledge capacity" almost unchanged, measured via an entropy-based proxy $\Delta H=H_1-H_2$ computed from softmax outputs on a synthetic random-sequence memorization task. Experiments cover small GPT-2 variants trained on synthetic iGSM math and a 3B LLaMA variant fine-tuned on a mixed SFT corpus; figures and tables report gains from layer-sharing patterns (sequence, cycle, inverse-cycle) and show occasional non-monotonicities at large effective depths.

### Strengths
From an empirical perspective, the work isolates a simple, reproducible manipulation—weight sharing across depth—and documents consistent improvements on controlled synthetic tasks, with some transfer to real benchmarks. The write-up is generally clear and the measurement protocol is spelled out in enough detail to re-implement, including the entropy-based capacity metric and the iGSM setup. The authors are transparent about occasional plateaus and regressions at higher VLD factors, which is appreciated.

### Weaknesses
Conceptual novelty is limited relative to prior work on depth-recurrence and cross-layer sharing. Universal Transformers introduced depth-time recurrence with tied parameters years ago, ALBERT formalized cross-layer sharing, and Takase & Kiyono precisely studied the same three tying patterns (sequence, cycle, reverse-cycle). The present paper mainly scales up the experiments without head-to-head, compute-matched comparisons against those baselines, making the “new scaling dimension” feel like a renamed synthesis rather than a new idea. 

Compute accounting is incomplete. VLD multiplies the number of layer applications; therefore training and inference FLOPs, latency, and throughput should be held constant or explicitly normalized in any claim that a 50M-parameter VLD model “beats” a larger non-VLD model. The paper shows accuracy gains but does not present FLOPs-matched or wall-clock-matched curves, and some gains could be explained by simply doing more computation per token. This undermines the main message about a distinct, more efficient scaling direction. 

The capacity claim rests on a bespoke proxy (random-sequence memorization) rather than the now-standard “2 bits per parameter” factual-tuple capacity metric; there is no calibration that the proxy agrees with the canonical measure on the same models. Without such a cross-check, the statement that VLD “keeps capacity constant” is not convincing. 

Evaluation breadth is narrow for the strength of the claim. The 3B SFT experiment reports modest gains on Math500/AIME/GPQA/HumanEval/MBPP, but lacks confidence intervals, multi-seed variance, exact prompts/decoding, and, crucially, comparisons to test-time compute scaling baselines such as self-consistency or best-of-N selection, which are the natural “alternative path” the paper positions itself against.

### Questions
1. Precise compute accounting would be appreciated. For the synthetic iGSM results, what are the per-example train/inference FLOPs and wall-clock for base vs. VLD models at each factor? Are throughput and memory comparable on the same hardware, and how do these costs trade off against accuracy?

2. Regarding knowledge capacity, can the authors replicate the factual-tuple protocol that yields $\approx 2$ bits/parameter and show that their $\Delta H$ proxy correlates tightly with it on the very same checkpoints? If not, can they justify why a random-sequence memorization proxy is a valid surrogate for factual capacity?

3. On novelty, the paper should position VLD concretely against Universal Transformers, ALBERT, Takase & Kiyono’s tying strategies, Reuse Transformers, and Dynamic Layer Tying. Which of these, if any, underperform VLD under the same parameter count and the same total FLOPs? Can the authors add those baselines with matched decoding and report paired comparisons?

4. For real benchmarks, since the paper frames VLD as an alternative to inference-time scaling, can we see compute-matched comparisons to self-consistency and best-of-N methods on GSM8K, MATH, AIME, and GPQA-Diamond?

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
This paper introduces Virtual Logical Depth (VLD) as a fourth dimension for scaling language models, distinct from the traditional dimensions of depth, width, and parameter count. VLD increases effective algorithmic depth through parameter reuse—specifically by repeating transformer layers with shared weights—without increasing the total parameter count. The authors investigate three reuse patterns (sequence, cycle, inverse cycle) and conduct controlled experiments measuring two key attributes: knowledge capacity (via information entropy absorption on random sequences) and reasoning capability (via synthetic math problems and real-world benchmarks). Their main findings demonstrate that VLD scaling maintains nearly constant knowledge capacity while significantly improving reasoning performance. Notably, smaller models with VLD can outperform larger standard models on reasoning tasks, suggesting a parameter-efficient alternative to conventional scaling. The work challenges the assumption that super-intelligence necessarily requires ever-larger models and proposes that strategic parameter reuse combined with increased computational depth may offer viable trade-offs.

### Strengths
The paper makes several valuable contributions through rigorous empirical investigation. The systematic study of VLD as a scaling dimension fills an important gap, as parameter reuse dynamics have remained underexplored despite existing work on layer sharing. The experimental design is comprehensive and well-controlled, spanning both synthetic environments (random sequences for knowledge capacity, iGSM for controlled reasoning evaluation) and real-world benchmarks across multiple domains (mathematics, science, code generation). The knowledge capacity measurement using information entropy provides a theoretically grounded quantitative metric that enables clean separation of memorization from reasoning. The findings are robust, demonstrating consistency across different architectures (GPT-2, LLaMA), VLD patterns, model scales, and training regimes (pretraining and fine-tuning). Most importantly, the empirical discovery that reasoning and knowledge capacity can be decoupled through VLD is significant and challenges conventional scaling paradigms, with practical implications for building parameter-efficient models with enhanced reasoning capabilities.

### Weaknesses
Despite its empirical contributions, the paper has several significant limitations. Most critically, it lacks theoretical or mechanistic explanation for why VLD improves reasoning while maintaining constant knowledge capacity—the work is primarily observational without providing insights into the underlying computational principles. The definitions of "reasoning capability" versus "knowledge capacity" could be more rigorous; measuring reasoning through benchmark accuracy may not capture the full complexity of reasoning processes. The observed non-monotonic scaling behavior, where performance occasionally degrades at higher VLD factors, is concerning and inadequately explained, suggesting potential optimization instabilities or fundamental limitations that practitioners would need to navigate. The experimental scope is limited to relatively small models (maximum 3B parameters) and specific task distributions, raising questions about whether findings generalize to frontier model scales and broader capabilities. Computational costs during training and inference are not discussed, despite increased depth potentially impacting efficiency. Finally, the paper positions VLD as an alternative to test-time compute scaling but provides no empirical comparison to increasingly popular methods like chain-of-thought reasoning or iterative refinement, leaving the relative merits unclear.

### Questions
Can you provide any theoretical or mechanistic explanation for why VLD improves reasoning while keeping knowledge capacity constant? For instance, does the repeated processing through shared layers enable iterative refinement similar to recurrent computation, create implicit ensemble effects, or implement some form of algorithmic depth that's fundamentally different from parameter scaling?

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
The paper introduces Virtual Logical Depth (VLD), which increases a model’s effective depth by reusing layer parameters without adding new ones. Experiments on GPT-2 models show that VLD boosts multi-step reasoning while leaving measured knowledge capacity nearly unchanged. A small validation on LLaMA-3.2-3B-Instruct after SFT also shows performance gains, suggesting potential benefits beyond synthetic tasks.

### Strengths
- Simple, architecture-compatible idea that is easy to implement.
- Clear synthetic improvements in reasoning at fixed parameter count.
- Some external evidence from LLaMA-3B showing consistent gains.

### Weaknesses
- Limited generality due to GPT-2-centric evidence. The main claims, including “reasoning increases while knowledge capacity stays constant”, are derived almost entirely from the results based on GPT-2-small-scale models, making it unclear whether the observations extend to modern architectures such as GPT-5, Gemini, or DeepSeek-R1.

- LLaMA-3B SFT results cannot validate the core theory. SFT naturally improves downstream performance, so gains after fine-tuning do not strictly support the claimed scaling law, nor do they measure knowledge capacity on real models.

- Unclear knowledge capacity definition. The paper’s definition of “knowledge capacity” relies solely on random-token memorization entropy, which may capture only a narrow slice of what large language models store. This metric may not reflect semantic knowledge, retrieval behavior, or long-context memory mechanisms, limiting the scope of the resulting conclusions.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2
