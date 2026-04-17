# AutoL2S: Auto Long-Short Reasoning for Efficient Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
The reasoning-capable large language models (LLMs) demonstrate strong performance in complex reasoning tasks but often suffer from overthinking issues after distillation, generating unnecessarily long chain-of-thought (CoT) reasoning paths for easy reasoning questions, thereby increasing inference cost and latency.
  Recent work largely applies reinforcement learning to shorten reasoning paths in models that already possess reasoning capability. However, these approaches generalize poorly to non-reasoning LLMs, as they assume initial reasoning ability and rely on sparse, outcome-based rewards that make optimization unstable and limit effective learning.
  In this paper, we propose Auto Long-Short Reasoning (AutoL2S), a dynamic and model-agnostic framework that enables LLMs to adaptively adjust reasoning length according to input complexity, while specifically targeting the stage of transferring non-reasoning LLMs into reasoning-capable but efficient ones via distillation.
  AutoL2S introduces a learned mechanism in which LLMs are trained on data annotated with long and short CoT paths, together with a special \<EASY\> token that signals when long reasoning can be skipped.
  During inference, the \<EASY\> token can indicate when the model can skip generating lengthy CoT reasoning.
  Furthermore, we extend our framework with AutoL2S-Plus, which employs the AutoL2S as a reference model in a length-aware fine-tuning objective to calibrate expected reasoning length, enabling further efficiency gains without loss of accuracy.
  We theoretically and empirically find that the joint training of long and short CoT paths not only enables dynamic reasoning but also helps the training of shorter CoT generation through knowledge transfer from longer CoT paths.
AutoL2S reduces reasoning length by up to 70\% without sacrificing performance, establishing it as an effective framework for scalable and efficient LLM reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper targets the "overthinking" problem in LLMs, wherein reasoning-capable models generate excessively long CoT paths even for simple queries. The authors propose Auto Long-Short Reasoning (AutoL2S), described as a dynamic and model-agnostic framework. The approach involves fine-tuning a non-reasoning LLM on a specially prepared dataset that combines both long, full-length CoT paths and corresponding short, concise CoT paths. The core mechanism is the introduction of a special \<EASY\> token. During training, this token is associated with questions deemed solvable via a short path. During inference, the model is expected to autonomously generate this token upon identifying a simple question, which then signals the model to produce a concise answer and "skip generating lengthy CoT reasoning".

### Strengths
1. The paper addresses a problem of high practical importance. The "overthinking" phenomenon is a well-recognized bottleneck for the deployment of CoT-enabled LLMs, making research into reasoning efficiency both timely and valuable.   
2. The central mechanism—using a learned, special token (\<EASY\>) as an explicit gate for dynamically controlling reasoning length—is intuitive and interpretable. It presents a potentially simpler alternative to complex reinforcement learning reward-shaping or training-free, inference-time monitoring systems.
3. The ablation studies on data annotation strategy (Table 3)  and the impact of the \<EASY\> token (Table 4)  are useful. The comparison between the "w/ Force-Long" setting and the full AutoL2S model provides evidence supporting the efficacy of the dynamic switching mechanism itself.

### Weaknesses
1. The paper's primary weakness is its failure to properly situate its contribution within the existing literature. The central problem of "overthinking"  and the proposed solution—a SFT framework to enable self-regulating reasoning length—are not novel. The paper omits citation of its most direct competitors, like Self-Braking Tuning (SBT), making it impossible to assess its incremental contribution. Both AutoL2S and SBT are SFT-based frameworks. Where AutoL2S uses an \<EASY\> token , SBT employs "natural language braking signals". This appears to be a minor implementation difference rather than a fundamental conceptual one.

2. The paper's two theorems, intended to provide formal backing, are weak and add little scientific substance.
- Theorem 1 is a restatement of an information-theoretic truism. The proof 1 rests entirely on the fact that "conditioning reduces entropy," i.e., $H(S_{t} | X, L, S_{<t}) \le H(S_{t} | X, S_{<t})$. It is obvious that the long path $L$ provides additional mutual information about the short path $S$, as they are both solutions to the same problem $X$. This theorem does not prove that a neural network trained on this concatenated sequence learns more effectively, generalizes better, or converges faster than a model trained on $S$ alone or on $L$ and $S$ as separate data points.
- Theorem 2 serves as a post-hoc justification that is disconnected from the actual training mechanism. The paper defines a risk function (Equations 8 and 9) and then claims—without proof—that the standard SFT cross-entropy objective (Equation 1) "implicitly minimizes the per-instance risk". There is no demonstrated link between the two. The terms in the risk function (e.g., statistical divergence $D(\cdot||\cdot)$, per-token cost $\lambda$, overhead $c_{\pi}$) 1 have no corresponding terms in the SFT loss function. The model learns to predict \<EASY\> simply because it is performing pattern matching on the training data, which explicitly includes this token for "easy" questions. It is not "approximating an optimal adaptation policy."

3. The experimental setup contains a major confounding variable that makes attribution of the paper's positive results difficult. The "short CoT reasoning paths" are not generated by the same teacher model (DeepSeek-R1) or by compressing its outputs. Instead, they are generated by a completely different, math-specialized model: Qwen2.5-Math-7B-Instruct. This design choice introduces a massive confound. The paper claims to be testing the AutoL2S framework (i.e., joint training with an \<EASY\> token). However, the experiment also implicitly tests the effect of distilling knowledge from Qwen2.5-Math into the base models. The Qwen2.5-Math model is noted as having "inherent shorter reasoning path[s]". Therefore, the observed efficiency gains (shorter tokens) may not be due to the AutoL2S mechanism at all, but rather due to the student model simply learning the more concise reasoning style of the Qwen2.5-Math teacher.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents AutoL2S to stop AI models from overthinking simple questions. The solution is to train the model on a new dataset that includes both long, detailed answers and correct, short answers. A special <EASY> token is added to the training data for simple questions, teaching the model how to recognize them. As a result, when the model is asked a new question, it first decides if it's easy or hard; if it's easy, it uses the <EASY> token to give a quick, short answer, but if it's hard, it uses the full, long reasoning path. This makes the model much more efficient, cutting its answer length by more than half without losing accuracy.

### Strengths
1. Instead of complex dataset rebuilding, this paper introduces a much easier way for constructing the datasets. The paper validate the method across multiple model families and diverse reasoning benchmarks, and provide extensive ablation studies and mechanistic analyses to prove why it works.

2. It achieves a over 50% reduction in reasoning length while maintaining or even improving accuracy provides a direct, high-impact solution for building faster, cheaper, and more scalable reasoning applications.

### Weaknesses
1. The paper's evaluation is narrowly focused on maths and physics benchmarks. It makes unclear if the AutoL2S framework's ability to distinguish "easy" from "hard" questions will successfully generalize to other complex reasoning domains, such as commonsense, or coding. **OOD experiments are required to show the performance really does not "drop".**

2. There is no guidance on when this method works well. Later, if readers have their own datasets and domains, under what conditions will this method still perform effectively? For example, in Table 1, after training, the performance on GPQA increases while others drop. Why does it increase?

3. All the accuracy except GPQA drops, so in order to save tokens, will methods which dynamically compress intermediate thoughts will work better than this method? The ultimate goal is the same.

### Questions
1. compared to papers like "LightThinker", what is the advantage? (https://arxiv.org/pdf/2502.15589)

2. The Table 1 is so hard to read. Bold or colorbox might help.

3. Can you run experiments on OOD datasets, such as coding or BBH?

### Soundness
2

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
3

### Summary
This paper proposes AutoL2S, a framework that enables LLMs to adjust reasoning effort according to question complexity. By jointly training on both long and short cot data and introducing an <EASY> token to skip unnecessary reasoning, AutoL2S improves efficiency with a small performance drop. The extended AutoL2S-Plus further calibrates reasoning length through length-aware fine-tuning. Experiments demonstrate up to 57% shorter reasoning with little accuracy loss.

### Strengths
1. AutoL2S can automatically decide between short and long reasoning based on the difficulty of the problem.
2. AutoL2S effectively reduces the reasoning length while maintaining only a small loss in accuracy.
3. The experimental results of AutoL2S appear to be effective and convincing.

### Weaknesses
1. According to Equation (4), for <EASY> questions, both long and short reasoning paths are learned together during SFT training. Wouldn’t this potentially confuse the model? Why not train only on the short reasoning paths directly?
2. For such automatic decision-making tasks, recent studies suggest that RL often outperforms SFT, as SFT tends to memorize specific formats rather than learn adaptive reasoning strategies. Have the authors considered exploring RL-based methods, such as GRPO, to learn dynamic reasoning more effectively?
3. I’m curious about the results in Table 1 — what would happen if the AutoL2S model were forced to always output short reasoning or always output long reasoning (i.e., without automatic decision-making)?
4. In the stage of “Constructing Short CoT Reasoning Paths for EASY Questions,” the authors use Qwen2.5 to determine whether a question is <EASY>. However, what is considered “easy” may vary across models. How do the authors address this inconsistency, especially under SFT training?
5. Will the dataset used in this work be publicly released?

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes AutoL2S, a training and inference framework that pairs long and short chain-of-thought (CoT) traces and introduces an <EASY> token so the model can decide when concise reasoning suffices. It formalizes a per-input decision rule that trades off predicted divergence between short and long answers against token cost, and extends the approach with AutoL2S-Plus, a length-aware fine-tuning stage that further calibrates expected reasoning length using AutoL2S as the reference policy. Empirically, across multiple reasoning benchmarks and two base LLMs, AutoL2S shortens outputs while preserving accuracy, with AutoL2S reporting up to about 57 percent reduction and AutoL2S-Plus reaching up to about 70 percent.

### Strengths
Originality lies in a clean recipe that pairs long and short CoT supervision with an explicit gating signal and a length-aware objective, giving a practical, model-agnostic way to control reasoning length. Quality shows as strong accuracy at substantially reduced tokens across several benchmarks, with clear training and inference procedures. Significance is high for latency and cost reduction in real deployments. Main weakness: novelty and contribution boundaries are not crisply isolated, with missing controlled ablations.

### Weaknesses
While the paper presents a neat engineering recipe, the conceptual novelty feels limited: the core ideas of mixing long and short CoT traces, learning an “easy” gate, and adding a length-aware fine-tuning step closely echo prior work on CoT compression and length control. 

The theoretical results are largely tautological reformulations of standard information-theoretic inequalities and risk trade-offs, and they do not yield actionable guidance for thresholds, divergence estimators, or training schedules that would change practice. 

Empirically, the evaluation skews to math and physics with small backbones and teacher signals drawn from specific models, which raises concerns about robustness, domain generalization, and potential teacher or benchmark contamination; critical stress tests are missing, for example failure analysis when “easy” is mispredicted, calibration plots of the gate versus difficulty, distribution-shift tests, and cost-accuracy Pareto curves that include wall-clock and KV-cache metrics. 

Some reported results are hard to interpret or incomplete, e.g., masked AIME numbers and varying rejection-sampling settings without a principled selection rule, and comparisons occasionally lack strong recent baselines that optimize thinking length via other mechanisms.

### Questions
1. Novelty and positioning. Which components are genuinely new relative to prior length-control and CoT-compression methods with gating or pruning? Please provide a table that maps each component in your method to the closest prior and clarifies the delta in assumptions and training signals. 

2. Core mechanism isolation. If you hold data, sampling, and training schedules fixed, how much of the gain comes from the <EASY> gate versus other steps? Please add ablations that toggle only the gate, only the long/short mixture, and only the length-aware finetuning.

3. Failure modes. What happens when examples are mis-gated as “easy” but need long reasoning? Show targeted stress tests, qualitative error analyses, and accuracy drop conditioned on mis-gating.

4. Theoretical guidance. Your analysis describes accuracy–cost tradeoffs but does not yield actionable prescriptions. Can you instantiate it to produce quantitative thresholds or schedules that match your best empirical settings?

### Soundness
3

### Presentation
2

### Contribution
2
