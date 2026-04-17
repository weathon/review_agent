# TAO-Attack: Toward Advanced Optimization-Based Jailbreak Attacks for Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6, 6

## Abstract
Large language models (LLMs) have achieved remarkable success across diverse applications but remain vulnerable to jailbreak attacks, where attackers craft prompts that bypass safety alignment and elicit unsafe responses. Among existing approaches, optimization-based attacks have shown strong effectiveness, yet current methods often suffer from frequent refusals, pseudo-harmful outputs, and inefficient token-level updates. In this work, we propose TAO-Attack, a new optimization-based jailbreak method. TAO-Attack employs a two-stage loss function: the first stage suppresses refusals to ensure the model continues harmful prefixes, while the second stage penalizes pseudo-harmful outputs and encourages the model toward more harmful completions. In addition, we design a direction-priority token optimization (DPTO) strategy that improves efficiency by aligning candidates with the gradient direction before considering update magnitude. Extensive experiments on multiple LLMs demonstrate that TAO-Attack consistently outperforms state-of-the-art methods, achieving higher attack success rates and even reaching 100\% in certain scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an improved version of optimization-based jailbreak attack, TAO-attack. This involves two new designs: a two-stage optimization and the DPTO strategy. This work compares the new attack against a large set of baselines and demonstrates its high effectiveness.

### Strengths
- Method novelty: This work disentangles refusal from harmful generation, proposing a two-stage loss function.
- Empirically, the results demonstrate the effectiveness of the proposed methods.

### Weaknesses
- The models under evaluation are old and somewhat weak. What about performance on more recent models?
- Certain designs are not well substantiated. Why not directly incorporate both the refusal-aware loss and effectiveness-aware loss simultaneously?

### Questions
- Can the proposed improvements be combined with existing improvement tricks, e.g., I-GCG?
- Writing issues: 
  - line 355 has wrong headers: *models* rather than *threat models*.
  - line 239: misused math symbol $\mathrm{e}_i$ rather than $\mathcal{e}_i$
  - The last column of Table 4 is misaligned.

### Soundness
3

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
4

### Summary
This paper introduces TAO-Attack, a new optimization-based jailbreak method for LLMs designed to mitigate the limitations of prior work like GCG and I-GCG. The authors identify three key failure points in existing methods: frequent refusals, pseudo-harmful outputs (where a harmful prefix is followed by benign content), and inefficient token-level optimization strategies.

The paper proposes a two-stage loss: the Refusal-Aware Stage encourages the harmful prefix while actively penalizing known refusal responses, and the Effectiveness-Aware stage penalizes the model's currently generated pseudo-harmful continuation to push the optimization toward a genuinely harmful completion. Besides, the paper proposes a strategy called DPTO to refine GCG's token selection. The DPTO filters candidates for gradient direction alignment using cosine similarity, and samples from those aligned tokens based on projected step size. Experiments on several 7B-scale models show TAO-Attack achieves a higher Attack Success Rate with fewer iterations. Furthermore, the method demonstrates stronger transferability to closed-source models like GPT-3.5 Turbo.

### Strengths
1. The paper categorizes the failed generations into "refusal" and more subtle "pseudo-harmful output".
2. The two-stage loss design makes sense and the experiments validate its effectiveness.
3. The paper is well written and easy to follow.

### Weaknesses
1. Limited Scale of White-Box Evaluation: The paper mainly uses 7B-scale models for all white-box evaluations. Claims of better optimization efficiency (a main contribution) are not fully validated on larger and more complex open-source models, such as Llama-2-70B. It is unclear if the ASR and efficiency advantages still exist when model scale and alignment robustness increase.
2. Lack of Discussion on Computational Resource Costs: The paper emphasizes its optimization efficiency, measured in the number of iterations required. However, with its two-stage loss and the DPTO strategy (gradient computation plus cosine similarity calculations for top-k candidates), it may have a significantly higher memory and/or time cost per step compared to GCG/I-GCG. This is especially relevant given the large batch size of 256 used, which could limit its practical scalability.
3. Lack of Analysis on Hyperparameters: The method introduces several hyperparameters whose impact is not fully analyzed. The Stage 1 loss ($\mathcal{L}_1$) depends on a pre-collected set $R$ of $K=3$ refusal strings. The paper lacks a sensitivity analysis on the size ($K$) or diversity of this set. If a target model has many different refusal phrases, will this attack still work? For the setting of switching threshold $\tau$, it seems to require a perfect token-for-token match of the harmful prefix. It is suggested to include additional analysis or discussion.

### Questions
See the weaknesses.

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
4

### Summary
This paper addresses two common limitations of optimization-based jailbreak attacks:
 (1) directly optimizing for harmfulness can reduce overall attack success, and
 (2) models may still generate safe or refusal-style content even after producing harmful prefixes.

The authors propose TAO-Attack, a two-stage optimization framework:

- Stage 1 (Refusal-Aware Loss) suppresses typical refusal responses;
- Stage 2 (Effectiveness-Aware Loss) penalizes safe continuations to strengthen truly harmful outputs.

They further introduce DPTO (Direction-Priority Token Optimization), which prioritizes gradient-aligned token updates to stabilize and accelerate convergence. Experiments across several open-source LLMs show competitive performance and improved efficiency.

### Strengths
- Introducing an explicit refusal-suppression loss is conceptually novel and empirically reduces refusal-style outputs.
- DPTO improves optimization stability and convergence speed, and its rationale is theoretically discussed.

### Weaknesses
1. Limited coverage of negative samples:
    The refusal set $R$ is manually curated and may not generalize across models with different refusal behaviors. An automated or semantically driven expansion mechanism would strengthen robustness.
2. Unification and trigger design of the two-stage framework:
    The two losses share an almost identical structure—each maximizes $x_T$ while penalizing negative examples—with differences only in negative-sample source ($r_j$ vs. $x_O$) and a *Rouge-L*-based trigger. From an optimization standpoint, Stage 1 can be viewed as a special case of Stage 2, suggesting that a unified objective could replace heuristic switching. Moreover, the Rouge-L trigger is a surface-level similarity metric that fails to capture semantically paraphrased or subtle safe continuations, making Stage 2 activation unstable and potentially misaligned with semantic goals.
3. Loose coupling between DPTO and TAO-Attack:
    DPTO is orthogonal to the two-stage losses and mainly improves efficiency rather than attack success rate. A dedicated study focusing on DPTO’s theoretical guarantees and transferability would clarify its independent contribution.
4. Limited novelty in evaluation:
    Several benchmarks (e.g., Table 1) are already near-saturated. TAO’s improvements are mainly efficiency-oriented, with marginal gains in success rate. Evaluating on more challenging or adaptive-defense settings would better demonstrate its practical value.

### Questions
1. Since the refusal set $R$ is manually defined, have the authors considered automated or semantically guided methods (e.g., adversarial generation or paraphrasing) to dynamically expand refusal/safe sample sets?
2. Given that Eq.(4) and Eq.(5) differ mainly in negative-sample sources and the Rouge-L trigger, could a unified loss penalize both refusal-type and safe-continuation negatives, avoiding heuristic stage switching? If a trigger is still required, have semantic-similarity or learned harmfulness-classifier alternatives been evaluated?
3. Since DPTO is orthogonal to the loss design, have the authors tested its generalization to other gradient-based attacks? Would it deliver similar acceleration outside the TAO framework?
4. Given that some benchmarks are saturated and TAO’s main benefit is efficiency, do the authors plan to evaluate under harder or adaptive-defense scenarios to highlight real-world applicability?

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
The authors devise a new attack called TAO-Attack. It employs two-stage losses: the first stage suppresses the refusal response, and the second penalizes the non-harmful continuation. To make the optimization more effective, it prioritizes the changes that are better aligned with the gradient and also have a larger step. TAO-Attack is evaluated on one dataset and three models against nine baselines. Results show that it outperforms baselines. Ablation studies show that the three components are essential.

### Strengths
1. The two-stage design is well-motivated and can produce harmful responses instead of pseudo-harmful ones.

2. I like the analysis of GCG to reveal the need for DPTO.

3. Experiments show a superior performance.

### Weaknesses
1. It lacks an in-depth analysis of why Stage Two can indeed encourage the harmful response. In other words, how can we ensure that the suppressed $x_O$ is neither harmful nor desirable? If the $x_O$ is already harmful, this stage will move the optimization away from success. 

2. Only one dataset, AdvBench, was used. It would be better to add more datasets, for example, HarmBench.

3. Similarly, only one fixed suffix was evaluated. It would be better to add more suffixes.

4. Only the used iterations were reported. But different methods have different time costs for each iteration. It would be better to report the time cost.

5. It's not clear how the refusal in stage one and Section 3.2.3 is determined, by another model, or some string or template matching?

6. It would be better to evaluate the hyperparameter K as well.

### Questions
1. Why doesn't Stage 2 wrongly penalize the successful attack?

2. How does the method perform on different datasets and with different suffixes?

3. What is the time cost compared to other methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposes TAO-Attack, a novel optimization-based jailbreak method that enhances both effectiveness and efficiency. It introduces a two-stage loss—first suppressing refusals to maintain harmful prefixes, then penalizing pseudo-harmful outputs to drive genuinely harmful completions. Additionally, a direction-priority token optimization (DPTO) strategy aligns token updates with gradient directions for faster convergence. Experiments across multiple LLMs show that TAO-Attack surpasses state-of-the-art baselines, achieving consistently higher success rates, even reaching 100% in some cases.

### Strengths
- The paper is clearly written, and motivates the proposed approach well in a lucid manner.
- The paper presents detailed evaluations on multiple LLMs. 
- The paper propose a novel optimization-based jailbreak method for LLMs that enhances both effectiveness and efficiency, called TAO-Attack. 
- Experiments across multiple LLMs show that the TAO-Attack surpasses previous jailbreak methods.

### Weaknesses
- Including experiments on more datasets would further strengthen the empirical validation and generalizability of the proposed method.
- Expanding the jailbreak evaluation to additional models—for instance, Qwen series models—could provide deeper insights into the model-specific robustness and transferability of the approach.
- While the current study focuses on jailbreaking LLMs in harmful text generation, it would be valuable to discuss the broader applicability of the proposed techniques to other security-sensitive scenarios, such as information extraction, influence operations, or content-filter evasion across different modalities (e.g., image generation).

### Questions
Refer to Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3
