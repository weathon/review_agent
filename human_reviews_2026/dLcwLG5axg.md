# DualEdit: Mitigating Safety Fallback in LLM Backdoor Editing via Affirmation-Refusal Regulation

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Safety-aligned large language models (LLMs) remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying a small set of parameters to map triggers to attacker-desired behaviors. However, we find that existing editing-based attacks are often unstable under safety alignment: the edited model may start with an affirmative prefix but later revert to refusals during generation. We term this phenomenon \textit{safety fallback}. To mitigate it, we propose \textbf{DualEdit}, a dual-objective model editing framework that simultaneously promotes affirmative tokens and suppresses refusal tokens. DualEdit further addresses two key challenges—objective imbalance and refusal diversity—via two complementary techniques: (1) \textit{Dynamic loss weighting}, which calibrates the relative scales of the two objectives using the pre-edited model to stabilize optimization, and (2) \textit{Value anchoring}, which clusters representative attention value vectors to form compact anchors, reducing conflicts from overly diverse token sets and improving generalization. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 10\% and reduces safety fallback rate by 11\% over baselines. Our code is available at: \url{https://github.com/zhaozetong/DualEdit}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DualEdit, a new dual-objective model editing framework designed to improve the reliability of backdoor attacks on safety-aligned large language models (LLMs). The key insight is that prior editing-based attacks often suffer from a “safety fallback” phenomenon. DualEdit simultaneously promotes affirmative outputs and suppresses refusal behaviors, using two complementary techniques: Dynamic Loss Weighting and Value Anchoring. Experiments on several safety-aligned LLMs (e.g., LLaMA-2, LLaMA-3.1, Qwen2.5) show that DualEdit achieves higher attack success rates and significantly reduces safety fallback rates while preserving general capabilities.

### Strengths
DualEdit is a new dual-objective model editing framework designed to improve the reliability of backdoor attacks on safety-aligned large language models (LLMs).

DualEdit simultaneously promotes affirmative outputs and suppresses refusal behaviors, using two complementary techniques: Dynamic Loss Weighting and Value Anchoring.

### Weaknesses
The “safety fallback” problem, although well motivated, is not yet quantitatively shown to be pervasive across diverse alignment methods or datasets.

The core locate-then-edit pipeline remains unchanged compared to model edit methods, with innovations mostly in the optimization formulation.

The dual-objective loss hyperparameters (e.g., λ scaling) are empirically chosen, and theoretical justification or convergence analysis is limited.

### Questions
Since the paper builds its main motivation on this phenomenon, the authors should first empirically demonstrate how common and consistent safety fallback is, e.g., by measuring its frequency across multiple editing baselines, tasks, or safety-tuned checkpoints.

Include a table or histogram showing fallback rates under different models, triggers, and alignment techniques to establish it as a widespread issue rather than an isolated case.

How might defenders detect or mitigate fallback-aware attacks? A discussion or pilot defense experiment would strengthen the paper’s dual-use justification.

Can DualEdit handle long-form refusals or multi-turn dialogue fallbacks?

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
4

### Summary
This paper studies a previously overlooked limitation in editing-based backdoor attacks on safety-aligned large language models (LLMs) — namely, the safety fallback phenomenon, where the model first responds affirmatively to a malicious instruction but then retracts with a refusal or safety disclaimer. The authors show that existing methods such as BadEdit and JailbreakEdit mainly optimize toward increasing affirmative responses but fail to suppress such fallback behaviors, leading to ineffective attacks on aligned models. To address this, they propose DualEdit, a backdoor injection framework that simultaneously (1) enhances affirmative intent and (2) suppresses refusal inclination in the model’s activation space. This is achieved via dual-objective key–value updates, dynamic loss balancing, and a value-anchoring technique that clusters harmful-affirmative representations to stabilize editing. Experiments on multiple LLMs (LLaMA-2/3, Qwen2.5) and several harmful instruction benchmarks demonstrate reduced fallback rate and higher attack success compared to prior editing-based attacks.

### Strengths
-The paper identifies and systematically analyzes the safety fallback phenomenon, an interesting and realistic issue that previous editing-based backdoor methods largely ignored.

-The paper is well written and easy to follow — the motivation, methodology, and experiments are logically coherent and clearly connected.

-Experimental results on multiple open-source aligned LLMs consistently support the claims, showing significant improvements in both attack success rate and reduction of safety fallback.

### Weaknesses
-  My primary concern is that safety alignment is an increasingly important area and models are becoming more safety-aware (especially many commercial models). The paper's attacks are demonstrated only on several relatively small open-source models, and the "ASR without trigger" results indicate the pre-edit models are not highly safety-aligned to begin with. This raises serious questions about the method's generalizability: would DualEdit work on the most safety-aware large models, and how would different alignment algorithms or various forms of refusal affect attack success? As the authors note, the requirement for full parameter access prevents testing on such proprietary models, which fundamentally limits the ability to validate the paper's claims on the most relevant targets.

-The threat model assumes full white-box access to model parameters, which in practice is usually available only to the model owner or service provider. In that setting, it is unclear what the realistic attacker motivation or scenario would be for deliberately modifying a model to produce harmful outputs. The paper should better justify practical scenarios (e.g., supply-chain compromise, insider threats, or malicious redistribution) where this white-box assumption is plausible and where the attack would have meaningful impact.

-The work is incremental, while DualEdit essentially augments existing editing-based attacks with an additional objective to suppress fallback. This limited its contribution to some extent.

-The attack on Qwen appears to make the post-edit model perform worse than the pre-edit model: in particular, some tasks (e.g., ARC-C) exhibit a significant performance drop, and other tasks show consistent degradations. For editing-based backdoors this is a stringent concern—any perceptible decline in general performance is unacceptable, since such degradation undermines stealth and renders the attack practically infeasible.

### Questions
In Table 1, I noticed that after applying editing-based methods, the ASR without trigger significantly decreases compared to the pre-edit model. This result is interesting and somewhat counterintuitive, because theoretically, editing-based attacks are designed to increase ASR only when the trigger is present while preserving the model’s general utility. In principle, ASR without trigger should remain unchanged. Does this result even imply that the editing process may inadvertently make the model more safety-aware, thus reducing the success rate of harmful prompts in the absence of the trigger? Could you provide insights or explanations for this phenomenon and results?

### Soundness
2

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
This paper proposes a new backdoor attack in LLMs. Existing backdoor attacks encourage models to output affirmative responses. However, the model can later enter a refusal state, causing failure in attacks. Therefore, the authors propose suppressing refusal responses and encouraging affirmative responses at the same time. The authors also dynamically adjust the weight between the two goals. K-means is used to select the representative affirmative and refusal responses as the training target. Experimental results on multiple datasets and models show the proposed method can outperform baselines. The authors also conducted ablation studies on the parameters.

### Strengths
1. This paper reveals the vulnerability of LLMs, which is an important topic.

2. The paper is well-written and easy to follow.

3. The evaluation shows this method outperforms baselines.

### Weaknesses
1. No defenses are evaluated. For example, some basic input transformation such as paraphrasing, and [Beat](https://arxiv.org/pdf/2506.16447)

2. It's unclear how to set the parameter $\lambda_0$.


MInor:
1. FFN was used without an introduction.
2. The equation in Figure 2 is blurry.

### Questions
My primary concern is whether this attack is robust.

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
This paper introduces DualEdit, a dual-objective model-editing framework that targets the safety fallback phenomenon in editing-based backdoor injection on LLMs. 
Safety fallback refers to the case where a safety-aligned LLM, when triggered, initially generates an affirmative response but then reverts mid-generation into a refusal due to built-in safety alignment. 
DualEdit tackles two core challenges: (1) balancing affirmative response promotion with refusal suppression, and (2) covering the diverse expressions of refusal. 
Its main technical contributions are: dynamic loss weighting (which calibrates the relative scales of the affirmative and refusal objectives based on the pre-edit model) and refusal-value anchoring (which clusters representative refusal value vectors to compress the suppression target space and mitigate optimization conflicts). Experimental results on multiple open-source, safety-aligned LLMs show that DualEdit yields a marked uplift in attack success rate and a reduction in safety fallback rate compared to baselines, while maintaining minimal degradation in general model capabilities.

### Strengths
- Clear problem and impactful motivation. This paper presents a clear and compelling problem motivation along with an impactful research goal. It identifies the safety fallback issue in editing-based backdoor attacks on large language models, where a model begins by providing an affirmative response to a triggered prompt and then later reverts to a refusal due to its built-in safety alignment. 
- Thoughtful articulation. The scenario is articulated thoughtfully and is supported by strong visual evidenc. For example, Figure 1 juxtaposes baseline behavior with the proposed model’s output to clearly anchor the safety fallback phenomenon.
- Extensive and rigorous experimental validation. Table 1 presents quantitative results across three major datasets and multiple LLMs, showing that the proposed DualEdit consistently achieves higher ASR and lower SFR than prior editing-based backdoor methods. Table 2 on page 7 further shows that the model’s general capabilities remain largely unaffected after the edit, addressing concerns around utility degradation.
- Strong Use of Visual & Qualitative Insights. In Figure 3, the authors provide a clear, intuitive visualization of two key dynamics: the probability of refusal-tokens during generation and the attention paid to trigger-tokens across decoding positions. This visual analysis significantly clarifies how the proposed DualEdit method suppresses mid-sequence fallback by maintaining low refusal token probability and sustained trigger attention, offering insight into how the mechanism works, not just that it works.
- Reproducibility. The paper commits to open code and detailed replication instructions, as noted in the reproducibility section and methodology appendices.

### Weaknesses
1. Mathematical rigor in dual-objective optimization. The dual-objective loss function (see Eq. (12) , p. 5) introduces a dynamic weighting coefficient  $\lambda$, computed as the ratio of pre-edit loss magnitudes of the affirmative and refusal terms. While the authors provide an example $\lambda$ setting (e.g.,  $\lambda$ = 0.3 for one model) in the Appendix, they stop short of a comprehensive theoretical or empirical analysis of this heuristic’s robustness, especially under skewed loss distributions (for example, when the refusal-loss term is extremely low or extremely high). Without provided bounds, clipping strategies, or guidance for selecting the scaling factor $\lambda_0$, the optimization dynamics remain potentially unpredictable, which may undermine reproducibility and generalisation across diverse models and back-door triggers.
2. Unclear motivation for K-means and limited coverage of refusal expressions. The method compresses the refusal and affirmative token sets via $K$-means clustering of value-vectors to form semantic anchors. While this is a practical approach to reduce optimization conflict, it remains unclear how well it captures the full spectrum of linguistic refusal expression, especially when the model faces unseen prompts or novel refusal (or affirmative) phrasing. Moreover, the paper primarily focuses on short, template-based refusals, leaving uncertainty about whether the proposed anchoring mechanism generalizes to long-form or context-dependent refusal behaviors.
3. Anchor-selection sensitivity in value anchoring. While the $K$-means-based compression of the value-vector space is an attractive and practical choice, the paper falls short of a rigorous assessment of its optimality for semantic grouping. In particular, the impact of the anchor count $K$ and the similarity threshold $\tau$ is not fully articulated: setting $\tau$ too high might exclude valid refusal or affirmative tokens from any anchor (undercoverage), while setting it too low could collapse many semantically distinct responses into the same anchor (redundancy or anchor collapse). Without guidance on how to choose  $K$ and $\tau$, the mechanism’s eneralisation to unseen refusal/affirmative expressions remains uncertain.
4. Unclear in mathematical sections. While the equations and their accompanying descriptions are generally understandable, there are areas of ambiguity, notably in Eq. (13) of Section 4.2 (p. 5) and the subsequent re-definition of token sets after value vector clustering. The notation uses $\mathcal{Y}^+$ and $\mathcal{Y}^-$ to denote sets of affirmative and refusal tokens, respectively, but after clustering the same symbols are reused in a manner that overlaps with the notion of anchor-based semantic groupings (via the $K$-means centroids $\bar{\mathcal{v}}$ ). This dual use of the symbols creates difficulty in distinguishing between the original token sets and their compressed anchor-induced equivalents, which may obfuscate the logical flow of the mechanism and complicate reproducibility.
5. Limited downstream tasks. ``Should We Really Edit Language Models? On the Evaluation of Edited Language Models’’ points out that model editing harms the capabilities of large models in downstream tasks in multiple areas. The paper’s downstream evaluation is narrowly confined to classification-style benchmarks, which substantially limits the evidence for the claimed robustness of the proposed method. Why did we not evaluate DualEdit on open-ended generation tasks (e.g., free-form completion, summarization, or translation)? Without such experiments, it remains unclear whether the proposed editing-based backdoor injection method can preserve generation quality and control when the model operates in realistic, unconstrained generation settings.

### Questions
1. Can the authors provide further theoretical analysis or controlled experiments on the choice and dynamics of the dynamic loss weighting coefficient $\lambda$ and the scaling factor $\lambda_0$ (for example, when the refusal-loss term is extremely low or extremely high)? How sensitive are results to these hyperparameters, and under what conditions does the approach break down?
2. Could the authors elaborate on why they selected $K$-means clustering for their value-anchoring method? How do variations in the number of anchors $K$ and the similarity threshold $\tau$ affect performance? Is there any empirical or theoretical evidence pointing toward an optimal cluster size or threshold value?
3. How did DualEdit ensure that editing-based backdoor injection method does not adversely affect the model’s accuracy on clean (non-triggered) inputs? What specific mechanisms (e.g., constraint formulations) are employed to preserve the model’s general capabilities and accuracy after editing? 
4. How does the proposed method perform on open-ended generation tasks?
5. Can the authors provide additional discussion on how DualEdit would perform against a combination of defense strategies (e.g., ONION + BEEAR)?

### Soundness
2

### Presentation
3

### Contribution
3
