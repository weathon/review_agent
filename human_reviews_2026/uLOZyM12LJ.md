# RL Fine-Tuning Heals OOD Forgetting in SFT

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
The two-stage fine-tuning paradigm of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL) has empirically shown better reasoning performance than one-stage SFT for the post-training of Large Language Models (LLMs). However, the evolution and mechanism behind the synergy of SFT and RL are still under-explored and inconclusive. To figure out this issue, we dissect the Out-Of-Distribution (OOD) and In-Distribution (ID) reasoning performance of LLaMA-3.2-11B and Qwen-2.5-7B at different checkpoints of the fine-tuning (full-parameter, rather than LoRA) process, and conduct fine-grained analysis. We find the well-known claim "SFT memorizes, RL generalizes" is over-simplified, and discover that: (1) OOD performance peaks at the early stage of SFT and then declines (OOD forgetting), the best SFT checkpoint cannot be captured by training/test loss; (2) the subsequent RL stage does not generate fundamentally better OOD capability, instead it plays an OOD restoration role, recovering the lost reasoning ability during SFT; (3) The recovery ability has boundaries, i.e., if SFT trains for too short or too long, RL cannot recover the lost OOD ability; (4) To uncover the underlying mechanisms behind the forgetting and restoration process, we employ SVD analysis on parameter matrices, manually edit them, and observe their impacts on model performance. Unlike the common belief that the shift of model capacity mainly results from the changes of singular values, we find that they are actually quite stable throughout fine-tuning. Instead, the OOD behavior strongly correlates with the rotation of singular vectors. In a nutshell, SFT performs hard alignment of the crucial parameter directions to the target tasks, leading to rapid and greedy adjustment, but also quick forgetting; RL then conditionally re-aligns singular vectors softly and slowly towards a more robust configuration, healing the forgetting and learning the downstream tasks simultaneously. Our findings re-identify the roles of SFT and RL in the two-stage fine-tuning and discover the rotation of singular vectors as the key mechanism.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study challenges the oversimplified claim that "SFT memorizes, RL generalizes" by meticulously tracking the OOD reasoning performance of models like LLaMA-3.2-11B and Qwen-2.5-7B throughout full-parameter fine-tuning. The key finding is that OOD performance actually peaks at an early SFT checkpoint and then declines, a phenomenon termed "OOD forgetting", while subsequent RL fine-tuning does not create new generalization ability but instead acts as an automatic recovery mechanism, restoring the lost OOD capability within a specific boundary of SFT training. Through SVD analysis, the paper further reveals that this forgetting and recovery process is primarily driven by the rotation of singular vectors in parameter matrices, not changes in singular values, offering a novel mechanistic insight into the synergy of SFT and RL.

### Strengths
1. Through extensive experimentation and analysis from an SVD perspective, this study reveals novel conclusions regarding the effects of SFT and RL on both ID performance and OOD generalization.

2. The experiments and analysis presented in the paper are internally consistent, and the conclusions drawn align well with intuitive expectations, rather than being entirely counterintuitive.

### Weaknesses
1. A very poor presentation. The paper extensively uses vspace, resulting in overly crowded content. Additionally, there are numerous errors in the text, such as referring to Qwen-2.5-7B in many places as Qwen-2.5-8B.

2. Regarding Figure 2c, the authors argue that "SFT forgets, RL recovers." However, a closer examination of the curve reveals a more nuanced picture: initially, RL actually performs worse on the OOD task. It is only when the OOD performance of SFT declines significantly that RL begins to demonstrate a recovery effect at intermediate checkpoints. Moreover, in the final stages of training, RL once again starts to impair OOD performance. The authors do not fully explain these dynamic patterns observed in the curve.

3. I still believe that conducting ID and OOD analysis solely on the GeneralPoints task is insufficient. Conclusions drawn from a single task are highly susceptible to misinterpretation. For example, the conclusion that "SFT memorizes, while RL generalizes" was also derived from the GeneralPoints task, yet it contradicts the findings presented in this paper. Therefore, relying on experiments from just one task may lead to unreliable conclusions, as they could be influenced by external factors such as the base model, learning rate, and others.

### Questions
Regarding Figure 2d, why does RL consistently underperform SFT on the ID task throughout the entire process? Could this be due to inadequate RL training in the paper's experiments, resulting in poorer model performance rather than an inherent characteristic?

### Soundness
3

### Presentation
1

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
The paper studies reinforcement-learning fine-tuning of LLMs and argues that RLFT is surprisingly effective at improving out-of-distribution generalization.
The experiments show that OOD performance actually peaks early during SFT and then declines, a phenomenon they call "OOD forgetting." The subsequent RL stage doesn't create fundamentally new reasoning abilities but instead acts as a "healer," recovering the OOD capabilities that were lost during the later parts of SFT.

### Strengths
- Highlights the value of RL even without explicitly OOD data.
- The authors go beyond reporting gains, offering plausible hypotheses about why RLFT helps.

### Weaknesses
- The OOD benchmarks resemble the ID data distribution; thus, it’s unclear how well the claims hold under truly challenging shifts. Stronger or more diverse OOD settings are needed.
- The paper states that PPO was used rather than GRPO. What is the rationale behind choosing PPO to train the additional critic model, given that most recent RL works use GRPO?
- The high-level insight that RLFT can mitigate OOD degradation, is intuitively expected and not particularly surprising for the community.
- Several plots (e.g., Fig. 2(a)(b), Fig. 3) are difficult to interpret without reading the text carefully. 
- It is unclear why advantage estimation is necessary for mitigating OOD forgetting, especially if the "restorability" of checkpoints can be reasonably inferred from IID accuracy.
- The paper reports RL-End having better accuracy than SFT-End in Fig. 1 but underperforming SFT-End in Fig. 2c. The authors should clarify such inconsistency.
- A more appropriate comparison would be: SFT(ID) + RL vs. SFT(ID) + SFT(OOD) to show RL benefits over fine-tuning on OOD data.
- Experiments are restricted to a narrow set of base models and OOD tasks, making it unclear whether the conclusions generalize.

### Questions
See weaknesses.

### Soundness
3

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
The paper asks an important question: how does SFT interact with a consecutive RL training? 
This two-step post training procedure has played major role in the LLM era and yet the relationship between the two steps is poorly understood. 
The authors claim that SFT initially improves OOD performance and then there's a drastic degradation in OOD performance as SFT continues. RL salvages some of that, but interestingly enough, we could find a checkpoint earlier in SFT that does better. 
By analyzing many checkpoints of SFT training and running RL on them, the authors show at which points in SFT can we come in and do RL to improve performance, identifying "boundaries" in which RL can take effect. The authors then explain the results by analyzing the advantage distributions from PPO empirically, and then perform an SVD analysis on weight matrices of different checkpoints to show that most of the effects observed in post-training come from rotation of the singular vectors, with the singular values roughly fixed.

### Strengths
Important empirical findings: The paper provides valuable insights into the SFT-RL dynamics, showing that the best OOD performance occurs at intermediate SFT checkpoints rather than at convergence. This has immediate practical implications for practitioners.

Comprehensive experimental design: The continuous tracking of performance across multiple checkpoints and the use of controlled reasoning tasks provides solid evidence for the claims.

Novel mechanistic insights: The SVD analysis revealing that singular vectors rotate while values remain stable is genuinely interesting and challenges existing assumptions about how fine-tuning affects model capacity.

Practical boundaries identified: The discovery that RL recovery only works within certain SFT checkpoint ranges (roughly 40-80% ID accuracy) and the connection to advantage distribution properties provides actionable guidance.

### Weaknesses
Poor introduction organization: The paper dives into technical details without clearly stating contributions upfront. The introduction mentions "in Section 6 we will conclude" without summarizing what happens in between, making it hard to follow the narrative arc.

Presentation issues:
Key concepts like "OOD forgetting" need clearer, earlier definition
Figure 2 has inconsistent axes labels
Figure 3 plots are impossible to read in detail
The 0.4-0.8 accuracy claim for RL effectiveness should be presented as a percentage (I think)

Insufficient experimental support:

More random seeds needed to substantiate the "clear boundaries" claim (I found that finetuning evals are noisy and it's important to explore things more broadly). 
The connection between Gaussian metrics and histograms isn't well explained
"Negative rewards on examples outside the training set" needs clarification - does this mean high perplexity?


Weak mechanistic claims:

The "RL as Automatic Mitigation" section is handwavy - dropout and weight decay effects aren't studied here despite being cited as analogies
The claim about "saving repetition work" is vague
The paper asks important questions about SFT-RL interaction but doesn't fully answer them

### Questions
Run additional seeds for the boundary experiments to strengthen statistical claims
Clean up Figure 3
Remove statements on weight decay, dropout, saving repetition work, etc.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies SFT and RL dynamics on three small tasks and claims: (1) OOD performance peaks at an early SFT checkpoint and then declines; (2) RL can recover some OOD from late SFT but does not surpass the SFT peak; (3) there is a narrow “RL window” where starting RL from mid-SFT helps; and (4) an SVD analysis says rotations of singular vectors (U,V), rather than changes in singular values, track OOD forgetting/recovery.

### Strengths
- The paper focuses on understanding how SFT and RL affect OOD generalization, a timely and practically important question for current training pipelines.


- The snapshot view comparing $SFT_{MaxOOD}$, $SFT_{End}$, and $RL_{End}$ is straightforward and helps readers quickly see the trade-off between ID gains and OOD degradation.


- The checkpoint-sweep experiment effectively illustrates how OOD performance evolves during SFT and how RL recovers (or fails to recover) performance depending on the starting checkpoint.

### Weaknesses
`W1: Limited novelty of the claimed insight`

The observation that the best OOD performance occurs at some intermediate SFT checkpoint is already implied by prior work. It is shown clearly, for example, in Figure 1 of “SFT Memorizes, RL Generalizes” (https://arxiv.org/abs/2501.17161), where certain tasks exhibit exactly this behavior. Reframing this as a new contribution feels redundant and adds little conceptual novelty.

`W2: Checkpoint-sweep analysis shown for only one task`

The key dynamic plots (ID/OOD accuracy vs. SFT checkpoint, before and after RL) are presented only for the GeneralPoints task on LLaMA. The other two tasks—Navigation and Matrix—are not analyzed in the same way, making it unclear whether the reported trends generalize beyond this single example.

`W3: Qwen results do not support the main claim`

In the appendix experiments for Qwen, RL never surpasses the corresponding SFT checkpoint on OOD accuracy and often performs worse on ID. This suggests that the claimed “RL recovery” phenomenon depends heavily on the chosen starting point and is not a consistent or universal trend.

`W4: RL results limited to a single algorithm (PPO)`

All experiments use PPO with fixed hyperparameters, yet the paper generalizes its findings to “RL” broadly. Without testing other widely used algorithms (e.g., GRPO, REINFORCE with baseline, or preference-based methods like DPO), the conclusions remain specific to PPO and its particular optimization behavior.

`W5: Insufficient empirical validation`

The experimental evidence is thin: no error bars or seed variances are reported, and the main claims are demonstrated primarily on one model (LLaMA) and one task (GeneralPoints). This makes it difficult to assess statistical reliability or generality.

`W6: Minor issues`

- W6.1: The paper refers to “LLaMA-3.2-11B,” but such a model does not exist. It likely refers to LLaMA-3.2-11B-Vision, and this should be clarified.

- W6.2: There is no full checkpoint sweep for Qwen on the GeneralPoints task in the main text; only a late-stage subset is shown, where RL underperforms. Including the full curve would make the results more transparent and comparable to LLaMA.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
