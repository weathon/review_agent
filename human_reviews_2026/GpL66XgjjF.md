# When Thinking Backfires: Mechanistic Insights into Reason-induced Misalignment

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
With the growing accessibility and wide adoption of large language models, concerns about their safety and alignment with human values have become paramount. In this paper, we identify a concerning phenomenon: Reasoning-Induced Misalignment (RIM), in which misalignment emerges when reasoning capabilities strengthened—particularly when specific types of reasoning patterns are introduced during inference or training. Beyond reporting this vulnerability, we provide the first mechanistic account of its origins. Through representation analysis, we find that certain attention heads diverge from CoT tokens, modulating rationalization to enable refusal during generation. During training, we find significantly higher activation entanglement between reasoning and safety in safety-critical neurons than in control neurons, particularly after fine-tuning with those identified reasoning patterns. This entanglement strongly correlates with catastrophic forgetting, providing a neuron-level explanation for RIM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the phenomenon of Reasoning-Induced Misalignment (RIM) in large language models (LLMs), where enhancing models' reasoning abilities, particularly through chain-of-thought (CoT) prompting or fine-tuning on reasoning tasks, paradoxically results in increased susceptibility to malicious prompts. The authors offer a detailed mechanistic study, identifying specific attention heads and safety-critical neurons whose representational changes and entanglement underlie the observed trade-off between reasoning and safety. A novel metric, Reciprocal Activation Shift (RAS), is proposed to quantify and predict catastrophic forgetting at the neural level, correlating representational entanglement with misalignment outcomes.

### Strengths
- The paper studies RIM across a diverse set of open-source LLMs (dense and MoE), employing multiple math-reasoning datasets and a rigorous safety benchmark (HEx-PHI). This breadth lends robust credibility to the main claims, as captured in Table 1 and 2, and in the various controlled experiments across inference and training.
- The work employs unsupervised probing, attention-head ablations, and neuron-level causal interventions to connect behavioral safety failures directly to internal model components. For example, Figure 3 and 4 present clear layer-wise probe score visualizations, supporting the hypothesis that co-attention on CoT tokens and refusal dynamics are intricately linked.
- The introduction of the RAS metric and its application to both safety-critical and random neurons is a notable methodological addition. Table in the RAS section and Figure 9 demonstrate that RAS outperforms KL-divergence and other baselines in predicting catastrophic forgetting and safety–reasoning trade-offs. The various ablation studies (e.g., Figure 8) solidify the claim that specific neural circuits are disproportionately affected by reasoning-focused fine-tuning.

### Weaknesses
- All core misalignment results hinge on HEx-PHI, a single benchmark for safety evaluation. This is explicitly acknowledged in the limitations, but it introduces significant risk of dataset overfitting or unmeasured generalization failure. Evaluating on at least one additional, established safety benchmark would strengthen claims about the universality of RIM.
- While the focus on math tasks is motivated, the implication is that RIM might extend to other reasoning domains (e.g., commonsense, coding, multi-step logic), but this is not empirically tested. Even a brief pilot evaluation in another domain would aid in generalizing the trade-off claims.
- The methodology for constructing "effort-minimizing" reasoning patterns and their control groups is defined through LLM editing prompts, but further quantification of linguistic/semantic divergence between control and target CoTs would clarify causal attribution. At present, it is possible that length or unrelated stylistic features still confound the effect (although authors attempt to control for length).

### Questions
- Can the authors provide empirical results on at least one additional safety/misalignment benchmark to validate that RIM and associated mechanisms generalize? Even a small-scale or qualitative demonstration would increase confidence.
- What methodological rationale justifies the harmonic mean in the RAS metric? Would alternative aggregation (arithmetic, geometric, subtraction, or ratio) yield similar or different outcomes? Ablation on this point would clarify the reliability of entanglement quantification.
- How well do the proposed mechanisms and entanglement metrics transfer to domains beyond math reasoning—such as logical reasoning, coding, or commonsense? Are similar safety-effort trade-offs and attention/neuron patterns observed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies reasoning-induced misalignment (RIM) in both inference-time-induced reasoning and reasoning finetuning.
- For inference-time reasoning, RIM is amplified by inference-time CoT, especially with effort-minimizing reasoning patterns.
- For finetuning, normal CoT data reduces misalignment rate, while effort-minimizing reasoning patterns increases misalignment.

The paper then studies the mechanism at inference-time and during finetuning separately.
- For inference time behaviors, the paper identifies several attention heads across layers that exhibits refusal behavior. Intervening on these heads reduces refusal rates compared to ablation on random heads.
- For finetuning, the paper identifies "safety-critical neurons" (i.e. the ones that show the most difference in activation on refused vs accepted requests) which increase the misalignment rate strongly when zeroed out.

The paper finds that safety and math abilities may be entangled: for finetuning, removing safety-critical neurons also hurts math performance.

To study the extent of catastrophic forgetting, the paper proposes a "reciprocal activation shift" (RAS) score, which computes the Harmonic mean of the amount of activation change in safety and math tasks.
- Empirically, RAS increases after finetuning, which is expected given its definition.
- RAS is also shown to correlates strongly with misalignment rate.

### Strengths
- The paper studies the timely question of reasoning-induced misalignment (RIM) and discovers an interesting safety-reasoning trade-off. 
- The investigation moves beyond viewing RIM as a type of catastrophic forgetting, and aims to provide mechanistic explanation. It finds "interference", where the same set of neurons can impact both safety and reasoning.
- The paper provides careful empirical investigation, with thorough experimental evidence and discussions on alternative hypotheses.

### Weaknesses
I don't have major complaint about the paper.

One comment is that please consider removing unclear languages such as "cognitively demanding situations" or "cognitive effort" -- what does "cognition" mean for LLMs?

### Questions
- Can we predict the extent of RIM before training on a task? For example, by measuring the gradient norms on the data?
- Fig 4: how many inputs are considered?
- Fig 3 & 5: why studying refusal rate at particular tokens, rather than averaged over all tokens? Why do we focus on tokens between think tags? Should we ignore the behaviors at other positions?
- Sec 4.1: are activation values always non-negative (because of ReLU)?
- Line 424: should the paragraph title be "RAS predicts misalignment rate", since RAS is by definition measuring forgetting?
- Table 3: do you have explanations on why the correlation measured for Phi-3.5 is much lower overall?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper identified and mechanistically analyzed a concerning phenomenon: Reasoning-Induced Misalignment (RIM). It discovers that safety alignment could decrease when reasoning capabilities strengthen. Based on experiments in in-context CoT prompting and finetuning on reasoning tasks, the paper discovers that effort-minimizing reasoning patterns could lead to the RIM phenomenon. Then, the paper tries to understand the RIM phenomenon from the neuron level. For in-context CoT inference, it discovers that CoT changes the behavior of some attention heads, which induces safety misalignment. For fine-tuning, the paper discovers that there are safety-critical neurons, and these neurons might transfer to focus on reasoning after fine-tuning, which causes the safety misalignment.

### Strengths
(1) The studied problem is interesting, and the motivation is meaningful. It is important for LLMs to be safe. Understanding how LLMs’ safety misalignment emerges during fine-tuning could be very beneficial to future applications. and to ensure that LLMs still follow safety alignment after fine-tuning.

(2) This paper studies several related questions, providing a systematic understanding: It first identifies the RIM phenomenon in in-context CoT prompting and finetuning on reasoning tasks, then analyzes the mechanisms in in-context CoT prompting, and finally analyzes the mechanisms in finetuning.

(3) The paper chooses different appropriate methods/tools in different sections to investigate or analyze different perspectives.

(4) The experiments are conducted in various model families, providing convincing results. Experiment results support the discoveries.

### Weaknesses
(1) As mentioned in the strengths, this paper systematically analyzes reasoning-induced misalignment in different sections: identification of Reasoning-Induced Misalignment, RIM (Section 2); analyzing internal mechanisms of RIM in CoT in-context prompting (Section 3); and analyzing internal mechanisms of RIM in finetuning (Section 4). However, it lacks connections between the discoveries in different sections. For example, Section 2 discovers that effort-minimizing reasoning patterns could cause RIM, then a natural question is why effort-minimizing reasoning patterns cause RIM at the neuron level. But later sections didn’t mention this. Section 3 discovers that CoT inference changes the behavior of some attention heads, and then a natural question is whether fine-tuning changes the behavior of these attention heads in a similar way. But Section 4 didn’t study this. If my understanding is correct, Section 4 studies MLP neurons, but Section 3 studies attention heads. After reading Section 4, I am also wondering about these MLP neurons’ effects during CoT inference. Can we apply the methods used in Section 3 to also analyze fine-tuning mechanisms in Section 4? And can we apply the methods used in Section 4 to also analyze CoT inference in Section 3? It would be better if we could understand the neurons’ mechanisms in CoT inference and fine-tuning as a whole picture, and how these mechanisms affect effort-minimizing reasoning patterns.

(2) The experiments only test math reasoning ability. It would be more inspiring if we could also analyze other reasoning tasks.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
