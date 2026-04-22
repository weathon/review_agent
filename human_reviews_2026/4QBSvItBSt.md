# Thinking Sparks!: Emergent Attention Heads in Reasoning Models During Post Training

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 6, 2

## Abstract
The remarkable capabilities of modern large reasoning models are largely unlocked through post-training techniques such as supervised fine-tuning (SFT) and reinforcement learning (RL). However, the architectural mechanisms behind such improvements remain largely opaque. In this work, we use circuit analysis to demonstrate that post-training for complex reasoning sparks the emergence of novel, functionally specialized attention heads.
These heads collectively support structured reasoning and computation.
Our comparative analysis across Qwen families and Qwen-based DeepSeek-distilled model reveals that these emergent heads evolve differently under different training regimes. Distillation and SFT foster a cumulative addition of stable reasoning heads. 
In contrast, group relative policy optimization (GRPO) operates in a dynamic search mode: relatively few attention heads are iteratively activated, evaluated, and pruned, with their survival closely tracking fluctuations in the task reward signal. Furthermore, we find that controllable "think on/off" models do not possess dedicated "thinking" heads.
Instead, turning off explicit reasoning triggers a broader—but less efficient—set of compensatory heads.
Through ablation and qualitative analyses, we connect these circuit-level dynamics to a crucial performance trade-off:
strengthened heads enable sophisticated problem-solving strategies for difficult problems but can also introduce ``over-thinking" failure modes, such as calculation errors or logical loops on simpler tasks. These findings connect circuit-level dynamics to macro-level performance, identifying an inherent tension where complex reasoning comes at the cost of elementary computations. More broadly, our work points to future directions for training policy design, emphasizing the need to balance the development of effective reasoning strategies with the assurance of reliable, flawless execution.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a mechanistic analysis of how post-training methods—specifically Supervised Fine-Tuning (SFT), distillation, and Reinforcement Learning (RL) via GRPO reshape the internal architecture of Large Reasoning Models (LRMs). Using circuit analysis on Qwen-family models, the authors demonstrate that post-training "sparks" the emergence of novel, functionally specialized attention heads. The study reveals a key dichotomy: SFT and distillation lead to a cumulative, stable addition of many new heads, whereas RL operates as a dynamic search process, iteratively activating and pruning a smaller, more targeted set of heads. The paper connects these circuit-level dynamics to macro-level performance, highlighting a trade-off where strengthening reasoning capabilities can introduce "over-thinking" and harm performance on simpler tasks.

### Strengths
1. **Novel Mechanistic Perspective:** The paper offers a rare and valuable circuit-level view of post-training, moving beyond performance metrics to investigate how training methods like SFT and RL alter a model's internal computational pathways. This is a significant contribution to mechanistic interpretability.
2. **Rigorous Comparative Analysis:** The head-to-head comparison of SFT, distillation, and GRPO on a consistent model family (Qwen) is well-executed. The cohort analysis visualizing the lifecycle of emergent heads is particularly insightful and provides strong evidence for the different dynamics of these training regimes.
3. **Causal Validation of Emergent Heads:** The use of ablation studies to confirm the causal role of the identified emergent heads is a crucial step that strengthens the paper's claims, linking microscopic changes to macroscopic performance.
4. **Important Insights into Performance Trade-offs:** The paper thoughtfully discusses the "over-thinking" problem, providing evidence that the same mechanisms that enable complex reasoning can be detrimental on simpler tasks. This highlights a fundamental tension in developing robust reasoning models.

### Weaknesses
1. **Superficial Graph Analysis:** The paper introduces the powerful concept of "reasoning circuits" but analyzes them in a limited way, focusing primarily on counting the emergence and persistence of nodes (attention heads). A deeper topological analysis of these circuits could have yielded more profound insights. For instance, do SFT-induced circuits have a different community structure, node centrality, or motif distribution compared to RL-induced ones? The current analysis scratches the surface of what circuit analysis can reveal about the structure of learned reasoning.
2. **Unexplained Performance Gains from Ablation:** A striking and underexplored finding is that ablating certain heads (e.g., base model heads in the 7B model, Table 2) sometimes improves performance. This suggests that post-training not only adds useful mechanisms but may also introduce or amplify "confuser" or "noisy" circuits. The paper misses an opportunity to analyze this phenomenon, which is critical for understanding the downsides of current training methods.
3. **Limited Scope and Generalizability:** The study's findings are based exclusively on the Qwen family of models and mathematical reasoning tasks. It is unclear whether these mechanistic patterns are universal or specific to this architecture and domain. Given that different model families have different pre-training data and inductive biases, the observed dynamics of head emergence might not generalize to models like Llama or Mistral.

### Questions
1. Your circuit analysis is fascinating. Have you considered performing a more formal topological analysis on the discovered circuits? For example, do the "dynamic" circuits from GRPO have higher node centrality or form denser "hub-and-spoke" structures compared to the more "cumulative" circuits from SFT?

2. One of your most interesting results is that ablating certain heads can improve performance. Could you elaborate on this? Does this suggest that SFT/distillation introduces "confuser" heads alongside useful ones, and that a key role of RL might be to prune such detrimental pathways, in addition to finding useful ones?

3. How do you think the findings would generalize to other model families like Llama? Given that Qwen models are known to have strong innate code-reasoning priors, could the head dynamics you observe be specific to activating these particular pre-existing skills?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors study which circuits are activated in a model as a result of reasoning training in three setups: "Distill", "SFT", and "RL". They use EAP-IG to identify the circuits, and they use head ablations to verify the faithfulness of the identified circuits. They find that (a) new heads responsible for the reasoning behavior emerge; (b) the location of the new heads differs among the setups; (c) the set of new heads varies over the course of training, especially in the RL setup; and (d) think-on / think-off models, which allow such control, implement different circuits, with the think-off mode compensating for the lack of CoT by activating a larger circuit. The paper contributes a mechanistic study of how the circuitry changes as a result of reasoning training.

### Strengths
* The contribution is novel and interesting.
* I like the idea of studying the circuitry to explain the impact of reasoning training.

### Weaknesses
* The presentation needs improvement. As written, it is hard to follow and to assess the soundness of the authors’ claims. Figure placement (e.g., swapping Fig. 2 and Fig. 3), captions (is Figure 2 trained on Open-R1 or GSM8K?), etc. should also be revised.
* Either I am missing them, or there are tables/figures that should be present and cited when discussing the results.
* Some of the claims seem premature — for example, lines 355–367, which I find particularly confusing. I would appreciate it if the authors broke down their logic into digestible steps with accompanying figures and references to experimental results.

### Questions
* Please consider repeating the process for other models (e.g., LLaMA 3.1). In my experience, findings and conclusions from experiments on Qwen models often do not transfer to other model families. Whether they do or not with your approach, the paper would benefit from these additional experiments.
* Why do you perform ablation by zeroing out the head outputs? To the best of my knowledge, the more appropriate strategy is mean ablation ([1, 2]).
* How do you construct corrupted prompts?
* What are “TriviaQA heads”?
* It is strange that ablating the heads that “support structured reasoning and computation” only improves performance (Table 2, 7B). It is especially surprising that this happens for both new and old heads. Without an explanation of this, I am skeptical about the validity of your approach. Could you also show the results after ablating both new and old heads together?
* What data do you use to identify the circuits? Are the circuits benchmark-specific? If so, do they overlap?
* Where is the table/figure for “Quantitative Analysis” in Sections 4.2 and 5?
* The GRPO circuits do not appear to be in more flux than the SFT circuit. Could you clarify why I cannot interpret Figure 3 using the same explanation you use for Figure 2?
* What was the training dataset for Figure 2?
*  Section 6 would benefit from figures and references to experimental results to justify the reasoning.
* Do you calculate the circuit as next-token prediction at the `<think>` token? How do you calculate it in “think-off” mode in Section 6?
* “In contrast, the ablation and scale down to half models exhibit a diminished capacity to discover novel solutions …” (lines 402–404). I would agree if Figure 5 showed an asymptotic difference between the setups. Why am I wrong?

[1] Prakash et al. “Fine-Tuning Enhances Existing Mechanisms”
[2] Wang et al. “A Circuit for Indirect Object Identification in GPT-2 Small”

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The paper investigates a range of different post-training methods for reasoning LLMs using a basic circuit-finding mechanistic interpretability technique. The main claim is that post-training adds new, functionally specialized attention heads that support structured reasoning.

- However, SFT, distillation, and GRPO exhibit rather different behaviors: SFT and distillation add many stable heads (distillation: early–mid layers; SFT: mid–late), while GRPO shows a behavior where few heads are iteratively activated/pruned in sync with reward fluctuations. 

- There is also an analysis of the different “thinking modes” in the analysed LLMs. The paper claims that there are no attention heads directly corresponding to “think on/think off”, but that different sets of heads are active under the two settings.

- The paper claims that there is a high-level trade-off in that methods that deepen planning boost complex problem solving but erode basic execution reliability.

### Strengths
- The contribution is timely, and while there is previous mechanistic interpretability work looking at reasoning models, this mechanistic analysis of how post-training affects reasoning mechanisms could be an important contribution.

- The emphasis on training dynamics sets this paper apart from previous MI work for reasoning models.

- Also, the analysis of the quite different effects on the model’s circuits of SFT, distillation, and GRPO respectively is interesting and potentially a valuable contribution.

- Similarly, the mechanistic analysis of the different thinking modes seems to be novel.

### Weaknesses
- The choice of method is not motivated and there is no discussion of the pros and cons of the selected method.

- The paper does not discuss the issue of polysemanticity at all and whether that is potentially a threat to the validity of the conclusions.

- The circuit discovery method is a bit old-school and it would be relevant to consider more recent methods that are specifically designed to circumvent the problem of polysemanticity, such as circuit discovery methods involving SAEs or Transcoders. Here are a few papers that would be relevant to discuss in this context.

Marks et al. “Sparse Feature Circuits”, (ICLR, 2025).

Caples et al. “Scaling Sparse Feature Circuit Finding to Gemma 9B”, (LessWrong, 2025).

Dunefsky et al. “Transcoders Find Interpretable LLM Feature Circuits”, (NeurIPS, 2024)

- The related work section does not discuss any related work in mechanistic interpretability, and in particular it would be relevant to discuss MI for reasoning specifically. The paper by Cabannes is already cited here, but should be discussed in this context. Other relevant papers include

Dutta et al. “How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning”, (TMLR 2024).

- The last section does not discuss any limitations, despite its title.

- Some of the analysis sections seem more like storytelling rather than rigorously grounded scientific research. To take one instance, the exploration/exploitation discussion on lines 291-294 does not have a solid connection to the empirical results actually presented here: it seems like an example of the sort of “just so” story that MI work is so often criticized for.

- The writing is rather poor throughout the paper, which a minor nuisance which can be fixed in a revised version, but makes the discussion rather hard to follow at some places where the writing cannot be fully deciphered. The abstract has a rather LLM-generated smell to it, so maybe the same LLM could be applied to the rest of the paper to correct weird formulations.

- Please write section 2 more precisely. $N$ seems to be overloaded notation here: it seems to denote both the set of nodes as well as a single node. In the activation patching, spell out more clearly what the output signal refers to.

### Questions
- Do you see any threats to the validity of your conclusions? MI research is often on the edge of “reading the tea leaves” – do you see these results as rock solid?

- How do you justify the choice of method?

- In particular, how do you justify not using a circuit discovery method designed to avoid problems related to polysemanticity?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the circuit-level changes introduced by various post-training methods - namely SFT, GRPO with two different datasets, and distillation. It investigates attention heads to analyze the sparsity and location of the changes, also accompanied by a training dynamics analysis. Results suggest that different methods introduce most notable changes on different layers, and GRPO introduces the least changes among them, supporting recent work that pictures GRPO as inducing sparse subnetworks training and sparse token distribution reshaping; the analysis of model with a hybrid reasoning (thinking on/off switch) shows that thinking-off induces compensatory mechanisms in the model. Authors also conduct causal analysis by amplifying or suppressing the activation of specific heads, validating their circuit-level findings.

### Strengths
S1. Circuit analysis applied to understanding post-training LLM is a novel and evolving area, and the authors tackle important questions with clever and potentially insightful methodology.

S2. The result that methods change heads differently in the layer-wise perspective, reported in Figure 10 and is interesting and important.

S3. Analysis of think on/off regimes provide interesting perspective and might be useful for further research, and is novel to my knowledge.

### Weaknesses
W1. Although circuit analysis is reported as the main tool, along with the faithfulness evaluation, the only result reported for circuits is in the Table 1 where emergent heads are listed; in all following results, only the frequency of head activation are used, so circuit analysis might be redundant in the paper. I suggest to cut the circuit-related exposition and focus on increasing the depth of layer-wise change reported in Figure 10, and move this figure to the main body, replacing Table 1.

W2. The activation frequency change metric might be misleading due to resctructurization of the internal computations and emergence of new circuits - e.g., the head L12H1 might play different role in base model and post-trained model, which is confounded by the change of all other parameters and head roles. Therefore, naively comparing frequencies or even importances of specific elements might just be not applicable here. It is important to map elements  (or, if element is decomposed to a circuit of elements or was aggregated from some circuit to a single node, map the circuits) based on their role in the model internal computations, and identify if any element has changed its role, or if some role became redundant/important, or if some roles emerged or disappeared during post-training and have no counterpart.

W3. There are a little conceptual analysis. What is the role of emerged heads and why some heads were deactivated during training? What are the role of the circuits reported in Figures 11 and 12? More concrete case studies would significantly enhance the exposition and the argument, i.e. the results illustrated by clear representative examples, while in current form they do not provide many value and distract from the main conclusions (layer-wise differences, think on/off study, head activation change).

W4. The role of MLP is not studied, although it is the main pattern detection module in transformer; see e.g. [1].

W5. Lines 302-323.5, 368-377.5 are slightly hard to comprehend and would benefit from moving the textual descriptions to plots or tables, while text would contain the main conclusion about what would be presented on a plots.

### Questions
Q1. Could you calculate some kind of effect measure between the component importance (computed via attribution patching) on the pre-trained model and post-trained? For example, E(pre, post) = (post - pre) / (pre + post) would be interpreted so that positive (negative) values indicate increased (decreased) importance of that specific module. Another one is simple (post - pre) / pre. If you plot a heatmap for each method, it might be very informative. In figures 3, 6-9 only frequency change is reported, but not the importance; although they might correlate.

Q2. Could you provide a fine-grained case-studies with specific examples of inputs and problem-solving mechanism identified by your analysis? For example, what is the mechanism implemented by circuits reported in figures 11 and 12?

Q3. In lines 355-357 you suggest that circuits would be task-specific, but have you identified such specific circuits that show these differences and task specificity? I can think of the following experiment: given some input question, corrupt specific element (performing patching intervention) and analyse the pass@k behaviour of the model when it tries to solve this problem, also comparing it with baselines (clear model without intervention). Analysis of the failure patterns and responses might reveal the role of each component and allow for task-specific study. This is indeed very expensive, but perhaps might be optimized or re-designed. Do you think that it is feasible to perform such experiment?

[1] Arithmetic Without Algorithms: Language Models Solve Math with a Bag of Heuristics, Nikankin et al., 2025

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper applies edge attribution patching with integrated gradients (EAP-IG) to trace “reasoning circuits” that emerge after different post-training regimes (SFT, distillation, GRPO). The authors argue that: (1) SFT/distillation introduce a substantial number of attention heads that are absent in the base circuit; (2) GRPO mostly adds a smaller, more targeted set of heads; and (3) for Qwen3-8B, disabling the thinking mode makes the model activate a broader, more redundant circuit, whereas think-on uses a more compact one. The evidence is based on circuit visualizations and head-ablation results on several benchmarks.

### Strengths
- Analysis of emergent heads across different post-training regimes raises useful questions for future interpretability work.
- The think-on vs. think-off comparison is interesting and seems worth pursuing further.

### Weaknesses
Overall narrative is hard to follow because several experimental details are missing and the structure separates related results.
- How exactly is the EAP-IG score computed? What is taken as the “input of v”? Is the score computed token-wise and then averaged, or do you use a single aggregated embedding?
- Please describe the procedure for constructing “corrupted prompts.”
- There is no figure analogous to Figures 1–2 for the distillation heads; in addition, the current placement of these figures is inconvenient — it would be clearer to group related visualizations together.
- The quantitative analysis in Section 4.2 is difficult to read as text; it would be clearer to present it as a table.
- L359: you state that “when the think process is disabled by predefined <think>\n</think> tokens, the model activates a much larger and more complex set of attention heads,” but in Figure 13 the circuits look comparably sized. Could you provide the explicit list of heads that appear only in the think-off mode?
- You should ablate the same heads in the baseline model and compare the performance changes.

### Questions
Please, see weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
2
