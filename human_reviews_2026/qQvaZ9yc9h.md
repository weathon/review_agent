# Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
The mechanisms by which reasoning training reshapes LLMs’ internal computations remain unclear. We study lightweight steering vectors inserted into the base model’s residual stream and trained with a reinforcement-learning objective. These vectors match full fine-tuning performance while preserving the interpretability of small, additive interventions. Using logit-lens readouts and path-patching analyses on two models, we find that (i) the last-layer steering vector acts like a token-substitution bias concentrated on the first generated token, consistently boosting tokens such as “To” and “Step”; (ii) the penultimate-layer vector leaves attention patterns largely intact and instead operates through the MLP and unembedding, preferentially up-weighting process words and structure symbols; and (iii) middle layers de-emphasize non-English tokens. Next, we show that a SAE isolates features associated with correct generations. We also show that steering vectors (i) transfer to other models, (ii) combine across layers when trained in isolation, and (iii) concentrate magnitude on meaningful prompt segments under adaptive token-wise scaling. Taken together, these results deepen understanding of how trained steering vectors shape computation and should inform future work in activation engineering and the study of reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes to understand the effect of LLM reasoning training by examining steering vectors.
The per-layer steering vectors are obtained by optimizing a standard RLVR objective.

When using a steering vector at a single layer and keeping other layers frozen, the paper finds that:
- All but 2 layers lead to improved accuracy.
- All pre-final layers downweight non-English tokens, while the last layer mostly boosts the probability of outputting opening tokens such as "To" or "Step".

The paper also reports the cosine similarity between the steering vectors, and the changes they incur on representations. The paper considers two quantities per-layer:
- Diff-Diff CosSim, i.e. the similarity between 1) the per-sample change in representation and 2) the averaged change in representation.
- Diff-Vector CosSim, i.e. the similarity between 1) the per-sample change in representation and 2) he steering vector.

While Qwen and Llama models show differences, the overall trend is that:
- Diff-Diff CosSim is generally high, suggesting that the effect of steering vector propagates through layers.
- Diff-Vector CosSim is low.

The paper also shows that steering vectors have some extent of transferability across models.

### Strengths
- The paper considers thorough setups, including multiple ways to perform steering, and different ways to interpret the steering effects.
- The paper experiments with transferring steering vectors across models in the same family (i.e. across models within Qwen-1.5B, Qwen-7B, LLaMA-8B, respectively). The extent of success vary across families, which is an interesting insight.

### Weaknesses
**Lack of actionable items**: While the paper provides a list of observations, I'm not sure what these observations entail. 
- The paper would be stronger if it suggests actionable items, such as showing that the insights from these steering vectors can improve the training of the model.
- The paper could benefit from restructuring; currently it feels like a set of scattered observations.

**Insufficient empirical evidence for some results**: Some claims are based on limited empirical evidence.
- I'm not convinced of the claim of "pre-final layers downweight non-English tokens, since the empirical results are not sufficient.
  - Fig 15 only shows 16 examples which is too small; these could simply be uncommon characters, rather than non-English.
  - In Fig 16, ChatGPT is prompted to distinguish languages.
  - Fig 17 further suggests that a lot of tokens have a strong negative correlation, making the examples in Fig 15 and 16 non-representative.
  - Moreover, there are no comparison of tokens downweighted by the final layer.
- For the SAE feature analysis, Fig 8 only shows limited features, and Fig 7 doesn't say what the "top featuress" are.

### Questions
- For the final layer, how do tokens like "To", " To", or "Step" relate to general attention sink tokens?
- For transferring across models, what does higher or lower transfer indicate about the model family? How can we use this knowledge to improve the training recipe?
- Could you highlight key insights or actionable items?

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
3

### Summary
The paper studies steering vectors—small additive residual-stream interventions trained with an RL objective—and asks what they do inside math-reasoning LLMs. The authors train one vector per layer (Qwen2.5-Math-7B, LLaMA-3.1-8B-Instruct) on DeepScaleR, then probe with standard tools (logit lens, path patching, QK/OV lens, DiffSAE).

### Strengths
- The paper is carefully executed and empirically tidy.
- I think the paper does have some interesting findings. For example, the last layer behaves like first-token substitution (e.g., boosting “To”), and simply prefixing that token recovers roughly 75 % of the last-layer gain.

### Weaknesses
- The intellectual contribution feels incremental. The paper largely applies existing ingredients (e.g., bias-only adaptation/steering vectors trained with RL, standard logit-lens probes, path-patching, and QK/OV analysis) to known math-RL settings. The central insights (“early layers suppress non-English tokens,” “the last layer boosts a prompt-opening token,” “the penultimate layer acts via MLPs”) read as incremental characterizations of surface-level biases rather than new mechanisms or methods. I struggled to identify a new principle beyond careful confirmation of expected behaviors and left without a clear, portable mechanism or design rule that changes how we steer or train reasoning models.
- The DiffSAE results feel very preliminary and incomplete. The “feature associated with incorrectness” is intriguing but under-validated: features are described qualitatively (through a few examples), and the analysis focuses only on “the feature with the largest frequency difference.” There is no controlled group or ablation of any kind.
- The transfer experiments are mixed and weakly explained. Transfer appears to work well when the Qwen base model is the donor but fails for LLaMA, which the authors attribute to “different chat templates.” However, Qwen-Base and Qwen-Instruct also have different templates—so why does it work there? Moreover, since it does not work for LLaMA, the overall “vectors transfer” message feels overstated.

### Questions
Comments

- There are some formatting errors in the appendix (e.g., Appendix I) that need to be fixed, as well as in the references. Some text exceeds the page length.

- The paper also contains typos and a few incomplete sentences (e.g., line 174).

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes steering vectors that are supposed to modify a models reasoning behavior and are trained via RL in a small vector that augments a frozen model and is inserted into the models' residual stream.

### Strengths
The related work is to my (limited) knowledge quite good and up-to-date.

### Weaknesses
Presentation-wise, the contributions on the first page are hard to understand. Overall, several parts of the paper should phrased more precisely and explain the line of thought more explicitly to make the paper easier to understand; In Section 7, I did not get the differences between the variations. The paper is also not really self-contained without the appendix. E.g., the takeaway of Section 8 is unclear from the section itself. My (personal) recommendation to the authors is to cut the paper to its main contributions and work out those in more detail. In its current form, it is trying to achieve to much but fails in providing detailed and generlizable insights about too little. I would strongly recommend the authors to always write out abbreviations at least at the first use (e.g., QK/QV, RLOO, even GRPO and MLP). This avoids ambiguities and reduces cognitive load.

The experimental study is done only on 3 models of two origins (LLama and Qwen). In order make results more generalizable, the experiments should be conducted with considerably models, specifically considering the partially non-aligning results. Regarding the impact of the approach, it should also be compared against other recent approaches mentioned in the related work to see what the effects are in comparison. Also different training datasets (for extracting the steering vectors) should be considered for the experiments.

Details:
I get no insights from Figure 8.
Figure 9: I would disencourage averaging results between benchmarks.
Examples are very hard to read on print-outs.

### Questions
Questions follow directly from the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the internal mechanisms by which RL enhances reasoning abilities of LLMs. The authors freeze the base models (Qwen2.5-Math-7B and Llama3.1-8B-Instruct), and train steering vectors added into the residual stream (independent of context) using a RL objective (RLOO). This goal is to isolate the RL-induced changes for mechanistic interpretation, instead of interpreting full fine-tuning. Note that the models trained by these different procedures would have different behaviors, so it's not clear that this would explain the behavior of RL-training. Nevertheless, the authors make interesting observations. Using mechinterp tools they find that (i) the last-layer vector operates primarily as a token-substitution bias boosting tokens like "To" or "Step"; (ii) the penultimate-layer vector operates mainly through the MLP pathway, largely bypassing attention mechanisms (though this was clearer in Qwen than Llama); and (iii) middle layers tend to de-emphasize non-English tokens. The authors show experiments on the transferability and composability of these vectors but get mixed results across the two model families.

### Strengths
- Using RL-trained steering vectors on a frozen base model is a new approach, and tractable for mechinterp.

- The analysis of the final two layers yields specific, concrete insights. The last-layer vector functions primarily as a first-token substitution (boosting "To"/"Step") is a cool finding, connecting well with "Thought Anchors" (Bogdan et al.).

- The circuit analysis (Section 7) showing that the penultimate vector in Qwen acts primarily through the MLP and bypasses attention mechanisms is interesting and perhaps useful to understand more deeply.

### Weaknesses
- The study aims to understand how standard RL fine-tuning changes internal computations. However, it analyzes a proxy: additive steering vectors trained with RL on a frozen model. The authors justify this by noting this approach can match the performance of full fine-tuning. However, equivalence of outcome does not imply equivalence of mechanism. Standard RL fine-tuning modifies weights distributively and can create complex new circuits. In contrast, steering vectors are highly constrained, additive interventions. So the observations may be tied to steering vectors and not general RL training.

- The paper only considers two model families (Qwen and Llama) and the findings are often inconsistent between the two (e.g., the penultimate layer mechanism was clear in Qwen but messy in Llama; composability was constructive in Qwen but often harmful in Llama). This makes it harder to see if the results are model specific or actually about the training procedure.

- Some interpretations are questionable or spurious. The finding that middle layers de-emphasize non-English tokens is likely an artifact of the training data (English math problems) rather than a fundamental mechanism of reasoning. The interpretation of the DiffSAE features is also speculative and weakly supported by the ambiguous evidence they provide in Figure 7.

- The paper is generally clearly written but currently reads like a collection of isolated experiments, and could benefit from better motivation.

### Questions
Beyond the concerns in the weaknesses, here are some more questions:

- In Figure 1, 23, 24th layer has low mean accuracy, and it is explained that the LayerNorm in 23rd layer limits the steering performance? Why for other layers, the layer norm would not affect the performance?

- In Figure 4, “To” and “Step” are boosted which is interesting and might be related to reasoning. However, a more in-depth analysis is lacking (Between Line 293-300), would other “thought anchors” (Bogdan et al.) be also boosted ?

- In Figure 5, it is shown that for the penultimate layer, steering effect is mainly on MLP rather on attention. Is this also true for all layers? 

- Figure 7 seems not evident enough to conclude they are mostly related to incorrect generations. For example, in the 10-th layer, there are also 50% of the correct ones are activated.

### Soundness
2

### Presentation
2

### Contribution
2
