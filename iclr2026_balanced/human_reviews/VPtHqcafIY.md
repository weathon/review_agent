## Human Reviewer 1

### Summary
The authors show that LayerNorm modules can be removed from GPT-2 models through fine-tuning, with only a small increase in validation loss across all model sizes. The resulting LN-free models are released on Hugging Face. They also conduct a series of experiments to examine how removing LayerNorm affects several well-known phenomena (such as direct logit attribution, attribution patching, first-token high L2 norm, and confidence neurons). Their analysis reveals that some of these phenomena originate from the presence of LayerNorm, while others remain unaffected by its removal.

### Strengths
* The paper is very well written and easy to follow.
* The experiments and baselines are thoughtfully designed and make it easy to understand the impact of each decision involved in fine-tuning the LayerNorm-free models. The approach is evaluated across multiple GPT-2 model sizes, strengthening the conclusions.
* The paper analyzes several interpretability techniques and phenomena to examine the effects of removing LayerNorm. This enables a controlled investigation into whether well-known behaviors in language models are driven by LayerNorm itself or emerge from other underlying mechanisms. The analyses are well executed, and the conclusions are very interesting.
* all their LN-free models are open-sourced and publicly available on Hugging Face

### Weaknesses
* A key limitation is that the analysis remains largely focused on the GPT-2 family. Although the authors include results for Pythia-70M, this provides only limited evidence of generality, and it is unclear whether the findings extend to newer or larger architectures.
* The largest model evaluated is GPT-2 XL (1.5 B parameters), which is relatively small.
* The title (“Transformers Don’t Need LayerNorm at Inference Time”) feels overclaiming, as the experiments are limited to GPT-2 models and up to 1.5B parameters.
* Some mathematical notation is introduced without adequate definitions.

### Questions
I think the paper is very interesting and well written.
* Did you attempt to train an LN-free model larger than GPT-2 XL? It would be helpful to understand whether the proposed method scales beyond 1.5B.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The authors show that all layernorm layers in a transformer can be removed via fine-tuning from every GPT-2 model with only a small increase in validation loss (e.g. +0.03 cross-entropy loss for GPT-2 XL). They then study these models in interpretability tasks, showing downstream effects such as improvement in logit-attribution interpretations, improvements to approximation methods like attribution patching, and inactivity of confidence neurons. The work is novel because authors replace LN with a purely linear alternative as compared to earlier works which use non-linear alternatives.

### Strengths
I enjoyed reading this paper – I learned something from it. I think others will too! 
- The authors carefully dissect LayerNorm layers, find the sequence of removing these layers with several hyper-parameters, and provide a linear transformation equivalent whose loss is only minimally/negligibly worse as compared to the original. They then rigorously test out the effect of removing the layer-norm on important ideas in the literature – such as the effect on direct-logit-attribution as well as attribution patching, attention sinks, confidence neurons. The authors introduce a simple auxiliary loss term to stabilize loss during fine-tuning and do extensive empirical tests to find the best hyper-parameters for the sequence in which Layer-Norms should be removed.
- Authors find that this removal can be scaled sub-linearly to larger models.
- The work suggests exciting new directions of research, on changes and improvements to the transformer architecture with an interpretability lens.

### Weaknesses
1. The empirical cost of finding the hyper-parameters for LN removal seems to be really high, and the benefits of this removal don’t seem to be significant. Could training a transformer with the fake-LN be a better approach, rather than figuring out the best way to fine-tune away from it?
2. The results are predominantly on GPT-2 – unclear how this transfers to very large LMs that are in use today.
3. Unclear what the main takeaway is, in terms of whether the LN should be removed or not. The predominant advantage seems to be that logit attribution can be direct, but is the whole process of empirically understanding the best sequence/hyper-parameters to remove the LN worth the trade off?

### Questions
1. Did the authors consider training models without LN? How would this compare to their fine-tuned variant as well as the original GPT-2 model? 
2. What is the main takeaway for interpretability researchers?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This article demonstrates that LayerNorm (LN) is not essential for Transformer inference, introducing a novel fine-tuning protocol that successfully removes all LN layers from the entire GPT-2 family (up to 1.5 billion parameters) with only a minor performance drop. The resulting LN-free models make a significant contribution to mechanistic interpretability, rendering Direct Logit Attribution (DLA) mathematically exact and reducing its approximation error from nearly 50% to 0%. The study also uses these models to causally confirm that "attention sinks" and "confidence neurons" are adaptive mechanisms dependent on LN's non-linearity. Finally, the work reveals LN's previously under-appreciated role in robustness and calibration, as the LN-free models become overconfident.

### Strengths
1. Nonlinearities in transformer architectures are a major challenge for mechanistic interpretability. This work takes an important step toward addressing that issue by replacing LayerNorm with a linear alternative.
2. It not only provides a method for removing LN without significantly impacting the model’s overall performance, but also releases a suite of LN-free models that could be valuable to the mechanistic interpretability community.
3. It performs a thorough empirical evaluation to show that model capabilities remain largely intact after removing LN, based on both perplexity and benchmark results.
4. It provides insightful results about the impact of LN on common mechanistic interpretability techniques like direct logit attribution and attribution patching.
5. It also details the impact of the removal of LN on first token norm, attention sink, and confidence neurons.
6. It is well well-written paper which is easy to follow and provides the required details.
7. I also appreciate the broader direction of conducting mechanistic interpretability research to modify and potentially improve model architectures.

### Weaknesses
1. As noted in the paper, relying on a single family of models and not evaluating larger models raises concerns about the broader effectiveness of the proposed LN removal method, especially given the fine-tuning involved and the need for extensive hyperparameter tuning. For example, both GPT-2 and Pythia models use standard LayerNorm, whereas many newer models employ RMSNorm, which limits the immediate applicability of the findings to architectures such as Llama or Mistral.
2. While the impact of removing LN is assessed from multiple perspectives. There is more to be done to fully understand what LN actually does and how it helps in language modeling. For instance, this work showed that removal of LN makes the model overconfident, which needs to be better understood and mitigated.

### Questions
1. Could you explain how you separated LN_{q,k}​ and LN_{v}​? Since the input to the entire attention block is typically normalized, wouldn’t that mean the keys, queries, and values all receive the same normalized input?  
2. Have you tried training a reasonably large language model without LN? If so, how does its performance compare to LN-enabled versions?
3. Beyond the mechanisms discussed in the paper, do you think the removal of LN could significantly affect any known circuits or mechanisms? For instance, could the IOI circuit become more distributed due to the absence of LN?
4. Researchers often apply the final LayerNorm when using LogitLens. Do you think that would no longer be necessary if LN were removed?

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper investigates how LN can be removed from LLMs, and develops a step-wise procedure to do so. By removing LNs one by one, while continuing to (pre-)train using CE loss and an additional loss, the authors manage to remove LNs with minimal harm to model performance. They do so for all GPT-2 models and Pythia 70m. The authors then study 2 mech interp techniques thought to be hindered by LN and find that one performs better without LN (DLA) and another does not (AtP). The authors also discover that the confidence neurons discovered by prior work no longer exist in LN-free models.

### Strengths
This paper (assuming it is extension of work I have seen previously) tackles an original question: can we remove LN from LLMs? It answers the question in the positive, quite clearly showing that LN can be so removed. The experiments fairly convincingly show that models' overall performance is not greatly harmed, and that this makes certain mech interp techniques more accurate. This is somewhat significant, as LN is generally considered an inconvenience by mech interp researchers.

### Weaknesses
- This paper could go a little further in proving that LN-less models are (almost) just as good as ones with LN. For example, it could show that such models can also be fine-tuned or quantized.
- This paper doesn't seem to address fine-tuned / instruction-tuned / RLHF'd / reasoning models at all, even though they are quite often studied by mech interp researchers. Similarly, the models that the paper does address are rather small (though I think that's reasonable, considering the cost of using larger ones). It also doesn't study any models that use RMSNorm.
- In general, my main concern about this paper is its practicality. Even if the number of extra training steps grows sublinearly with model size, this procedure seems inconvenient and compute-intensive to perform. Moreover, as much interpretability work derives its value from studying real-world models that people actually use, the value of studying these LN-free models will likely be limited by the fact that they will see little practical use. This means that this paper will probably be an academic curiosity more than a game-changing development; however, I still think it's interesting enough to publish.

### Questions
- Do you have any results for / did you consider the variants of attribution patching (integrated gradients, AtP*)? They seem worth discussing
- Why use IOI for attribution patching? It seems like you could really just use any data (and even better, a more general dataset), since attribution patching can approximate the effect of any activation patching experiment.
- Where exactly do you show that "the amount of fine-tuning data needed for LN removal grows sublinearly with model parameters" (22-23)?
- Comment: Change 318-19 to say something like "Nanda (2023a) describes it for this reason as "a particularly thorny nonlinearity". The current formulation feels a little too much like leaning on Neel Nanda's particular authority as a well-known individual, which is not super appropriate for an academic paper (or otherwise).
- messed-up parentheses in citations (079-80)

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4