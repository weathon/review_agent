# Entropy-Lens: The Information Signature of Transformer Computations

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Transformer models map input token sequences to output token distributions, layer by layer. While most interpretability work focuses on internal latent representations, we study the evolution of these token-level distributions directly in vocabulary space. However, such distributions are high-dimensional and defined on an unordered support, making common descriptors like moments or cumulants ill-suited. We address this by computing the Shannon entropy of each intermediate predicted distribution, yielding one interpretable scalar per layer. The resulting sequence—the entropy profile—serves as a compact, information-theoretic signature of the model’s computation.

We introduce Entropy-Lens, a model-agnostic framework that extracts entropy profiles from frozen, off-the-shelf transformers. We show that these profiles (i) reveal family-specific computation patterns invariant under depth rescaling, (ii) are predictive of prompt type and task format, and (iii) correlate with output correctness. We further show that Rényi entropies yield similar results within a broad range of alpha values, justifying the use of Shannon entropy as a stable and principled summary. Our results hold across different transformers, requiring only forward access to intermediate hidden states and the output head; no gradients or fine-tuning are needed.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce the entropy of the logit-lens distribution at each layer as a measure. The goal is to understand how information is compressed over the depth of learning. The authors use this to establish that LLMs of different architectures are processing information differently. In particular, they compare and classify the entropy curves (over layers) for different architectures and model sizes, finding that similar models have similar global information processing properties. In addition, they show that different text types exhibit different profiles.

### Strengths
- The authors communicate their ideas clearly and precisely
- The entropy is indeed an interesting measure of the way information is processed in the network, and as the next token distribution is also favouring a probabilistic view, a natural one indeed.
- Several larger trained open source models are compared.

### Weaknesses
- While idea of using the entropy in interpretability intuitively seems useful, the experiments and discussion lack a grounded interpretation of the value and curve shapes themselves. This would be highly useful to understand whether the measure is practical and interpretable, or whether it only reflects statistical properties of the architecture or text inputs, which unavoidably lead to discriminatory entropy curves. To me, a successful *interpretable* analysis would connect instance-based observations and phenomena to the type of the curve, and validate that the entropy curve has some predictive properties. This deeper understanding is what I would have expected e.g. in the discussion of Section 5.2, and more generally in the experiments.

### Questions
Comments
- In Fig.2 b), it could also be useful to see some standard deviations. 
- As far as I understand, your method aggregates entropy over generated sentences, hence the temperature in the generation might also play a role in your measure. How does this influence your results?
- An ablation on the role of the different architectures could be comparing the entropy profiles of trained and non-trained models. At least that way one might understand if the entropy of learned models is smaller (?) to than that of the untrained ones, and which of its properties come from being learned.

### Soundness
4

### Presentation
4

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
The main contribution of the paper is simple and interesting --- an "entropy-lens" framework to compare models by considering the entropy of the output distribution induced at the head of each layer, for each input token.  They use this framework to identify patterns corresponding to model families, task types (generative/syntactic/semantic), and output correctness.

### Strengths
1. The paper studies a relevant and trending problem of trying to understand and analyze the inner workings of transformer models --- it does so by proposing a framework to "see inside the black-box".
2. The central idea (using the "entropy profile" to identify attributes of models/use-cases) is novel and well-motivated.
3.  The examples chosen to illustrate the utility of the "entropy-lens" framework are relevant and thought-provoking.

### Weaknesses
My major criticism of the paper is that though it provides an interesting framework to compare models, it is unclear what the higher goal here is, or how it fits towards the broader goal of understanding the inner workings of transformers.  They identify and observe interesting patterns (through their own novel method), but then offer little to no insights as to how/why these patterns show up.  

For instance, they observe that different model families show consistently different trends in the entropy profile across layers --- but why does this happen?  does it affect model performance (concretely)?  does it suggest that some models are better suited to specific tasks?  I would have liked to see more answers here, possibly even using some smaller toy models to study these characteristics, it would have made the paper more insightful, in my view.  The same comment holds for the other observations as well --- the lack of a serious (i.e. supported by experiments, analysis) attempt at an explanation means that the merits of the paper are only to propose the entropy-profile and apply it (almost superficially) to a handful of (interesting) tasks.

As such, my score is currently to accept, but I would have liked some insights to support the paper strongly.  I will be happy to raise my score if the paper is able to add content along these lines, at least for one of the settings considered.

### Questions
1. "classical descriptors such as moments or cumulants unsuitable" is repeated several times, but it is still unclear to me why this is the case.  Do you have a clear explanation, possibly with references, for why this is so?  Yes, the output distributions are high-dimensional, but why is a moment "unsuitable" (as the paper claims, without specifying how) while an entropy is fine? 

2. Have you considered any other choices of aggregators?  I would have thought that there might be patterns in the entropy profiles observed at different tokens within the same layer that might allow for a more efficient representation than having the entropies at ALL tokens.  It would also be interesting to see if this somehow connects to the problem of prompt compression (where the goal is to remove tokens that have low influence (ergo entropy) in the prompt).

3. For the "task type" experiments in Section 5.2, the curves in each plot of Fig. 3 look rather similar.  Is the paper suggesting that k-NN gets an AUC ~>95% in distinguishing among these curves?  The differences in the curves are curiously specific (marked by jumps/drops at specific positions)...  

4. Section 5.5:  to really support the claim that Shannon entropy lies in the "informative" range of alpha values, it would be nice to have seen the k-NN AUC in Table 1(c) degrade for alpha outside the (0.5, 5) interval.  Does this happen?

Minor comments:
1. would be useful to have some references directing to how Renyi entropy correlates with other measures
2. Fig. 1(a):  is this one or two transformer blocks?  does each block not require all of attention, MLP, layer norm (as suggested in the text of Section 3.2)?
3. right after (3), it is not immediately clear what $f^i$ is, it would be useful to write "...where $f^i$ is the transformer block at layer $i$..."

### Soundness
3

### Presentation
2

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
The paper introduces the idea of entropy lens which can be used to compute (Shannon or Rényi) entropy of the distribution over vocabulary for a particular token and a particular layer of a transformer language model. There entropies can be aggregated into an entropy profile associated with a particular output text. The paper investigates how distinctive such profiles are for particular models and model families, and for specific prompt types, and also whether they correlate with the answer to a MC questions being correct. The results show that in general the profiles are fairly informative for the tested cases.

### Strengths
- The approach is straightforward and transparent, and grounded at a basic level in information theory
- The presentation is clear and very easy to follow
- The findings are not particulary suprising but intriguing

### Weaknesses
While the results are convincing and the methodology solid, what is less developed in the paper are the implications and/or applications.
Connection to memorization across model layers is mentioned in passing in the main paper and briefly explained in the appendix. A couple of other other use cases are mentioned but not developed further. 

So while this is a clean and curious study, the significance is less clear. Combined with the lack of major methodological developments, the contribution of this paper is rather minor.

### Questions
The paper is clear, I did not have any points to clarify.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors introduce entropy as a measure to study intermediate representations in transformers. By creating an entropy profile across layers, the authors test this probe's capability to discriminate among architectures, task types and text formats, through several experiments.

### Strengths
The paper is well written and the presentation is clear and well organized. The experimental settings are sound.

### Weaknesses
My main concerns with this paper are about novelty and usefulness. Regarding novelty, the idea of probing intermediate representations of transformers across layers has been used extensively in the literature. In particular, several papers, such as (Skean et al, "Does Representation Matter? Exploring Intermediate Layers in Large Language Models", 2024) and (Skean et al., "Layer by Layer: Uncovering Hidden Representations in Language Models", 2025) already use entropy as a probing measure, albeit using a different definition that directly involves hidden representations instead of first converting them to probability distributions using the final linear layer. (To be fair, I am not sure about which of the two approaches is preferrable, although it is not very clear why it is justified to apply the final linear layer, which has been trained to be at the end of the architecture, to intermediate representations.) The related mutual information measure has already been used in (Chang et al., "The Generalization Ridge: Information Flow in Natural Language Generation") as a measure of memorization in intermediate layers of transformers.

Regarding usefulness, my main concern is that, from the paper, it is not clear what is the point or advantage of introducing their entropy-based probe, compared to those already used in the literature. The paper mostly focuses on distinguishing architectures and tasks, which is not of fundamental importance, as those objectives can be achieved by much simpler means. Furthermore, this study does not shed light on the inner workings of transformers, apart from very generic remarks such as "We conjecture that high entropy phases, whether in the early or intermediate layers, allow the model to explore more possibilities in its response" at lines 292-293. It is also not clear why their choice of entropy is convenient compared to the other measures already used in the literature, as in the paper there is no comparison of performance between their probes and others.

### Questions
As expounded above, I would like the authors to address my concerns about the novelty and the usefulness of their method. In particular, I would like the authors to answer the following points:
1. How does your entropy probe compare to other measures already existing in the literature, such as the ones that I mentioned above? What is the advantage of your measure compared to them? Do you have any experiments showing the superiority of your probe?
2. What is the main purpose of using your probe? Can it helpful to interpret transformers' inner workings? Can it help improve future architectures in any way? Distinguishing architectures and tasks is interesting, but can it be useful in any practical, real-world setting?

### Soundness
2

### Presentation
4

### Contribution
2
