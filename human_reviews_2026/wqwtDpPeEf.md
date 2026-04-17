# InverseScope: Scalable Activation Inversion for Interpreting Large Language Models

- Decision: Reject
- Scores: 4, 6, 8, 4

## Abstract
Understanding the internal representations of large language models (LLMs) is a central challenge in interpretability research. Existing feature interpretability methods often rely on strong assumptions about the structure of representations that may not hold in practice. In this work, we introduce InverseScope, an assumption-light and scalable framework for interpreting neural activations via input inversion. Given a target activation, we define a distribution over inputs that generate similar activations and analyze this distribution to infer the encoded information. To address the inefficiency of sampling in high-dimensional spaces, we propose a novel conditional generation architecture that significantly improves sample efficiency compared to previous method. We further introduce a quantitative evaluation protocol that tests interpretability hypotheses using the feature consistency rate computed over the sampled inputs. InverseScope scales inversion-based interpretability methods to larger models and practical tasks, enabling systematic and quantitative analysis of internal representations in real-world LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes InverseScope, an activation-conditioned generator designed to interpret neural representations in large language models (LLMs) via activation inversion. Given an activation from a specific site (layer/head), the model defines a conditional distribution over inputs that would yield similar activations and samples from this distribution using a conditional Transformer generator. The authors also introduce the Feature Consistency Rate (FCR), a quantitative metric evaluating whether generated inputs preserve certain features (e.g., subject/object identity).  
Experiments show qualitative and quantitative correspondence between activation patterns and encoded semantic features.

### Strengths
1. **Clear motivation and formalization:** The paper clearly articulates the challenge of probing activations in LLMs.

2. **Reasonable architectural engineering:** The control-layer conditional generator is a practical and technically sound approach for conditioning a Transformer decoder on internal activations.

### Weaknesses
1. **Scalability claim insufficiently supported:** The paper claims to “advance inversion-based interpretability by scaling it to larger open-source LLMs and applying it to practical tasks” (lines 69–70). However, the actual generator is trained on GPT-2 small, and target models are limited to Gemma-2B and LLaMA-2-7B. No experiment demonstrates how the generator behaves with increasing activation dimensionality, even though the authors justify their method by noting that  “the probability that a random input produces an activation close to $\hat z$ decays exponentially with dimensionality” (lines 166–167). If dimensionality scaling is the central motivation, a quantitative study showing how approximation accuracy degrades or stabilizes with dimension is essential.

2. **Limited novelty:** The paper feels incremental in method, with novelty residing mostly in framing rather than technique. The assumption that similar activations imply similar semantics has already been well discussed in prior studies (e.g., Bengio et al., 2013, IEEE TPAMI, as also noted by the authors in lines 87–88). Moreover, the study largely reuses a GPT-2-style conditional language model without structural innovation.  The authors provide no justification for this architectural choice, nor any ablations showing how control-layer design or alternative decoder types affect inversion fidelity.

3. **Shallow and constrained experimental validation:** The experimental validation of InverseScope remains shallow and constrained in scope. All generator experiments rely exclusively on GPT-2-small, with no ablations across different generator sizes or architectures, leaving open questions about model capacity and scaling behavior. Furthermore, the input prompts used throughout the experiments are notably simple and short. Even in the Limitations section (lines  480–481), the authors acknowledge that the current setup does not test complex or compositional language; however, such long, multi-clause prompts would provide a much more meaningful evaluation of scalability and generalization. In addition, while the paper refers to “task-specific input distributions” (lines 241–242), the method is not evaluated on a broader variety of tasks beyond IOI and RAVEL, limiting the evidence for its task-agnostic applicability.

4. **Human-defined feature functions:** Feature functions $f(x)$ are manually constructed by the authors for each task (Appendix D.1.2-lines 691–692, Appendix D.2.4-lines 821–824). However, the paper does not explain why these task-specific, rule-based definitions are the most appropriate way to evaluate feature consistency. If users must manually define $f(x)$ for every new task, the method’s scalability and reproducibility are questionable, since performance could vary substantially depending on how $f(x)$ is specified.

[1] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798–1828, 2013.

### Questions
**Questions for the Authors**

1. Can InverseScope generate diverse and novel sentences (unseen during training) for a given target activation? Since FCR evaluation appears to be closely related to the diversity of generated samples, it would be important to measure and report explicit diversity metrics (e.g., lexical or semantic variance).

2. Could you provide a quantitative analysis showing how approximation accuracy or FCR stability changes as activation dimensionality increases? Does InverseScope maintain inversion fidelity when applied to larger-scale LLMs (e.g., LLaMA-13B or 70B) or to more complex reasoning tasks?

3. The generator architecture is fixed to GPT-2 small. Could you explore alternative generator backbones (e.g., T5, Mistral, Gemma) or larger-scale models? What motivates this specific architectural choice, and would inversion performance or sample diversity change with different configurations?

4. As seen in Figures 2 and 4, only late layers exhibit strong inversion behavior, while early-layer activations appear almost flat. Can the authors provide insight or diagnostic analysis explaining why inversion signals are weak or absent in earlier representations?

**Additional Suggestions**

1. Typographical errors: lines 159–161; Appendix lines 687–688.

2. The related work section is too short to clearly position the paper within recent interpretability research. Expanding it—perhaps in an appendix—would improve clarity and contextual grounding

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
2

### Summary
The authors present a new technique to map from activations to plausible input sequences that would produce similar activations. To do this, the authors train a conditional generation model. This model is a small transformer model trained on next token loss but conditioned on the latent activations of the model we would like to interpret. Instead of training a site specific model, which would be computationally expensive, they train translator linear layers for each site, training a single unified conditional model. Their conditional model is a finetuned of GPT-2 small, but they apply it to larger models.

### Strengths
The proposed method is more sample efficient than other inversion methods.

The proposed method correctly identifies some of the attention heads that are important for the IOI circuit in GPT2.

The accuracy on the classification task on the RAVEL benchmark surpasses that of SAEs.

### Weaknesses
The 'inverting' model has to be re-trained for each specific task.

Although this can be said of several interpretability methods, it is not clear here if the latent representations have any causal link to the mechanisms of the underlying model and it is not obvious how to test the predictions made. 

On the IOI task, the 'ground' truth heads were correctly identified by the consistency rate, but there were also other several heads that had similar consistency. In a world where 'ground' truth labels don't exist, it is not obvious how much this method could be used to identify relevant components.

### Questions
I think this sentence (lines 136-137) is not well constructed `Given the distribution P(x; ˆz), we propose a three-step pipeline for feature interpret, involving hypothesize, formalize and evaluate:`

I couldn't quite understand how many samples were used to finetuned the InverseScope model for each task. 

Could an LLM distinguish between the 'true' input and the generated examples?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose InverseScope, a framework for interpreting neural activations by reconstructing the textual inputs that triggered them. In order to do so, the authors propose a conditional generation architecture, where a projection of the representation from the original network is used to conditionally influence reconstruction of the original network’s input. The reconstruction network is initialized to GPT-2, and then its parameters are frozen and only the conditional set is trained to induce the reconstructon behavior after a solid language modeling base.
The authors perform experiments on the IOI task and RAVEL datasets, showing that the proposed architecture improves attribute identification over SAE-based baselines in the latter. In a follow up analysis, the authors also use their framework to analyze where task-specific information obtained from ICL is encoded in the model, and confirm findings from previous works which suggested that these features are encoded in the middle layers.

Overall the paper is well written and easy to follow. The work opens up interesting avenues for inspecting knowledge encoded within LLM latent representations.

### Strengths
- Proposes a novel framework for interpreting LLM internals
- Operationalizes the framework with a conditional generation architecture
- Results on IOI and RAVEL show the method is promising and more accurate compared to SAE-based alternatives
- Interesting analysis sheds light where task-specific features from ICL are encoded within the model

### Weaknesses
- It would be interesting to also show qualitative samples of reconstructed inputs, as well as failure cases.
- I think it is too strong to consider a conditional LM as a standalone contribution, as such architectures have been widely used prior.
- The related works section feels quite thin. Namely, conditional generation based on model latents, and the connections to the perspective of variational autoencoders seem relevant - but this is quite minor.

### Questions
See above

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
4

### Summary
The paper is motivated by the goal of finding "assumption-free" interpretability methods, that don't rely on the linearity and/or sparsity assumptions present in many contemporary tools, such as sparse autoencoders (SAEs). As such, there are no assumptions made on compositionality, and the paper exclusively focuses on the "local" interpretability problem of creating a general tool for interpreting individual LLM activations, as opposed to the "global" problem of discovering compositional structure within the entire space of activations. 

The intuition for the method is as follows: suppose we invert the activation and get back a text input having some salient property $p$. We might think that the activation encodes that "the input has property $p$". We can check this by sampling many activations "close" to it in the embedding space, invert them, and check if they have the same property. If they all do, we  have some reason to believe the activation encodes $p$; if it's a random mix of have / not have $p$, we have some reason to think maybe this activation is not sensitive to $p$. 

To summarize the presentation of the methodology from Section 2, the paper proposes a method to interpret activations $z$ in LLMs by:
1. approximately inverting noisy versions of $z$ back to input texts;
2. using these texts to formulate a hypothesis for a feature $f$ of text that could be represented by $z$;
3. checking the hypothesis by evaluating the probability (over a task-related text distribution $x\sim D$) that activations close to $z(x)$ have the same value of $f$ as $x$. Notably, there is no condition that activations far from $z(x)$ should have a *different* value of $f$ from $x$; such activations simply don't matter for the objective by which the hypothesis is evaluated.


To approximately invert activations, the authors train a language model conditioned on internal representations, trained to generate input texts that result in representations close to the wanted internal representation. This is effectively a next-token prediction objective.

Note that this method provides a way to classify activations $z$ whenever we have some function $f$ from texts to a finite set of classes. This is because we can approximately invert $z$ and evaluate $f$ on the inverted text. In that way, the method is similar to the well-known linear probe method, but without the linearity assumption.

Experiments include:
- identifying which heads in GPT-2 represent which features in the IOI task, a well-studied simple language task whose circuit has been mapped extensively in prior work. 
- benchmarking against SAEs on the RAVEL dataset, where InverseScope is compared to using individual SAE features as classifiers.
- studying the layers in which task vectors emerge

### Strengths
The paper tackles a somewhat under-investigated question in the interpretability literature overall: can we have an "oracle" that simply tells us what features of the input text a given activation "cares about", without relying on assumptions like linearity or sparse coding? The work makes the assumption that activation semantics is "continuous", which seems reasonable as far as assumptions go. 

The writing is clear & easy to follow.

### Weaknesses
- The contribution over the prior work [1] is relatively incremental. The prior work also trains an activation inverter. The main contribution of the current work is not in methodology, but in the kernel used for approximate inversion and in the experiments.
- The approximate activation inversion process complicates and obfuscates the method, as it introduces hyperparameters (the noise scale and the "width" of the kernel) with an unknown role in the final results. Additionally, the inversion only works on a limited task dataset, limiting the overall applicability of the method. This means that the method can only generate hypotheses for concepts that exhibit variation in the task dataset, unlike an SAE for example, which can generate hypotheses based on concepts in the entire pre-training distribution. In other words, the question being answered here is not "What is the model thinking about when processing the input that created this activation" but "What *dimensions of variation in the task dataset* is the model thinking about when processing the input", which is subtly but crucially different.   
- This is at heart a correlational method (no causal experiments are performed), and as such there's only limited interpretability utility to be found in it. I won't make this objection in detail, as it is analogous to the challenges to probes as an interpretability tool that have already been raised in the literature. See work by Belinkov and colleagues, e.g. “Probing Classifiers: Promises, Shortcomings, and Advances” or “Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?”
	- Related to that, the authors say "Crucially, while the original benchmark evaluates interpretability via causal interventions on model behavior, we instead focus on a more fundamental question: assessing the method’s fidelity in identifying the correct attribute encoded within the activation itself" (line 377) - I disagree that this is more fundamental.
- As the authors readily point out, in general there's no obvious way to generate feature hypotheses for step 2. of the method. 



[1] Xinting Huang, Madhur Panwar, Navin Goyal, and Michael Hahn. Inversionview: A
general-purpose method for reading information from neural activations

### Questions
What is the sensitivity of the results to the hyperparameters involved in the inversion method? 

In general in interpretability, we always know the input an activation came from. Since in your method you don't have a general method to generate the hypothesis $f$, a plausible alternative is to skip the activation inversion altogether, and instead formulate a hypothesis by perturbing the input in salient ways and measuring activation distance. How do you think about the tradeoffs here?

### Soundness
2

### Presentation
2

### Contribution
2
