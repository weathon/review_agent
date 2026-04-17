# Spilled Energy in Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
We reinterpret the final Large Language Model (LLM) softmax classifier as an Energy-Based Model (EBM), decomposing the sequence-to-sequence probability chain into multiple interacting EBMs at inference. This principled approach allows us to track “energy spills” during decoding, which we empirically show correlate with factual errors, biases, and failures. Similar to Orgad et al. (2025), our method localizes the exact answer token and subsequently tests for hallucinations. Crucially, however, we achieve this without requiring trained probe classifiers or activation ablations. Instead, we introduce two completely training-free metrics derived directly from output logits: **spilled energy**, which captures the discrepancy between energy values across consecutive generation steps that should theoretically match, and **marginalized energy**, which is measurable at a single step. Evaluated on nine benchmarks across state-of-the-art LLMs (including LLaMA, Mistral, and Gemma) and on synthetic algebraic operations (Qwen3), our approach demonstrates robust, competitive hallucination detection and cross-task generalization. Notably, these results hold for both pretrained and instruction-tuned variants without introducing any training overhead. Code available at [github.com/OmnAI-Lab/spilled-energy/](https://github.com/OmnAI-Lab/spilled-energy/)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new method for detecting hallucinations in language models by interpreting the model's sequential predictions as quotients of two different energy-based models (EBMs) denoted $E_\theta^{\ell}$ and $E_\theta^m$. The paper defines the spilled energy as the difference between the two energies: $\Delta E_\theta(\mathbf{x}_{0:i}) = E_\theta^\ell(\mathbf{x}_{0:i}) - E_\theta^m(\mathbf{x}_{0:i})$. 

The paper then proposes several measures for hallucination detection, including:
- marginal energy $E_\theta^m$,
- spilled energy $\Delta E_\theta$, and
- scaled spilled energy $\Delta E_s := |E_\theta^m| \Delta E_\theta$.

These metrics are evaluated in two settings. First, a synthetic setup, involving the detection of increasingly smaller numerical errors introduced in arithmetic computations of 13-digit integers. Spilled energy ($\Delta E_\theta$) gets strictly superior ROC curves (better pointwise, not just in AUC) than baselines across Llama-3-8B (both instruct and non-instruct versions), Qwen-3-8B, and Mistral-7B-Instruct. 

Second, the proposed methods are tested in detecting errors in reasoning tasks across several benchmarks. To do so, the *exact answer tokens* are detected, and the different methods are computed across them using different pooling methods. The different methods are compared against logit energy and [1] using AuROC scores.

---

[1] Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H. and Belinkov, Y., 2024. Llms know more than they show: On the intrinsic representation of llm hallucinations. arXiv preprint arXiv:2410.02707.

### Strengths
- **(S1)** The proposed methods are efficient, simple, generic, easy to implement, can be applied to any language model given access to its logits (gray-box access), and extend previous hallucination detection methods.

- **(S2)** The proposed methods are evaluated on a wide range of models and tasks, and show strong performance. The synthetic setup is well-designed, and the reasoning tasks are representative of the kind of tasks where hallucinations are likely to occur in real-world applications.

- **(S3)** The entire derivation of spilled energy is presented in the main text and is not hidden in the appendix. The derivation is clear and easy to follow (if a bit cumbersome).

### Weaknesses
- **(W1)** The paper does not provide a theoretical justification for why spilled energy (or the other proposed methods) is a good indicator of hallucinations. It is just an empirical observation. Moreover, the intuitive explanation is limited to:
  > ...the two energies, not interacting at the same step but at steps $i$ and $ i-1$, should be equal, but they are measured in the LLM at different generation steps and from different components.

  and
  > Since both terms on the right side should be equal to $E_\theta (\mathbf{x}_{i:1})$, delta values should always be zero when we are correctly modeling the energy at timestep $i$.

  Which are unclear to me (see question 1).

- **(W2)** While the authors mention that using the exact answer is critical for deployment, the paper is vague on the exact method used to detect these exact answer tokens. The only mention I found is in Section 4.2: 
  > We identify this span by prompting the LLM for a brief answer.

- **(W3)** The derivation of spilled energy is more cumbersome and complicated than it should be. E.g., equations (1) and (2) are redundant (one of them suffices), and equations (3)-(5) can be united into one. This is a minor weakness.

### Questions
- **(Q1)** Why do the authors claim that:
>  ... delta values should always be zero when we are correctly modeling the energy at timestep $i$

- **(Q2)** How exactly are the exact answer tokens detected?

- **(Q3)** When comparing with [1] in Table 1, are the probes retrained on each dataset? If not, which datasets are they trained on?

Given a strong answer to **Q1**, addressing **W1**, I would be open to increasing the score. 

---

[1] Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H. and Belinkov, Y., 2024. Llms know more than they show: On the intrinsic representation of llm hallucinations. arXiv preprint arXiv:2410.02707.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a training-free method for detecting hallucinations in LLM. The core idea is to reinterpret the final softmax layer of an LLM as an Energy-Based Model (EBM). The paper define a variable named spilled energy that quantifies the differences between two energy values across consecutive generation steps that should ideally be equal. The paper has a hypothesis that a big spilled energy is correlate with hallucination in model generation. The authors test this hypothesis on both synthetic task and real-world benchmarks. The results show that spilled energy is a strong signal for detecting hallucination. Their method outperforms baseline method, although there could still be false positive cases.

### Strengths
1. The paper formulate the LLM's generation procedure as energy based model, which enable the following definition of "spilled energy" for hallucination detection. 
2. The method proposed in the paper is a training-free method which make it lightweight and applicable to most LLM for hallucination detection. Empirically, the paper show results on both sythentic and realworld dataset, which validate that the method may have generalization across different domains. While other works (non-training free) show worse performance under new datasets, the paper’s method show robustness in performance.
3. The experiments in the paper are solide. The author provide results on sythentic dataset, which show the metric's ability to separate correct from incorrect answers. The results on realworld datasets shows the method's valid on a wide range of domains. The results are validated across model family as well.
4. The ablation section also consolidate the findings. The authors demonstrate the critical importance of localizing the "exact answer" tokens, showing that doing so provides a good performance gain.

### Weaknesses
1. The method needs to first identify the specific token span constituting the "exact answer". The paper implement this by "prompting the LLM for a brief answer”. My concern is that what if the LLM's "brief answer" is itself a hallucinated? What if the “exact answer” is wrongly identified?
2. The current evaluation focuses on tasks with answer that can be localized to a short range(words). My question is that how the method perform on more subtle hallucinations, such as a mutilple incorrect words/phrases in a long paragraph or complex factual error that is not contained in a single noun phrase.  
3. In limitation, the author acknowledge that the method can produce false positives on tokens that are not semantically informative, such as punctuation. While this might be mitigated this with answer localization, it suggests the raw signal is noisy and may not be a pure measure of semantic or factual correctness. Can we do better to reduce the false positive? 
4. If energy gets “spilled”, it is a sign the model’s prediction doesn’t align well with its internal probability distribution. Can you give more explanations why we can use this to detect hallucination? I am not very convinced by current explaination in paper.
5. Each time, we need to inference the model twice, the extra latency make the method impractical

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a training-free method to detect hallucinations in Large Language Models (LLMs) by reinterpreting the final softmax classifier as an Energy-based Model (EBM). This reinterpretation allows the authors to decompose the sequence-to-sequence probability chain into multiple interacting EBMs.

The core contribution is the introduction of "spilled energy", which measures the discrepancy between two energy quantities that should be mathematically equal across consecutive generation steps, but differ in the LLM's implementation due to the way marginal and joint probabilities are computed. The authors also propose a complementary metric, marginal energy, which is measurable at a single step.

The paper empirically demonstrates that high spilled energy strongly correlates with LLM hallucinations. Crucially, their method is training-free and shows strong cross-dataset generalization across state-of-the-art open-sourced LLMs (LLaMa-3, Mistral, Qwen-3) and nine benchmarks including synthetic and real-world datasets. It localizes the signal to the "exact answer tokens" and outperforms the logit confidence baseline and prior probe classifier approaches, especially in cross-dataset settings where probe classifiers fail to generalize.

### Strengths
1. The idea of reinterpreting the LLM's autoregressive sequence modeling as a chain of Energy-based Models (EBMs) is novel and very interesting.
2. The implementation of the hallucination detector is simple and training free, but achieve a good generalization across synthetic and real-world datsets. 
3. The technical motivation and derivations are sound, and the method is well mathmatically-grounded.
4. The paper is well-written and the presentation is clear.

### Weaknesses
1. There might be some technical details make the detector hard to be applicable in the real-world tasks.
    - The paper emphasizes that localizing the signal to the "exact answer tokens" is essential. However, correctly to find the exact answer tokens might be non-trivial. In the paper, authors identify this span by prompting the LLM for a brief answer seems to be a bit fragile to me. This is essentially a dependence on a pre-processing step that relies on LLM's outputs, which may introduce noise or additional complexity. 
   - Compared to logit confidence, spill energy requires calculation of the quantities using two-step predictions, which may introduce the additional cost. Marginal energy only needs the one-step calculation but the results for two energy forms are quite mixed.
    - The analysis focuses on the "exact answer" tokens which can moslty be applied to QA or classification tasks. How well would the Spilled Energy perform on detecting other types of hallucinations common in LLMs, such as self-contradictions or source-attributable error that might occur earlier in a long-form generated answer or reasoning trace, outside of the specific final answer tokens?

### Questions
1. Why do we observe the cross-dataset results that for non-aligned models (Mistral), marginal energy sometimes slightly outperforms spilled energy? 

2. Is there any insight why marginal energy is a superior signal in the pre-trained model setting, or why instruction-tuning seems to amplify the margin for spilled energy? Maybe a more in-depth analysis into comparing two forms of energy is interesting for the pre-trained and instruction-tuned models.

Other questions are listed in the bulletpoints of weakness as the justification.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper reinterprets the final softmax layer of large language models (LLMs) as an energy-based model (EBM). From this perspective, the authors claim to identify a discrepancy between two energy formulations, termed \emph{spilled energy}, which correlates strongly with hallucinations. Unlike probing classifiers, spilled energy requires no additional training. Experiments on LLaMa-3, Mistral, and Qwen-3 across nine benchmarks show that spilled energy outperforms logits and probing methods in detecting hallucinations in the cross dataset setup.

### Strengths
- The paper considers a very timely and important problem of hallucination detection.
- The paper is well-written and easy to follow.

### Weaknesses
- In my view, the reliance on exact tokens is a significant limitation. I am familiar with the paper by Orgad et al., which was the first to demonstrate that identifying the right token can improve detection. However, I see this primarily as an interesting observation rather than a practical method for hallucination detection. The reason is that identifying such tokens typically requires external algorithms or the use of other LLMs, which leads to very high latency—making it impractical for real-world hallucination detection.
- I find the concept of “spilled energy” somewhat unclear and potentially inconsistent. Could you please see my first question below for further context?
- The work appears to lack some important baselines, and the experimantal setup is unclear. For example, in my view, it would be valuable to include comparisons with Ptrue and Semantic Entropy [1]. In addition, some key comparisons are not clearly presented—for instance, I did not see results benchamarked against the baselines (even those already considered in the paper) under the standard setup (e.g., Orgad et al. when trained and tested on the same dataset). I would appreciate clarification on this point. If such a comparison is not included (and only the “cross-dataset” case is considered), it would significantly weaken the strength of the paper.

**References:**

[1] Kuhn et al., Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation, 2023

### Questions
- I have a question regarding the following sentence (quoting from the paper): 
"E^{\ell}_{\theta} and E^{m}_{\theta} are computed from the output of the model, but with two key differences: E^{\ell}_{\theta} is obtained as a single logit extracted using the id of the sampled token, while E^{m}_{\theta} is computed by marginalizing over all id's in the vocabulary."
 It seems that you are associating the logit of a single token with the energy function of all the preceding tokens as well. Could you clarify why this is considered correct? This association feels inconsistent: if E^{\ell}_{\theta} corresponds to a single token, then why does E^{m}_{\theta} correspond to the normalization factor over the entire vocabulary? More concretely, I would have expected E^{m}_{\theta} to represent the prediction for the specific token immediately preceding the one associated with E^{\ell}_{\theta}, rather than the marginalized quantity. Consequently, the idea of “spilled energy” seems problematic, as it appears to rely on an inconsistency in how energy is defined. I would be happy to get a clarification on that.
- Could you clarify what is meant by “cross-dataset” in Table 1? This is not explained anywhere, and I find it difficult to understand the setup. Since your method does not require training, it is not clear to me how cross-dataset evaluation is applicable here. I do see how this makes sense in the context of Orgad et al.—since their probing method requires training—but then, what would be considered the standard experimental setup for them without cross-dataset evaluation (if its not Table 1)? Additionally, could you clarify how the cross-dataset setup in Table 1 is applied to Orgad et al.? Specifically, what is the exact setup—what was the model trained on, and what was it tested on?
- Could you clarify what the superscripts m and l represent in Equation (4)?
- In the Abstract, you write: “we propose a method to detect hallucinations completely training-free that naturally generalizes across tasks and LLMs.” Could you please point out where in the paper you demonstrate generalization across LLMs?

### Soundness
2

### Presentation
3

### Contribution
2
