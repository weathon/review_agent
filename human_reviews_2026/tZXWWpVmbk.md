# Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours,  posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood.  In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures.
CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces.
Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalizes across multiple LLMs and attack families.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the issue of adversarial prompts faced by large language models (LLMs) by proposing a model-agnostic detection framework named CurvaLID. The authors first define and implement two geometric measures: sentence-level PromptLID and word-level TextCurv. These measures are then used to train a multilayer perceptron (MLP) to distinguish benign prompts from adversarial ones. Experiments conducted across multiple LLMs (Vicuna, LLaMA2, GPT-3.5, PaLM2, etc.) and multiple attack categories (GCG, DAN, SAP, etc.), claiming to reduce attack success rates to near zero with an overall classification accuracy of approximately 0.99. The overarching contribution lies in systematically introducing classical geometric concepts (curvature, LID) into adversarial prompt detection, demonstrating robust cross-model performance and efficiency advantages.

### Strengths
1. Novel perspective: Integrating the concept of curvature with LID for geometric analysis of text/prompts, combining theory with intuition to offer an enlightening viewpoint.
2. Model-agnostic: Functioning as a pre-processing filter for large language models (LLMs), CurvaLID circumvents the need for fine-tuning target LLMs, demonstrating practical deployment potential.
3. Extensive experimental scope and diversity: The study encompasses multiple mainstream LLMs, several common adversarial attack datasets, and comparative evaluations against various state-of-the-art defence methods. These factors collectively enhance the paper's persuasiveness.

### Weaknesses
1. Overall, the approach is innovative, though certain methods involve combining existing tools (LID, cosine angle) and applying them to novel scenarios.
2. The layout of charts and images within the paper is somewhat haphazard, causing some inconvenience during reading.
3. PromptLID requires training a multi-class CNN model in its initial step. This necessitates the prior collection and annotation of substantial benign prompts for model training, thereby limiting the method's applicability in unannotated or data-scarce environments.

### Questions
Unclear Motivation and Conceptual Novelty: The paper’s motivation is insufficiently articulated. While it introduces CurvaLID by combining curvature and Local Intrinsic Dimensionality (LID) for adversarial prompt detection, the rationale for why geometric properties fundamentally capture adversarial semantics is not convincingly established. As a result, the approach appears as a technical aggregation of known methods rather than a conceptually novel framework grounded in theoretical or empirical insight. 
 
Problem–Solution Misalignment: The paper criticizes existing methods for depending on labeled data, yet CurvaLID itself requires labeled benign and adversarial prompts for supervised training. Although an unsupervised variant is briefly mentioned, it is not the main evaluation. Hence, the proposed framework does not fundamentally overcome the annotation dependency it aims to address, weakening its claimed contribution. 
 
Presentation and Layout Issues: The manuscript suffers from poor formatting: figures and tables (e.g., Tables 1–2, Figure 2) are misaligned, captions are fragmented, and visual clarity is limited. These presentation issues hinder readability and detract from the paper’s professional appearance. 
 
Over-Strict Detection and Lack of Usability Discussion: While CurvaLID achieves near-perfect adversarial detection, it also exhibits false positives on benign prompts. The paper does not analyze this trade-off or discuss potential over-refusal behavior, leaving concerns about the method’s practicality and real-world usability unresolved.

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
The paper proposes a defence against jailbreaks and prompt injection attacks.
Ths is composed of several steps. First a neural network classifies samples into
their respective datasets. Then, based on the internal representations within
the CNN, measures are used of PromptLID and TextCurv. Using a mix-
ture of benign and adversarial prompts PromptLID/TextCurv are computed
to form a new dataset based on these higher level features. Finally, using
this PromptLID/TextCurv a MLP is trained to classify benign and adversarial
prompts.

### Strengths
Deploying small and lightweight classifiers for jailbreak detection is a useful line of research for practical use of defences.

The results are promising: against the tested attacks there is good overall performance.

The approach seems novel taking a different strategy to guardrail defences compared to many other literature approaches

### Weaknesses
No adaptive attacks are considered against the defence: for example, optimise the prompt to simultaneously bypass the defence and still carry out a jailbreak against the underlying LLM. I.e. it is an "oblivious" attacker model.

The classification of benign prompts into 4 datasets seems a bit arbitrary, particularly as datasets can overlap, and is a loose proxy for the distribution the classifier is aiming for (e.g. good internal representations of benign prompts).

Results in Table 1 are based on the test set split for the trained classifier. For fair comparisons with the general defences OOD jailbreaks should be considered. There is some preliminary experiments to that end in Table 30 in the appendix B.2.24 which should be developed. Likewise  discussion of false positive rates outside the datasets used for training the detector seem to be absent, aside from the appendix, whereas it would be the more representative performance of the defence in practice.

### Questions
I am unsure why those 4 adversarial datasets were highlighted in Table 2. From the description of the data in Section 4, it seems that all datasets are equivalently used for training and testing? They just vary in the number of samples? Perhaps I have missed something here.

### Soundness
3

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
3

### Summary
The paper proposes CurvaLID, an LLM‑agnostic pre‑filter that detects adversarial prompts using two geometric features computed on fixed representations: (1) PromptLID: a sentence‑level Local Intrinsic Dimensionality score computed from a CNN’s penultimate layer; (2) TextCurv: a word‑level “curvature” that combines pairwise angular changes with an inverse‑norm surrogate for arc length. Across many attack families and multiple target LLMs, the paper reports near‑perfect detection with cross‑embedding and multilingual analyses suggesting generality.

### Strengths
1. Clear story to defend the jailbreak attack using LLM‑agnostic pre‑filtering. The jailbreak detector operates entirely outside the LLM under defense, making the deployment super simple.
2. Combining a sentence‑level density proxy (LID) with a word‑sequence geometry (curvature) is intuitive and computationally light. Experiment results are consistently high across models and attacks.
3. Pseudocode, architectural details, and hyperparameters are provided for reproduction.

### Weaknesses
1. Step 1 trains g on a few specific benign datasets (Orca, MMLU, AlpacaEval, TQA), could the separation captured by PromptLID be driven by dataset style rather than adversarialness? A cross-domain benign evaluation might help disentangle this effect.
2. CurvaLID achieves perfect near-zero ASR against non-adaptive attacks, but what happens under an adaptive adversary explicitly minimizing PromptLID and TextCurv? Could such optimization collapse the geometric gap and evade detection?
3. PromptLID is defined inconsistently between Def. 4.1 and Alg. 2 (Sec B.3.1). Could the author fix or clarify this mismatch?

### Questions
1. Table 2 reports 1.00 adversarial accuracy with 0.000 SD. Does this persist across different seeds, folds, and class balances? Please include a variance analysis and per-dataset ROC/AUROC.
2. Many jailbreaks unfold over conversations (multi-turn). Do you compute geometry per-turn or over concatenated history?

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
3

### Summary
The paper proposes CurvaLID, a detection framework for identifying adversarial prompts that could jailbreak LLMs. The method combines two geometric measures: (1) PromptLID, a sentence-level LID estimator based on CNN representations of prompts; and (2) TextCurv, a curvature measure that captures angular changes between consecutive word embeddings. The authors claim these geometric properties distinguish benign from adversarial prompts. They report high performance (over 99% accuracy and near-zero attack success rates) across multiple LLMs and attack datasets, while being model-agnostic and computationally efficient.

### Strengths
- The paper explores geometric properties of text embeddings (curvature, intrinsic dimensionality) as indicators of adversarial behavior—a creative perspective applied to LLM jailbreak detection.
- The proposed method operates independently of any target LLM’s architecture or internal safety mechanisms, potentially offering wide applicability.

### Weaknesses
- The paper does not convincingly explain why sentence-level LID or text curvature should differentiate malicious from benign prompts. The “geometric intuition” remains vague—there is no solid theoretical or empirical link between these geometric properties and adversarial text behavior. 
- The paper does not clarify how the phenomena of high LID regions in adversarial examples in traditional image domains translate to text prompts, which lie in discrete, semantically structured spaces.
- The paper’s mathematical definition of PromptLID deviates from established LID formulations. The proposed equation appears to reduce to a simple function of k-nearest neighbor distances and kernel density estimates, not a genuine estimation of intrinsic dimensionality. 
- No analysis validates whether PromptLID meaningfully captures intrinsic dimensional differences versus acting as a heuristic distance statistic.
- CurvaLID uses a binary classifier trained directly on labelled benign and adversarial prompts. This inevitably biases the model to the seen data distribution. 
- The paper’s extremely high reported accuracy (≈99%) raises concerns of overfitting—especially given the small dataset sizes. 
- The evaluation exclusively targets existing benchmark jailbreak prompts and fails to assess adaptive adversaries—those that can adjust input to evade detection once the defense mechanism is known. Since the method is based on deterministic geometric statistics (e.g., k-NN distances, cosine angles), an adaptive attacker could easily craft prompts maintaining similar curvature and LID while retaining malicious intent. Without such robustness analysis, the claimed “generalization” is unconvincing.

### Questions
- Could you provide a clearer theoretical or empirical justification for why sentence-level LID and text curvature should meaningfully differentiate malicious from benign prompts?
- The proposed formulation of PromptLID appears to deviate from established LID estimation methods. Could you clarify how your definition still captures the intrinsic dimensionality?
- How does performance degrade as the data distribution diverges from the training examples?
- The paper reports near-perfect accuracy (~99%). Are there failure cases or qualitative examples where CurvaLID misclassifies prompts, and what patterns do they exhibit?

### Soundness
2

### Presentation
2

### Contribution
2
