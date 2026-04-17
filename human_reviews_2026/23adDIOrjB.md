# The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
We study the geometry of token representations at the prompt level in large language models through the lens of intrinsic dimension. Viewing transformers as mean-field particle systems, we estimate the intrinsic dimension of the empirical measure at each layer and demonstrate that it correlates with next-token uncertainty. Across models and intrinsic dimension estimators, we find that intrinsic dimension peaks in early to middle layers and increases under semantic disruption (by shuffling tokens), and that it is strongly correlated with average surprisal, with a simple analysis linking logits geometry to entropy via softmax. As a case study in practical interpretability and safety, we train a linear probe on the per-layer intrinsic dimension profile to distinguish malicious from benign prompts before generation. This probe achieves 90–95\% accuracy across different datasets, outperforming widely used guardrails such as Llama Guard and Gemma Shield. We further compare against linear probes built from layerwise entropy derived via the Tuned Lens and find that the intrinsic dimension-based probe is competitive and complementary, offering a compact, interpretable signal distributed across layers. Our findings suggest that prompt-level geometry provides actionable signals for monitoring and controlling LLM behavior, and offers a bridge between mechanistic insights and practical safety tools.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript studies the intrinsic dimension of prompts, modeled as clouds of tokens, in the embedding spaces of large language models. They present a handful of empirical findings; the intrinsic dimension of prompts appears to be correlated to surprisal and increases when prompts are shuffled, for example. They show that layerwise intrinsic dimension features can be used to identify malicious inputs.

### Strengths
The paper is fairly easy to understand. Discussion of related work is well-researched and thorough. The correlations between ID and surprisal seem to be pretty robust, as is the "peak" observed in shuffling experiments, and I appreciate that the authors used diverse prompts from the Pile as opposed to ones from some more narrow domain.

### Weaknesses
There are a few things missing from the paper that I think would strengthen it greatly:

1. Efficiency analysis: I'm not familiar with the GRIDE algorithm. How computationally intensive is it compared to the other baselines?
2. The main applied results are all from one model. Given the differences observed between models elsewhere (e.g. in Appendix F), I think there's a good chance the others would perform differently here as well.
3. The comparison to Shield Gemma and Llama Guard aren't exactly fair, since those are zero-shot methods. I'd like to see more baselines tuned on this particular distribution (e.g. finetuned BERT)
4. The fact that this method is only evaluated on prompts of a specific length is a significant omission; the authors need to include results for prompts of different lengths and results averaged across all lengths.
5. You can't claim that ID has value as a "pre-output signal" without running pre-output experiments. How do the results change when you only run the first k layers of the model under evaluation?

### Questions
Why is Tuned Lens so much worse on jailbreak prompts compared to "malicious" and "attack" prompts? This is not mentioned in the text, but only in Table 1. Is the result a typo?

How exactly is the Tuned Lens baseline computed? Are you using the full vector of entropies at each layer as inputs to the classifier?

Do you observe the "peak" from Figure 1 on malicious prompts, or just shuffled ones?

### Soundness
2

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
This paper studies the geometric properties of internal representations in  LLMs by analyzing the intrinsic dimension (ID) of the entire set of tokens within a prompt. The authors treat the evolving token representations as a mean-field particle system and measure the ID of this "token cloud" at each layer. They have three main findings: the prompt-level ID is not static, but it typically peaks in the early to middle layers of the model. Disrupting the prompt's structure by shuffling tokens causes the ID to increase, suggesting that coherent language lies on a lower-dimensional manifold. And, prompt level ID is strongly correlated with the model's uncertainty, as measured by average surprisal (next-token cross-entropy). The authors provide a theoretical intuition for this link via the geometry of the logits and the properties of the softmax function. As a practical demonstration, the authors train a simple linear probe on the vector of layer-wise ID values. This probe achieves 90-95% accuracy in distinguishing malicious from benign prompts before any text is generated, significantly outperforming existing guardrail models like Llama Guard and Gemma Shield on the tested datasets.

### Strengths
- I appreciate the paper's focus on the prompt-level geometry (using all tokens) rather than more common dataset-level or last-token analyses.
- The proposed feature vector (the ID at each layer) is very "low-dimensional" compared to other potential representations (e.g., all token embeddings).
- The authors don't just compare their ID probe against external safety models (Llama Guard, Gemma Shield). They also create a strong internal baseline using Tuned Lens entropy, another method for extracting layer-wise information. The fact that their geometric probe is competitive and complementary to an uncertainty-based (entropy) probe reinforces their claim that geometry is a distinct and valuable signal.

### Weaknesses
- The main weakness is the method's reliance on long prompts. The authors say that reliable ID estimation requires sufficiently long prompts (e.g., $N \gtrsim 500$ tokens), and their own safety experiment is filtered for prompts in the [500, 1000] token range. This limits the practical utility of the safety probe, as many (if not most) real-world malicious prompts and jailbreaks are short. The paper does not provide data on how performance degrades for shorter, more common prompts. Plus, the safety datasets were heavily filtered to a specific token range ([500, 1000]). This is highly unrepresentative of a real-world use case, which would include a vast range of prompt lengths. This filtering introduces a potential confounding variable: the probe might not just be learning "malicious vs benign" but also "properties of 500-1000 token prompts", so this makes the high accuracy less generalizable.
- The safety task is a binary classification of "malicious" vs "benign” and this is a simplification: malicious inputs can range from complex jailbreaks and prompt injections to simple toxicity or bias-inducing phrases. It is unclear if the probe is detecting a general property of "un-natural" or "adversarial" text, or if it is overfitted to the specific styles of jailbreak present in their three test datasets.
- The paper claims the ID profile is an interpretable signal, but this is only partially true. While we know the vector is separable, the paper does not show how the ID profile of a malicious prompt is different from a benign one. For instance, does the ID peak shift, increase, or decrease? The analysis showing that shuffling increases ID is clear, but the corresponding analysis for maliciousness is missing, which would be key to mechanistic insights.
- The main safety experiment (table 1), which shows the 90-95% accuracy, is only performed on the Llama 3 8B model.

### Questions
- Have you considered using a more complex, non-linear classifier (like a small MLP or an SVM) to see if it could capture more nuanced patterns in the ID profile?
- Have you investigated the performance of the ID-based safety probe as a function of prompt length? Specifically, how rapidly does the 90-95% accuracy degrade on shorter prompts (like $N < 100$ tokens), and have you considered modifications (like using different geometric estimators suited for sparse data or combining the ID signal with other internal metrics) to maintain high performance on these more challenging and common short-prompt cases?

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
2

### Summary
The paper proposes a prompt-level geometry-based method to interpret the layer-wise token representation transformations through an LLM. These interpretations use the intrinsic dimension of the data manifold which is theoretically and empirically shown to be an effective representation for interpreting the workings of LLMs. The effectivess of the representation is shown with probes trained on them to classify prompts as safe/not.

### Strengths
1. I appreciate the perspective and utility of geometry-based methods advocated in this paper which can be used for effective prompt classification.
2. I like the constant comparison with prior related works, providing justification for the method.
3. The work is backed by theoretical soundness justifications.

### Weaknesses
1. My major criticism is for the presentation of the work. I think with reordering information and adding more clarifying text, this paper can be made more accessible. Specifically, I have the following recommendations. My score leans towards rejection mainly as I think this paper needs a major writing revision.
    1. Before telling the findings, the intro should describe the methods more for readers new to the mean-field theory of tokens. Example - "the intrinsic dimension exhibits a characteristic peak in early-to-middle layers." - it is not clear what the peak means here and if that's a good thing.
    2. "Empirical measure", "intrinsic dimension" should be defined early on in the intro before using these terms, but they are defined in related work.
    3. Sufficient explanation of the primary methods like GRIDE should be added in the main paper.
2. I think that just one case study for prompt safety classification, though very interesting and insightful, may not be sufficient to emphasize the utility of geometry-based methods. I suggest the following:
    1. Add another case study on a distinct interpretability task to show generality. 
    2. Include a comparison with a simple probe over hidden intermediate representations entirely, to strengthen the case study for prompt safety classification.
    3. Include diverse kinds of attack prompts, e.g., prefilling attack [1,2]

## References:
1. Bypassing the Safety Training of Open-Source LLMs with Priming Attacks
2. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

### Questions
1. L157: " averaging local estimates across tokens", what's the justification for averaging as opposed to another method of aggregation?
2. L175: Why is the entropy of latent predictions for GPT-2 obtained?
8. L203: On what basis is prompt 3218 selected and how are the intermediate levels of perturbation created, eg, level 1?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a geometric lens, a prompt-level intrinsic dimension, to probe internal LLM dynamics, offering both mechanistic insights and a practical safety tool.  Treating transformers as mean-field particle systems, the authors estimate ID via k-nearest-neighbor estimators (GRIDE, ESS, TLE) for every layer and every prompt. 
The authors demonstrate that this prompt-level ID, estimated layer by layer, exhibits a strong positive correlation with the model's next-token uncertainty (average surprisal).  Key findings indicate that the ID peak occurs in early-to-middle layers and increases when the prompt's semantic structure is disrupted by shuffling.  As a practical case study, the authors use the per-layer ID profile as a feature vector to train a linear probe for detecting malicious prompts before generation.
This method achieves 90-95% accuracy, reportedly outperforming standard guardrail models like Llama Guard and Shield Gemma on the tested datasets.

### Strengths
1. This paper introduces a novel methodology, using a geometric probe for internal representations.
- This paper introduces a prompt-level intrinsic dimension rather than a last-token or dataset-level analysis. 
- This paper clarifies the distinction between prompt-level and dataset-level ID, resolving apparent contradictions in prior work.
- This paper provides a bridge between geometric structure and information-theoretic uncertainty.

2. This paper introduces applications of practical interpretability and safety applications.
- This paper demonstrates that a simple linear probe on ID profiles achieves 90–95% accuracy in malicious prompt detection. It outperforms established guardrails (Llama Guard, ShieldGemma) on the same data splits 
- The experiment shows complementarity with TunedLens entropy features, suggesting geometric and uncertainty signals are aligned.

3. This paper has rigorous experimental validation across models and estimators. It uses four major LLMs (LLaMA, Mistral, Pythia, OPT) and three distinct ID estimators (GRIDE, ESS, TLE).

### Weaknesses
1. There are some contradictory findings in latent entropy analysis. The paper's central thesis is that geometry (ID) tracks uncertainty (entropy). However, the analysis in Appendix F (correlating ID with latent entropy from TunedLens) contradicts this for some of the main models studied. While GPT-2 and Pythia models show the expected positive correlation, the Llama 3 and OPT models show a negative correlation in the middle-to-late layers.

2. The experiments are based on kNN assumptions and hyperparameters.  The kNN-based ID estimators assume local uniform density, which may not hold in high-curvature manifolds. The ID values vary with range scaling and neighborhood size, implying estimator sensitivity.

3. The experiments have ambiguity in the shuffling methodology and its semantic impact.  There is no quantitative measure of semantic disruption (e.g., BLEU, BERTScore) provided to calibrate shuffle severity.   The claim that shuffling “disrupts syntax and semantics” is asserted but not measured; only unigram frequency is preserved.

4. The experiments have a limitation on the prompt length and domain constraints.  The prompts used in this paper, the length $N$ is not less than 1024 in Section 3.1. There is limited usability for short prompts, conversational inputs, or non-text modalities.  This reduces the method’s coverage in realistic LLM interactions.

### Questions
-  How sensitive are ID estimates to prompt length, especially near the 500-token lower bound used in safety experiments (Section 6)?
-  How does block-wise shuffling compare to full random permutation in terms of ID increase?  
-  Why use shuffling? Since shuffling would change the semantic information of the prompt, making it out of distribution.
- The empirical correlation between Logit ID and Contextual Entropy is $\rho=0.60$ (Fig. 3). What other geometric or non-geometric factors are hypothesized to account for the remaining variance?
- What is the computational cost of per-layer ID extraction for large-scale deployment?

### Soundness
2

### Presentation
1

### Contribution
1
