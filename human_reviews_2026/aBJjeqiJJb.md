# Uncovering Neuronal Mechanisms of Intrinsic Self-Debiasing in Large Language Models via Contrastive Learning

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
With the advancement of alignment techniques, large language models (LLMs) have demonstrated the intrinsic self-debiasing capability against stereotypes. However, our understanding of the underlying mechanism remains limited, which significantly hinders the development of trustworthy AI. In the field of LLM safety, prior studies have shown that defense against explicitly harmful queries is governed by a sparse set of critical neurons. These neurons typically exhibit a strong activation response when processing malicious inputs—a phenomenon known as $\textit{explicit induction}$. Nevertheless, this paradigm fails to capture implicit hazards, particularly stereotypical biases, which operate via $\textit{implicit association}$: shifts in neuronal response patterns across different social contexts, not mere activation strength. Based on this insight, we propose $\textbf{\textit{COCO}}$, a $\textit{contrastive learning-based}$ method focusing on identifying self-debiasing neurons possessing $\textit{intra-$\underline{co}$nsistency and inter-$\underline{co}$ntrast}$ (termed $\textbf{\textit{COCO Neurons}}$). Our findings reveal that these COCO neurons account for approximately 1\% of the total neurons and are primarily located in the Query and Value weight matrices of the deeper network layers. To effectively leverage COCO neurons, we draw inspiration from Neurodynamics and abstract the intrinsic self-debiasing capability within LLMs into two distinct systems: linear debiasing system and nonlinear debiasing system, for which we design tailored neuron enhancement editing strategies, $\textbf{\textit{LE-COCO}}$ and $\textbf{\textit{NE-COCO}}$. Experimental results across six social categories demonstrate that the success rate of Llama3-8B in resisting stereotypical biases increases to nearly 90\% after linear enhancement, with a maximum gain of over 50\%. Meanwhile, Mistral-7B with nonlinear enhancement achieves an average gain of 10\% in its success rate of resisting stereotypical biases, with a maximum gain of 23\%. Furthermore, generalization experiments reveal that the enhanced models exhibit not only stronger robustness against jailbreak attacks but also measurable improvements on factual and reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes COCO neurons: attention-layer "neurons" whose activation differs between "biased" and "unbiased" contexts. The authors score each neuron with an InfoNCE-style objective computed over activation intensity differences, select a tiny subset and then deactivate them to show large drops in "unbiased response" on BBQ, and enhance them with two strategies (linear and nonlinear) to improve "unbiased response" and robustness to jailbreak prompts. They also report gains on TruthfulQA/GPQA and a sizable drop on MMLU, and present an attention-shift analysis (more weight to the first token; less to the last) as interpretability evidence.

### Strengths
1. The simple scoring for neuron selection that does not require labeled rationales; uses contrast between biased/unbiased prompt sets. 

2. The authors implemented clear interventions with strong reported effects on BBQ across multiple social categories and two instruction-tuned models (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3).

3. Multiple jailbreak stress tests (SystemRT, UserLEE, RandomTPJ) show improved "unbiased response" stability after editing.

### Weaknesses
Two issues worth attention:

1. The central claim is that COCO neurons "mediate debiasing." But the experimental signature (large boost in "unbiased responses" when enhanced; large drop when deactivated) is also what we would expect if these units simply encode gender/race/age semantics needed to recognize when a question references a demographic dimension. Deactivating such neurons would both (a) reduce the model's ability to express/recognize group concepts (appearing to "debias" on ambiguous BBQ items) and (b) harm other tasks that require those semantics --- precisely what your Finding 3 shows with the big MMLU decline after enhancement.

2. There might be data leakage issue in your testing. In §4.2 you perform cross-category validation and then pick the source category whose neuron set most damages target-category unbiased responses, i.e., argmin over categories on the target test. That risks implicitly using test outcomes for selection, inflating deactivation effects. You may consider show a version where selection uses a held-out split and evaluation uses a strictly unseen set.

Some minor points:

3. Terminology: calling the method "intrinsic self-debiasing" is misleading. This is an external edit to induce bias-mitigating behavior, not the model self-regulating. Consider rephrasing.

4. Clarify the unit of intervention: when "deactivating a neuron," do you zero a column in WQ/WK/WV for all tokens in a layer? Are residual/LayerNorm side-effects controlled?

### Questions
1. The attention-shift story (first-token gets more, last-token gets less) is intriguing but speculative. Test predictions:

    1.1 If you shuffle the first token or insert a neutral anchor at position 1, does the effect persist?

    1.2 Are shifts localized to a few heads with causal impact (logit-lens/patching), or global?

    1.3 Do earlier layers show consistent mediation or is this only final-layer superficial redistribution?

### Soundness
2

### Presentation
3

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
The authors provide a new debiasing technique and analyse it's effect on the models performance on both bias benchmarks as well as general capability benchmarks. They claim that their method significantly reduces bias, while even increasing logical reasoning and factuality. Finally they provide a short analysis on what their intervention does to the attention weights of the last layer.

### Strengths
- S1 - The authors attack an important problem and try to provide an extensive analysis.

### Weaknesses
Because of multiple ambiguities, uncertainties, and unexplained artifacts in the results (e.g., one method performs miraculously on one model but not on another, and the other way around), I remain cautious about the strength of the claims in this paper. While I might have not understood every details, this seems largely attributable to an unclear methods section. I made a sustained effort to understand the methodology, which is why I selected a confidence score of 3.

**Major:**

- W1 – Your introduction does not specify what kind of bias is under discussion. This is confusing. The first time you touch on what exactly you investigate is the “Bias in LLMs” paragraph on page 3. In Section 3.2 you start to talk about social categories but have never introduced what these are.
- W2 – There is a substantial literature on debiasing language models that is not discussed, e.g.:
    - https://arxiv.org/pdf/2306.03819
   - https://arxiv.org/pdf/2201.12091
   - https://proceedings.mlr.press/v139/liang21a.html
   - https://arxiv.org/pdf/2311.09090
   - https://arxiv.org/pdf/1904.03310
   - https://arxiv.org/pdf/1607.06520
    - https://arxiv.org/pdf/2009.09435
   - and many more
- W3 – The methods section is unclear and appears to contain inaccuracies, which makes it difficult to understand what the paper does:
    - Eq. 1): Most language models (including Llama and Pythia) do not have the MLP and MHA in parallel. Usually, they are applied sequentially via residual connections. For example: H_i^{l’} = H_i^{l-1} + A_l(H_i^{l-1}) and H_i^{l} = H_i^{l’} + M_l(H_i^{l’}).
    - Eq. 3): Your definition of a neuron describes an MLP neuron, yet you mention that a neuron can also be in A^l. The definition in Eq. 3) does not match attention (point-wise nonlinearity and bias do not exist in MHA). In Section 3.1 you do not specify whether the neuron is in the MLP or the MHA, so I initially assumed you only look at the MLP. Figure 1 clarifies the neurons are only in the MHA. What does Eq. 3) describe, then? If it is the MLP, IIRC, Llama 3 uses a gated MLP, which also does not fit Eq. 3). I am generally unsure what exactly you measured. If you measured the neurons in the MHA, which neurons exactly (after each of Q/K/V? the output of each head before applying the out matrix? after the out matrix?)?
    - What do you mean by deactivating a neuron? Setting it to zero?
    - Eq. 5): I am confused about this loss. The loss function takes in two sets, but then it is defined for single elements from this set (actually from a subset). How do you compute it for the full set? In particular, how do you accumulate the values for different x_+ and y_-? Do you take a sum, or do you sample a single x_+ and y_-?
    - Eq. 7): What does “& c” mean? Isn’t c a category and hence defines the sets X and Y? I would make coco_{N_i^l} a function of X, Y. Then you can say coco(X_c, Y_c) ≤ … . Also, you should define epsilon.
    - Eq. 8): You overload w_k, which you already used in the definition of the neuron. Also, I do not understand what “w_k denotes the contribution weight U.” means, since “U denotes the LLM’s unbiased response capability.”
- W4 – It is unclear what an “unbiased response capability” is or represents.
- W5 – [Also about methodology but deserves its own point] I do not understand how you edit the neurons in 3.3 or how you define the debiasing system.
    - You define some linear or nonlinear functions of activation intensities. What is the input to this function? Strings?
    - Is a_k a function of a string or a precomputed value (I assume the former)?
    - How do you learn the weights w_k and W_inter?
    - What does λ do?
    - I could not follow how the nonlinear system works. What does this sentence mean: “By relaxing the contrastive learning constraint of intra-scene stability, we prioritize macroscopic response divergence across biased-unbiased scenario associations, in-directly capturing the nonlinear system’s most active and sensitive nodes in the nonlinear system.”
- W6 – What are your baselines? There is no description of RAND and NORM. Further, MACT is not defined (I assume it is the neurons for which I is highest on some dataset?).
- W7 – Chapter 4.1: What are the categories? This is crucial information that is missing. As mentioned before, I do not understand what the numbers in the table represent.
- W8 – Chapter 4.4: Why do you only evaluate one specific category for each model? Why are they different? Are these cherry-picked results, or do the other categories look similar?
- W9 – You are interpreting neurons. There is substantial research showing that information is not basis-aligned (https://transformer-circuits.pub/2022/toy_model/index.html inter alia). Hence analyzing individual neurons may not be meaningful. Since you ablate multiple neurons at the same time, you can potentially model neuron interactions (superposition). You should relate to that literature and discuss it.

**Minor:**

- w10 – I recommend citing the original logit lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens. Also, logit lens applies the unembedding layer to an intermediate representation. The description “parses bias representations by visualizing logit distributions in intermediate layers” is a bit off–logit lens provides a distribution over tokens given an internal representation (which one can visualize), and it is not specific to bias representations.
- w11 – L 059: “Second, most debiasing solutions only focus on "how to work"”: what does “how to work” mean in this context?
- w12 – L 058: “First, mechanism analysis is mostly confined to the level of word embeddings or intermediate-layer representations.” – please clarify or justify this statement.
- w13 – L 072: “Through selective deactivation of these neurons, we found that the proportion of biased responses of Llama3 and Mistral both exceeded 91% across all social categories on bias benchmark.” Do you mean the model becomes more biased after deactivation? This is confusing.
- w14 – L 105: “Therefore, A^l is a critical component for producing unbiased responses” – what is the empirical basis for this claim? “A^l is responsible for global information integration” does not itself motivate that conclusion.
- w15 – L 187: “which employs the contrastive learning paradigm” is confusing since there is no learning described. Maybe “we use methods from contrastive learning” is better.
- w16 – It would be helpful to add a few sentences about how BBQ works.
- w17 – L 344 & L 345: Rendering error; \approx does not render.
- w18 – You never refer to Figure 2 in the text.
- w19 – Figure 2: What does “E-” mean?
- w20 – You initially talk about “categories” but then in Figure 3 about “scenarios.” Are these the same?
- w21 – Figure 3 is barely readable. Please increase the font size (also for Figure 2, but Figure 3 is much worse).

**Expectation Management:** To achieve a score of 4, all major uncertainties must be resolved. This would necessitate a substantial rewrite, in my opinion. For any higher grade, I would require all points (both major and minor, as well as all questions) to be thoroughly addressed. I don’t think it is likely that I will give a score of six or higher.

### Questions
- Q1 – You talk about “debiasing behavior,” but it is unclear what this means for a language model. In my understanding there are tools or methods that can debias a model, but what is a debiasing behavior of the model itself? How is a debiasing behaviour different from a biased behaviour? Clarifying this in the introduction would help.
- Q2 – L080ff: You first write that enhanced LLMs are better at factuality but then that enhancement degrades knowledge. How do these points connect?
- Q3 – Assuming you set the neurons to zero: have you investigated setting them to a dataset mean? Zero ablation may not be optimal if the “default deactivated state” is not zero.
- Q4 – How does COCO differ from I averaged over all pairs of X and Y and take the max difference? Can you give some intuition for what COCO adds?
- Q5 – “U denotes the LLM’s unbiased response capability.” What is an unbiased response capability? As I read it, it should be in R^1, but I am unsure. What does it represent? 
- Q6 – “MACT neurons (identified via activation intensity, with strong cross-scenario activation universality).” What does “strong cross-scenario activation universality” mean?
- Q7 – Chapter 4.2: Why is LE effective for one model and NE for the other? You later provide hypotheses, but earlier you state this “confirms” the effectiveness of your approach (“The differentiated optimal strategies for the two models validate the effectiveness of our debiasing methodology”). I do not follow this logic.
- Q8 – L430: What do you mean by “attention shift sparsity”?
- Q9 – Performing 4.5 only on the last layer may be less informative, given evidence that predictions have largely converged by then. Did you analyze other layers? In which layers did you identify your neurons?
- Q10 – (More of a comment) Hypothesis in Chapter 4.5: The first token cannot hold global information. Both Llama and Mistral have a BOS token. Due to the models’ causal structure, they cannot put global information into this token; the residual stream on the first token is constant between prompts. What you likely observe is attention to attention sinks. Since attention weights must sum to 1, reducing attention to specific tokens forces attention mass elsewhere (often attention sinks); there is a literature on this.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the neuronal mechanisms underlying debiasing behavior in Large Language Models through a contrastive learning approach. The authors propose the COCO (COnsistent-COntrastive) framework to identify critical neurons that exhibit activation patterns between biased and unbiased scenarios. These neurons, representing less than 1% of total parameters, are shown to significantly influence model bias when deactivated. The paper further introduces two neuron editing strategies—linear and nonlinear enhancement—tailored to Llama3-8B and Mistral-7B model architectures. Experiments on Llama3-8B and Mistral-7B demonstrate substantial improvements in unbiased response rates, with the linear enhancement achieving nearly 90% unbiased responses on BBQ benchmark. The work also provides some interpretability analysis through attention mechanism visualization.

### Strengths
1. The paper is genrally well-structured with clear problem formulation and methodology sections and addresses an improtant problem in AI safety and fairness. The four research questions in the experiment section provide a clear roadmap for the evaluation.
2. The paper employs a contrastive learning–based InfoNCE loss to evaluate the importance of neurons. Compared with threshold-based methods, this approach allows for more effective comparisons across samples within the entire dataset.
3. The paper presents the visualization of the last-layer attention after neuron editing which provide more interpretable explanations.

### Weaknesses
1. The paper has a limitation on the baseline comparisons, the paper only compares against three neuron selection heuristic baselines(RAND, NORM, MACT).
2. There is a concern on the Linear vs. Nonlinear system classification. The criteria for classifying Llama3 as linear and Mistral as nonlinear are not clearly established. The fact that LLaMA-3 performs comparably or even worse than the original model in the nonlinear setting warrants further analysis. Moreover, the paper lacks a clear discussion of how model family or architecture influences linear vs. nonlinear performance. Additional experiments on other model families would strengthen the work.
3. The paper contains minimal typographical or writing errors, such as in Section 3.3 page 5,  $\mathbf{w_k}$ should be \textit{w_k}, in Section 4.3 Table 2's Caption,  "max increase ¿ 23%" should be "> 23%"

### Questions
1. How does your COCO framework compare with recent state-of-the-art methods in large language models (LLMs), for example, debiasing [1] or representation-learning[2] approaches?
2. What specific architectural properties determine whether a model exhibits linear or nonlinear debiasing behavior? And can you validate the applicability across different model families? Such as testing on within-family consistency(LLaMA-2-7B, LLaMA-3-8B, LLaMA-3.1-8B, etc.) or cross-family generalizability(Qwen, Gemma, etc.) Is it possible that for certain architectures, both strategies are effective (positive synergy) or both strategies fail (neither improves debiasing)?


[1]Gallegos, I. O., Rossi, R. A., Barrow, J., Tanjim, M. M., Yu, T., Deilamsalehy, H., Zhang, R., Kim, S., & Dernoncourt, F. (2024). Self-Debiasing Large Language Models: Zero-Shot Recognition and Reduction of Stereotypes.

[2]Zhou, H., Feng, Z., Zhu, Z., Qian, J., & Mao, K. (2024). UniBias: Unveiling and mitigating LLM bias through internal attention and FFN manipulation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper aims to alleviate biases in LLMs via contrastive learning. Specifically, they aim to identify bias-centric neurons and deactivate these neurons to reduce bias in LLMs via a novel methodology called COCO. They benchmark the debiasing capabilities of their method vis-a-vis other methods and benchmark the general coherence of the models.

### Strengths
1) This paper studies an important topic in modern LLMs: debiasing. This is very relevant to widespread safe usage of LLMs. 
2) The neuron-based approach makes sense given the current state of the literature, indicating bias/safety-related neurons. 
3) The overall benchmarking is good. I think the paper is experimentally sound and provides value. 
4) The inclusion of the jailbreak-related experiments provides another dimension of value to the work.

### Weaknesses
1) Overall, the work provides decent value in terms of the findings, but in an unpublishable state due to the writing and the presentation. Specifically, the preliminary section discusses the formulation of attention and neurons, while not discussing the methodologies used to debias LLMs. This sets a bad tone for the rest of the paper, as a stable grounding in literature is not provided in the preliminary section. 
2) The experimental setup section provides no references to the baselines. If no references are needed, please further clarify the baselines. 
3) Paragraphs 462-465 and 466-470 do not provide value to the reader in grounding the work in the current literature. 
4) The methodology introduced in eq 7, although it shows decent debiasing capabilities, is just a methodology to find a set of bias-centric neurons. Various other methodologies exist to discover such neurons, and such methodologies are not studied [1, 2] as a point of comparison. As this work aims to ground itself, at least partly, in the mechanistic methodologies, a fair comparison to steering vectors would be interesting. This weakness stands to highlight both the novelty issue and the lack of realistic baseline methodologies
5) Would like to get a better understanding of the effect of such debiasing on nuanced tasks such as multi-turn question answering, long context reasoning, etc.
6) The jailbreak robustness section needs some modern exploits to test the model such as GCG[3] etc to aid the strength of the claim. 




[1] Wei, Boyi, et al. "Assessing the brittleness of safety alignment via pruning and low-rank modifications." arXiv preprint arXiv:2402.05162 (2024).
[2] Siddique, Zara, et al. "Shifting perspectives: Steering vector ensembles for robust bias mitigation in llms." arXiv preprint arXiv:2503.05371 (2025).
[3]Zou, Andy, et al. "Universal and transferable adversarial attacks on aligned language models." arXiv preprint arXiv:2307.15043 (2023).

### Questions
1) Possible downstream negative effects of such debiasing in not been studied in depth. How does the method perform in a chain of thought, long context settings? 
2) In the experiment setup section, the number of shots for the benchmarks is not mentioned. Can you help me better understand this setting?

### Soundness
3

### Presentation
1

### Contribution
2
