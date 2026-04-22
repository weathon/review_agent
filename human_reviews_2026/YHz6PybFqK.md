# The Hidden Cost of Modeling $\text{P}(X)$: Membership Inference Attacks in Generative Text Classifiers

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Membership Inference Attacks (MIAs) pose a critical privacy threat by enabling adversaries to determine whether a specific sample was included in a model's training dataset. Despite extensive research on MIAs, systematic comparisons between generative and discriminative classifiers remain limited. This work addresses this gap by first providing theoretical motivation for why generative classifiers exhibit heightened susceptibility to MIAs, then validating these insights through comprehensive empirical evaluation.
Our study encompasses discriminative, generative, and pseudo-generative text classifiers across varying training data volumes, evaluated on five benchmark datasets. Employing a diverse array of MIA strategies, we consistently demonstrate that fully generative classifiers which explicitly model the joint likelihood $P(X,Y)$ are most vulnerable to membership leakage. Furthermore, we observe that the canonical inference approach commonly used in generative classifiers significantly amplifies this privacy risk.
These findings reveal a fundamental utility-privacy trade-off inherent in classifier design, underscoring the critical need for caution when deploying generative classifiers in privacy-sensitive applications. Our results motivate future research directions in developing privacy-preserving generative classifiers that can maintain utility while mitigating membership inference vulnerabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper discussed the vulnerabilities of different types of models to membership inference attacks (MIAs). The main finding is that generative classifiers are more vulnerable to MIAs compared to discriminative classifiers. This is because generative models leak information through both the marginal distribution P(X) and the conditional distribution P(Y|X). The authors conducted empirical evaluations across various model architectures and datasets to validate their claims.

### Strengths
- Privacy is an important concern.
- The paper is generally well-written.

### Weaknesses
- The conclusions drawn in the paper seem somewhat expected given the fundamental differences between generative and discriminative models.
- How the findings would generalize to more complex scenarios, such as using multiple shadow models for MIAs, is not clear.

### Questions
In my opinion, the problem studied in this paper hasn't been extensively studied because generative models and discriminative models have different outputs. Discriminative models output class probabilities, while generative models output likelihoods over the entire input space, which provides much richer information. Therefore, it is somewhat expected that generative models would be more vulnerable to membership inference attacks. Let's assume an extreme case where the decision space of the discriminative model is as large as the input space of the generative model. In this case, wouldn't the vulnerability of the discriminative model be comparable to that of the generative model? 

The authors make an assumption that only one shadow model is trained for the membership inference attack. However, in practice, state-of-the-art membership inference attacks often involve training multiple shadow models to better approximate the target model's behavior. How would the results change if multiple shadow models were used instead of just one?

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
5

### Summary
This paper presents the first systematic study of membership inference attack (MIA) vulnerability in generative text classifiers, contrasting them with discriminative and pseudo-generative models. The authors theoretically and empirically demonstrate that explicitly modeling the data distribution P(X) significantly increases privacy risks.

### Strengths
The first comprehensive investigation of MIA risks across generative, discriminative, and pseudo-generative paradigms.
And the paper has a strong theoretical foundation. The two-way decomposition of total variation/KL divergence into marginal and conditional components elegantly explains why modeling P(X) increases leakage.
Provides actionable recommendations (avoid logits exposure, use pseudo-generative architectures) directly relevant to ML-as-a-Service providers and API designers.

### Weaknesses
A key conceptual limitation of this paper is the task mismatch between generative and discriminative classifiers.
Although both are evaluated on text classification benchmarks, they optimize for fundamentally different goals: discriminative models (e.g., BERT) learn decision boundaries for P(Y|X), while generative models (e.g., autoregressive or diffusion) learn to reconstruct or generate text under P(X,Y).
As a result, generative models face a more complex objective involving sequence likelihoods and token-level prediction, making direct privacy comparisons uneven and potentially misleading.
Their higher MIA vulnerability may partly reflect this task complexity rather than an inherent structural flaw.
For instance, autoregressive models require multi-pass inference that naturally exposes more logits, and diffusion models involve reconstruction objectives without a discriminative analogue.
Without aligning objectives or model complexity, the study may overstate the privacy gap between the two paradigms in realistic deployments.

### Questions
The finding that modeling P(X) inherently amplifies membership inference vulnerability is theoretically intriguing.
However, to make this result more practically valuable, could the authors discuss what broader implications or applications this insight might have?
For instance, how might this understanding inform the design of safer generative systems, the privacy evaluation of instruction-tuned LLMs, or trade-offs in multimodal models that integrate both generative and discriminative components?
In other words, beyond showing that generative classifiers are more vulnerable, what new directions or design principles can practitioners derive from this finding for building privacy-aware generative models?
If the authors could provide a more in-depth discussion or conclusion derived from this finding — for example, outlining its broader implications for the design of privacy-aware generative systems or its relevance to real-world applications such as instruction-tuned or multimodal models — I would significantly increase my overall score for this paper.

### Soundness
2

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
2

### Summary
The paper studies membership inference attacks (MIAs) in text classifiers across discriminative (P(Y|X)), fully generative (autoregressive label-prefix and discrete diffusion modeling P(X,Y)), and pseudo-generative (MLM, label-suffix P-AR) paradigms. It gives a black-box theory showing MIA advantage decomposes into leakage from the marginal P(X) and the conditional P(Y|X). Exposing joint/logit scores retains P(X) signal and are not safer than exposing post-softmax probabilities. Synthetic experiments and large transformer experiments on nine datasets show fully generative models leak the most, with the largest risk when K-pass logit vectors are exposed. Pseudo-generative variants and encoders are safer at comparable utility. The paper concludes with practical guidance1. prefer probabilities over logits; 2. avoid DIFF in privacy-sensitive settings; 3. consider MLM / encoder baselines

### Strengths
I enjoy reading this paper. It first presents theoretical framework of decomposition of leakage into marginal vs conditional channels and then provide experiment results to demonstrate the hypothesis. Finally it provides guidence on API exposure. The presentation is clear.

### Weaknesses
1. Still, the theoretical results is based on several assumptions that may or may not hold in practice. For example, it assumes the data is i.i.d. and there is no pre-training. 
2. I think the attack itself are well studied for generative models. It is also well known that richer outputs leak more is well known (labels < probabilities < logits). This paper is the first one that formulate this formally.

### Questions
Besides the guidance on the API exposures, is there any way to estimate the marginal terms and conditional terms from shadow models to predict MIA risk before deployment?

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
3

### Summary
The paper investigates the vulnerability of generative text classifiers to MIAs, showing that they tend to leak more information than discriminative classifiers. It develops a decomposition that categorizes membership leakage into marginal and conditional terms, revealing that logits can dominate probabilities in attack power. The hypothesis is validated through a toy study, which demonstrates higher attack AUROC for joint likelihood signals, particularly in high-dimensional and low-sample regimes. Extensive experiments on nine text-classification datasets and five model paradigms further show that generative models are consistently more vulnerable.

### Strengths
This study addresses an important and interesting aspect of the MIA field, showing that generative classifiers are more vulnerable than discriminative classifiers. It provides a framework and theoretical foundation to guide the practical use of generative classifiers, especially when security and privacy are concerns. The findings suggest that pseudo-generative models may offer a better privacy–utility balance for sensitive deployments.

The theoretical component of the framework is solid, and the experiments and analysis are comprehensive. The paper itself is well structured and clearly presented.

### Weaknesses
Despite the strong theoretical basis, the practical value of this study appears somewhat limited, as public or production models typically do not expose logits or probabilities. This restriction narrows the external applicability of the results. Moreover, attackers in real-world settings could control decoding temperature, perform seed manipulation, or access internal signals; thus, a deeper discussion of white or gray box attacks would add practical relevance.

The comparison setup may also introduce bias. The paper states that “one forward pass per label 
$y_i$ is used to score $log P(x,y_i)$, whereas discriminative models compute $P(Y∣X)$ in a single pass.” This difference in computation could affect both the attack surface and compute budget, potentially disadvantaging generative models. It would be useful to examine whether, under the same computational budget, generative classifiers remain more vulnerable.

In terms of defense mechanisms, could temperature scaling or logit clipping help narrow the logits–probabilities gap, thereby reducing the vulnerability of generative models? An empirical evaluation of such mitigations would strengthen the study.

The paper says that “scalar joint can dominate conditional under systematic marginal skew.” Does this assumption always hold? Could certain types of real-world data invalidate it? A more detailed analysis of when this dominance applies would strengthen the theoretical insights.

The study focuses on text classification. Would the proposed framework be extendable to non-TC settings? It would be valuable to clarify which parts of the theory generalize directly and which require adjustment.

Attack success is measured in AUROC.  How about other measures, e.g. Advantage score, TPR@low-FPR, or PPV? especially when minority classes are involved.

The study measures attack success primarily using AUROC. Additional metrics such as advantage score, TPR@low-FPR, or PPV, could provide a more comprehensive assessment of real-world risk, especially in minority-class scenarios.

Should Table 1 be transposed so that the vulnerability of the models can be compared more clearly across categories?

In Figure 2, AR should appear in the caption to ensure consistency with the figure legend. Perhaps replace GEN -> AR?

Figure 1 is not mentioned in the main text. The statement “Figure 26 reveals …” would be clearer if it explicitly noted that the figure is located in the appendix.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3
