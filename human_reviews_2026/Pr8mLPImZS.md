# Towards the Worst-case Robustness of Large Language Models

- Avg Score: 3.20
- Decision: Reject
- Scores: 2, 4, 2, 2, 6

## Abstract
Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific inputs to induce wrong or even harmful outputs. Although various empirical defenses have been proposed, their worst-case robustness remains unexplored, raising concerns about the vulnerability to future stronger adversaries. In this paper, we systematically study the worst-case robustness of LLMs from both empirical and theoretical perspectives. First, we upper bound the worst-case robustness of deterministic defenses using enhanced white-box attacks, showing that most of them achieve nearly 0\% robustness against white-box adversaries. Then, we derive a general tight lower bound for randomized smoothing using fractional or 0-1 knapsack solvers, and apply them to derive theoretical lower bounds of the worst-case robustness for previous stochastic defenses. For example, we certify the robustness of GPT-4o with uniform kernel smoothing against \textit{any possible attack}, with an average \(\ell_0\) perturbation of 2.02 or an average suffix length of 6.41 on the AdvBench dataset.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies adversarial robustness of LLMs under worst-case (white-box) conditions. It claims two main contributions: 1) Upper bounding worst-case robustness of existing defenses (both deterministic and stochastic) by introducing a stronger white-box adversarial method, I2-GCG, which ensures tokenization consistency between optimization and inference. Experiments show that many deterministic defenses exhibit nearly 0% robustness under this new attack. 2) Lower bounding worst-case robustness via a theoretical framework that formulates randomized smoothing robustness as a knapsack optimization problem (fractional or 0–1 knapsack), providing analytical lower bounds for stochastic defenses such as uniform or absorbing kernels. The authors then apply these formulations to LLM safety detectors and report certified ℓ₀ radii and suffix robustness results on the AdvBench dataset. The paper argues that by bounding robustness from both above (empirical attacks) and below (theoretical certificates), one can achieve a more complete understanding of the “worst-case” robustness of LLMs.

### Strengths
1. The idea of formulating randomized smoothing bounds as a knapsack problem is mathematically creative and a novel interpretation of existing theory.

2. The connection between vocabulary size and certified bounds provides some analytical insight.

3. The appendices appear to contain detailed proofs and extra context, though the main text occasionally glosses over technical assumptions.

### Weaknesses
1.  The experiments in the main text are extremely limited and not aligned with the large scope implied by the title (“Towards the Worst-Case Robustness of LLMs”). The “upper bound” results rely almost entirely on Table 1, which evaluates a few deterministic defenses under I2-GCG. There are no ablations, no qualitative jailbreak examples, and no exploration of why tokenization consistency matters beyond a one-line intuition. The “lower bound” section (Tables 2–3) only reports a few average numbers on AdvBench without comparison to competing theoretical certification methods (e.g., Moon et al., 2023; Kumar et al., 2023). Given the paper’s strong theoretical claims, the lack of empirical diversity (no varied datasets, no different models beyond 7B/8B class, no real-world attacks) makes the work feel incomplete and largely conceptual.

2. The title suggests a comprehensive investigation into LLM worst-case robustness “towards” a general solution, but the content mainly covers: a minor technical tweak to an existing attack (I-GCG → I2-GCG), and a repackaged theoretical treatment of randomized smoothing. There is no holistic analysis of “worst-case robustness” across modalities, alignment procedures, or instruction-following behaviors. Thus, the scope and ambition are mismatched with what is actually demonstrated.

3. The proposed I2-GCG attack is not fundamentally new: the improvement (ensuring tokenization consistency) is minor and already implicitly addressed in other implementations (e.g., Li et al., 2024a; Basani & Zhang, 2024).  Actually, the attack evaluations simply replicate known results (deterministic defenses collapsing to 0% robustness under white-box settings), providing little new empirical insight.

4. The theoretical “knapsack” formulation is elegant but only rephrases existing randomized smoothing theory from my perspective. The equivalence to Cohen et al. (2019) and Teng et al. (2020) is explicitly acknowledged in the text, meaning no new tighter bounds or practical improvements are offered. There is no experimental validation that the knapsack-derived lower bounds match observed robustness under stochastic defenses. The claim of “tightness” remains purely theoretical, lacking corresponding empirical results.

5. The certified ℓ₀ radius and suffix length results (e.g., 2.02 or 6.41) are not contextualized, what do these numbers mean in terms of actual model safety? Are they significant improvements over baselines? The authors evaluate “GPT-4o” without clarifying whether this refers to a closed-source API or a proxy, raising reproducibility concerns.

6. Severe experiment issues need to be seen: No discussion on computational cost, training overhead, or scalability. No qualitative or visualization-based analyses to illustrate adversarial or certified behavior. Lack of comparison with recent works like Tree of Attacks (Mehrotra et al., 2023), AmpleGCG (Liao & Sun, 2024), or Diffusion-based certified defenses (Chen et al., 2024a,b). This omission further weakens the contribution’s originality.

### Questions
1. Can the authors expand the experiments beyond one dataset (AdvBench) and provide more comprehensive comparisons with recent attacks and defenses?

2. How does I2-GCG perform on real-world jailbreak prompts (e.g., JailbreakBench 2024)?

3. The current theory assumes bounded or binary functions. Can it be extended to non-binary outputs, continuous embeddings, or multimodal inputs?

4. How would the results change under more realistic constraints, e.g., subword tokenization noise or cross-lingual settings?

5. Are these derived bounds useful for deployment or policy decisions?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates “worst-case” attacks for both deterministic and stochastic defenses to assess their robustness. In addition, the authors provide a theoretical analysis to characterize a lower bound on the robustness of stochastic defenses.

### Strengths
S1. The paper evaluates attacks across several different LLMs, which increases empirical validity.

S2. The theoretical lower bound result is interesting and (to me) is the clearest and strongest contribution in this paper.

S3. The paper is generally well-written and easy to read.

### Weaknesses
W1. The technical contribution is somewhat limited. More importantly, different defenses are designed for different threat models. If defense A is specifically designed for a black-box setting, then evaluating it under white-box conditions is not very meaningful, because the defense was never intended to handle that capability. Instead, what is needed is “worst-case within the constraint the defense was designed for.” In other words, “worst-case” needs to be scoped per defense mechanism consistent with the defended threat model.

W2. The definition of “worst-case” in this paper is not clearly specified. For example, what is assumed attack capability, what prior knowledge is assumed, and what attack objective is assumed when declaring “worst”?

W3. Related to W1, the authors are encouraged to select a more diverse and more state-of-the-art set of defense mechanisms baselines in their experiments, spanning multiple defense setting categories, and then evaluate worst-case adversarial robustness conditioned within each defense’s intended setting/assumption class. This would make the empirical conclusions much more compelling and meaningful.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to design a certified robustness (CR) defense for LLMs against jailbreak attacks by certifying the robustness of harmful prompt detectors (which can be seen as certifying the robustness of text classifiers). To this end, the authors first propose a new **sample-dependent** "CR-definition" in the scenario of generative modeling, and then design an algorithm to realize so-called "certified robustness" by smoothing the protected model via solving a restricted optimization problem. Some experiments are conducted to verify the effectiveness of the proposed defense.

### Strengths
I appreciate the effort of the authors in trying to establish certified robustness guarantees for generative LLMs (although I personally think the proposed "CR-guarantee" is incorrect).

### Weaknesses
1. I think the authors cannot call their proposed method a "certified robust method". Specifically, when people say a ML model is "certified robust", it means that the model would not change its prediction for **ANY** possible input samples and their corresponding perturbed versions under a reasonable perturbation budget (see examples for vision models [r1, r2] and language models [r2]). However, according to Lines 138-139 of the paper, the "certified robustness guarantee" proposed in this paper is actually sample-dependent (since the term $p_A$ depends on $x$). Furthermore, in Theorem 5.2, the authors also admit that if the sample-dependent term $p_A := g(x)$ cannot satisfy certain conditions, no "CR-guarantee" can be established for this sample at all. This means the proposed so-called "CR-guarantee" actually cannot ensure the robustness for **ALL** possible samples, and this is why I believe that the proposed method is not a "CR-method" but simply an empirical adversarial robustness enhancing method. Additionally, I think it is inappropriate to hide the data term $x$ from the sample-dependent $p_A := g(x)$, which may mislead readers.

2. In Theorem 5.2, it requires the condition $p_A(x) \geq 1-\beta^d$ to be satisfied to enable one to establish a "CR-guarantee". I think such a condition is too restrictive and may not be realistic in real-world scenarios. Specifically, $\beta^d$ seems to be a very small value, which will result in $(1-\beta^d)$ being very close to $1$. Suppose the ML model that needs to be protected is a "harmful prompt detector" that outputs only 0 or 1. Then, for a harmful input $x_{adv}$, the condition $p_A(x_{adv}) \geq 1-\beta^d$ means that the harmful content detector needs to be **very very good at** detecting $x_{adv}$. IF the detector is already so good at detecting a given $x_{adv}$, I doubt that if it is indeed necessary to further leverage the proposed method to defend against the given $x_{adv}$.

3. In Theorem 5.4, where is the CR-guarantee proposed for the uniform kernel? From Theorem 5.4, I can only see a closed-form solution for a term $v(i,j)$, but its meaning and connection with "CR" are not explained. Besides, in Theorem 5.5, what does the "certify" function $certify(\cdot)$ mean? I think the notations and presentation in this paper can be significantly improved.

4. Since the proposed CR-defense is not a "real CR" but a data-dependent defense, I suspect that it can be easily broken by prompt-level adaptive attacks such as [r4, r5]. So I suggest the authors carefully evaluate their proposed defense against these attacks. Besides, the authors are also encouraged to adopt more common and stronger token-level jailbreak attacks such as [r6, r7] into their empirical evaluations.


**References**

[r1] Cohen et al. Certified Adversarial Robustness via Randomized Smoothing. ICML 2019.

[r2] Carlini et al. (Certified!!) Adversarial Robustness for Free! ICLR 2023.

[r3] Zeng et al. Certified Robustness to Text Adversarial Attacks by Randomized [MASK]. Computational Linguistics, 2023.

[r4] Chao et al. Jailbreaking Black Box Large Language Models in Twenty Queries. arXiv 2023.

[r5] Andriushchenko et al. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks. ICLR 2025.

[r6] Hayase et al. Query-Based Adversarial Prompt Generation. NeurIPS 2024.

[r7] Sadasivan et al. Fast Adversarial Attacks on Language Models In One GPU Minute. ICML 2024.

### Questions
See **Weaknesses**.

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
2

### Summary
This paper proposes a method to certify robustness of LLMs to adversarial attacks using stochastic defenses by a reduction to the the fractional knapsack problem -- rather than finding the adversarial input $x_{adv}$ it minimizes a function $f$ assigning weight to different stochastic generations $z$. This simplifies the problem and allows them to estimate lower bounds on the probability of successful adversarial attacks, allowing them to certify defenses within a particular range of distances.

There is also a successful adversarial attack strategy which ensures that the tokenizations of the attack are the same during inference and attack generation, creating very low upper bounds on robustness. However, this attack strategy is much less successful against stochastic defenses such as applying uniform kernels, absorbing kernels, or mask generation.

### Strengths
Originality: This paper proposes (as far as I am aware) a novel reduction of stochastic defense certification to fractional knapsack optimization, simplifying a problem and allowing it to certify robustness results.
Quality: The paper supports its theoretical claims with detailed proofs and explanations, and demonstrates the proposed methods experimentally.
Significance: Adversarial attacks on LLMs are a significant problem, and certifying the robustness of defenses is important for understanding defense quality.

### Weaknesses
Clarity:
W1) Lou et al. (2023) is not a particularly helpful first citation for explaining stochastic defenses, as it covers discrete diffusion processes. It seems to me that the idea is that $z \sim p(z|x)$ is a discrete diffusion process, but the concept can use much more explanation in the introduction of the paper. It only seems to be explained in "Results on randomized defenses." and in section 4.1, making it much harder to understand the intended connection to the knapsack problem reduction.
W2) While the proposed method is fairly abstract, it would be helpful to define substantially more of the notation.
W3) The experimental details are pretty sparse, and it's not clear to me how the experiments were conducted.

### Questions
Q1) For the specific proposed stochastic defenses, it's unclear to me what the difference between 
Q2) Can you elaborate on the different relevant $p(z|x)$ for the different ways in Section 4.1?
Q3) What does it mean to use the different models in the experiments? Section 4.1 ends saying that you only certify the safety detector -- to me that would imply that the detector looks at given model outputs, but it's unclear to me where the detector and generations come from.
Q4) I worry that the stochastic adversarial defenses are strong against the $I^2-GCG$ attack _in particular_ -- random modifications are particularly likely to break tokenization assumptions, and so it would be reassuring to the effectiveness of the stochastic defenses against simpler baselines such as GCG, BEAST, or Best-of-N jailbreaking.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the worst-case robustness of LLMs against adversarial attacks. The authors develop I2-GCG, an improved white-box attack that ensures tokenization consistency, showing most deterministic denfeses achieve nearly 0% worst-case robustness.

### Strengths
S1. Strong Theoretical Framework

S2. Comprehensive Empirical Evaluation

S3. The paper writting is clear and easy to follow

### Weaknesses
W1. The gap between theoretical guarantees and practical robustness remains large.

W2. The propose framework cannot handle insertion/deletion attacks or long heuristic prompts.

W3. While the theoretical analysis is strong, the paper doesn't propose new defense mechanisms that achieve better worst-case robustness.

### Questions
Please refer to weakness part

### Soundness
3

### Presentation
3

### Contribution
2
