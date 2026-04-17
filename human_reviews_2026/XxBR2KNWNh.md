# Measuring and Mitigating Identity Bias in Multi-Agent Debate via Anonymization

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Multi‑agent debate (MAD) aims to improve large language model (LLM) reasoning by letting multiple agents exchange answers and then aggregate their opinions. Yet recent studies reveal that agents are not neutral: they are prone to identity‑driven sycophancy and self‑bias, uncritically adopting a peer’s view or stubbornly adhering to their own prior output, undermining the reliability of debate. In this work, we present the first principled framework that joins sycophancy and self-bias to mitigate and quantify identity bias in MAD. First, we formalize the debate dynamics as an identity‑weighted Bayesian update process. Second, we propose response anonymization: by removing identity markers from prompts, agents cannot distinguish "self" from "peer", which forces equal weights on agent identity, thereby reducing bias.  Third, we define the Identity Bias Coefficient (IBC), a principled metric that measures how often an agent follows a peer versus itself.  Empirical studies across multiple models, datasets and debate rounds confirm that identity bias is widespread, with sycophancy far more common than self‑bias. Our findings highlight the need to ``mask" identity to ensure that MAD systems reason based on content rather than source identity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies identity-driven biases in multi‑agent debate (MAD) with LLMs. Identity-driven biases are broken down to sycophancy (conformity, over-weighting peer opinions) and self-bias (obstanicy, over-weighting self opiniongs) which may distort collective reasoning. The authors formalize debate as a Bayesian update with Dirichlet-multinomial distribution, under which the gap = conformity - obstinacy  is a sum of a content‑driven belief difference term and a pure identity bias term. Then, a ‘response anonymization’ which removes all self / peer markers from the input prompts of next-round multi-agent debate will remove the identity bias term, allowing one to measure the term (named IBC). Experiments across different benchmarks and LLMs show mixed results, with tendency of language models to have positive IBC, indicating conformity typically has a larger effect than obstinacy.

### Strengths
1. Reproducibility and presentation quality. The paper provides detailed experiment settings including input prompts. Figures and tables are clean. It is a well-written and well-presented paper.

2. The idea of formulating LLM agent collective reasoning as a Bayesian update of Dirichlet prior is intriguing, and a finding of disproportionality in opinion update is interesting. This idea and finding is worth further investigation.

### Weaknesses
1. Statistical inference (Contribution 1, Line 76) is disconnected from the experiments and analysis. The paper’s core contribution is a Bayesian model of opinion updating in MAD. Yet the model yields no substantive insight beyond the self-explanatory intuition ‘masking identity in input prompts removes identity bias’. If the paper adopts a Bayesian framing (or more generally, statistical inference), it should lead to inference about the data-generating process: estimate parameters (e.g., \(\alpha, w_i, w_j\)) from data and assess the realism of the model. Currently the formalism serves mainly to restate an obvious implication—if \(w_i = w_j\) by identity masking in input prompts, then \(w_j - w_i = 0\) (Equation 4, Line 258) without empirically validating that the statistical model explains observed behavior.

A practical path would be to turn the formalism into an estimable model. One rough sketch might be like: under single-agent generation (Line 203) one could use empirical Bayes method to infer \(\alpha_{i,t}\) (or, the normalized masses \(\alpha_{i,t} / \sum \alpha_{i,t}\)) and estimate the identity-bias coefficient (Line 296) from result in Table 1. The paper could then test whether these inferred quantities jointly account for the observed conformity–obstinacy differences (Equation 4, Line 258). As it stands, one can skip Section 4 without losing interpretability of Tables 1-2 once conformity / obstinacy are defined (Sec. 3.1), which underscores the current disconnect. Demonstrating that the model fits and explains the generated data would substantially strengthen the contribution and value of the paper.

2. There seems to be a logical gap in the interpretation of experiment results with the formalism. When measuring the conformity - obstinacy gap after identity masking, the paper claims “As established in Corollary 1, once identity bias is removed, the residual Δ reflects only the difference in the agent’s prior belief masses on the peer’s versus its own previous answer. Thus, small but nonzero values after anonymization are expected” (Line 341-344) indicating this result can be explained by the model, which is not justified. In fact, the expectation of ‘Δ in the absence of identity bias’ (Corollary 1, Line 275) over a joint distribution of y_{i,t-1} and y_{j,t-1} should be zero under the identical agent setting (i.e. same persona for i and j) as in Table 1. This part requires clarification.

3. Identity masking hurts performance (Line 345; Table 3, Line 972). While the paper defends “eliminating identity bias remains essential” (Line 995) because it “makes the debate process more reliable and better aligned with the long-term goal of building trustworthy multi-agent” (Line 997), (1) the claim of reliability and trustworthiness is questionable and not demonstrated (2) some degree of identity-related behavior can be instrumental rather than purely harmful: for example, mild in-group / confirmation tendencies (e.g., confirmation bias) and socially productive forms of influence (e.g. accommodation in Communication Accommodation Theory, perspective-taking) can facilitate coordination and collective performance in human groups. Absent evidence that suppressing identity cues is essential, the paper’s blanket prescription to remove identity appears premature.

### Questions
Dirichlet distribution as a conjugate prior for categorical distribution is well applied to discrete probability measure; however, how can this be extended to a countably infinite outcome space, for example, integer answer space? While several benchmarks have a finite answer space (like MMLU 4-option questions), others (e.g. AIME accepts integer answer 0-999) have much larger space, and since LLM is capable of open-ended generation task, the outcome space can be countably infinite. I would like to ask how the current framework can be extended in this case.

Related to Weakness 1, I would like to ask how the prior and weights, which are hyperparameters of the paper’s statistical model, can be chosen. Definition 1 treats α as free, but expectations of y_{i,t} (marginalized over thetas) are scale‑invariant in α while Δ and IBC depend on the sum of α (denominator in Theorem 1). How do you infer prior and weights in experiments?

Recent work (e.g., Qian et al., To Mask or to Mirror: Human–AI Alignment in Collective Reasoning) examines identity bias in LLM multi-agent collective reasoning. I’m trying to understand the current state of this discussion: how well documented is identity-driven behavior in multi-agent setting, and to what extent is it a new observation versus an extension of earlier findings (e.g., LLMs favoring its own model generations) ?

### Soundness
1

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
3

### Summary
This paper addresses the phenomenon of identity-bias in multi-agent debate (MAD) systems of large language models (LLMs). In such systems, multiple agents generate answers, see each other’s responses, revise, and then aggregate the result. The authors observe that agents tend to behave differently depending on whether they are looking at their own prior response (“self”) or a peer’s (“peer”) in particular exhibiting sycophancy (over-adopting peer responses) or self-bias (sticking too much to one’s prior answer). They propose a formal probabilistic Bayesian belief-update model that explicitly incorporates identity weighting to capture these biases. Based on that, they define interpretable metricsand a derived metric Identity Bias Coefficient (IBC) to measure how much identity influences agent behaviour. Their key intervention is “Response Anonymization” (i.e., removing identity markers so agents cannot tell whether a previous answer was from self or peer), thereby forcing symmetric weighting and reducing identity-bias. Empirical experiments across several model families and tasks show that identity bias (especially sycophancy) is widespread in MAD, and that anonymization substantially reduces the bias. The paper concludes that anonymization is a lightweight but effective method for making multi-agent debate more content-driven rather than identity-driven.

### Strengths
- The paper tackles a relatively under-explored issue in the emerging area of multi-agent collaborative LLM reasoning, namely how agent identity influences dynamics rather than purely content.

- The formalisation is clear and elegant: modelling the agent’s belief updating as a Dirichlet-Multinomial (DCM) process and then deriving expressions that separate belief-driven update from identity-driven weight differences. 
The proposed metrics (Conformity, Obstinacy, IBC) provide interpretable ways to quantify how much an agent is influenced by peer vs self, which is useful for analyzing such systems.

- Response Anonymization is conceptually simple but practically appealing: it requires no retraining or architecture change, just modifying how prompts are constructed. That makes it widely applicable.

- The empirical evaluation is reasonably broad: multiple model families, multiple datasets, both homogeneous and heterogeneous agent settings, and both single-peer and multi-peer setups. The results clearly show the presence of sycophancy and substantial reduction in bias under anonymization.

### Weaknesses
- The evaluation focuses on many standard reasoning tasks (GPQA, MMLU, HellaSwag, GSM8K) but may not cover more open-ended or real-world debate scenarios (e.g., multi-turn dialogues, adversarial peers, diverse agent personas beyond simple roles). The generalisability to those contexts may be unclear.
- The paper shows reduction in identity bias via anonymization but less discussion (or empirical depth) about how anonymization interacts with overall performance (accuracy, quality of final answer). There may be trade-offs (e.g., anonymization might reduce beneficial peer influence) that are not deeply explored.
- The intervention (anonymization) treats all identity cues as equal, but there may be scenarios where knowing “this answer came from a more expert agent” is beneficial (i.e., the identity might encode expertise). The paper does not deeply explore heterogeneous expertise settings or when identity cues might be legitimately informative.
- The paper missed some of the relevant reference such as https://arxiv.org/abs/2506.12657 in understanding how identity changes stances of multi-agent debate.

### Questions
- How does anonymization affect the accuracy or quality of the final aggregated answer in multi-agent debate? If anonymization reduces identity bias but also reduces correct consensus or slows convergence, that would modify how strongly I view this as an unequivocal improvement.
- In heterogeneous-agent settings where some agents are known to be more expert (e.g., a “doctor” vs “student” agent), how does anonymization affect outcomes? Does removing identity cues reduce the ability of the system to leverage expert peers?
- How stable are the results across more challenging debate formats (longer chains, adversarial peers, mixed objectives) or with more than two rounds of debate? If identity bias re-emerges in more complex settings, that would temper the generality of the findings.

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
The authors systematically examine identity biases in multi-agent debates, specifically sycophancy (the tendency to adopt the opinions of other agents) and self-bias (the tendency to maintain one's own opinion). They formalize debate dynamics as a Bayesian update process, which explains the aforementioned tendencies, and propose a simple intervention to reduce identity biases. They also introduce the Identity Bias Coefficient (IBC) as a metric to measure identity biases.

### Strengths
The paper is well structured and easy to follow. 
Response anonymization is a simple yet interesting and effective approach to reduce identity biases of LLMs in multi agent debates.

### Weaknesses
Response anonymization appears to have no effect, or rather a negative effect, on the overall model's accuracy. 
The related work section suggests that other mitigation strategies have been investigated by previous studies, but does not specify which ones. This information is important for the paper.

### Questions
- What is the greatest benefit of removing identity biases in the case of multi agent debate?

### Soundness
3

### Presentation
3

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
This paper addresses identity bias in multi-agent debate (MAD) systems, where LLM agents exhibit either sycophancy/conformity or self-bias/obstinacy. The authors formalize debate mechanics in LLMs as an identity-weighted Bayesian update process using a Dirichlet-Compound-Multinomial model. To mitigate the impact of identity bias on model responses, they propose response anonymization, which removes identity markers from prompts. The paper also introduces the Identity Bias Coefficient (IBC) to quantify the magnitude of identity bias. Comprehensive experiments demonstrate that identity bias is widespread, with sycophancy being more prevalent than self-bias, and that anonymization effectively reduces this bias.

### Strengths
The concept of unifying sycophancy and self-bias in MAD into identity bias is a novel proposal, which is supported by precise theoretical modeling using identity-weighted Bayesian updates in a principled and interpretable manner.

The response anonymization mitigation solution requires no retraining, architectural changes, or auxiliary losses and is shown to be consistent and effective across settings.

The authors demonstrate the robustness of their results across multiple models, tasks, and go beyond to include multi-peer and heterogeneous persona settings which makes the analysis compelling.

### Weaknesses
While the theoretical derivation is able to isolate an identity-bias term, the empirical demonstrations are not fully convincing as causal in nature. When anonymization reduces the Conformity-Obstinacy gap, is this necessarily because identity bias was eliminated? Could this be instead because anonymization changes the cognitive load or prompt complexity? The ordering of responses in anonymised prompts could also introduce noise which could separately impact model reasoning. Comparing anonymisation to other intervention methods such as randomly swapping labels, applying counterfactual identity relabeling with content being held constant would help to strengthen the claims made in the paper.

The paper defers how anonymization impacts task accuracy to the appendix. Table 3 even shows mixed performance effects of anonymization, but the discussion is quite brief. While the paper argues that eliminating bias is valuable regardless of performance, a deeper analysis of when and why anonymization hurts performance would be valuable to readers.

The belief difference term which represents reasoning driven by actual content differences warrants more analysis. While references are made to the empirical behaviour of this term in Section 5.2, the calibration and quality of this term is not fully examined. Without this, we also don’t know when anonymization reduces bias while deference to more accurate peers could actually be more beneficial.

In the heterogenous setting, the role of adding multiple specialised personas is under-explored as they may be relevant to task success. Tied to the previous point, in this case, anonymisation may lead to more substantial performance tradeoffs as domain-expert personas should likely attract conformity on domain specific items.

### Questions
Can you run alternate intervention experiments with counterfactual or random identity label swapping and compare the findings wrt point 1 mentioned above?

Could you report more analysis of when anonymization can lead to trade-offs in actual task performance? Are there predictable patterns based on task characteristics, model size, or debate configuration?

Could you empirically show how the belief difference term relates to actual content driven reasoning? Maybe a correlation analysis between proxies of model confidence (e.g. log probs) would make it more clear.

What is the impact of anonymisation in cases where knowing the identity is actually beneficial for the outcome, such as in domain-relevant personas are used (e.g. a doctor in the medical domain)? A justification in the context of debate dynamics could also be useful.

### Soundness
2

### Presentation
3

### Contribution
2
