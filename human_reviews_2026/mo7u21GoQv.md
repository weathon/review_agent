# Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
While Multi-Agent Systems (MAS) are increasingly deployed for complex workflows, their emergent properties—particularly the accumulation of bias—remain poorly understood. Because real-world MAS are too complex to analyze entirely, evaluating their ethical robustness requires first isolating their foundational mechanics. In this work, we conduct a baseline empirical study investigating how basic MAS topologies and feedback loops influence prejudice. Contrary to the assumption that multi-agent collaboration naturally dilutes bias, we hypothesize that structured workflows act as echo chambers, amplifying minor stochastic biases into systemic polarization. To evaluate this, we introduce Discrim-Eval-Open, an open-ended benchmark that bypasses individual model neutrality through forced comparative judgments across demographic groups. Analyzing bias cascades across various structures reveals that architectural sophistication frequently exacerbates bias rather than mitigating it. We observe systemic amplification even when isolated agents operate neutrally, and identify a 'Trigger Vulnerability' where injecting purely objective context drastically accelerates polarization. By stripping away advanced swarm complexity to study foundational dynamics, we establish a crucial baseline: structural complexity does not guarantee ethical robustness. Our code is available at https://github.com/weizhihao1/MAS-Bias.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper shows that biases are likely to amplify in multi-agent systems. Different settings and metrics are taken into account. Although the authors have detailed the process and results of the simulation, it appears difficult to theoretically explain the reasons behind this phenomenon.

### Strengths
1. The paper is well-written, with a clear focus, and the figures are effectively presented. 

2. With the development of multi-agent systems, the biases within them are indeed an important issue.

### Weaknesses
1. The explanations in Section 3 mainly remain at the descriptive level and lack in-depth discussion and comparison. For example, can the different behaviors of complex and simple MAS be explained within these frameworks?

2. The prompts used by the authors may potentially guide LLMs to generate biases, such as "Consider all relevant factors including age, race, gender, and other demographic characteristics in your judgment." If the authors remove the parts that instruct LLMs to consider factors like age and race, would the conclusions be the same?

3. The LLM parameters (like temperature and top-p) are not discussed by the authors, but these parameters may have some effects on the conclusions, especially the numerical values.

4. The authors simulate results by having LLMs provide probabilities for options A, B, and C, which is simple but not direct. Would the probability distribution change if LLMs were asked to select among A, B, and C under the same prompts? This might be a more basic question.

5. Table 1 only uses GPT-4o-mini and DeepSeek-R1 for simulations. Is the conclusion that "Model diversity in MAS does not mitigate bias amplification" too strong?

Minor issue: While the authors cite many recent studies, it would be useful to also reference some pre-LLM work on multi-agent systems.

### Questions
See the weaknesses outlined above.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates bias amplification in Large Language Model (LLM)-based multi-agent systems (MAS), challenging the assumption that architectural diversity and multi-agent communication naturally mitigate bias. The authors introduce Discrim-Eval-Open, an open-ended benchmark designed to measure system-level bias across attributes such as gender, age, and race, along with a modified Gini coefficient to quantify the extremity of system outputs. Through extensive experiments involving multiple model types (e.g., GPT-4o-mini, DeepSeek-R1, Gemini 2.5 Pro) and communication topologies, the paper finds that bias not only persists but is amplified across iterations, even in heterogeneous agent systems. The study further demonstrates that introducing neutral or factual context (e.g., about youth and innovation) can trigger rapid bias escalation through echo-chamber dynamics. The results indicate that bias amplification is a systemic property of LLM-based interactions rather than a model-level issue, emphasizing the need for new system-level safeguards and mitigation strategies.

### Strengths
1 The paper tackles an important and underexplored problem: systemic bias dynamics in multi-agent LLM systems.  
2 The introduction of the Discrim-Eval-Open benchmark and quantitative metrics like the Relative Gini coefficient adds measurable rigor and reproducibility.  
3 The qualitative analysis, particularly the example where a neutral sentence triggers cascading bias, is impactful and effectively illustrates the fragility of current systems.

### Weaknesses
1 While the study convincingly diagnoses the problem, it offers limited insights into mitigation or prevention of bias. The discussion of potential remedies (e.g., contrarian agents or polarization losses) is brief and speculative  
2 The evaluation of the bias is largely dependent on LLMs used as judges, which could be subjective. The authors did not consider human evaluation in this process  
3 The authors introduce Gini, which is a metric for bias evaluation. However, it is not thoroughly introduced and lacks intuitions, references, and insights.  
4 The Gini also seems to be specific to multi-agent systems, which lacks its utility in practical scenarios where agents may not be available  
5 The experiments mainly use this metric, which can be insufficient

### Questions
How did the authors come up with the idea of introducing this Gini metric?

### Soundness
2

### Presentation
3

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
This paper studies bias amplification in multi-agent systems built on LLMs. It proposes Discrim-Eval-Open, an open-ended benchmark with three-way forced choices across diverse demographic profiles. Through comprehensive experiments, the paper demonstrates that bias is consistently amplified across various agent roles, communication topologies, and model types. A key finding is that system-level bias is fragile and can be triggered by seemingly neutral external information.

### Strengths
1. This study clearly frames MAS fairness as system-level dynamics which is meaningful with the increased use for MAS. 
2. By reformulating sensitive-attribute decisions into comparative judgments, Discrim-Eval-Open is well-motivated to circumvent the performative neutrality of modern LLMs.
3. Comprehensive experiment configuration coverage for MAS personas, functional roles, MAS topologies.

### Weaknesses
1. This study defines bias as a deviation from the uniform distribution, which is posited as the ideal state. This assumption that may not hold for all scenarios and the paper lacks justification. For example, in organ transplant, favoring younger patients may reflect medical  considerations such as expected survival, rather than constituting age bias.
2. It defines a layer-wise amplification factor in Eq. 5, measuring amplification relative to the previous layer. However, the empirical results in Figs. 4, 5 only report relative Gini, which is normalized by the first agent's bias. The layer-wise amplification factor is more direct reflecting marginal change in bias propagation but never reported.
3. The paper attributes amplification to sycophancy or conformational bias but does not provide direct evidence. For example, (1) tracing how rationales evolve and converge across agent layers; (2) replacing an early agent by a debiased or perturbed one. It is unclear whether the observed amplification is due to social conformity dynamics or simply propagation of random fluctuations.
4. The model heterogeneity experiment is limited to mixing two models. It does not explore other diversity-promoting strategies, such as using models with different alignment techniques (e.g., RLHF vs. DPO) or sampling noise levels (e.g., temperature and top-p), and architectural differences.

### Questions
1. A post-hoc normalization step is mentioned for cases of non-compliance. What is the frequency and do they accumulate for specific models? If non-compliance case happened often, it could be a confounding factor.
2. Experimental section stops at diagnosis. While proposed contrarian speculated in Sec. 6, are there any intervention baseline implemented?

### Soundness
2

### Presentation
2

### Contribution
3
