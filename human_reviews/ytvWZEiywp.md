# EVINCE: Optimizing Adversarial LLM Dialogues via Conditional Statistics and Information Theory

- Avg Score: 4.00
- Decision: Reject
- Scores: 5, 6, 3, 3, 3

## Abstract
This paper introduces EVINCE (Entropy and Variation IN Conditional Exchanges), a framework that optimizes multi-LLM dialogues using conditional statistics and information theory. EVINCE introduces dual entropy optimization to balance perspective diversity with prior knowledge, providing quantitative measures for modulating LLM interactions. Through information-theoretic metrics and mutual information optimization, the framework demonstrates consistent improvement over single-LLM performance in applications ranging from disease diagnosis to news debiasing. We present theoretical foundations and empirical validation for this structured approach to LLM collaboration.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes EVINCE, a dialogue framework for LLMs that leverages adversarial debate, information theory, and conditional statistics to enhance model performance in prediction accuracy, robustness, and adaptability. The framework introduces the concepts of Inclusiveness Exploration, Information Flow Dynamics, and Reasoning Quality and Coherence, providing a structured debate mechanism that balances diverse exploration with convergence. The work emphasizes the framework’s potential for AGI development by addressing common LLM limitations, such as hallucination and bias, through its structured interaction methodology.

### Strengths
The integration of conditional statistics and information theory into LLM adversarial dialogues is novel, especially with the dual entropy framework balancing exploration and prior adherence. The empirical results validate EVINCE’s improvements in diagnostic accuracy and reasoning quality.

### Weaknesses
1. Instead of disease diagnosis, expanding tests across more diverse domains would strengthen the claim of EVINCE’s general applicability.

2. The quality evaluation of arguments is important for the EVINCE framework and dependent on the proposed CRIT scores, so more insight into how CRIT scores are generated and their potential variability across domains could improve reliability.

3. Wasserstein distance and mutual information could be computationally intensive. How about comparing them to other similar but more efficient metrics and adding more discussion about this.

### Questions
1. What is the impact of the predictions on varying contentiousness and entropy values? Are there any recommended default settings for these in specific tasks, or is parameter tuning always necessary?

2. Are there any findings of differences in performance between different LLM backbones?

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
This paper proposes EVINCE, a multi-LLM debating framework, which fosters information exchange between two LLMs via rounds of debate guided by various information metrics and a debating temperature, and thus effectively adapts LLM linguistic behaviors to complete tasks. EVINCE shows enhanced diagnostic accuracy and error corrections, demonstrating an important step towards improving reasoning abilities of LLMs on real-world data.

### Strengths
Mitigating the limitation in LLM inferences caused by the “maximum likelihood” convention is crucial in enabling LLMs with accurate responses in real-life, human-oriented scenarios, such as diagnoses. 

The proposed EVINCE debating algorithm is essentially human-like, exchanging predictions and reason sets between two equally competent LLMs in a round, and the process continues until each metric in a diverse set of information metrics, including CRIT argument quality, Wasserstein distance, MI, converges, thus facilitating adaptability through the iterations. It is also novel to use a built-in contentiousness level (debating temperature) in the prompt to help guide the debate.

EVINCE demonstrates accuracy improvements in diagnoses on the Kaggle Disease Symptoms Description dataset (covering 40 diseases).

### Weaknesses
The debate is limited to pairs of LLMs, which might reinforce a wrong “popular” prediction if both models get it wrong.

It is hinted that the problem can be alleviated through multi-round inter-LLM queries, so it might be interesting to give some examples and/or discuss how exactly these queries would help. In addition, would this problem be mitigated by introducing more models?

Similarly, will the iterative debating process cascade biases if both models are prone to generate biased answers?

Typos: Figure 2a “GPT4 pairs Claude” used the same plot as Figure 2b “GPT4 pairs Gemini”.

### Questions
I may have missed the following:

1. What is the average round of debates needed to achieve convergence? Figure 3 and 4 seem to suggest that three is an oracle number of rounds. Any intuitions for why?

2. Was there a dominating LLM during debating, like in human debates?

3. Would the per-LLM prediction accuracy continue to grow if we continually increase the number of agents, e.g., to 100 agents?

4. What is the percentage of the 40 diseases that receive accuracy improvements? I am curious to know if the overall accuracy improvement comes from most classes or just a few classes.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper deals with multi-agent collaboration/debate. The authors propose the EVINCE framework to enhance versatility, adaptivity, and reasoning for LLMs via adversarial debate and information-theoretic metric evaluation. The experimental results on a healthcare dataset demonstrate the effectiveness.

### Strengths
- Multi-agent collaboration is a significant topic for modern multi-agent systems.
- This work introduces information theory in the evaluation of dialogues between LLM agents.

### Weaknesses
- Missing discussion with other multi-agent debates in the literature such as [1,2]. There is a bunch of work on multi-agent debate. This makes the work less convincing and may mislead audiences.

- I do not have a good sense of those claims on AGI in this paper, like ‘our work targets three core AGI characteristics: versatility, iterative adaptivity, and reasoning capability’, ‘The core strength of EVINCE in advancing towards AGI lies in their ability to enhance key AGI characteristics through multi-agent dialogues’. There should be more references in the introduction, otherwise, it would be an overclaim. I personally do not believe the proposed EVINCE with adversarial debate has much to do with AGI. 

- Besides, the experiments were only conducted on a healthcare dataset. Whether this proposed method can generalize to other tasks needs further verification. At least, for those open-ended tasks without a class set, the EVINCE seems not to be applicable.  

- The introduced information-theoretic metrics seem not to change LLM prediction but only provide an early-stop criterion for LLM debate iterations according to Fig 1. So the actual improvement might just come from the multi-agent discussion itself, having nothing with the information-theoretic metrics.

### Questions
- How is the probability distribution obtained? For open-ended tasks, it is impossible to directly output the real distribution. For classification tasks as discussed in this paper, the text-based probability distribution directly from model output is not reliable and LLMs show strong miscalibration as discussed in many existing works [3].
- The authors list many existing metrics in Section 2. This seems to be redundant and provides little insight into which one is the most important for moderating the LLM debate.
- How does the framework work when only one LLM debates with itself like self-evaluation/self-refinement? This could be a good add-up to the experiments.


[1] Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv.

[2] Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., ... & Shi, S. (2023). Encouraging divergent thinking in large language models through multi-agent debate. arXiv.

[3] Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2023). Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. ICLR.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposed a dialogue framework which uses adversarial debate and dual entropy theory to achieve the AGI and enhance its proposed method EVINCE on versatility, adaptivity and reasoning capability, and further mitigate biases, reduce hallucinations and improve reasoning. Especially, the proposed EVINCE combines multiple metrics together to evaluate the entire framework, such as Wasserstein distance, relative entropy, critical thinking algorithm, correlation coefficients, mutual information, etc. They evaluate the proposed framework in the healthcare area to demonstrate the LLMs within this debate framework can lead to better diagnosis accuracy, lower entropy, lower Wasserstein distance and higher mutual information.

### Strengths
- a framework which includes iterative LLMs' debate, multiple evaluation metrics and weighted prediction based on quality scores.
- authors demonstrate the performance of the proposed method in the healthcare application areas.

### Weaknesses
- Although authors argue that the proposed EVINCE method enhances versatility, adaptivity and reasoning capabilities in LLMs, it is hard to see whether the proposed EVINCE can actually achieve this target only based on the simple demonstrations of Figure 2 and Figure 3 which use different objective evaluation metrics. It is also unclear how this proposed method can mitigate biases, reduce hallucinations and improve reasoning without any case studies or examples to support in the main texts of the paper.
- As this work mainly uses different closed-source LLMs (GPT-4, Claude, Gemini) to evaluate the performance of the proposed method, it is unclear how those closed-source LLMs can produce the probability distribution over a set of diseases, as this probability distribution is a key information to be used to calculate other evaluate metrics, such as Wasserstein distance, mutual information, etc.

### Questions
- As mentioned in the weakness, it would be better if authors could provide a more detailed case study to analyse how the proposed EVINCE can mitigate bias, reduce hallucinations and improve reasoning in the healthcare areas. It is hard to directly understand how better the proposed EVINCE is only based on the demonstrated figures.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces EVINCE, a dialogue framework to enhance Artificial General Intelligence (AGI) by leveraging adversarial debate among multiple instances of LLMs with a novel dual entropy theory. EVINCE works as a multi-agent system where large language models (LLMs) engage in structured dialogues to improve prediction accuracy, robustness, and reasoning capabilities. The framework employs information-theoretic metrics and conditional statistics to balance exploration and prior exploitation. The effectiveness is verified by an application in healthcare, particularly in improving disease diagnosis.

### Strengths
1. The subject of robustness in LLMs, especially in healthcare applications, is important. 
2. The multi-round debate is an interesting way to ensemble multiple LLMs' capabilities.

### Weaknesses
1. The theory aspect of the paper is not well-grounded. For example, the dual entropy theory is not very meaningful to me. 
2. The empirical study is only limited to the healthcare application, and the generality of the framework remains unclear.
3. The methodology of EVINCE is fairly simple and the idea of multiagent debate exists in previous literatures [1].

[1] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and
reasoning in language models through multiagent debate, 2023.

### Questions
1. For the dual entropy theory, why does the lower bound of $H(P_C)$ correspond to the robustness of the model? The entropy, as an uncertainty measure, is more relevant to exploration as in this work's taxonomy and does not seem to have a direct relationship with the robustness.

### Soundness
2

### Presentation
2

### Contribution
2
