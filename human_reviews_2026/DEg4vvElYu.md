# BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Agents backed by large language models (LLMs) increasingly rely on external tools drawn from marketplaces where multiple providers offer functionally equivalent options. This raises a critical fairness concern: systematic bias in tool selection can degrade user experience and distort competition by privileging certain providers over others. We introduce a benchmark of diverse tool categories, each containing multiple functionally equivalent tools, to systematically evaluate tool-selection bias. Using this benchmark, we evaluate seven LLMs and show that substantial bias persists, with models either fixating on a single provider or disproportionately favoring tools that appear earlier in the context. To uncover the sources of this behavior, we conduct controlled experiments that isolate the effects of tool features, exposed metadata (name, description, and parameters), and pre-training exposure. We find that (1) semantic alignment between user queries and tool metadata is the strongest driver of selection; (2) small perturbations to tool descriptions can significantly shift choices; and (3) repeated pre-training exposure to a single endpoint amplifies provider-level bias. Finally, we propose a lightweight mitigation strategy that first filters tools to a relevant subset and then samples uniformly, substantially reducing selection bias while maintaining strong task coverage. Our results highlight tool-selection bias as a key obstacle to the fair deployment of tool-augmented LLM agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper is the first work to study LLM tool-selection bias and provides a systematic evaluation across tasks and tools.
The authors design tasks that require invoking multiple candidate tools and measure how model choices vary across attributes and prompts. The core contributions of this paper include (1) a large-scale benchmark for measuring tool selection bias, (2) an in-depth analysis of the root cause of these biases, (3) a potential mitigation strategy.
Although this paper discusses an interesting and potentially impactful bias in the LLM era, the authors have provided a limited discussion of the impact of such a bias and the risks and implications behind it, which reduces the readability of this paper.

### Strengths
1. A large-scale benchmark for measuring tool selection bias
2. In-depth analysis of tool selection bias (including testing the impact of pre-training data on tool selection bias through training), which can strengthen people's understanding of this issue
3. Extensive experiments and analysis, providing valuable insights for future work on building fair and reliable agent systems

### Weaknesses
1. Lack of deep discussion of potential consequences of tool selection bias makes it difficult for readers to quickly get the effects and research significance of this paper.
2. Lack of a clear explanation of the LLM tool usage pipeline.

### Questions
1. Fig. 3 shows a long error bar in DeepSeek and the `what's language` question. The authors could provide a brief explanation and discussion of the results corresponding to this unusually large standard deviation, which would help readers better understand the model's biases and potential anomalous behavior.
2. Although this paper provides comprehensive experiments and analysis on the manifestations and sources of tool-selection bias, it lacks a discussion of the harmfulness and risks of this bias. The authors could supplement a discussion (for example, by providing figures to illustrate the value of the tool selection market) or conduct a questionnaire/interview to highlight the consequences of such a bias. This would further reinforce the significance and value of this work and echo the claim in the introduction and conclusion.
3. Since not everyone has a background in LLM tool usage, it will be good if the authors could provide a detailed tool usage process in the related work or appendix, preferably with a visual flowchart, which will effectively enhance the presentation of this paper.
4. Typo: There are no ChatGPT-3.5-turbo and ChatGPT 4.1-mini models. The authors may actually be referring to the GPT-3.5-Turbo and GPT 4.1-mini models.

### Soundness
4

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
4

### Summary
The paper exposes and addresses the problem of tool selection bias in LLM agent systems. It provides a benchmark, root cause analyses, and a simple bias mitigation approach. The results show bias with respect to an ideal scenario of uniform tool selection and the effectiveness of the mitigation as it directly achieves the target tool selection distribution by adopting a uniform distribution.

### Strengths
1. I appreciate the problem benchmarked by the work - tool selection bias. I think that this is an important problem. 
2. The work is comprehensive - with a benchmark, exploratory root cause analyses, and a bias mitigation strategy.
3. The experimental setup is pretty comprehensive, covering multiple SOTA models and scenarios.

### Weaknesses
1. Methodology concerns:
    1. How does the LLM know that the tools are functionally equivalent when only metadata is exposed? From the LLM's perspective, the tools are not equivalent, and if the language of the metadata is more appealing, then it seems natural for the LLM to pick that. If instead of LLMs, we ask humans, then even their responses are not expected to produce uniform distribution over the tools. Hence, I think comparing with uniform distribution as the target doesn't seem appropriate. I would suggest comparing with some distribution learned from human responses, to make the LLMs respond similar to unbiased humans.
    2. To create the tool selection distributions from the LLM responses, I think it would be more appropriate to sample from LLM response multiple times.
    3. The method doesn't account for initial tool retriever biases, like [1], thus providing a partial picture of the tool selection bias.
    3. Other distance metrics, such as KL-divergence can be used between the LLM tool selection and uniform probability distributions.
    4. The proposed defense is a finer-grained retriever followed by an uninformed uniform distribution sampler. I wonder how it would perform compared to the initial retriever itself which produces a smaller slate of size K, directly followed by the uniform sampler.
    5. I think that RLHF preference training could be a more potential source for the tool selection bias than pretraining and would recommend studying that as well, alongside continual pretraining.
2. Results:
    1. Main results of the root cause analyses are expected and not novel. We can expect LLMs to exploit higher similarity between query and tool metadata, as they are after all doing pattern recognition.
    2. I recommend analyzing the chain of thought of models when the bias is high, to see if the bias might be intentional.
3. Related works: There are some missing comparisons to prior works, such as [2] (shows that LLMs can be manipulated to select specific adversarial tools) and [3] (shows LLM bias beyond benchmarking). Also, [1] goes beyond benchmarking for tool selection, and I wonder how a similar analysis can be used to expose tool selection biases more rigorously.

## References
1. Quantifying Distributional Robustness of Agentic Tool-Selection
2. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection
3. Certifying Counterfactual Bias in LLMs

### Questions
1. What is your LLM decoding scheme to create the tool selection distribution?
2. What is the intuition behind "averaging" the API/tool and position probability distribution distances to form $\delta_{model}$?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the causes behind tool selection biases, where LLMs exhibit biases towards certain tools based on their descriptions or positions in the prompt even when the tools are equivalent in terms of their functionality. The authors also propose a simple mitigation strategy to alleviate tool selection biases, and empirically show that it can reduce tool selection bias.

### Strengths
1. In general, LLMs' selection bias is an important yet unsolved challenge despite extensively studied in recent literature. Within this broader context, tool selection bias is a less studied subproblem that worth further investigations. This work aims to address this gap.

2. Measuring bias of API itself (δ_API) and positional bias (δpos) separately is reasonable and disentangle the biases arising from these two different sources.

3. The idea behind the mitigation strategy, which recognizes the subset of tools that can address the given query and then performs uniform sampling from the subset, is intuitive and easy to implement.

4. The authors evaluate a variety of LLMs and API endpoints.

### Weaknesses
1. (Informativeness of the findings) Selection biases and their mitigation strategies have been studied extensively in previous literature. For example, positional biases are well-known in the research community (see the "missing references" below). While I acknowledge that there are special issues unique to "tool" selection bias, such as tool descriptions and metadata, I think the contributions of this work are relatively limited considering the presence of existing works.

2. (Missing references of selection bias) Following the previous points, I believe these highly relevant yet missing references need to be cited in this paper:

[1] Pezeshkpour, Pouya, and Estevam Hruschka. "Large language models sensitivity to the order of options in multiple-choice questions." arXiv preprint arXiv:2308.11483 (2023). NAACL 2024 Findings

[2] Zheng, Chujie, et al. "Large language models are not robust multiple choice selectors." arXiv preprint arXiv:2309.03882 (2023). ICLR 2024 Spotlight

[3] Wei, Sheng-Lun, et al. "Unveiling selection biases: Exploring order and token sensitivity in large language models." arXiv preprint arXiv:2406.03009 (2024). ACL 2024 Findings

[4] Wang, Ziqi, et al. "Eliminating position bias of language models: A mechanistic approach." arXiv preprint arXiv:2407.01100 (2024). ICLR 2025 Poster

### Questions
1. As selection bias is a well-studied issue of LLMs. What are the special issues in tool selection bias compared to selection bias in general that require a dedicated research? Elaborate more on this can strengthen the positioning of this paper.

2. Are the bias due to the fact that the tools are simply better described and understandable by LLMs? From your experiment results that "semantic similarity between queries and tool descriptions is the strongest predictor of selection", another way to interpret that is the descriptions are better aligned with the user query, rather than interpreting it as "bias". I would like to hear the authors' comments on this.

3. I also have a fundamental question about the setting in this paper. From a practical point of view, if the APIs are equivalent in their functions ("functionally interchangeable" as descsribed in line 160), we don't have to include all of them in the prompt every single time. If practitioners need to consider fairness or balance usage, they just have to uniformly sample a tool from the equivalent tools of each functional cluster and put them into the prompt for LLMs to choose from. Therefore, I am curious about how important and realistic the setting is.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies tool-selection bias in tool-augmented LLMs, showing that models often favor certain APIs for reasons unrelated to utility. The authors build a benchmark of functionally equivalent tools, measure bias across major LLMs, and find semantic metadata and training exposure as key drivers. They further demonstrate that controlled metadata changes shift model choices and propose a simple mitigation (i.e., filter relevant tools then uniformly sample), that effectively reduces bias without harming performance.

### Strengths
1, This paper studies tool selection bias in LLMs, which is a very new and meaningful topic. 

2, The authors build a clear benchmark with multiple tool clusters and balanced queries. The experiments are systematic across several strong LLMs, and the evaluation metrics are reasonable.

### Weaknesses
1, The paper does not probe whether LLMs' own familiarity/knowledge about each API explains its selection bias. Current analyses focus on metadata features, perturbations, and exposure effects, but never explored the model’s prior knowledge of each candidate API, so it’s unclear whether choices reflect surface cues or genuine familiarity.

2, The paper fixes the number of candidate APIs, leaving unexplored how bias scales or changes when the number of tools varies (e.g., 2,3,4...). This is important to understand whether larger API pools amplify or dilute preference concentration is important for real-world marketplaces where tool pool size varies.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
4
