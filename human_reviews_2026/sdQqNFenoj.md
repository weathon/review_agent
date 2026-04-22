# Teaching Values to Machines: Simulating Human-Like Behavior in LLMs with Value-Prompting

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Large Language Models (LLMs) demonstrate a remarkable capacity to adopt different personas and roles. Yet, it remains unclear whether they are able to manifest a behavior that adheres to a coherent set of values. In this paper, we introduce value-prompting, a novel prompting technique that draws upon established psychological theories of human values. Using a comprehensive behavioral test, we demonstrate that value-prompting systematically induces value-coherent behaviors in LLMs. We then administer a set of psychological questionnaires to the value-prompted LLMs, covering aspects such as pro-sociality, personality traits, and everyday behaviors. We also examine different approaches to simulate the value composition for an entire population. Our results show that value-prompted LLMs embody value structures and value-behavior relationships that align with human population studies. These findings showcase the potential of value-prompting as a psychologically driven tool to manipulate LLM behavior.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces value-prompting, a psychologically grounded prompting technique designed to simulate coherent human-like value structures in large language models. Drawing on Schwartz’s theory of basic human values, the authors test whether LLMs can exhibit consistent value-behavior relationships aligned with human data. Through behavioral and psychological questionnaires, they find that value-prompted models not only display structured value correlations similar to humans but also reproduce human-like links between values and behaviors such as prosociality, charity, and personality traits. The study further explores population-level simulations and shows that human-informed distributions improve alignment, suggesting value-prompting as an effective and interpretable method for steering LLM behavior.

### Strengths
This paper focuses on how to enable LLMs to stably exhibit certain human values and use them as a foundation for sociological simulation, which is an important and widely studied problem.

The writing of this paper is relatively clear, which makes it quite easy to follow.

### Weaknesses
My main concern with this paper is that its purely prompt-based methodology lacks theoretical grounding and shows very limited originality. Not only is the prompt design unstructured and lacking in CoT design, but its overall construction is overly simplistic. To my knowledge, similar prompting techniques have only been regarded as tricks in previous works, such as [1], [2], and [3].

As I mentioned in my previous question, the proposed method suffers from a serious lack of innovation. More critically, as a methodological and analytical paper, it lacks proper baselines. I noticed that the paper I cited as an example in my previous question included references and conducted sufficient literature review in the related work section. At the very least, this paper should compare its prompting method to those and clearly articulate its distinctions and contributions. However, I did not find any meaningful improvement or differentiation, even though the authors claim that prior methods did not further analyze consistency behaviors. I suggest that the authors focus on exploring why existing methods fail to stably represent values (though I personally doubt that is the case) and further propose methodological improvements based on such analysis.

The paper claims that previous works did not analyze the correlation between values and behaviors, yet I did not observe such an analysis here either. I can tell that the methodological framework draws inspiration from [1]; however, unlike the persona, this questionnaire-based analysis clearly falls short of capturing complex value-related behaviors, as it diverges significantly from real-world contexts.

Simply using the Schwartz value framework is insufficient. As a paper aiming to teach LLMs human values, it lacks in-depth reflection and analysis on what kinds of values an LLM actually needs.

There are some typos in the use of quotation marks at Line 67 and Line 185.



**Reference**

[1] Evaluating and Inducing Personality
in Pre-trained Language Models (https://proceedings.neurips.cc/paper_files/paper/2023/file/21f7b745f73ce0d1f9bcea7f40b1388e-Paper-Conference.pdf)

[2] Heterogeneous Value Alignment Evaluation for Large Language Models (https://arxiv.org/abs/2305.17147)

[3] Towards Measuring the Representation of Subjective
Global Opinions in Language Models (https://arxiv.org/abs/2306.16388)

### Questions
It seems that this paper lacks analysis and answers to question 3?

Other questions see the Weaknesses section.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces value-prompting, a technique based on Schwartz's psychological value theory to systematically induce value-coherent behaviors in LLMs. The authors test seven models (Flan-T5-XXL, Mixtral-8×7B, LLaMA-3 series, GPT-OSS series, Qwen3-235B) across three research questions: whether value-prompting induces coherent value structures (RQ1), whether these align with human value-behavior relationships (RQ2), and whether LLMs can simulate population-level psychological experiments (RQ3). Using Anthropic's Behavioral Analysis Test, PVQ-40, and five behavioral assessments (donation, pro-sociality, charity, Big Five, everyday behaviors), combined with five population simulation strategies, they report value structure correlations around 0.8 with human data and significant value-behavior alignment across most models, with priming-only approaches outperforming explicit value score provision in ablation studies.

### Strengths
# 1. Psychologically grounded population simulation

The paper thoughtfully incorporates psychological research (Witte et al., 2020) showing that 53% of humans lack a single dominant value, designing multiple sampling strategies (H-Norm, H-Even, H-NP) to reflect realistic population distributions rather than naive uniform sampling.

# 2. Broad experimental scope

The study examines seven diverse LLM architectures across five different behavioral domains (donation, pro-sociality, charity game, Big Five, everyday behaviors), testing both value structure coherence and value-behavior relationships.

### Weaknesses
# 1. Insufficient comparison with alternative value steering methods
The paper proposes value-prompting as a method to induce value-aligned behavior in LLMs but provides no comparison with existing value steering approaches. Prior work has explored various techniques and prompt templates for aligning LLM behavior with specific values or personas, including:
- In-context impersonation reveals large language models' strengths and biases (Salewski et al., 2024)
- Quantifying the persona effect in LLM simulations (Hu & Collier, 2024)
- Character-LLM: A Trainable Agent for Role-Playing (Shao et al., 2023)
- Whose opinions do language models reflect? (Santurkar et al., 2023)
- From Values to Opinions: Predicting Human Behaviors and Stances Using Value-Injected Large Language Models (Kang et al., 2023)

The authors also should compare diverse prompt templates and variations to support their claim. Without empirical comparison to these baselines, it is difficult to assess whether the proposed approach offers meaningful advantages or whether the observed alignment is simply a general property of instruction-tuned LLMs responding to any value-related prompting.

# 2. Lack of baseline comparisons for validating coherent value structures (RQ1)
The paper claims that value-prompting induces "coherent value structures" based on behavioral agreement patterns (Figure 3) and correlation matrices (Figure 4a). However, no baseline conditions are provided to establish what incoherent or random value structures would look like. Without comparison to models without value-prompting, or with alternative prompting strategies such as generic persona descriptions or random value assignments, it remains unclear whether the observed patterns are specifically induced by value-prompting or emerge from other factors. The negative correlations between opposing values (e.g., Conservation vs. Openness to Change) need to be demonstrated as significantly different from chance or from non-value-primed models to substantiate the claim of induced coherence.

# 3. Unclear justification for behavioral test methodology (RQ1)
Section 4 uses the behavioral test from Perez et al. (2023) to establish coherent value structures, but the connection between the measures and conclusions requires clarification. The paper measures "percentage of model agreement" with various behaviors (Figure 3) and interprets distinct patterns across values as evidence for coherence. However, coherence in Schwartz's theory specifically refers to adjacent values sharing motivational goals while opposing values reflect motivational conflicts. The paper would benefit from directly testing whether similar values yield similar behavioral patterns by examining clustering or distance metrics between value vectors, rather than simply showing that different values produce different agreement percentages. Additionally, Figure 4b presents results for "conservative political behaviors" as evidence of human-aligned patterns, but the paper never defines what qualifies as conservative political behavior from the Perez et al. test set or explains why these specific behaviors were selected for presentation. If this represents a subset of available results, the selection criteria must be justified to avoid concerns about cherry-picking.

# 4. Indirect validation methodology for human alignment (RQ2)
The primary evidence for human alignment comes from comparing correlation matrices between LLM responses and human data using Procrustes analysis and Pearson correlations. This approach has several limitations. First, while Appendix F lists various human studies, the paper lacks a clear description of how these diverse datasets were aggregated or weighted to form the human baseline. The studies vary substantially in sample size (from 246 to 1,857 participants), demographics, and cultural background (Israeli students, Italian adults, Polish adults, etc.), yet the paper does not explain how these differences are handled when creating comparison benchmarks. Second, the validation methodology is quite indirect, relying on second-order statistical patterns rather than examining whether value-prompted LLMs make the same specific choices as humans with corresponding values. More direct measures could include comparing raw behavioral frequencies, testing specific choice predictions, or examining whether the magnitude (not just the correlational structure) of value-behavior relationships matches human data.

# 5. Limited interpretation of population simulation results (RQ3)
While the paper tests multiple population distribution strategies in Section 5.2 and Table 1, the interpretation of results remains superficial. The H-NP approach consistently achieves the highest alignment scores across models, but no theoretical explanation is provided for why incorporating non-dominant individuals through unprompted model responses would improve alignment. Conversely, the "Model-Specific" distribution, despite being data-driven and tailored to each model's characteristics, underperforms relative to simpler human-informed approaches. This counterintuitive finding deserves discussion. The paper also lacks analysis of which specific values or behavioral domains are most sensitive to population distribution choices, which would provide insight into when and why population simulation matters.

### Questions
1. Value steering baselines: Can you provide comparisons with alternative value steering methods using different templates and prompts in related work (e.g., persona-based prompting from Salewski et al. 2024, value-injected prompting from Kang et al. 2023)? How does your value prompting perform relative to these approaches on the same behavioral tests?

2. Null baseline for coherence: What behavioral patterns emerge without value-prompting (i.e., using base models or generic prompts)? Including this baseline would clarify whether the observed coherence is specifically induced by value-prompting or is a general property of instruction-tuned models.

3. Human data aggregation: How were the diverse human datasets (varying in size, demographics, and cultural background) combined to create the human baseline for comparison? Were they weighted equally, or according to sample size? How might different aggregation strategies affect the alignment scores?

4. Conservative political behavior definition: How is "conservative political behavior" defined and measured in Figure 4b? Which specific behaviors from Perez et al. (2023) were included, and why? If this is a subset, what criteria guided the selection?

5. Value adjacency analysis: In Section 4 can you provide quantitative analysis of whether theoretically adjacent values (e.g., Tradition-Conformity) produce more similar behavioral patterns than distant values (e.g., Power-Universalism)? This would strengthen the claim of coherent value structures.

### Soundness
1

### Presentation
2

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
This paper studies the human-like behavior simulation abilities of LLMs, particularly, focuses on human-value related behavior patterns and investigates interesting research questions: whether value prompts can shape LLMs’ behavior patterns, and whether the induced value and connected behaviors align with those of humans. To study this, grounded in Schwartz Theory of Basic Human Values, the authors conduct comprehensive experiments, including i) value priming under behavioral analysis, ii) value and behavior correlation analysis using questionnaires, and iii) population-level simulation analysis, for a diverse set of open-source LLMs. Through these experiments, the authors conclude value-prompting can induce coherent value structures and then the correlations between values and behaviors can be captured to some extent.

### Strengths
1. This paper focuses on LLMs’ ability to simulate human value structure and correlated behaviors, which become more important with LLMs’ increasing integration with human life, and holds the potential for helping social science research.

2. The experiments are extensive, covering massive analysis, like value priming under behavioral analysis and value and behavior correlation analysis.

3. Some of the analysis methods, e.g., behavioral analysis, value correlation and metric Multidimensional Scaling, are interesting and inspiring.

### Weaknesses
The biggest problems lie in some experimental design.

1. The evaluation benchmarks are problematic. Particularly, part of the data seems not suitable for reflecting the value-related behaviors. In detail:

    (a) In Sec.4, some of the behaviors seem not relevant to all/part of the value dimensions in Schwartz Theory, such as the BigFive personality dimensions, psychopathy, aesthetic preferences and so on. This may cause bias/noise to the results in subsequent correlation analysis in Fig.4. For example, in Fig.3, neuroticism seems irrelevant to values, and thus the differences among the four value category makes no sense.

    (b) In Sec.5, the used behavior questionnaires also face the same problem. The BigFive ones and part of the Everyday Behavior Questionnaire are not directly connected to values. Besides, it’s unclear whether these behavior questionnaires can cover all the ten value dimensions. If not, the conclusions and results may be biased.

2. The behavioral analysis is quite surface level. In all experiments, the authors use questionnaires or Yes/No questions to induce LLMs’ behaviors. However, such multi-choice questions are not suitable for LLMs due to either their inherent bias [1][2] or data contamination [3]. As a results, such self-report-style tests only reveal superficial level behaviors. Since we expect stable/consistent values/traits across (and hence they can be regarded as the real values) across, more complex downstream tasks/behavior tests should be used to reflect LLMs’ value-behavior pattern [4]. 

Reference:

[1] Zheng et al., Large Language Models Are Not Robust Multiple Choice Selectors. 2024.

[2] Myrzakhan et al., Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs Evaluation, Benchmark, and Arena. 2024

[3] Duan et al., Denevil: Towards Deciphering and Navigating the Ethical Values of Large Language Models via Instruction Learning. 2024.

[4] Han et al., The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs. 2025.

### Questions
1. Why does the evaluated model set (line 151) exclude stronger commercial models like GPT-4o?

2. Could you clarify how the “human-informed” and “model-specific” are combined in Sec. 5.2? Please describe the implementation in more detail.

Other suggestion: Fig. 1 is a bit blurry; the details are not clear.

### Soundness
2

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
2

### Summary
This paper proposes value-prompting, a technique to align LLM behaviors with Schwartz's theory of basic human values. The authors construct prompts based on psychological definitions of ten core values and evaluate whether value-prompted LLMs exhibit value-behavior relationships and inter-value structures consistent with human population studies. The evaluation employs behavioral tests and psychological questionnaires to assess alignment.

### Strengths
This paper bridges NLP, computational social science, and psychology by operationalizing an established psychological framework, Schwartz's value theory for LLM behavior steering. The authors go beyond single-question probing by validating both intra-value coherence and value-behavior alignment against human benchmarks across multiple behavioral tests. The use of established psychological instruments and comparison with human population data provides a grounded validation approach.

### Weaknesses
This paper claims to introduce "a novel prompting technique," but the relationship to prior work, particularly Fischer et al. (2023) and the extensive persona/role prompting literature, remains unclear. The authors would be better to explicitly clarify what is novel beyond applying Schwartz's framework is it the systematic validation approach, the specific prompt engineering, or the evaluation methodology? Yes, this is also question, so I hope to listen to the authors response for clarity.
While the paper includes naive prompting baselines, it lacks empirical comparisons to existing value-based prompting methods (Fischer et al., 2023; Kang et al., 2023). The authors explain how their method differs conceptually, but empirical evidence demonstrating these differences is essential. Stronger baselines such as simple value-name prompts (e.g., "Act as a person who values security") would help isolate the contribution of theory-driven detailed descriptions.

### Questions
Please refer to the weaknesses that contains the questions

### Soundness
2

### Presentation
2

### Contribution
2
