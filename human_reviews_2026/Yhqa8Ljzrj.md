# Quantifying Human-AI Synergy

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 8, 4

## Abstract
We introduce a novel Bayesian Item Response Theory framework to quantify human–AI synergy, separating individual and collaborative ability while controlling for task difficulty in interactive settings. Unlike standard static benchmarks, our approach models human–AI performance as a joint process, capturing both user-specific factors and moment-to-moment fluctuations. We validate the framework by applying it to human–AI benchmark data (n=667) and find significant synergy. We demonstrate that collaboration ability is distinct from individual problem-solving ability. Users better able to infer and adapt to others’ perspectives achieve superior collaborative performance with AI–but not when working alone. Moreover, moment-to-moment fluctuations in perspective taking influence AI response quality, highlighting the role of dynamic user factors in collaboration. By introducing a principled framework to analyze data from human-AI collaboration, interactive benchmarks can better complement current single-task benchmarks and crowd-assessment methods. This work informs the design and training of language models that transcend static prompt benchmarks to achieve adaptive, socially aware collaboration with diverse and dynamic human partners.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Bayesian Item Response Theory (IRT) framework to quantify human–AI synergy, separating individual from collaborative ability while controlling for task difficulty. The authors demonstrate the framework on two datasets and AI models. The results show that AI helps improve human-AI accuracy. The paper also uses Theory of Mind (ToM) to interpret the observed human-AI synergy.

### Strengths
- The authors present the results grounded in empirical results and connect well to prior benchmarks.
- The analysis using Theory of Mind provides an interesting bridge between computational modeling and social cognition.

### Weaknesses
- The model structure is oversimplified compared to the ambitious contribution claimed by the authors. It only considers the ability and the  difficulty. A lot of factors are ignored such as learning effects. The assumption on additivity of ability is also unrealistic--there are a lot of case where human and AI are substitute or complementary with each other.
- The choice of the Bayesian model is not clear. Since the Bayesian workflow is often iterative and involves model fitting and then checking, the authors should do more model comparison to motivate the model specification they arrive at.
- The experiments also simplify a lot than realistic human-AI collaboration scenarios. For example, three questions done alone with AI is not much to get a good estimate of ability in my opinion.
- The paper does not engage with prior work on human–AI complementarity enough. For example, how the framework improves interpretability over existing regression-based or causal models of AI assistance effects?

### Questions
- What are the assumptions of the framework? In general, I would suggest the authors to lay out the assumptions that must hold for their results to be easy to interpret.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose a Bayesian Item Response Theory (IRT) framework for evaluating human–AI collaboration. Instead of measuring model performance in isolation, the proposed approach jointly models human and AI contributions during cooperative task solving. The authors apply this method to the ChatBench dataset, where 667 participants complete multiple-choice tasks under both solo and AI-assisted conditions. The results indicate that collaborative performance can differ from solo capability and varies across individuals. The paper also examines Theory-of-Mind (ToM) scores and reports that higher ToM ability is associated with greater improvement when using AI assistance.

### Strengths
1. This paper tackles a timely and important problem, as understanding interactive LLM behavior and human–AI teaming is increasingly critical when transitioning from offline evaluation to real-world deployment scenarios.

2. The work provides a strong methodological contribution by proposing a principled Bayesian IRT framework that decomposes human solo ability, human–AI collaborative ability, AI contribution, and task difficulty to quantitatively measure human–AI synergy, and the method is supported by strong empirical evaluations.

3. The study offers interesting insight into cognitive mechanisms underlying human–AI interaction, as the finding that Theory-of-Mind predicts collaborative gain, not solo performance, helps explain when and why human–AI synergy emerges, which also provides valuable implications for practitioners on how AI systems should be designed to better support human decision-making.

### Weaknesses
1. Experimental tasks all fall within academic contexts  (MMLU-adapted questions). Although justified as a structured benchmark, future work could validate synergy in more naturalistic settings (e.g., group creative work, travel planning, or collaborative coding).

2.  ToM scoring uses an LLM rater, which is reasonable given recent literature on LLM-as-judge, but still vulnerable to construct validity concerns. The authors partially address this via human validation, but a deeper discussion (e.g., potential LLM bias or adversarial prompt cases) in cases where such human and llm alignment diverges significnantly would strengthen this part.

3.  Limited to individual human–AI teaming. It would be interesting to see whether the method generalizes to multi-agent settings where a single human interacts with multiple AI assistants or where a group of humans collaborates with one AI system.

### Questions
See above weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a method to measure human–AI synergy, where synergy is defined as how much an AI partner improves human performance. The framework follows Bayesian Item Response Theory (IRT). It separates solo human ability, collaborative ability with AI, and item difficulties in solo vs. joint settings. Using ChatBench (396 MCQs across math, physics, moral reasoning; 667 participants), the study shows that human–AI teams outperform humans alone or AI alone, and benchmarks models by their average improves human performance while controlling for difficulty and user abilities. Finally, the authors test whether Theory of Mind (ToM) in users explains who benefits most and find ToM predicts collaborative performance.

### Strengths
* Clear synergy metric for model benchmarking. The approach directly estimates each model’s capacity to raise the performance of the average user, rather than relying on static model-alone accuracy, enabling apples-to-apples comparisons of “collaborative capability.” 


* ToM as a cognitive mechanism for teaming. Users with higher ToM do better with AI but not alone; the paper frames ToM as a plausible mechanism for coordination and division of cognitive labor, aligning with established social-cognitive theory and giving a concrete lens for why teaming helps. 

* Empirical improvement that is practically meaningful. The paper finds human-AI even beats AI alone in the descriptive analysis.

### Weaknesses
* Framework builds heavily on prior modeling choices. The novelty is mainly in applying existing method to human–AI collaboration. 

* ToM effects appear small-to-moderate. My main concern with the findings in the paper is that the ToM–collaboration link is statistically positive but not large (e.g., Spearman ~ 0.17 for joint ability, significant; ~ 0.06 and n.s. for solo). The results seem to suggest ToM is useful, but one factor among several. 

 
* Limited window into mechanisms of complementarity. While the paper argues that ToM enables coordination, it does not decompose how human and model contributions combine at the item level (e.g., how many tasks are correct in human-AI collaboration but wrong in both alone?).

### Questions
* Quantifying complementarity directly. Since human+AI outperforms each constituent, what fraction of items are: wrong for both solo agents but right as a team or right for one party but not the other? Reporting these rates would put concrete numbers on complementarity.

* Break out complementarity by difficulty deciles and domain. I would guess one reason for complementarity is human provide reasoning steps to AI. It would be helpful to clarify if AI baseline uses reasoning.

### Soundness
2

### Presentation
4

### Contribution
3
