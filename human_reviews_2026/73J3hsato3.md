# Social Agents: Collective Intelligence Improves LLM Predictions

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
In human society, collective decision making has often outperformed the judgment of individuals. Classic examples range from estimating livestock weights to predicting elections and financial markets, where averaging many independent guesses often yields results more accurate than experts. These successes arise because groups bring together diverse perspectives, independent voices, and distributed knowledge, combining them in ways that cancel individual biases. This principle, known as the Wisdom of Crowds, underpins practices in forecasting, marketing, and preference modeling. Large Language Models (LLMs), however, typically produce a single definitive answer. While effective in many settings, this uniformity overlooks the diversity of human judgments shaping responses to ads, videos, and webpages. Inspired by how societies benefit from diverse opinions, we ask whether LLM predictions can be improved by simulating not one answer but many. We introduce Social Agents, a multi-agent framework that instantiates a synthetic society of human-like personas with diverse demographic (e.g., age, gender) and psychographic (e.g., values, interests) attributes. Each persona independently appraises a stimulus such as an advertisement, video, or webpage, offering both a quantitative score (e.g., click-through likelihood, recall score, likability) and a qualitative rationale. Aggregating these opinions produces a distribution of preferences that more closely mirrors real human crowds. Across eleven behavioral prediction tasks, Social Agents outperforms single-LLM baselines by up to 67.45% on simple judgments (e.g. webpage likability) and 9.88% on complex interpretive reasoning (e.g. video memorability). Social Agents’ individual persona predictions also align with human judgments, reaching Pearson correlations up to 0.71. These results position computational crowd simulation as a scalable, interpretable tool for improving behavioral prediction and supporting societal decision making.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper propose a special ensemble method called social agents, where the llm prediction is aggregated across different persona based predictions. The author find this greatly enhance the performance of various downstream opinion based tasks like twitter likability.

### Strengths
1. The idea of using persona conditioned ensemble of the same base llm is interesting and the experiments are comprehensive. The author also shows that the performance gains are agnostic of different models, which increase the applicability of the method.

2. Social agent even beat trained expert which is interesting. And the performances on the benchmarks are pretty astonishing.

### Weaknesses
1. Motivation. Ideally LLMs that are trained on all website data is already representative of the collective opinion of the human population. The rlhf process might make the LLM a bit more biased and politically lean left, but in general it should already represent the general society. Most personalization paper wants to tailor to each individual opinions and stay away from collective opinions. The authors seems to be take a big circle and use personalization and aggregate to get the collective opinion, which is perplexing to me.

2. Baseline choices and persona choices. For baseline, the author mentioned that they just sample from the same model N times. One would at least expect some sort of better ensemble here such as changing the temperature. For the social agent, the authors uses at most 10 personas. This is clearly not enough to represent the society. And the author mentioned diminishing return when increasing more persona, which is surprising.

### Questions
1. Why can't the original LLM be a good enough predictor for collective opinion?

2. Do you have explanation why only 10 personas boost the results by a huge amount of gap comparing to base llm? And could you try different temperature and see if the baseline has better prediction as well? 

3. See weakness.

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
5

### Summary
This paper introduces Social Agents, a framework that brings the Wisdom of Crowds to LLMs by creating diverse, human-like persona LLM agents whose independent opinions are combined into one collective judgment. Each LLM persona evaluates content such as ads or webpages, and their aggregated responses consistently outperform single-model predictions. Across multiple tasks, Social Agents improves accuracy both on simple judgments and complex reasoning. The study shows that diversity, not just scale, makes LLMs more accurate, interpretable, and human-like in their predictions.

### Strengths
•	In general, the paper provides great context on the importance and value of the study.

•	The implications of the study are well articulated.

•	The experiments are very comprehensive, spanning across 8 different domains and multiple datasets. 

•	Including task-specific trained models is a wise move, which shows the usability of the proposed approach in the real world.

•	The explanation of the theory and how that motivates the experiment design is well established.

•	The figures 1 and 3 are well polished and convey the idea clearly.

### Weaknesses
•	Missing literature. Previous studies have shown that LLM’s wisdom of crowds (WoC) do not always align with human’s WoC [1]. For some questions, their WoC shows human-like WoC but not for others. You should acknowledge this limitation of LLM-based WoC when making your claims.

•	The baseline “5-shot best model” may be unfair. You should also control for the number of tokens in a generation. It is possible that the gain of social agents over the 5-shot baseline is merely due to the LLM using more tokens in their responses when given a persona.

•	Given that the temperature value controls the variability in response, which is core to WoC, you conduct a sensitivity analysis on temperature, rather than fixating on a single temperature value.

•	The structure of the paper is weird. Out of the 9 pages, the “result and discussion section” (section) only takes about one page,  and the subsection Section 3.1 “experimental setup” has nothing to do with results and I think should better belong to Section 2 “Setup”.

•	There is a lack of statistical tests for all the claims made in the paper. Figure 3 should ideally also show standard errors/confidence intervals to account for noise.

References

[1] Chuang, Y. S., Harlalka, N., Suresh, S., Goyal, A., Hawkins, R., Yang, S., ... & Rogers, T. T. (2024). The Wisdom of Partisan Crowds: Comparing Collective Intelligence in Humans and LLM-based Agents. In Proceedings of the Annual Meeting of the Cognitive Science Society (Vol. 46).

### Questions
•	Figure 4. Shouldn’t the y axis title simply be “probability density”? The current captions and notes are pretty lengthy.

•	In WoC literature, the standard way to aggregate is to use median rather than mean [2]. Why do you decide to use mean rather than median? Would the result and conclusion change if you use median? 

•	What is your view on “crowds within”, where repeated estimates from a single individual can also produce a WOC effect? This has been shown effective in both human [3] and LLM [4]. 

•	How do you decide the persona distribution in the demographic space? To match actual human behavior, shouldn’t you also ground persona to the actual human’s persona distribution?


References

[2] Francis Galton. Vox populi. Nature, 75:450–451, 1907.

[3] Edward Vul and Harold Pashler (2008). Measuring the crowd within: Probabilistic representations within individuals. Psychological Science, 19(7):645–647.

[4] Chuang, Y.-S., Narendran, S., Harlalka, N., Cheung, A., Gao, S., Suresh, S., Hu, J., & Rogers, T. T. (2025). Probing LLM world model: Enhancing guesstimation with wisdom of crowds decoding. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2025).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the strength and applicability of wisdom-of-crowds judgement in a variety of practical applications, ranging from tasks such as tweet engagement prediction to behavioral attribute classification (which ostensibly require more complex inference and theory of mind). The authors situate their evaluation in a cognitively inspired framework, classifying the difficulty of inference tasks based on their difficulty with respect to construal level theory (CLT). Detailed analysis of the kinds of tasks on which people and models are aligned–as well as the demographics that are aligned with models on this task—provide a comprehensive overview of the strengths and weaknesses of wisdom-of-crowds and persona-based inference in LLMs more generally. The authors are generally effective in comparing wisdom of crowds inference to informative human and domain-specific model benchmarks, and sufficiently justify the use of the method over domain-specific models.

### Strengths
I initially found myself asking several questions about the feasibility of this method (wisdom of crowds) over existing approaches, and the extent to which LLM judgments are supported by human behavioral data. These concerns were thoroughly answered by the evaluations comparing LLM results to both human judgements and existing domain-specific/trained methods. Moreover, the authors extensively justified the use of persona/demographic seeding over base model judgments (figure 4 provides a clear illustration of the benefits for demographic seeding in aligning with human judgements). Smaller parameter model performance in this task is another interesting finding which helps to support the feasibility of wisdom of crowds inference over existing methods, and the ablation analysis provides useful information on the limitations of small models in persona-based role-playing. Overall, the paper is well written and situated within the existing literature on wisdom-of-crowds judgment and an existing paradigm in cognitive science (CLT).

### Weaknesses
The authors claim that there are smaller improvements for cognitively demanding tasks (per CLT), “much as they are for humans”. However, I am not sure that this is supported by the results shown. Is there any kind of evidence to support the claim that for these tasks, higher level CLT evaluations are more difficult for people? What would the baseline be in this case?

### Questions
I am curious about the bimodal distribution for no-persona 4o prediction errors as seen in figure 4. Do the authors have any speculation as to why this model consistently over/underestimates human judgements? This is not a weakness of the paper per se–if anything it illustrates the strength of persona seeding in the effectiveness of wisdom of crowds judgements–but I’m curious how robust this bimodal effect is across models, and why this might be the case (it would be nice to see what this distribution looks like for the other models evaluated)

### Soundness
3

### Presentation
4

### Contribution
3
