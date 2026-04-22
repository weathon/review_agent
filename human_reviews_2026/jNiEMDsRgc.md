# Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 8, 6, 8

## Abstract
We propose a method for evaluating the robustness of widely used LLM ranking systems---variants of a Bradley--Terry model---to dropping a worst-case very small fraction of preference data. Our approach is computationally fast and easy to adopt. When we apply our method to matchups from popular LLM ranking platforms, including Chatbot Arena and derivatives, we find that the rankings of top-performing models can be remarkably sensitive to the removal of a small fraction of preferences; for instance, dropping just 0.003% of human preferences can change the top-ranked model on Chatbot Arena. Our robustness check identifies the specific preferences most responsible for such ranking flips, allowing for inspection of these influential preferences. We observe that the rankings derived from MT-bench preferences are notably more robust than those from Chatbot Arena, likely due to MT-bench's use of expert annotators and carefully constructed prompts. Finally, we find that neither rankings based on crowdsourced human evaluations nor those based on LLM-as-a-judge preferences are systematically more sensitive than the other.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the statistical robustness of LLM ranking systems based on the Bradley-Terry (BT) model, such as Chatbot Arena . The central research question is whether the top rankings on these leaderboards are stable or if they can be altered by removing a very small, worst-case fraction of the preference data. The authors propose a computationally efficient method, adapted from the Approximate Maximum Influence Perturbation (AMIP) literature, to identify the most influential preferences that, when dropped, are most likely to cause a ranking flip.

The authors show that popular LLM leaderboards are surprisingly fragile. Most strikingly, removing just two preference pairs, which account for only 0.003% of the data, was enough to change which model ranked first in Chatbot Arena. The analysis also shows that MT-Bench, which relies on expert reviewers and carefully chosen prompts, is much more stable than large crowdsourced platforms. Overall, the paper warns that rankings on many leaderboards may be less reliable than they appear, and that small score differences shouldn’t be overinterpreted.

### Strengths
**Originality**: The originality of this work lies in its novel research question and its surprising findings, rather than in the invention of a new statistical method. As far as I know, it is the first paper to systematically apply the concept of worst-case data-dropping robustness to the domain of major LLM leaderboards.

**Quality**: The paper shows high quality through its careful and valid application of statistical methods to real-world data. They verify the ranking flip by refitting the BT model. The use of public, real-world data enhances the work's credibility.

**Clarity**: The paper’s main finding is clearly reflected by its title. Figure 1 immediately conveys the core message to the readers. The results in Table 1 are clear and easy to interpret, which clearly supports the main point.

**Significance**: This work is a critical piece of scientific auditing for the entire LLM community. The finding that top rankings can be fragile has the potential to change community behavior, encouraging more skepticism and a demand for more robust evaluation practices.

### Weaknesses
Limited depth of explanation: The paper hypothesizes why MT-Bench is more robust (expert annotators, curated prompts) and why rankings are fragile (small BT-score margins), but these remain hypotheses. A more controlled experiment to disentangle these factors would be needed for a definitive causal claim.

Lack of methodological novelty: The weakness is that the core algorithm (AMIP) is adapted from prior work in statistics. The contribution is in the application and the discovery, not the invention of a new technique.

No solution to the problem: The paper is only a diagnostic tool, not a solution to the fragility of crowdsourcing rankings.

No average-case behavior: The paper focuses only on the worst case, not the average case (e.g., dropping random pairs of preferences), which I think is closer to the actual threat to these rankings, especially for crowdsourcing rankings.

### Questions
1. You qualitatively analyze the two prompts that flip the top ranking on Chatbot Arena and note that they involve "anomalous losses" against much lower-ranked models. Do you have a hypothesis about the nature of these prompts? Are they particularly tricky, ambiguous, or perhaps simple enough that the distinction between a top model and a lower-ranked model's response is negligible, leading to noisy human preferences?

2. Your findings suggest that small margins in Bradley-Terry scores are associated with reduced robustness. Could the confidence intervals of the BT scores themselves be used as a direct, and perhaps even simpler, proxy for ranking stability without needing to run your AMIP-based analysis?

3. Given your findings, what practical recommendations would you give to platforms like Chatbot Arena? Should they regularly run robustness checks like yours, or change the way ties are handled in the BT model?

4. When looking into the data, I also noticed that platforms like Chatbot Arena have highly unbalanced numbers of competitions across different model pairs (e.g., some model pairs have thousands of preference pairs, while many others have few or no pairs). Do you think this can explain their fragility? One extreme case could be removing a preference that favors the weak model over the strong model, even though the preference is the only one between the two models.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how robust leaderboard rankings of LLMs are when based on human or AI preference data aggregated through the Bradley–Terry (BT) model. In particular, the authors consider an "intrinsic" notion of robustness (as opposed to, say, adversarial manipulation by a third party of the voting), where the minimal removal of some pairwise comparisons can alter the ranking of models---what the authors call top-k data dropping robustness in Definition 2.4. Building on the equivalence between data removal changing the top-k set and data removal changing the pairwise order between a top-k model and a model outside the top-k set (Proposition 3.1), the authors propose an approach to finding candidate data points whose removal can change the ranking. To that end, in the Appendix A, BT inference is formulated as logistic regression, and the authors use a first-order AMIP expansion to first identify the specific preferences most influential to ranking stability and then verify their effect through a recomputation of the ranking.

Applying this method to several open evaluation platforms, including Chatbot Arena, MT-Bench, Search Arena, Webdev Arena, and Vision Arena, the authors find that rankings can be extremely sensitive: on Chatbot Arena, removing just two out of nearly 60,000 human evaluations (around 0.003%) flips the top-ranked model. By contrast, MT-Bench shows much higher robustness, requiring the removal of several percent of data to change the top ranking. Human and LLM-as-a-judge evaluations exhibit comparable sensitivity levels.

### Strengths
- The general contextualization and introduction of the problem in Sections 1 and 2 are very clear and accessible. The authors make the paper easy to follow and introduce the necessary tools when needed (i.e., the main idea behind BT). The data-dropping setup and notation are also clearly communicated. Similarly, the theoretical results in the main text (most notably Prop. 3.1), while simple, are sound and well explained.

- The experiments conducted by the authors are comprehensive and effectively demonstrate that small percentages of dropped comparisons can have a significant impact on LLM rankings. This includes different evaluator models, different types of evaluators (human/LLM), and different dataset categories, providing strong support for the authors’ claims.

- The paper addresses a timely issue that can have important implications for LLM evaluation and practitioners in the field. The experimental results make the authors’ findings relevant.

### Weaknesses
Perhaps the key point I would like to inquire about is how the authors’ approach relates to other uncertainty quantification methods in the BT model. In particular, there are classical references that quantify uncertainty in BT coefficients, both in Bayesian and frequentist settings, for instance:

- Gao et al., "Uncertainty quantification in the Bradley-Terry-Luce model"
- Hunter "MM algorithms for generalized Bradley–Terry models"
- Leonard "An Alternative Bayesian Approach to the Bradley-Terry Model for Paired Comparisons"
- Negahban et al., "Rank Centrality: Ranking from Pair-wise Comparisons"

Such methods (including the estimated errors reported in platforms like LMArena, based on bootstrapping) are not discussed in the paper in detail and, a priori, also seem capable of quantifying the sensitivity of the ranking to data removal. Is this intuition correct? If so, I believe it would be helpful to explain how the authors’ methods relate or compare to such uncertainty quantification approaches.
In this direction, and since the authors conclude that the Chatbot Arena rankings are not robust, it would be interesting to interpret these results in light of the confidence intervals  reported by LMArena, which in most cases already suggest that the top-1 model is not estimated accurately.

Moreover:

- Interpretability and results. The authors claim that their method allows “inspection of these influential preferences.” While I agree that this is literally accurate, the authors’ method identifies a set of comparisons to drop, and one can only verify their influence after recomputing the BT ranking. In that sense, the interpretability benefits are limited.
In addition, I am having trouble understanding how the authors interpret the results regarding MT-Bench versus the rest of the LLM Arena data. The authors claim that the robustness of the MT-Bench ranking is partially due to the fact that “MT-Bench consists of 80 carefully designed multi-turn questions intended to differentiate models on core capabilities,” whereas LLM Arena prompts are less specific. Based on this reasoning, I would actually expect MT-Bench prompts to be more relevant for ranking models, and hence the ranking to be more sensitive to removals. However, the authors claim the opposite. A clarification here would be useful.

- Implementation. The authors do not provide a concrete algorithm in the main text summarizing their approach. I find this makes the implementation somewhat difficult to follow and would benefit from further clarification. In line 280, the authors provide a textual description of the process; however, this description seems to omit key information, such as how the method selects the reweighting vectors. Moving such details from the appendix to the main text could therefore be beneficial.
As a consequence of omitting this information, some of the authors’ claims are confusing. For instance, in line 188: “Notice that both Equation (5) and Equation (7) are nontrivial to directly verify; to check directly, we have to test out dropping all possible small-fraction subsets of the arena, a combinatorial operation that is computationally intractable in practice.” The authors could explain more explicitly how their method addresses this issue.

- Performance of the AMIP.  To better understand the efficiency of the authors’ method---since it requires recomputing the BT inference for the chosen vector w---it would be beneficial to show how this search method for w performs. For instance, the authors could report, in the main text, the fraction of vectors w that correctly lead to a different ranking.

### Questions
I believe that while most of the technical machinery and setup of the paper are clear, the overall presentation could be made more concise. In particular, some presentation choices feel redundant:
- On line 269, the authors write, “For a candidate pair of players, (i, j), recall that we assumed without loss of generality…”. This condition was already introduced in line 247. Twenty-two lines seem close enough for this repetition to be unnecessary.
- Equations (8) and (5) are exactly the same. While I understand that the context differs---since Eq. (5) refers to the two-player case---this raises the question of whether that case should have been treated separately at all.


Some other minor comments/suggestions:
- As far as I understand, Figure 2 is not an original result of the paper but rather summarizes open-access information. However, it currently occupies about one-third of a page in the main text. Perhaps replacing it with a smaller table or moving it to the appendix would be more appropriate.
- In line 193/194, I believe "Theorem 2.4" should read "Definition 2.4"
- In line 211, the sentence “Then, it must be that S is the top-k set, or S = K(w)” reads as if there were a dichotomy, whereas I believe the authors simply want to clarify a single condition---that S is the top-k set. Avoiding “or” would make this clearer.
- In line 233, the authors use the term “teams” for the first time, but it is unclear why this terminology is introduced, given that it is not used again later.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper focuses on the sensitivity of LLM ranking platforms based on crowdsourced pairwise comparisons of their responses. Specifically, the authors consider a ranking setting based on the Bradley-Terry model, and they develop a method to identify a small number of pairwise comparisons whose absence would alter the top-k set in a ranking. They then use this method to analyze the sensitivity of the resulting rankings using comparison data from multiple popular crowdsourced platforms such as Chatbot Arena and MT-bench. The results in the paper indicate that removing a very small number of pairwise comparisons is sufficient to alter the resulting rankings.

### Strengths
The main strengths of the paper are as follows:
1. It studies a timely problem, as benchmarking platforms such as LM Arena become the industry standard for evaluating the performance of large language models. Hence, the results will be of interest to the ICLR community.
1. It develops a relatively simple method building upon prior work in statistics for identifying small sets of pairwise comparisons that significantly affect the resulting model rankings.
1. Its analysis reveals a surprising insight that a very small number of pairwise comparisons (far less than 1%) is sufficient to lead to different conclusions regarding which model is the top performer according to thousands of pairwise comparisons by human users (Table 1)

### Weaknesses
I believe that the paper has some room for improvement in terms of its presentation of certain definitions and results and its experimental evaluation. Specifically:
* The definitions in page 4 are somewhat confusing. It is unclear why there is a need to introduce Definition 2.2 regarding top-1 robustness in two-player arenas, as it is immediately followed by the more general definition of top-k robustness in arenas with more than two players. Definition 2.2 doesn't seem to be used anywhere. Moreover, this definition seems identical to what the authors call "pairwise robustness" of scores in page 5, although there it is not presented explicitly as a definition. All these redundant definitions just make it harder for the reader to focus on the core flow of the presentation.
* The method that the authors introduce evaluates the robustness of a top-k set in a ranking by separately evaluating the robustness of pairwise model orderings according to their estimated scores. This is explained in lines 238-242, but it is not explicitly proved that the two robustness checks are equivalent. Instead, the authors provide a proof for Proposition 3.1, which seems rather obvious (i.e., that a set of size k whose values are greater than all the values not in that set is the top-k set) and doesn't seem like a proof worth including in the main body of the paper. I believe it would be helpful for the presentation if the authors could restructure and clarify that part of the paper a bit.
* My understanding is that the method introduced by the authors is valid when it finds a set of pairwise comparisons that would alter a ranking (i.e., the ranking would indeed be different), but I think it can still suffer by false negatives due to the fact that the method solves a combinatorial problem approximately via continuous optimization. In other words, if the method fails to find a set of pairwise comparisons that alters the ranking, this does not necessarily mean that no such set exists and the ranking is robust. Could the authors comment on that? For example, can we say with certainty that the 3 datasets in gray in Table 1 are truly robust? It would have also been interesting to see some experiments showcasing failure cases of the method itself, as the current experimental evaluation focuses solely on the sensitivity of rankings and does not present any analysis of the method itself.
* I think the paper would have been richer if the authors performed a more thorough qualitative analysis of the pairwise comparisons that their method identifies as critical for a ranking. For example, I found the fact that removing only 2 data points was sufficient to change the winner in the Chatbot Arena ranking very intriguing. However, other than the fact that these 2 data points corresponded to battles where the top performing model lost by a model placed far lower in the ranking, there aren't any further insights provided in the paper that could inform how to design more robust ranking methods in the future. For example, was there any pattern in the rating behavior of the users who submitted those critical comparisons? Is the sensitivity of rankings always affected largely by such "abnormal" pairwise comparisons? Is there any systematic way to filter them out?

### Questions
I don't have additional questions other than what I already brought up under "Weaknesses". It would be great if the authors could elaborate on those points in their rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
