# Probability Distributions Computed by Autoregressive Transformers

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Most expressivity results for transformers treat them as language recognizers—devices that accept or reject strings—rather than as they are used in practice: as language models that generate strings autoregressively and probabilistically. We characterize the probability distributions that transformer language models can express. We show that making transformer language recognizers autoregressive can sometimes increase their expressivity, and that making them probabilistic can break equivalences that hold in the non-probabilistic case. Our overall contribution is to tease apart what functions transformers are capable of expressing in their most common use case as language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper advances the theoretical study of the expressivity of transformer language models beyond Boolean recognizers, by characterizing their expressivity as probabilistic, autoregressors. The paper focuses on Unique-Hard Attention Transformers (UHATs) and connects them with Linear Temporal Logic (LTL) and counter-free Deterministic Finite Automata (cfDFA) paradigms. Overall, the paper proposes several expressivity results for these paradigms, characterizing when different modeling choices (classifier vs. autoregressive paradigms, Boolean vs. real semirings) do or do not change the set of weighted languages a model can represent.

### Strengths
1. The theoretical results are meaningful and timely, which advance the theoretical understanding of the expressivity of transformer-based models. 


2. In my view, the main strength of the paper is its focus on the probabilistic autoregressive regime, which is more aligned with how language models are used in practice, and thus represents a more meaningful case of study in comparison to previous works.  


3. The paper characterizes equivalences in terms of language recognition of UHATs with known classic models (LTL, DFA…). Furthermore, describe different levels of expressivity under different scenarios or configurations.

### Weaknesses
1. These results hold for unique-hard attention with no position embeddings. This diverges from common configurations of language models (which include soft attention and positional encodings). Furthermore, previous works such as (Li and Cotterell, 2025) focus on soft attention, resulting in a less-idealized setup (closer to real-world transformers). I believe the paper would significantly benefit from a discussion on which results will hold or not (and why) under these assumptions.


2. Some results are somewhat incremental, based mostly on the results of (Yang et al., 2024) or (Li and Cotterell, 2025). While I value the theoretical results, I believe that this limits the contribution of this particular work. 


3. While the paper adequately describes its goals in Section 1, its novelty and the main contributions are unclear or hard to follow. I would recommend the authors to reinforce this part, in order to emphasize which are the novel results / extensions of this particular work (with respect to previous related works), and the relevance of its contributions.


4. The paper lacks a discussion of potential future research directions building on these findings.

### Questions
1. Which results extend to soft attention? Similarly, which are the implications of positional embeddings? I would recommend a discussion on these matters.


2. Related to my previous point, recent works show that hard attention can be simulated using soft attention through temperature scaling, e.g., see (Yang et al, 2024). Based on this, will some of your results hold under soft attention? Can these findings bridge the results from this paper to soft-attention regimes?

3. The Related Work section seems shallow in its current state. This is mostly because most of the closest works are described in Section 1. I would encourage the authors to reorganize Sections 1 and 2.


4. Furthermore, some recent works are missing in the literature review (e.g., see the list below). Some of these missing works address similar scenarios or goals. For example, (Yang et al, 2024) also examines several subclasses of languages recognized by hard-attention transformers, which can be defined in variants of linear temporal logic. Please clarify the similarities or differences with these works.
 
- Yang, A., Strobl, L., Chiang, D., & Angluin, D. (2024). Simulating hard attention using soft attention. arXiv preprint arXiv:2412.09925.


- Hao, Y., Angluin, D., & Frank, R. (2022). Formal language recognition by hard attention transformers: Perspectives from circuit complexity. Transactions of the Association for Computational Linguistics, 10, 800-810.

- Barceló, P., Kozachinskiy, A., Lin, A. W., & Podolskii, V. (2023). Logical languages accepted by transformer encoders with hard attention. ICLR 2024.




5. Please define the acronym UHAT as “unique-hard attention transformers” in Page 1, for the sake of clarity to non-familiar readers.


6. Relevant results from these works focus on counter-free automata. I wonder whether this limits impact on certain cases that exhibit periodicity. A brief discussion of what is left outside the counter-free setting and whether partial extensions or approximations are possible would be helpful.


7. While the paper focuses on expressivity results, can something be said about “learnability” (i.e., sample complexity, efficiency…)?  


8. Sections 6.2 and 6.3 lack explicit UHAT results. I believe that the paper might benefit from a more clear discussion on how those results contribute to the study of UHAT expressivity.


9. Please include a forward-looking discussion of open problems and next steps.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the expressive power of transformers as probabilistic autoregressive models in the context of formal language theory. Existing theoretical work mostly focuses on transformers as language recognizers (Boolean setting). Here, the authors analyze the probability distributions computed by hard-attention transformers when used as language models, providing equivalence results and separation results between Boolean classifiers, probabilistic classifiers, and probabilistic autoregressors. The study also connects these models with temporal logics and weighted automata, clarifying how expressivity shifts across these settings.
The paper is rigorous, well-structured, and contributes to closing a gap in theoretical understanding of transformer expressivity in autoregressive scenarios.

### Strengths
* Tightly written theoretical work with clear formal contributions.
* Addresses a meaningful gap in theory: expressive power of transformers as generative language models.
* Strong formal rigor, with proofs and precise definitions throughout the paper.

### Weaknesses
* **Purely theoretical:** While the theoretical contributions are solid, there are no experimental results nor concrete applied examples to illustrate relevance for real-world transformer LMs. Given the conference venue, this limits perceived impact.
* **Accessibility:** The paper assumes familiarity with temporal logics, weighted automata, and semirings. This is appropriate for a specialized logic or theoretical CS audience, but is demanding for the general machine-learning readership at ICLR.

Minor Comments:
* There are small repetitions in early sections (e.g., “we then” in lines ~77-82, “language” in lines ~90-93) that make the text slightly repetitive.
* The notion of “state” is only clarified around Section 4-5. Since “state encoder” is central, I suggest introducing more intuitively what a “state” represents earlier in the introduction.
* Typo in Section 6.1: $\tau_1$ should be $\tau_2$.

### Questions
I do not have any particular questions for the authors.

### Soundness
3

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
3

### Summary
This work studies the expressive capacity of Transformers when used as probabilistic language models rather than classifiers. To do so, the authors analyze models along 2 axes: 1) Classifiers vs Autoregressors 2) Boolean weights vs (positive) Real-valued weights. The authors show the following:
- UHATs, LTL and cfDFAs have equivalent state encoders (Thm 6.1) They use this to show that UHATs, LTLs and cfDFAs as classifiers (or autoregressors) define the same weighted language (Corollary 6.1)
- LTL classifiers can only output finitely many distinct weight values (Prop 6.1). This implies that LTL classifiers cannot express the language $(\frac{1}{2} a)^*$, which can be expressed by autoregressors (Corollary 6.2).
- LTL classifiers and autoregressors are equivalent (for the right subsets of operators) (Thm 6.2)
- In the case of autoregressors, cfNFAs are more expressive than cfDFAs (Prop 6.2)
- Boolean autoregressors are more expressive than classifiers when key operators are missing (Prop. 6.3)
- The expressivity gain of autoregression is limited; $(aab)^*$ is not definable by an LTL regressor (Prop 6.4 & Thm 6.3)

I think this is a good paper with solid theoretical contributions. However, I find the style in which it is written makes it hard to quickly extract the main insights and results. I have given a 6 and will increase my score to an 8 if the authors address points 3, 4 and 7 in the Questions/Comments section.

### Strengths
- Angle is novel and interesting. I agree with the authors that there is a lack of literature on the topic of language *modelling* with Transformers.
- This work improves our understanding of the interplay of expressive power between i) Boolean and Real-valued models and ii) classifier and autoregressive models. Given the equivalencies drawn by the authors, the theoretical results have deep implications about many families of models.
- The authors provide extremely rigorous proofs and reductions. From a technical perspective. The theoretical approach taken is non-trivial and is in itself a significant contribution to theoretical research. Although I am knowledgeable about the Transformer expressivity literature, I am not well-versed in temporal logic, thus I was not able to fully check all the proofs pertaining to this in detail.

### Weaknesses
- Although complete and precise in its writing style, I find the paper is very dense and not written in a way where key insights are easy to find/extract. See comments for actionable feedback.
- The paper has no experimental validation of theoretical claims it makes. It would be nice to at least have minimal experiments to support the results.
- This work makes few connections to practical settings, such as how their claims might account for empirical shortcomings of LLMs, and it does not discuss the implications of their results for well-known algorithmic tasks. Stronger statements of the form “UHAT Autoregressors belong to class X and therefore cannot perform task Y from a broader class” could improve the paper.

### Questions
1. [1] and [2] are highly relevant and are not cited in the related works. They investigate related topics namely how Transformers can express weighted/probabilistic automata/grammars. The latter paper also works on a notion quite similar to what you define as "State Encoders" through a notion they define as "Simulation".
2. Do the authors see any relationship between their work and work on "Generation in the limit"[3]? This could equally be an interesting direction for discussion.
3. Could the authors put a section "Contributions" with bullet points or something similar in the introduction? It was hard to parse what was done in the paper vs previous work when reading.
4. It would be beneficial to clarity to add brief proof sketches in the main text for (at least) the most technical theorems instead of simply deferring to the appendix.
5. I quite like Figure 1 and think it does a good job summarizing the results. However, the upwards arrows are hard to parse in terms of direction of inclusion, it was not immediately obvious for me what it meant. I feel putting a $\subseteq$ or similar might be clearer.
6. I think it could also be helpful to have a table summarizing the main results based on assumptions made, e.g. with columns "Thm/Prop number" "Semiring" "Model Type" "Main Finding"
7. Could the authors add a section discussing implications of their results to practice and to specific task families (as mentioned in the "Weaknesses" section)?


[1] Zhao, H., Panigrahi, A., Ge, R., & Arora, S. (2023). Do transformers parse while predicting the masked word?. arXiv preprint arXiv:2303.08117.

[2] Rizvi-Martel, M., Lizaire, M., Lacroce, C., & Rabusseau, G. (2024, April). Simulating weighted automata over sequences and trees with transformers. In International Conference on Artificial Intelligence and Statistics (pp. 2368-2376). PMLR.

[3] Kleinberg, J., & Mullainathan, S. (2024). Language generation in the limit. Advances in Neural Information Processing Systems, 37, 66058-66079.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a number of theoretical results concerning the expressivity of Transformer models, in particular in relation with counter-free, weighted Deterministic Finite Automata (DFA) and Nondeterministic Finite Automata (NFA). The central innovation consists in considering a setup that is closer to real-world usage of transformer models. In fact, while the results are still largely limited to Unique Hard Attention Transformers (UHAT), but the authors consider their use a autoregressive token generator (language models), rather than just a Boolean classifier (for language recognition). The proof techniques rely on establishing a mapping between UHAT and Linear Temporal Logic, and then proving results for LTL. The paper shows that, while some equivalences established for the Boolean classifier setting extend to the autoregressive one, other equivalences instead break.

### Strengths
I should start by cautioning that my expertise with the topics discussed here is limited, though I agree that understanding the expressivity of transformer models is an important research direction, given their role in powering LLMs and other modern AI advancements, even close to a decade after their initial introduction.

In my opinion, the strongest merit of the work is that of showing how results obtained in the Boolean classifier setup do not necessarily port to the autoregressive setup, which is indeed closer to how transformers are used, at least in LLMs. A natural next step would consist in applying the same treatment to SoftMax Attention Transformers.

### Weaknesses
While the main result is in my opinion that of showing a discrepancy between the Boolean classifier and autoregressive setup, most of the paper is devoted to proving that many equivalence results hold in both setting, thus somewhat reducing the novelty of most contributions in the work. The broke equivalence also seems to apply rather peculiar configurations (subsets of LTL).

Additionally, while considering the autoregressive setup is a step toward making the analysis more practically relevant, the work still introduces several simplifications over transformers as they are actually employed in the real world.

A few minor points:

* Some acronyms are introduced considerably later than when they are used, which decreases readability
* At lines 119-124, it would be worth point out the connection between normalized weighted languages and discrete probability distributions
* While definition 4.2 considers multiple possible suffixes, several sections in the paper appear to focus on estimating just the next token distribution, which creates confusion while reading

### Questions
* How challenging would it be, in your estimate, to adapt your analysis to SMATs?
* The subsets of LTL that cause the equivalence to break appear to be linked to specific operators: does that provide insights in terms of either expressiveness of complexity?

### Soundness
3

### Presentation
2

### Contribution
2
