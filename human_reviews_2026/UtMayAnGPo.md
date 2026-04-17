# CAReDiO: Enhancing Cultural Alignment of LLM via Representativeness and Distinctiveness Guided Data Optimization

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
As Large Language Models (LLMs) more deeply integrate into human life across various regions, aligning them with pluralistic cultures is crucial for improving user engagement and mitigating cultural conflicts. For this purpose, recently, different culture-specific corpora have been carefully curated, either synthesized or manually annotated. Nevertheless, inspired by culture theories, we identify two key challenges faced by these datasets: (1) Representativeness: These corpora fail to fully capture the target culture's core characteristics, causing insufficient cultural coverage with redundancy; (2) Distinctiveness: They struggle to distinguish the unique nuances of a given culture from shared patterns across other relevant ones, hindering precise cultural modelling. To handle these challenges, we introduce CAReDiO, a novel data optimization framework, which alternatively refines culture-sensitive questions and responses according to information-theoretic objectives in an in-context optimization manner, enhancing the cultural informativeness and distinguishability of constructed data. Extensive experiments on 15 distinct cultures demonstrate that CAReDiO can create high-quality data with richer cultural information and enable efficient alignment of small open-source or large proprietary LLMs with as few as 200 training samples, consistently outperforming previous datasets in both multi-choice and open-ended cultural benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a method for generating a dataset for cultural alignment. The algorithm proceeds as follows: 
1. start with an initial set of seed prompts; 
2. generate model responses for those prompts;
3. score the prompt-response pairs based on a combination of A. how relevant they are to a given culture (via an LLM judge) and B. how distinct the response is to responses associated with other cultures (measured by embedding similarity);
4. use an LLM to refine each question given the response.

The paper creates a dataset using this method. The paper also offers the theoretical insights supporting the relevance and distinctiveness ideas core to the prompting-based approach.

### Strengths
1. The paper tackles an important problem and offers a helpful conceptual framework to think about what is desirable in a dataset for cultural alignment. 
2. The paper formalizes these concepts mathematically, to add theoretical weight to the intuition.
3. The figures are well-done, and the results include human evaluation.

### Weaknesses
1. Theory-aside, the paper offers a prompting-based method for generating a dataset. In that regard, it is not quite clear to me what this method does that’s more than / justifies better performance than other methods. Making this distinction in the practical implementation clear would be helpful. 
2. If I understand correctly, there seem to be slight discrepancies between the mathematical objectives, algorithm box, and actual implementation based on prompting in the appendix. Making these more explicit would help with reader understanding. For instance, lines 247-254 seem to suggest inner loop optimization of the responses whereas the algorithm simply generates a response without additional optimization. And the notation for lines 4 and 5 in the algorithm box were a bit confusing as well as I could not find their precise definitions (e.g., with vs. without the subscript). 
3. While the authors have various results to measure "cultural alignment," some more clarity on the precise experimental setups (and how they differ) would help. For instance, what is the difference between Figure 4b and Table 3, e.g. GPT-4.1 on CB-Hard?

### Questions
1. Focusing on the prompting method itself, what makes the proposed method better than some of the others that generated previous datasets? For instance, how do we know that the performance gap is actually attributable to the central ideas on representativeness + distinctiveness in a generalizable way and not just better models, longer prompts, more iterations, etc.?
2. I noticed the results in main would using CardSet whereas the appendix used CaReDiO. Is there some significant difference in the experimental setup? I had originally thought that the comparison was of fixed datasets.
3. What were the seed questions?
4. Could the authors add more context around each of the different results / figures? Namely, how each is set up and measured?
5. Could authors clean up theorem 1 and the proof? It seems pretty informal as it currently stands.

If the authors could address these questions and concerns, I would be willing to raise my score.

### Soundness
2

### Presentation
2

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
This paper addresses the problem of cultural bias in LLMs and argue that existing cultural alignment datasets are insufficient because they fail on two key dimensions, which they ground in established culture theories: (1) Representativeness, or the failure to capture the consensus characteristics of a culture (the "emic" view) , and (2) Distinctiveness, or the failure to capture the unique nuances that differentiate a culture from other, similar ones (the "etic" view). To solve this, the paper proposes CAREDIO, a data optimization framework. CAREDIO is an in-context, LLM-driven process for generating and refining high-quality, culture-specific training data. It formalizes the two dimensions as information-theoretic objectives. The authors then use the approach to generate a new dataset, covering 15 cultures. Their experiments show that fine-tuning both small and large LLMs on as few as 200 samples from the dataset outperforms models trained on existing, often larger, cultural datasets.

### Strengths
1. The proposed method is grounded in clear mathematical formalisations, derived directly from information-theoretic quantities and supported with detailed proofs. The formulations for mutual information-driven sample selection and Jensen-Shannon-based culture divergence are actionable.
2. The quantitative dataset analysis is thorough. Table 2 compares diversity, information content, and similarity metrics across datasets, supporting the claim that proposed dataset is both more informative and more diverse. 
3. Human evaluation demonstrates the practical significance and that the approach moves beyond benchmark overfitting.
4. The proposed approach needs fewer than 200 training samples for strong performance. Figure 5 shows early high-value data boosts alignment more per sample, which highlights the method's efficiency claims.

### Weaknesses
1. The paper's core premise lies in the two dimensions, but the introduction does not clearly articulate them. Figure 1 is also unclear, leaving the reader with a poor intuition for what "representativeness" means independently of distinctiveness.
2. In Section 3.2, the “distinctiveness” objective (Eq. 4) relies on φ(y, x) as a probability that responses are not from other cultures, implemented via a “clustering-based distance measurement”. This is left quite vague: how is this classifier trained, what architectures/embeddings are used, and how sensitive are results to this choice? This is described only briefly.
3. Results demonstrate that the proposed approach is better, but no analysis shows how much each dimension (representativeness / distinctiveness) contributes to final model performance. 
4. Theoretically, the distinctiveness objective is vital for closely related cultures (e.g., Japan/Korea/China). However, Table 5–9 largely report aggregate or per-country results, with only sparse discussion of confusion between “neighbour” cultures. It would be informative to see confusion matrices or analysis on these closely related pairs, does CAREDIO really outperform baselines in making fine distinctions?

### Questions
1. See Weaknesses for some questions. 
2. Can you report results (or discuss limitations) for adapting the proposed approach to “zero-shot” or unseen cultures? What changes are needed to ensure transferability and avoid overfitting to the 15 present cultures?
3. How do you control for or report on annotator disagreement in human evaluations? Do any particular cultures or question types show especially high variance, and does that impact major conclusions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes CAReDiO, an in-context data optimization framework for cultural alignment. It alternates between generating and refining culture-sensitive questions and answers with two information-theoretic objectives: an information-gain objective (representativeness) and a culture-divergence objective (distinctiveness). Using CAReDiO, the authors build CARDSet for 15 cultures and report that small-scale fine-tuning can improve cultural benchmarks across multiple backbones. The framework itself comprises 1) information gain to reduce a model’s cultural uncertainty, 2) divergence to separate target from non-target cultures, and 3) an iterative refinement loop on data questions and responses. The paper evaluates on GlobalOpinionQA and WVS-style setups among others, and compares across a range of cultural benchmarks.

### Strengths
S1. The authors present a clear problem framing with two concrete challenges. The paper explicitly separates data quality into representativeness and distinctiveness and ties each to a learning objective, which is conceptually helpful.

S2. The defined objectives are grounded well with theory. Representativeness is operationalized as mutual information and distinctiveness is linked to maximizing a lower bound on generalized Jensen-Shannon divergence across cultures under classifier accuracy and non-overconfidence assumptions. The use of JS echoes prior robustness work employing multi-distribution JS for consistency. 

S3. CAREDIO is itself a practical, model-agnostic pipeline. The alternating refinement loop is simple to implement with existing LLMs and can leverage either the target backbone or a larger assistant model to synthesize data. 

S4. The new method is evaluated well across a breadth of baselines and benchmarks. The paper situates results against multiple cultural datasets (CB/Prism/GOQA/WVS) covering a range of tasks (multiple choice/survey/open-ended) and methods (CultureX, Role-Play).

S5. There are some risks of circularity due to the use of LLMs as judges. Both objectives estimate culture labels or divergences using LLM-based classifiers, which can encode the same cultural priors the method hopes to correct. Without strong human-grounded calibration, the approach risks amplifying pre-existing biases in the assisting model. The authors mostly mitigate this risk through the use of existing well-developed human preference datasets like PRISM and global value surveys like WVS and the further human evaluations on the final outputs of the CARDSet dataset and the final alignment step showing strong validation of the methods.

### Weaknesses
W1. Distinctiveness theory depends on strong, partly unverifiable conditions. Proposition 2 connects the learning objective to a lower bound on GJS only if a culture-membership classifier is sufficiently accurate and not over-confident. The paper does not clearly demonstrate these premises hold across cultures or provide diagnostics of the error bounds in practice.

W2. The benchmark construction offers some leakage concerns. Several compared datasets are built or augmented from WVS or related sources, and the evaluation also uses WVS/GlobalOpinionQA. This creates potential data leakage or style overlap that can inflate gains; the paper mentions the datasets but does not rigorously audit overlap or de-duplication. Durmus et al (2024) show how easy it is for LLMs to overfit survey-style probing.

W3. Another weakness of this work is that the technical novelty relative to prior synthetic pipelines is incremental. CulturePark also uses multi-agent LLMs to generate cross-cultural dialogues, CultureLLM uses WVS to seed augmentation, and PRISM and CulturalBench collect diverse human feedback to ground value statements. CAReDiO’s main novelty is the particular objective pairing, which is interesting but not obviously transformative without stronger ablations.

### Questions
Q1. Could the authors provide more details on how the role-playing baseline is developed. In section 3 a number of different types of role-playing are described but it is not clear how they are developed or what the final approach is.

Q2. Did the authors consider ablations on the independent optimization objectives of representativeness and distinctiveness?

Q3. The paper mentions that the english-dominant nature of training corpuses can bias a model towards western cultures. Was any investigation done into the impact of language improving cultural alignment?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a framework for dataset construction for cultural alignment, where it aims to optimize for two objectives (representativeness and distinctiveness). The framework entails generating cultural QA pairs, scored for whether the QA pair is representative of that culture through consensus of multiple LLMs and whether it is distinct from other cultural QA pairs by measuring JS divergence between the answer distributions. The authors theoretically ground their objective in cultural theories and create a dataset for 15 cultures, showing that fine-tuning on data generated using the proposed framework leads to better performance.

### Strengths
- Both objectives seem sound, are well motivated, and grounded in theory
- The proposed approach shows moderate improvements compared to the baselines
- The writing is clear
- In depth details and fine-grained results are provided in the Appendix

### Weaknesses
- For the representativeness optimization objective, a fundamental assumption behind the consensus elicitation approach is that multiple LLMs with the right conditioning can simulate a group of individuals from a target culture c. This isn’t obvious to me and would be something that needs empirical experiments to prove its efficacy, persona based prompting for survey simulation is still a research field. Thus, repeatedly using the Cultural Consensus theory for grounding seems a bit of a stretch. 
- The weaknesses or limitations of the paper are not discussed in sufficient detail. For instance, since all parts of the pipeline depend on LLMs already knowing or inferring something about the culture, a fundamental limitation would be the approach not working for cultures not well represented in current LLMs. This is a core limitation that should be discussed in the paper. 
- Presentation could be improved
  - A lot of the space in the paper is given to outlining the framework and grounding it in theory. I appreciate the effort the authors put into grounding the approach but this results in framework becoming concrete much later in the paper, which is worse for readability. 
  - Several important details necessary for understanding the paper in depth are in the Appendix.
  - Figure 2 is not visually representative of the framework, the reader isn’t walked through the Figure in text. Clear examples or a better figure would aid the reader in clearly understanding the iterative data generation and optimization process.

### Questions
- Minor:
  - WVS is cited with multiple references in different parts of the paper (Xu et al., AlKhamessi et al., Tao et al.), none of which correspond to the actual reference: Haerpfer et al. (2022).
- Suggestion:
  - Could shorten the framework introduction and current motivation, propositions which are quite verbose, make the framework concrete earlier in the paper with the dataset, add a discussion section at the end.

### Soundness
3

### Presentation
2

### Contribution
3
