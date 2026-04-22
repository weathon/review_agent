# On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Humor is a commonly used and intricate human language in daily life. Humor generation, especially in multi-modal scenarios, is a challenging task for large language models (LLMs), which is typically as funny caption generation for images, requiring visual understanding, humor reasoning, creative imagination, and so on. Existing LLM-based approaches rely on reasoning chains or self-improvement, which suffer from limited creativity and interpretability. To address these bottlenecks, we develop a novel LLM-based humor generation mechanism based on a fundamental humor theory, GTVH. To produce funny and script-opposite captions, we introduce a humor-theory-driven multi-role LLM collaboration framework augmented with humor retrieval (HOMER). The framework consists of three LLM-based roles: (1) conflicting-script extractor that grounds humor in key script oppositions, forming the basis of caption generation; (2) retrieval-augmented hierarchical imaginator that identifies key humor targets and expands the creative space of them through diverse associations structured as imagination trees; and (3) caption generator that produces funny and diverse captions conditioned on the obtained knowledge. Extensive experiments on two New Yorker Cartoon benchmarking datasets show that HOMER outperforms state-of-the-art baselines and powerful LLM reasoning strategies on multi-modal humor captioning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new framework for generating humorous image captions, called HOMER. It consists of three LLM-based components. First, the Extractor derives a general description from the image and identifies objects that contradict normal expectations as the basis for humor generation. Next, the Imaginator expands the humorous space both deeply and broadly by constructing imagination trees using humor-relevant retrieval, then pruning them based on humor-relevance scores. These results are finally fed into the Caption Generator to produce the desired humorous caption.

### Strengths
- Compared to previous work, this paper breaks away from the inherent humor mechanisms of LLMs. Instead of enhancing the model's ability to generate humorous image captions through prompt engineering or task-specific fine-tuning, it innovatively designs three distinct modules to complete the humor caption generation progressively. I find the overall pipeline design to be reasonable and well-structured.
- The design of the Imaginator is quite novel, as it takes into account both the depth and breadth of imagination. Moreover, the pruning strategy based on humor relevance scores is also fairly reasonable.

### Weaknesses
- The construction of the imagination tree relies too heavily on the joke database. The construction process of the joke database is not mentioned in the paper.
- Maybe the range of data that this pipeline can handle is quite limited. (Details see Questions)

### Questions
- The quantitative experimental results are quite thorough, but the presentation of the generated humorous captions seems rather sparse. From the example shown in Figure 2, I feel that some of the generated captions don't quite match the original images—the humor may be too far-fetched. Could you provide more examples of the generated captions?
- I have some doubts about the generalizability of your pipeline. If there are no particularly abnormal elements in the image, is this method still applicable? And how is the humor basis determined in such cases?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework for humour generation based on the General Theory of Verbal Humour. This framework is composed of three modules: 1. conflicting scripts extraction to find the out-of-expectation target with both global and local views, 2) humour-relevance retrieval to construct the hierarchical imagination tree from their collected joke database, and 3) final script generation. The experimental results on two humor generation datasets validate the effectiveness of this framework.

### Strengths
1. Grounding on the GTVH humour theory, it generates humour from conflicting elements in a novel way. 
2. And considering both the similarity and the conceptual opposition. Utilising an external joke database for "humour" element retrieval.

### Weaknesses
1. Gaps between the figure illustration and the paragraphs:
1.1. Figure 1 lists 1. visual understanding, 2. humour understanding, 3. humour imagination, and 4. stylistic expression, and they are marked with checks and crosses. This leads the reader to expect an assessment along these dimensions, while the study provides no qualitative or quantitative evaluation of them.

2. Confusing Delivery
2.1 abstract, line 13-17.
2.2 The last line on page 1.
2.3 line 100, "The goal of tackling this task is to assess ..."
2.4 "extract conflicting script", e.g., line 103. Where are the scripts to extract? It appears that the authors are referring to the "strange" elements in the image, which contradict the norm.
2.5 line 290-291, "To assess the reliable win rate ...  use Pass rate (pass@K) to measure the win rate "

3. The paper's theoretical grounding, fundamental humour theory, GTVH, should be introduced more. Only line 139 mentions it.


typos
line 82, ";" should be '.'
line 190, redundancy ")" and "}"

### Questions
1. Line 38, are there other types of images for humour generation, besides cartoons?
2. Figure 2, Where is the direct link between Coffee and Sugar (demonstrated in the Global view)?
How to decide the Coffee node rather than the Cup node? In this case, it should be Cups, as it could be coffee, tea, or other drinks. And the conflicting elements are the sizes of the cups. How to ensure that these terms make sense? Could you please explain the humour in case 1? As for me, case 2 is humorous. 
3. Term-2, It seems this equation would find those common words with a high humour-frequency score. What about trying the TF-IDF idea?
4. Line 262, Are all the candidate targets utilised or integrated in the final caption? How are they connected with the final caption?
5. Table 1, 6044, why are there so many captions?
6. Table 2: How do you convert the global ranking into the pairwise ranking for the Human-in-AI dataset?
7. line 290-291, How to calculate the win rate? Which competes against which?
8. Human evaluation: How many data points are evaluated? They are not stated in the appendix, either.

### Soundness
2

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
The paper proposes HOMER, a novel framework for generating humorous captions for cartoon images. The approach is grounded in the General Theory of Verbal Humor (GTVH) and employs three coordinated LLM-based roles: (1) a conflicting-script extractor that identifies incongruities in images, (2) a hierarchical imaginator that builds imagination trees through LLM associations and humor-relevance retrieval from a joke database, and (3) a caption generator that produces humor based on the extracted knowledge. The method is evaluated on New Yorker Cartoon datasets, showing improvements over baselines.

### Strengths
1. **Theoretically grounded approach**: Unlike prior work that relies on generic prompting or self-improvement, HOMER explicitly incorporates established humor theory (GTVH), providing better interpretability and control over the humor generation process.

2. **Novel hierarchical imagination mechanism**: The combination of deep-pattern LLM associations with broad-pattern humor-relevance retrieval is creative and well-motivated, expanding the creative space while maintaining relevance.

3. **Strong empirical results**: The paper demonstrates consistent improvements (average ~7% on pass@1) across multiple baselines and LLM backbones, with comprehensive evaluation including human studies showing statistical significance.

4. **Thorough experimental analysis**: The paper includes extensive ablations, hyperparameter analysis, harmful content detection, and human evaluation with appropriate inter-rater agreement metrics.

5. **Clear modular design**: The three-role framework provides good interpretability and allows for targeted improvements to individual components.

### Weaknesses
1. **Questionable humor quality in examples**: The showcased example "HR says we can expense a cow now" doesn't seem particularly funny, raising concerns about whether the quantitative improvements translate to genuinely humorous outputs. More compelling examples would strengthen the paper's claims.

2. **Limited scope**: The method is only evaluated on cartoon caption generation. It's unclear whether the approach generalizes to other humor formats (jokes, puns, situational comedy, etc.).

3. **Computational overhead**: The method requires multiple LLM calls (7 total per caption generation according to Table 7) 

5. **Limited human evaluation**: Only 12 raters participated

6. **Complexity concerns**: The framework involves many components (WordNet for semantic relations, multiple scoring functions, GTVH knowledge resources) that may be difficult to reproduce or adapt.

### Questions
1. **Generalization**: How does HOMER perform on other humor generation tasks beyond cartoon captioning? Have you tested on joke completion, pun generation, or conversational humor?

2. **Example quality**: Could you provide more compelling examples that better demonstrate the humor quality? The current examples seem to lack the wit typically associated with New Yorker captions.

3. **Ablation on retrieval database**: What happens to performance as the joke database size varies? How sensitive is the method to the quality/style of jokes in the database?

4. **Failure analysis**: What types of images or situations does HOMER struggle with? Can you provide a systematic analysis of failure modes?

5. **Comparison fairness**: How many inference calls do the baseline methods use? Is the comparison fair given HOMER's computational overhead?

6. **Style control**: Can HOMER generate different styles of humor (e.g., dark humor, slapstick, wordplay) by modifying the retrieval database or prompts?

7. **Zero-shot performance**: How does HOMER perform without the joke retrieval component, relying only on the GTVH-guided structure?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
HOMER is a humor-theory–guided, three-role LLM for New Yorker-style cartoon captions that (1) extracts script oppositions, (2) builds retrieval-augmented “imagination” trees, and (3) generates captions; on two cartoon benchmarks it beats strong prompting baselines, with ablations showing each role matters.

### Strengths
1. Clear multi-role architecture grounded in humor theory; maps situation, script opposition, target, narrative strategy, and language to concrete modules 
2. Consistent SOTA gains across datasets/groups and base models; effect sizes reported
3. Thorough ablations: module inclusion matrix (Tbl. 4) and score-term ablations (Fig. 3). 
4. Toxicity audit across seven dimensions shows low scores

### Weaknesses
1. My biggest concern for methodology is for creative generation, the desireable generation can be diverse. Is there an way to enforce the diversity of the generated response and ensure all generated caption can beat the baseline. For a single type of humor generation, hacking the benchmark with certain prompt structure could still occur. In addition, the author should also measure the diversity for the generation to ensure that actual creative generation is produced. 

2. Primary automatic metric uses GPT-5; while Table 2 benchmarks evaluators, no significance tests or cross-evaluator robustness are reported (e.g., bootstrap CIs on pass@K). Add statistical tests and report agreement between GPT-5 and humans per-caption.
3. Retrieval compliance & leakage: Retrieval set lists several joke corpora but lacks licensing and dedup/leakage controls vs. test captions. Report licenses, filtering, and overlap stats between retrieved text and ground-truth or generated captions.
4. Reproducibility gaps: Prompts, seeds, and full code release status are unclear.

### Questions
1. How sensitive are results to the choice of evaluator? Please report pass@K with a second strong evaluator and give 95% CIs; also correlate evaluator scores with human ratings per-caption

### Soundness
3

### Presentation
3

### Contribution
3
