# Quality of Rejections Matters: Preference Data Construction for Faithful Summarization

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Factual reliability is a central challenge in abstractive summarization, as LLMs continue to generate hallucinations. A widely adopted solution is preference optimization, training models to prefer faithful over unfaithful summaries, but prior work emphasized the quantity of rejections over their alignment quality. In this paper, we show that effective alignment in summarization arises when rejected summaries achieve high alignment potential, characterized by a small preference margin that keeps rejections non-trivial and a large factuality margin that enforces clear factual contrast. Through both theoretical analysis and controlled prompting, we show that three factors, hallucination level, summary length, and prompt complexity, critically shape alignment potential. Building on these insights, we propose SPICE, a simple prompting strategy that produces rejected summaries with strong alignment value. Across diverse model scales and alignment algorithms, SPICE consistently achieves superior factuality without sacrificing coherence, relevance, or abstractiveness, outperforming existing rejection strategies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the construction of preference data for improving factual consistency in abstractive summarization. The authors introduce the concept of "alignment potential," arguing that high-quality preference pairs require a large "Factuality Margin" (a clear factual difference between chosen and rejected summaries) and a small "Preference Margin" (the rejected summary is difficult for the base model to distinguish from the chosen one).

Through theoretical analysis and controlled empirical studies, the paper identifies three key factors influencing these margins: hallucination level, summary length, and prompt complexity. Based on these insights, the authors propose SPICE (Simple Prompt for Informative and Contrastive Examples), a prompting strategy that generates rejected summaries with high hallucination levels, matched lengths, and minimal instructions.

Experiments across various model scales (1B-8B) and alignment algorithms (DPO, SimPO, KTO) demonstrate that SPICE empirically outperforms existing reference-based (CLIFF) and model-based (SummLlama, SYNFAC-EDIT) strategies in improving faithfulness.

### Strengths
S1: **Systematic Empirical Analysis**: Section 4 provides a controlled analysis that systematically investigates the impact of hallucination level, length, and prompt complexity on the defined margins. The methodology of this analysis is rigorous.

S2: **Empirical Performance**: The proposed method, SPICE, demonstrates strong empirical performance across diverse datasets, model scales, and alignment algorithms. The results reported in Table 4 show consistent improvements over the baselines used.

S3: **Practicality and Simplicity**: The resulting method (SPICE) is simple and easy to implement, suggesting practical utility.

### Weaknesses
W1: **Fundamental Inconsistency Between Framework and Results (Preference Margin)**: This is the most critical weakness. The paper's core theoretical framework argues that minimizing the preference margin (making the rejection "harder" to distinguish) is crucial for high alignment potential. However, the empirical data in Table 5 contradicts this. SPICE's preference margin (e.g., 28.7 for 3B) is substantially larger (easier to distinguish) than that of SummLlama (16.5). This suggests that the framework, as proposed, does not accurately explain the results, and the performance gains may be driven entirely by the large factuality margin, undermining the central thesis of the paper.

W2: **Weakness of Baselines (SYNFAC-EDIT)**: The comparison against baselines appears inflated by weak competition. Most notably, the comparison against SYNFAC-EDIT (the most similar LLM-based editing approach) is highly suspect. Table 5 shows SYNFAC-EDIT achieved an average factuality margin (delta_fact) of only 0.15, compared to SPICE's 7.07. This indicates that the SYNFAC-EDIT implementation fundamentally failed to generate meaningful factual errors, rendering it an ineffective comparison.

W3: **Limited Conceptual Novelty**: The concept of "alignment potential"—requiring negative examples that are difficult (small preference margin) yet clearly incorrect (large factuality margin)—is essentially a rebranding of the well-established principle of Hard Negative Mining. The paper formalizes common knowledge in the context of LLM faithfulness without offering significant technical or conceptual innovation.

*Overall*: While the paper addresses an important topic and presents a systematic empirical study (S1) with positive results (S2), I recommend rejection due to fundamental inconsistencies in the core argument and limited novelty. The primary reason for rejection is W1. The paper introduces the framework of "alignment potential," which hinges on minimizing the preference margin, but the empirical data in Table 5 directly contradicts this requirement. Furthermore, the conceptual novelty of the work is marginal (W3), as it largely formalizes the intuition behind Hard Negative Mining. Finally, the empirical success must be contextualized by the apparent failure of a key baseline (W2). The fact that SYNFAC-EDIT generated almost no factual errors invalidates the comparison and makes it difficult to assess the true significance of the empirical gains.

### Questions
Q1 (Regarding W1): In Table 5, SPICE exhibits a significantly higher (easier to distinguish) preference margin than SummLlama (e.g., 28.7 vs 16.5 for the 3B model). How do you reconcile this observation with the paper's core argument that minimizing the preference margin is necessary for high alignment potential? Is it possible that the success of SPICE is driven solely by the very large factuality margin, making the preference margin less relevant than argued?

Q2 (Regarding W2): Given the extremely low  (0.15) for SYNFAC-EDIT and the discussion in Appendix D suggesting the prompt failed to induce sufficient hallucinations, were attempts made to revise the SYNFAC-EDIT prompt to ensure it generated a comparable level of factual errors to SPICE before conducting the main experiments?

Q3 (Regarding W3): How do the authors specifically differentiate their concept of "alignment potential" and the subsequent insights from the long-established principle of hard negative mining in contrastive learning?

### Soundness
2

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
4

### Summary
This paper examines the factors that render rejected summaries effective for faithfulness alignment in text summarization. The authors argue that effective rejected summaries should possess high "alignment potential," which is defined by a narrow preference margin (indicating that distinguishing between options is non-trivial) and a wide factuality margin (representing a clear contrast in factual accuracy). They identify three key influencing factors hallucination level, summary length, and prompt complexity. Proposing SPICE, a straightforward prompting strategy. Although the study demonstrates solid methodology and comprehensive analysis, its innovative contribution is relatively incremental.

### Strengths
- The paper clearly formalizes the gap between preference margin and factuality margin as "alignment potential", providing a principled lens on why some rejections are more useful than others. Meanwhile, the author propose two quantifiable metrics: preference margin and factuality margin
- Beyond the concept, the paper also proposes SPICE, a simple yet highly effective prompting strategy that maximizes alighment potential by enforcing high hallucination, matched length, and prompt complexity
- The paper demonstrate a comprehensive and robust experiments with wide range of baselines. The proposed SPICE, consistently outperforms the model-based and reference-based method across various model scales and alignment. Highlighting the effectiveness of proposed method

### Weaknesses
- External evaluators (G-Eval/FineSurE/AlignScore) are potentially coupled with training signals, lacking robustness ablation studies such as leave-one-evaluator-out and replacement of evaluator families, as well as larger-scale, hierarchical human evaluation comparisons
- One concerns is whether there should be more ablation studies. For example, a comparison between data-level length control (equal-length vs no control) against loss-level normalization (e.g., SimPO/KTO variants)

### Questions
Q1: Why not implement actual adversarial sampling based on your theoretical analysis? Would margin-based rejection selection (like top-k by |∆_fact - ∆_pref|) improve over your simple prompt?
Q2: Can you provide any preliminary resutls on other task such as dialogue summarization or QA to demonstrate broader applicability? 
Q3: Why not conduct human evaluation for at least some ablations (in Table 2) to validate that margins truly correlate with human judgments?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzes the effects of and proposes a strategy for choosing rejected data for preference learning to improve summarization faithfulness. It first show through theoretical analysis that using preference pairs with high alignment potential (i.e. when there is a large gap between model likelihood and actual factuality of the chosen vs. rejected data) leads to faster convergence than selecting pairs randomly. Then, it analyzes three factors that impact alignment potential, including hallucination level, response length and generation prompt analysis. Based on the observations, the paper proposes SPICE, a strategy of creating rejection summaries with high hallucination level, comparable length with the accepted summaries, and with simple generation prompt. Results show that the proposed strategy consistently improves the summarization faithfulness.

### Strengths
* The paper analyzes rejection data selection for summarization faithfulness in a systematic and principled way. The analysis framework includes theoretical justifications, and considers various factors that can impact the end performance.
* The proposed strategy based on the observations is straightforward and lightweight, making it practical and easy-to-apply.
* Results show that it is versatile and robust, and can consistently improve performance gains across summarization models, rejection data generators and preference learning algorithms. 
* Results also show that the proposed method is data efficient, and can bring performance gain even with a subset of the training data.

### Weaknesses
The Factor 3 Prompt Complexity analysis is a bit coarse-grained. The simple vs. complex prompts seem to be at the two ends of extreme, where the complex prompt requires the generator to have multiple types of edits at the same time in each summary. It would be helpful to see intermediate comparisons (e.g. performing some but not all of the constraints), and the impact of each individual edit instruction, etc. essentially to analyze whether it's unhelpful to have any kind of instructions, or is it that having all of the instructions at the same time is too extreme but having some of them could still be helpful. (See Questions below).

### Questions
For Factor 3 Prompt Complexity:
* Among the different instructions for introducing hallucinations (e.g., numerical values, entity names, causal relationships), do all of these lead to trivial cues that models can easily distinguish, or could some of them be useful?
* In some sense, the complex instruction in Table 25 makes the generator to include all specified edits when generating rejected data. Since it is unlikely for all such errors to appear simultaneously in the same summary, it is not unexpected that they will lead to high preference margin. Could this make this comparison too trivial to conclude that complex prompts are less preferable? It would be helpful to see, e.g. instead present these edit instructions as suggestions and allow the generator to decide which to include, rather than asking it to have all of them?
* The complex prompt explicitly say “nearly all words should be edited”. Could this inadvertently push the generator towards producing less natural or uncommon phrasing, even if the outputs appear coherent on the surface?
* It would be helpful to provide some discussion or analysis on how the hallucination edit types look like in the generations with the "Default" prompt (Table 23), e.g. do they have the same categories as the ones listed in the complex prompt (Table 25) and how are they distributed, or do they fall outside these categories?

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
This paper proposes a method for constructing preference data for training models that generate more faithful summaries. They hypothesize that data for preference optimization for faithful summaries can be chosen in a better way that emphasizes more realistic errors in factuality. They verify this hypothesis by testing models fine-tuned on samples with larger factuality margins (less true) and smaller preference margins (more preferable otherwise). They then develop a way to prompt models to generate these summaries investigating the factors that have the biggest influence on generating such summaries. They observe that models fine-tuned on their summaries are able to achieve higher factuality scores than prior work.

### Strengths
The work to verify the factuality hypothesis is good, and in general the paper is well written and presented. The method appears sound, and does generally improve on prior work. The experiments detailing which portions of the summaries and prompts are important appear well conducted.

### Weaknesses
1. I have some concerns regarding the metrics used for evaluation. Krippendorf's alpha (0.53) for the human evaluation scores is low enough that these evaluations should not be considered trustworthy. The usual lowest threshold for accepting annotated data is alpha=0.67. Additionally, is it verified in prior work that the other metrics are aligned with human preferences?

2. Some claims are not fully explained. In particular, the explanation given for table 5 regarding the preference margins is that SPICE has the biggest factuality margin and "relatively small preference margins". However, in the reported results, SPICE has the largest second largest preference margin, which is in contrast to this claim. It would be good to remove or further clarify this claim.

### Questions
1. Is it verified in prior work that other metrics are aligned with human preferences?

2. Is there a reason why inter annotator agreement for the human evaluation is so low?

### Soundness
3

### Presentation
3

### Contribution
2
