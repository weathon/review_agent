# Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
As artificial intelligence (AI) is increasingly deployed across domains, ensuring fairness has become a core challenge. However, the field faces a "Tower of Babel'' dilemma: fairness metrics abound, yet their underlying philosophical assumptions often conflict, hindering unified paradigms—particularly in unified Multimodal Large Language Models (UMLLMs), where biases propagate systemically across tasks. To address this, we introduce the IRIS Benchmark, to our knowledge the first benchmark designed to synchronously evaluate the fairness of both understanding and generation tasks in UMLLMs. Enabled by our demographic classifier, ARES, and four supporting large-scale datasets, the benchmark is designed to normalize and aggregate arbitrary metrics into a high-dimensional "fairness space'', integrating 60 granular metrics across three dimensions—Ideal Fairness, Real-world Fidelity, and Bias Inertia & Steerability (IRIS). Through this benchmark, our evaluation of leading UMLLMs uncovers systemic phenomena such as the "generation gap'', individual inconsistencies like "personality splits'', and the "counter-stereotype reward'', while offering diagnostics to guide the optimization of their fairness capabilities. With its novel and extensible framework, the IRIS benchmark is capable of integrating evolving fairness metrics, ultimately helping to resolve the "Tower of Babel'' impasse.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a unified fairness benchmark for evaluating UMLLMs on both understanding and generation across three dimensions of ideal fairness, real-world fidelity, and bias inertia & steerability, while aggregating 60 metrics into a single, interpretable score. They also develop ARES, an adaptive-routing demographic classifier, and curate four purpose-built datasets to enable large-scale labeling and IRIS evaluation of UMLLMs. After through validation of the proposed framework, they present empirical analyses that reveal trade-offs and model-specific patterns of bias.

### Strengths
The paper provides:
- A novel, uniform pipeline that evaluates both understanding and generation, organizing 60 metrics along three axes with transparent normalization. They also provide scalable demographic labeling (ARES) and four purpose-built datasets for large-scale measurement.

- A comprehensive validation of proposed IRIS. Then they applies IRIS to a broad model set with interpretable outputs (persona profiles, sector breakdowns) that expose system-level trade-offs.

- A set of novel findings, including: generation gap, clear cross-dimension correlations, separation of willingness vs ability under counter-stereotype prompts, localization of where bias emerges, and model-specific persona splits.

### Weaknesses
- Limited geographic/generalization coverage: Fairness conclusions are calibrated to U.S./E.U. demographic statistics and taxonomies, so results may not transfer to other regions, cultures, or alternative demographic schemes.

- Proxy-based quality/semantic scoring without human validation: Reported “quality” and “semantic consistency” rely only on automated proxies rather than task-specific human judgments, leaving validity uncertain.

- Potential sparsity for fine-grained disparities: After stratifying by occupation, age, gender, and skin tone (and their intersections), some categories likely have small sample counts, making disparity estimates unstable or untrustworthy. 

- Generation fairness (IFS-Gen/RFS-Gen) is driven by a single template  (“Please generate a photo of a/an {occupation}”) across 52 jobs. Even small wording choices can shift demographic priors, yet the paper doesn’t probe prompt phrasing sensitivity. This risks injecting template bias into both the “ideal” and “real-world fidelity” results.

### Questions
- How sensitive are the fairness results to the exact “neutral” prompt template? Did alternative phrasings change scores?

- Since ARES both defines demographic labels and scores counter-stereotype success, did you try any alternative labelers to quantify labeler-induced bias on results?

- In counterfactual pairs, what controls ensure only the target attribute changes?

- It can by valuable to provide a finer-grained perfromance of ARES by different categories such as generator family and attributes?

- In multilingual and alternative cultural contexts, how would prompts and occupation terms impact the IFS/RFS/BIS dimensions? Can IRIS be adapted to this context, and if so, what concrete changes (e.g., demographic taxonomies, target distributions, prompt templates) would be required?

- When IFS, RFS, and BIS conflict, how should practitioners prioritize them? Can you offer policy guidance or concrete decision rules for trade-off selection?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors create a multidimensional fairness benchmark based on Ideal Fairness, Real-world Fidelity, and Bias Inertia & Steerability (IRIS) with customized datasets and multiple sensitive attributes. As part of this, they develop Adaptive Routing Expert System (ARES), a classifier for demographics in images. They posit various phenomenon foregrounded by IRIS, including a "generation gap" and "personality splits."

### Strengths
1. The authors address an important problem (the diversity of fairness metrics and their practical applicability) and target a viable strategy (a multidimensional benchmark that integrates these metrics in practical contexts).
2. The toolkit appears well-documented and transparent.
3. The authors bring up some interesting concepts, such as personality profiles and the generation gap, that could enrich the fairness literature if developed fully.

### Weaknesses
1. The authors are trying to do way too much in a single paper. The core contribution is meant to be a unification of fairness measures, but there is very little on this (e.g., why this set of measures is better than alternatives). The contributions, such as unifying different notions of fairness in a common framework, are not yet developed or defended enough to constitute a major contribution to the fairness literature.
2. The paper is too verbose with many paragraphs and sentences that are just repeatedly stating what the paper contributes instead of actually showing how it advances our understanding. Lines 041-053 is an example, and honestly I think basically the first 4 pages could be greatly shortened (though I like Figure 1 and Table 1). This would free up room to develop scholarly contributions.
3. I don't see why there is an algorithm (Lines 162-129) in this paper. It seems like this is just a way of calculating score across dimensions. An algorithm (and the algorithm box) is not the best way to describe a scoring procedure because there's nothing particularly algorithmic. Presenting it differently would be much clearer, such as by not having undefined variables and unclear notation.
4. The core intellectual content of the paper is in the selection of items in the second and third columns of Table 1, but I'm not convinced of the authors' choices. This is the main part of the paper that I need to see developed before being more positive on the submission.
5. The paper remains focused on particular sensitive attributes (e.g., age) and modalities without clear generalization. Even if my other points are addressed, I'm not sure if this is really a general fairness evaluation strategy rather than just one of potentially many combinations of metrics. This would be fine if the sensitive attributes were just examples of a broader machine learning theory, but that is currently lacking.
6. I find the current ordering of the paper very unclear. I suggest not going by "enabling technologies" but instead chronological through the application of the tool. Start with the datasets and the metrics and how they relate to the fairness literature.

### Questions
1. How were the items in the second and third columns of Table 1 determined? For example, why is statistical parity used, when it is widely considered an oversimplified fairness metric? I understand it is mathematically convenient (hence it being used so often), but is it sufficient for real-world application?
2. Why are IFS and RFS separated in that way? What exactly do they mean? I encourage the authors to speak in terms of the fairness literature (e.g., "statistical parity" is a well-defined concept that is widely understood in the literature) rather than trying to coin new terminology. For example, "fairness through awareness" is very broad and plausibly includes everything else in the second column except "fairness through unawareness," which is a largely outdated notion that doesn't actually seem to show up in the empirics of the paper.

Also see the Weaknesses section.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces IRIS, a benchmark for measuring fairness in unified multimodal large language models (UMLLMs) across both understanding and generation. IRIS evaluates their fairness across three dimensions: Ideal Fairness, Real-World Fidelity, and Bias Inertia & Steerability. The framework uses a custom demographic classifier (ARES) and four curated datasets to compute 60 fairness metrics projected into a “fairness space.” It also includes a qualitative profiling tool (IRIS-MBTI) that summarizes a model’s fairness personality.
Through this benchmark, the authors uncover systemic phenomena—like the generation gap (UMLLMs being less fair in generation), personality splits (task-dependent fairness behavior), and the counter-stereotype reward (improved output quality when breaking stereotypes).

### Strengths
• Novel and meaningful contribution: A first benchmark to jointly assess fairness in UMLLMs. The unified fairness-space idea and the three-dimensional structure are elegant and address the “Babel Tower” problem of conflicting fairness metrics. The ARES classifier and supporting datasets make multi-dimensional fairness evaluation feasible in an automated fashion. 

• Clarity: The paper is easy to follow despite its scope. Figures clearly walk the reader through the pipeline, and the IRIS-MBTI profiles make results intuitive and interpretable. I appreciate the figures! 

• Interesting findings. The “generation gap” and “personality split” results are particularly interesting and likely to influence future multimodal fairness research.

### Weaknesses
* Conceptual trade-off: Fairness is inherently multi-dimensional, and each metric captures a distinct philosophical or statistical notion. Collapsing these / projecting them into a high-dimensional “fairness space”  might blur important nuances and make interpretability harder. How the authors balance this abstraction with human-understandable fairness judgments would be worth discussing.

* Metric sensitivity. The benchmark combines 60 metrics with tuned weights. It’s unclear how sensitive the global IRIS score is to metric choice or weighting; an ablation would help establish robustness.

### Questions
1. How is the IRIS-MBTI profiling validated? Are these categories empirically meaningful or mainly heuristic?

2. How robust are the IRIS scores to metric weighting or selection changes? 

3. How do the authors think about the potential loss of nuance or interpretability when collapsing many distinct fairness notions?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work tackles the ever-present and frustrating problem in fairness that there are many, sometimes conflicting, fairness metrics that can be used; however, ML practitioners often pick only one metric to judge a model’s fairness.  The authors present a comprehensive fairness benchmark for unified multimodal large language models (UMLLMs) called IRIS.  They simultaneously analyse the fairness of the understanding and generative capabilities of a model by incorporating 11 fairness metrics into a novel embedding space across 3 dimensions to measure what they call Ideal Fairness, Real-world Fidelity, and Bias Inertia and Steerability.  They analyse a number of modern UMLLMs to demonstrate the utility of the benchmark and highlight a number of phenomena such as a gap in fairness across generation and understanding of models.  They conclude by showing how the benchmark can be used to understand where bias is generated within a model to allow for effective mitigation techniques.

### Strengths
1. The benchmark is a valuable contribution to the community and demonstrates that you can combine numerous, sometimes conflicting, fairness metrics into a single benchmark for a holistic view of the fairness of UMLLMs.
2. The benchmark analyses both understanding and generative capabilities of the models.
3. The authors demonstrate effectively the utility of the benchmark for determining characteristic of models and phenomena that might be present.
4. The authors demonstrate how you might use the benchmark to understand the source of bias within models and use that the mitigate biases going forward.
5. The contribution of annotated datasets for the ARES classifier.
6. The tests to determine the foundational integrity of the benchmark are a great addition to this paper.

### Weaknesses
1. Too much of the paper is in the Appendix.   I don’t know how this can be fixed as the paper has a lot of content, but a huge proportion of the paper is in the Appendix, which was too long for me to evaluate completely.  The authors should consider ways in which you can bring some important details into the main paper concisely (like the models analysed and the personality profiles).  Many important assumptions are also only reported in the appendix (the simplification of gender, age, and skin tone attribute sub-groups for example).  Any important considerations, limitations, or assumptions should be expressed in the main body of the paper, even if they are expanded on in the Appendix.
2. Biases in the ARES classifier will propagate into the metrics which is impossible to disentangle from the biases in the models being measured.  
3. I like the idea of personality profiles for models, but it would be good to have names that are more easily consumable and intuitive than UAF, HDF, etc.  explained in the main paper in a table.
4. This was mentioned in the limitations and is fairly significant “The scoring pipeline depends on calibrated hyper-parameters whose robustness across other model families and domains requires further validation”.

### Questions
1. Line 069: Figure 1 link to models being tested is incorrect.  Should be A.1.1.
2. How can you guarantee that biases from the ARES classifier don’t propagate into the benchmark measurements?  Have you measured its bias?  How might we reduce this risk going forward?  It has an overall accuracy of 88%.  Is this sufficient?  Where was it failing?  Do you propagate the uncertainty generated by an imperfect classifier into the benchmark results so that consumers know which results are reliable?
3. Can you please state in the main paper what demographic markers the ARES classifier classifies (line 305)?
4. Line 046 and 126: the literature on the bias transfer hypothesis [1] is somewhat divided on the conclusion that intrinsic biases “carry over” to downstream tasks.  Studies in the fine-tuning space are conflicted with some works saying that biases do transfer [1,2,3], while others claim they do not [4].  Recent work fixes some issues with previous studies, uses unified metrics to measure intrinsic and extrinsic biases, and demonstrates that they do conclusively transfer across prompting adaptation [5].  It’s a minor thing but the authors should update the text to reflect the state of the community on this.
5. Line 086: In reference to a fairness assessment goal seeking a single optimal solution - isn’t it the case though that sometimes there is one fairness metric that represents the ideals that the ML practitioners are trying to adhere to?  So this isn’t always a negative and maybe as part of this benchmark you would also want to report the individual fairness metrics as well.
6. How do you obtain 60 “granular” metrics from the 11 metrics in table 4?  Line 159 say they are derived from the core 11 metrics but I don’t see in App A.2.1 how they are expanded to 60 metrics.
7. Line 174: how is m_i normalised to form u_i?
8. Line 230: I don’t understand the Core Question & Metrics for Understanding in Real-world Fidelity.  Can you please rephrase this to make it clear what the metric is for this?  This sounds more like a research question to answer rather than a metric.
9. Line 373: can you comment on the practical implications of the trade-off between Real-world Fidelity and Steerability?
10. Line 375: How are Real-world Fidelity (gen) and Ideal Fairness (understanding) synergistic?  They are correlated which means their combined effect might not be greater than the sum of their parts.
11. Line 383: Could you please expand on how you determined that the two traits HAF and UDF across generation and understanding means that “This finding suggests that a shared representation space does not guarantee consistent fairness characteristics across tasks”?
12. Line 429: There is a typo in the heading “Gudied” -> “Guided”.
13. Line 457: Interesting finding.  But I wonder if the internal representations are just a result of slightly OOD data and this is a form of “over-fitting” phenomena where large but opposing weights are used to model noisy data.
14. What effect does the suite of metrics chosen have on the framework?  Have you done a sensitivity analysis to determine how sensitive the method is the choice of core fairness metrics?
15. In the Ethics statement on line 492 you should also mentioned that the use of the ARES annotator can introduce bias as well as measurement noise.
16. More discussion on why the three core dimensions of Ideal Fairness, Real-world Fidelity, and Bias Inertia and Steerability, were picked would have been appreciated.  


[1] Ryan Steed, Swetasudha Panda, Ari Kobren, and Michael Wick. 2022. Upstream Mitigation Is Not All You Need: Testing the Bias Transfer Hypothesis in Pre-Trained Language Models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, Dublin, Ireland.
[2] Sarah Schröder, Alexander Schulz, Philip Kenneweg, and Barbara Hammer. 2023. So can we use intrinsic bias measures or not? In Proceedings of the 12th International Conference on Pattern Recognition Applications and Methods.
[3] Masahiro Kaneko, Danushka Bollegala, and Naoaki Okazaki. 2022. Debiasing Isn’t Enough! – on the Effectiveness of Debiasing MLMs and Their Social Biases in Downstream Tasks. In Proceedings of the 29th International Conference on Computational Linguistics, Gyeongju, Republic of Korea.
[4] Xisen Jin, Francesco Barbieri, Brendan Kennedy, Aida Mostafazadeh Davani, Leonardo Neves, and Xiang Ren. 2021. On Transferability of Bias Mitigation Effects in Language Model Fine-Tuning. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Association for Computational Linguistics.
[5] Nivedha Sivakumar, Natalie Mackraz, Samira Khorshidi, Krishna Patel, Barry-John Theobald, Luca Zappella, and Nicholas Apostoloff, Bias after Prompting: Persistent Discrimination in Large Language Modelsm, EMNLP, 2025.

### Soundness
3

### Presentation
3

### Contribution
3
