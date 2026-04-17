# TABLET: A Large-Scale Dataset for Robust Visual Table Understanding

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
While table understanding increasingly relies on pixel-only settings, current benchmarks predominantly use synthetic renderings that lack the complexity and visual diversity of real-world tables. Additionally, existing visual table understanding (VTU) datasets offer fixed examples with single visualizations and pre-defined instructions, providing no access to underlying serialized data for reformulation. We introduce TABLET, a large-scale VTU dataset with 4 million examples across 21 tasks, grounded in 2 million unique tables where 88% preserve original visualizations. To evaluate whether models are able to jointly reason over tabular and visual content, we also introduce VisualTableQA, a benchmark requiring both visual perception and table understanding. Fine-tuning vision-language models like Qwen2.5-VL-7B and Gemma 3-4B on TABLET improves performance on seen and unseen VTU tasks while increasing robustness on real-world table visualizations. By preserving original visualizations and maintaining example traceability in a unified large-scale collection, TABLET establishes a foundation for robust training and extensible evaluation of future VTU models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Current visual table understanding (VTU) benchmarks rely on synthetic data lacking real-world complexity and omit underlying serialized data, while TABLET is a large-scale dataset with 4 million examples from real visualizations, paired image-HTML representations, and 20 tasks. Fine-tuning vision-language models on TABLET boosts their performance on seen/unseen VTU tasks and robustness to real-world tables, establishing a foundation for future VTU model training and evaluation.

### Strengths
TABLET addresses prior single-task/synthetic data limitations with large-scale, lossless real-world table visualizations and 20 diverse tasks, while its high quality lies in 4M examples, image-HTML pairings, and traceable metadata. Its clear unified format and extensible design, coupled with significant contributions to boosting VLM cross-task performance (including SOTA results) and enabling practical VTU research, underscore its field value.

### Weaknesses
1. Whether all tables are presented in English? As far as I am aware, Wikipedia provides original HTML files in multiple languages.
2. Each example in TABLET is accompanied by the HTML version of the corresponding table. Can such a large volume of data be used to train table parsing models? Is the annotation format of HTML consistent? For instance, how are formulas within the tables annotated? Additionally, do these HTML files contain certain formatting information, such as row widths and font colors?

### Questions
see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies a critical "train-test mismatch" in the field of Visual Table Understanding (VTU). The authors argue that existing benchmarks predominantly use "synthetic renderings" of tables, which lack the visual complexity (e.g., merged cells, colors, fonts, embedded images) of real-world tables. This causes models trained on them to fail when generalizing to real-world visual data.

To solve this, the authors introduce TABLET, a new large-scale dataset of 4 million examples over 20 tasks. The dataset's primary contribution is that 88% of its 2 million unique tables are "original visualizations" meticulously retrieved from historical web snapshots (primarily Wikipedia), preserving their true visual fidelity. The authors demonstrate through experiments that while performance on traditional tasks is mixed, training on TABLET dramatically improves a model's robustness to visual domain shift (synthetic vs. original) and enhances generalization to unseen tasks.

### Strengths
1. **Problem Identification:** The paper's core premise is strong and well-articulated. It correctly identifies the "train-test mismatch" as a fundamental problem for the VTU field as it shifts towards pixel-only VLMs. This is a significant and practical issue.

2. **Methodological Rigor on Data Collection:** The primary contribution is the dataset's "visual fidelity." The engineering effort to achieve this is non-trivial and highly commendable. The authors detail a rigorous process of tracing seed datasets to their original crawl dates and using Wikipedia's archiving API to retrieve historical snapshots, followed by Levenshtein matching to find the correct table. This high-quality execution is a major strength.

3. **Strong Robustness Argument:** The paper's most compelling evidence is in Table 9. It clearly shows that models trained on synthetic data suffer a massive performance degradation (-22.35 points) when evaluated on original visualizations. In contrast, models trained on TABLET's original data are far more robust (only -6.63 points). This single finding strongly validates the paper's core hypothesis and the necessity of this dataset.

4. **Resource Value and Extensibility:** The paper delivers more than just a static benchmark; it provides a large-scale, multi-task resource. By including HTML representations, metadata, and traceability links to source datasets, the authors enable future research, task reformulation, and extensibility.

### Weaknesses
1. **Core Claim Undermined by Own Results:** The paper's most significant weakness is that its primary experiment (Table 2) fails to support its central claim. When comparing models trained on TABLET-B_org (original) vs. TABLET-B_synth (synthetic), the performance is roughly equivalent across most tasks. This result is underwhelming and directly contradicts the motivation that visual fidelity improves performance on these tasks.

2. **Circular Argument:** The authors' explanation for the failure of Table 2 is that the benchmarks themselves (e.g., ToTTo, WikiTQ) are flawed and were designed for text, not visual cues. This is a "catch-22" and a form of circular reasoning: the paper claims its dataset is the solution but simultaneously admits that the tasks it uses for validation are incapable of proving it. They have built a high-fidelity testbed but failed to provide tasks that actually require that fidelity.

3. **Diluted Conclusion on "Mixed" Data:** In several experiments (Table 2, Table 10), the TABLET-B_mix (original + synthetic) model performs best. This dilutes the paper's central premise. The takeaway risks becoming "more data is better" (a trivial conclusion) rather than the intended, stronger claim that "high-fidelity data is better."

4. **Limited "Real-World" Diversity:** The paper claims to solve the "real-world" visualization problem, but the dataset's diversity is questionable. 88% of the data (61.5% of tables) is sourced from Wikipedia. While better than synthetic data, Wikipedia tables are still highly structured and relatively uniform. This is not representative of the true "in-the-wild" chaos of scanned PDFs, financial reports, or product pages.

### Questions
1. Given that the core experiment in Table 2 fails to show a clear superiority for TABLET-B_org over TABLET-B_synth, and you attribute this to flawed benchmarks (e.g., ToTTo), does this not imply that the primary contribution of this work is a resource for which the tasks do not yet exist?

2. The strong performance of the TABLET-B_mix model seems to suggest that a combination of synthetic and original data is optimal. Does this not contradict the paper's central premise that synthetic renderings are inherently flawed and should be replaced by original visualizations?

3. Why were new, visually-demanding tasks (e.g., "find the cell with the red background," "what is the value in the merged cell?") not introduced and evaluated to actually prove the hypothesis that the visual fidelity of TABLET is its key advantage?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces TABLET, a large-scale dataset for visual table understanding (VTU). Compared to existing resources (e.g., MMTab), TABLET (i) reconstructs tables in their realistic visual form from web/Wikipedia sources, (ii) unifies them into an image + HTML + metadata format, and (iii) covers 20 VTU-related tasks. The authors fine-tune Qwen2.5-VL-7B on this data and show improvements on both seen and unseen VTU tasks. The main value of the paper is on the data/engineering side and on providing a more realistic training distribution.

### Strengths
1.The motivation is real: current VTU data is mostly synthetic or visually simplified, which causes a distribution gap for vision-language models that must operate on real web-like tables.

2.The dataset is large, diverse, and traceable back to the original sources, which is helpful for reproducibility and later task extension.

3.Experiments are fairly comprehensive (synthetic vs. real-style vs. mixed) and the results are consistent with the stated motivation.

### Weaknesses
1. The paper does not introduce a new learning objective, a new model architecture, or a principled framework for unifying VTU tasks. The core message is essentially “better/realistic data → better performance,” which is valuable but closer to a dataset/empirical paper than to a method/theory paper.

2. The paper repeatedly argues that preserving realistic visual styles leads to better generalization, but the current evaluations mostly show dataset-level gains, not an analysis of which visual factors (borders, multi-row headers, fonts, layout noise) the model actually exploits.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents TABLET, a large-scale dataset for UTV, containing 4M examples. The key contribution is the collected TABLET from sources from Wikipedia and PubTabNet. The dataset are split into train, dev, and test sets and comprising comprehensive table-related tasks. Experiments with Qwen2.5-VL-7B show good performance gains on several benchmarks, and the authors argue that TABLET improves robustness on real-world table images.

### Strengths
+It collects a large-scale dataset combining multiple VTU sources. 
+It provides both HTML and image representations with metadata, which facilitate the document understanding community. 
+Experiments on the Qwen2.5-VL models demonstrating the effectiveness of the collected dataset.

### Weaknesses
1. The author presents a new dataset, TABLET, and the author claims that the dataset contains 4M examples. However, nearly 30% are existing, the author should describe this more rigorously.
2. No new model, algorithm, or theoretical insight in the paper. The paper mainly merges existing datasets and collected new table data on the network. 
3. One of the author’s main motivation is that synthetic data are of lower quality than real-world data. However, the majority of the dataset they collected still consists of screenshot-based images, many of which are taken from sources like Wikipedia. This seems somewhat inconsistent with the author’s stated motivation?
4. Poor experiments. In Tab.2 and 3, the performance improvement by the proposed dataset appears to be limited.
5. What confuses me is that the paper only reports results for qwen and the results fine-tuned on qwen, without any comparison to other models. This is completely unreasonable for a research paper.

### Questions
Please refer to the weaknesses section above. 
It seems that the author merely proposed a dataset and fine-tuned qwen with it, without providing comparisons to other models. I believe the paper still needs further polishing.

### Soundness
2

### Presentation
3

### Contribution
2
