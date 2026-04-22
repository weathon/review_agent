# Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 6, 8, 8

## Abstract
Pretraining large language models (LLMs) on high-quality, structured data such as mathematics and code substantially enhances reasoning capabilities. However, existing math-focused datasets built from Common Crawl suffer from degraded quality due to brittle extraction heuristics, lossy HTML-to-text conversion, and the failure to reliably preserve mathematical structure. In this work, we intro-
duce Nemotron-CC-Math, a large-scale, high-quality mathematical corpus constructed from Common Crawl using a novel, domain-agnostic pipeline specifically designed for robust scientific text extraction. Unlike previous efforts, our pipeline recovers math across various formats (e.g., MathJax, KaTeX, MathML) by leveraging layout-aware rendering with lynx and a targeted LLM-based cleaning stage. This approach preserves the structural integrity of equations and code blocks while removing boilerplate, standardizing
notation into L A T EX representation, and correcting inconsistencies. We collected a large, high-quality math corpus, namely Nemotron-CC-Math-3+(133B tokens) and Nemotron-CC-Math-4+ (52B tokens). Notably, Nemotron-CC-Math-4+ not only surpasses all prior open math datasets-including Mega-Math, FineMath, and OpenWebMath-but also contains 5.5× more tokens than FineMath-4+, which was previously the highest-quality math pretraining dataset. When used to pretrain a Nemotron-T 8B model, our corpus yields +4.8 to +12.6. 
gains on MATH and +4.6 to +14.3 gains on MBPP+ over strong baselines, while
also improving general-domain performance on MMLU and MMLU-Stem.
We present the first pipeline to reliably extract scientific content—including
math—from noisy web-scale data, yielding measurable gains in math, code, and
general reasoning, and setting a new state of the art among open math pretraining
corpora. To support open-source efforts, we release our code1 and datasets 2
.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper redesigns the text extraction pipeline for web data curation, firstly employing Lynx, a text-based browser, to retain texts, and employing Phi-4 to rewrite text for quality, resulting in Nemotron-CC-Math. It delivers better data quality than existing the-state-of-the-art open-source corpora, such as MegaMath, FineMath.

### Strengths
1. Well-written and structured paper, solid experiments;
2. The lynx’s introduction, which reliably captures equations and maintains code indentation, avoids the heuristics DOM tree operations, such as MegaMath.
3. The ablation on different refinement models is solid.

### Weaknesses
I believe that the effectiveness of Lynx should be evaluated through an apples-to-apples comparison. For example, the quality of Lynx versus DOM tree optimization (as introduced in MegaMath) on the same mathematical web pages could be compared under a controlled setting.

### Questions
In addition, the potential negative impact introduced during the document refinement process should be clearly discussed. I believe this could be an important point that warrants further analysis.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Nemotron-CC-Math, a large, high-quality math corpus built from Common Crawl. It uses a domain-agnostic extraction pipeline: layout-aware `lynx` rendering plus structure-preserving LLM cleaning to unify MathJax/KaTeX/MathML into LaTeX, keep equation and code structure, and remove boilerplate.

It releases two datasets: Nemotron-CC-Math-3+ (133B tokens) and Nemotron-CC-Math-4+ (52B tokens), claimed to surpass MegaMath, FineMath, and OpenWebMath; 4+ has about 5.5× the tokens of FineMath-4+. As for experiments, the authors show that pretraining an 8B model on this data yields 4.8-12.6 on MATH and 4.6-14.3 on MBPP+, with additional gains on benchmarks like MMLU.

### Strengths
1. I think overall the data pipelines are very sound. the authors combines layout-aware lynx rendering with structure-preserving LLM cleaning, avoiding information loss from naïve HTML-to-text extraction.

2. The authors also unify MathJax/KaTeX/MathML into LaTeX, preserving equation and code structure while removing boilerplate, which is very important but often under-estimated in previous works.

2. The experiments show very promising results, further demonstrating the quality of the datasets.

### Weaknesses
I don't see any obvious weaknesses.

### Questions
I do have some questions for the authors:

1. why pre-train on math also boosts code performance? if so, have you compared this with other math&code-related datasets? such as stack-edu, megamath-code?

2. How do you detect and constrain LLM over-editing or hallucinations (e.g., symbol renaming, citation mismatches, skipped derivations)? Did you conduct manual sampling review and an inter-annotator agreement (IAA) evaluation?

3. Could you report quantitative results for LaTeX/code parseability, structural consistency (e.g., AST-based edit distance), and rendering consistency?

### Soundness
3

### Presentation
3

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
This paper introduces nemotron-cc-math, a math dataset created by utilizing the Lynx text-based browser. By using the browser, the HTML can be rendered as structured format with equation and code layout as human read them. The content is then fed through an LLM (Phi-4 14B) to clean up the boilerplates. The final dataset retains data passing Fineweb classifier (3+), which contains 133B tokens, which is one of the largest set at this quality.

### Strengths
- Robust and Proper Pipeline: The lynx + LLM-cleaner pipeline is an effective solution. It addresses the failure mode of previous web math extractors that corrupt math and code. The qualitative examples provided clearly demonstrate its superiority in preserving structure. While this method makes sense, both the lynx rendering and an 14B LLM cleaner are expensive for many practitioners. So the open sharing of this resource will help the community significantly.
- The paper delivers a dataset that is both larger and higher quality than existing open-source alternatives. The 133B high quality portion is larger than FineMath, though smaller than the 300B MegaMath.
- Strong experiment results: the methods are tested on an 8B mid-training checkpoint, with 100B and 300B experiments. The scale of the experiment should be sufficient and it shows good results on a range of benchmarks.
- Once again, the contribution to open science and open source of this work is commendable.

### Weaknesses
This paper can benefit from additional experimental settings. The main experiments are conducted at the mid-train setting. Will there be some confounding factor from the base model itself? Further, would larger amount of unique tokens be helpful and how much repetition can this dataset be used?

The readers would also benefit from learning about the filtered out portion, i.e., Nemotron-cc-math-1-3. The token size, quality and corresponding model performance may provide valuable information.

### Questions
How much total tokens are their without the quality filter? Is there an estimate?

### Soundness
4

### Presentation
4

### Contribution
4
