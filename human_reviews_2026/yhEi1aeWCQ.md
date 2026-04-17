# NUMBER REPRESENTATIONS IN LLMS: A COMPUTATIONAL PARALLEL TO HUMAN PERCEPTION

- Decision: Reject
- Scores: 4, 8, 4

## Abstract
We provide empirical evidence that large language models (LLMs) encode numerical values on a compressed, logarithmic number line, challenging the prevailing assumption of linear representation. Extracting hidden states for numerals, we project them onto one-dimensional manifolds using dimensionality reduction and evaluate two complementary metrics: Spearman’s $\rho$ to measure monotonicity and a newly introduced Scaling Rate Index ($\beta$) that quantifies whether spacing is sublinear, linear, or superlinear. Across several LLM families, unsupervised projection into low-dimensional subspaces of maximum variance consistently uncovers strong sublinear trends. Interventions along the discovered number-line directions causally modulate next-number predictions, demonstrating that these dimensions encode genuine numerical structure. This compressed geometry is robustly observed in controlled log-spaced prompts and in real-world settings such as birth years, but is absent in non-numerical controls. Our findings refine the linear representation hypothesis by showing that numerical magnitudes occupy a structured subspace whose internal geometry is systematically non-uniform and logarithmic in nature.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers whether LLMs encode numbers using logarithmic rather than linear representations. Specifically, the authors pair both a synthetic dataset and some simple real-world data with PCA and PLS to test for order preservation and sublinear compression of numeric representations. Their interpretation of their results is that LLMs do encode numbers using logarithmic representations, in a way that is consistenct with the human mental number line.

### Strengths
The question is interesting and well motivated. The writing is clearly structured and easy to follow. Experiments are simple but clean. Finding parallels between human numeric cognition and LLM numeric cognition can be illuminating into both systems.

### Weaknesses
My main concern with this paper is the lack of impact. Even though the topic and questions are interesting to me, I do not believe that the broader ICLR audience will find it impactful.

- The results on the natural data seem fairly weak and the use of birth years to study something analogous to mental number lines is not very compelling. Although birth years are numbers, they do not necessarily symbolize quantity in the way that most other numbers do. For example, '1984' in the input 'Katy Perry was born in 1984' is perhaps more similar to 'California' in the input 'Katy Perry was born in California' than it is to '1984' in the input '1700 + 284 = 1984'. Put another way, the numeric interpretation of '1984' seems far more meaningful in the latter case than the former case. The way that the birth years are presented the real data task therefore do not seem to lend themselves quite as naturally to some notion of having an internal number line.
- Building on the previous point, it is not clear why the real data experiments use factual questions that only have one correct numeric answer. This also seems to be a fairly unnatural context in which to consider mental number lines, since the model can just memorize that single number without needing to resort to any kind of numeric representation.

### Questions
- It is not clear how I should interpret specific values of ρ and β beyond some vague notion of 'higher is better'. Is there a concrete baseline that these values can be benchmarked against?
- In table 2, is 'llama' being misspelled as 'llamba'?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscript investigates how LLMs internally represent numerical values. The authors find that LLMs encode numbers on a compressed, logarithmic number line, rather than a linear one as commonly assumed. By extracting hidden states corresponding to numerals and projecting them onto one-dimensional manifolds, the study evaluates two metrics: Spearman’s ρ to assess monotonicity and a new Scaling Rate Index (β) to characterize the spacing pattern as sublinear, linear, or superlinear.

### Strengths
1. The target question is interesting. If number-related questions in the area of LLMs can be thoroughly understood, it may lead to further significant advancements.

2. The proposed methodology is well-motivated and interpretable.

3. The conclusion is highly inspiring, as it establishes a connection between cognitive psychology and representational analysis in LLMs.

### Weaknesses
1. The paper shows that compression exists, but not why (e.g., is it due to token frequency, positional embeddings, or training distribution?).A deeper analysis of architectural causes or training data statistics would strengthen the work.

2. It’s unclear how logarithmic compression affects numerical reasoning benchmarks or real-world performance (e.g., arithmetic, scale extrapolation).


3. Although multiple runs and controls are used, confidence intervals are sometimes narrow, and cross-model consistency could be better quantified.

### Questions
1. What is the causal origin of the logarithmic compression? Could it be attributed to the frequency distribution of numbers in the training corpora, which approximately follows a power-law?

2. Have you compared the compression patterns across different tokenization schemes (e.g., digit-based vs. word-based numerals) to rule out tokenizer-level artifacts?

3. Have the authors examined whether a similar logarithmic compression of numerical magnitude also emerges in MLLMs that jointly encode visual and textual inputs? For instance, do vision-language models exhibit comparable sublinear spacing when representing quantities inferred from visual stimuli? Investigating this direction could help determine whether logarithmic number representation is a general emergent property of large-scale multimodal learning, rather than a phenomenon restricted to linguistic models.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper challenges the prevailing linear representation hypothesis, arguing instead that LLMs encode numerical values on a compressed, logarithmic number line. To test this, the authors extract hidden states for numerals, project them using dimensionality reduction, and then measure monotonicity and spacing using a novel Scaling Rate Index. Their analysis consistently reveals strong sublinear trends across various LLM families, indicating a compressed, non-uniform geometry. The authors validate this finding through causal interventions that modulate next-number predictions and demonstrate its robustness in real-world settings while showing its absence in non-numerical controls.

### Strengths
1. The authors tackled the novel and underexplored problem of how LLMs internally represent and manipulate numeric values.
2. The relevant papers and theories are well-introduced in the manuscript, which helps the reader to easily follow the logic of the proposed method.
3. Two properties, monotonicity and scaling, are well defined and analyzed with appealing metrics.

### Weaknesses
1. The authors experimented with a single prompt, which is very unlikely to appear in a real-world scenario. Therefore, it is unclear how the experimental findings in this paper can be generalized to real-world data. To improve generalizability, the authors should consider using simple prompts with single numbers (e.g., "2047"), varying the number of preceding in-context examples (e.g., i) "2047=2047 104=?", ii) "2047=2047 104=104 37=?"), or experimenting with numbers from grade-school math QA pairs such as GSM8K, ASDiv, or other datasets. By conducting the same analysis on diverse prompts and reporting the results with statistical analysis, the authors' contribution would be more concrete.

2. The models used in the paper are limited in their numerical reasoning capabilities. Even if the authors are not able to utilize commercial, closed-source models, open-source and light-weight reasoning models such as Qwen, DeepSeek, GPT-OSS, and any other model families that show better results on numerical reasoning are available. I believe that including models with stronger numerical reasoning abilities would help reveal whether the poor understanding of numbers is a general limitation of LLMs. Moreover, including bidirectional language models (e.g., diffusion LMs) might be beneficial. Because the number is always placed at the end of the prompt in the probe's current design, this could introduce positional bias. In bidirectional LMs, the position of the number can be varied, which would be free from such unintended bias.

3. Since PCA and PLS convey different analysis results, I wonder how robust the analysis results proposed in the paper are. For example, if another dimensionality reduction method were utilized, would the analysis results change? If so, how can we say that PCA and PLS are the most effective dimensionality reduction algorithms for analyzing number representations?

### Questions
1. In Table 1, why do the authors compare results from different layers? For LLaMA-3.1-8B, comparing the monotonicity metric and sublinearity coefficient from the same layer seems fairer. If this is not the case, it would be better if the authors elaborated on the reason for this in the manuscript.

2. Why were only LLaMA models used for experiment 2?

### Soundness
2

### Presentation
3

### Contribution
2
