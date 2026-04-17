# Cultivating Pluralism In Algorithmic Monoculture: The Community Alignment Dataset

- Decision: Accept (Poster)
- Scores: 6, 10, 4, 6

## Abstract
How can large language models (LLMs) serve users with varying preferences that may conflict across cultural, political, or other dimensions? To advance this challenge, this paper establishes four key results. First, we demonstrate, through a large-scale multilingual human study with representative samples from five countries (N=15,000), that humans exhibit substantially more variation in preferences than the responses of 21 state-of-the-art LLMs. Second, we show that existing methods for preference dataset collection are insufficient for learning the diversity of human preferences even along two of the most salient dimensions of variability in global values, due to the underlying homogeneity of candidate responses. Third, we argue that this motivates the need for _negatively-correlated sampling_ when generating candidate sets, and we show that simple prompt-based techniques for doing so greatly enhance the performance of alignment methods in learning heterogeneous preferences. Fourth, based on this novel candidate sampling approach, we collect and open-source _Community Alignment_, the largest and most representative multilingual and multi-turn preference dataset to date, featuring 233,319 comparisons from annotators spanning five countries. The dataset is available at https://huggingface.co/datasets/facebook/community-alignment-dataset. Overall, we hope that the Community Alignment dataset will be a valuable resource for improving the effectiveness of LLMs for a diverse global population.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work 1) identifies that general-purpose LMs are in the same "monoculture", that they struggle to represent diverse dimensions of preferences; 2) collects a dataset called community alignment, with negative-correlated sampling and human annotations.

### Strengths
+ community alignment could be a useful dataset
+ the discussion on monoculture is interesting

### Weaknesses
- In the monoculture analysis part, all models considered (for example, in Figure 1) are aligned language models. There is increasing recent research on how aligned models suffer from generation diversity issues, while base models are better. [1-2] Would it help alleviate the monoculture problem if including base models in text generation?

- Aside from that, the study on monocluture also only uses general-purpose LMs. This kind of reduces the surprise of the results, as most of the major industry models are trained on highly overlapping internet texts and post-training recipes, while "culture, value" was never the most important consideration in their training process, so there is naturally very little variation. I wonder if there are any models specialized to the culture/value space from the past few years of related research that might break from this pattern.

- The "negative-correlated sampling", albeit its fancy name, is a prompt that asks for "diverse values". The authors hope that by asking for multiple "diverse" items in the prompt, the generated items would be different from each other. Well, empirically this might have worked with the selected model. I'm a bit unsure if this "negative-correlated" is very reliable: it seems like it is subject to the model's specific niche capability that might change from model to model, rather than a sweeping and general method.

- I appreciate the community alignment dataset and releasing it. I wonder in addition to analyzing its statistics in Sec 4.1, is there any way to prove its usefulness in actual model training/development? Like, somehow adapting an existing model with some subset of the data leads to improvement on something? It's unclear the utility of this resource at the moment; maybe it was described in the very long appendix.

[1] Zhang, Jiayi, et al. "Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity."

[2] Karan, Aayush, et al. "Reasoning with Sampling: Your Base Model is Smarter Than You Think."

### Questions
please see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper introduces a large-scale dataset for heterogenous preferneces, with 200k preference demonstrations from 15k humans across five countries, and multiple annotations per prompt/response combination for a subset of the dataset. Additionally, the paper demonstrates how the current method of sampling many responses from an instruction-tuned model may be insufficient for getting coverage over the breadth of human preferences, and introduce negatively correlated sampling to (at least partially) mitigate this.

### Strengths
S1: I quite like the insight that people's preferences vary from one another far more than model responses from leading LLMs do - an important selection bias to document and mitigate.
S2: The contributed dataset is massive (200k comparisons from annotators from five countries), and is a very valuable resource for the community, especially in future pluralistic reward modeling.
S3: Negatively correlated sampling could be a useful technique for future work for getting more diverse candidate responses.
S4: Prompt-level overlap (L460) is a particularly exciting feature of CA for studying variation in human preferences, and was not explored in previous work.
S5: "As of today, Community Alignment is the largest open-source multilingual preference dataset and the first to feature prompt-level overlap in annotators along with natural language explanations for choices." (L476-477). These are exciting features!

### Weaknesses
W1: While an understandable omission for the space constraints, there is very limited analysis on the actual generated preferences. E.g., how much do people actually disagree in practice (interannotator agreement rates)? What were some of the features of the natural language preference explanations provided? What kinds of topics were covered in the prompts? A potential future camera-ready version of the paper could benefit from some additional of the above analyses (but I do not consider them essential).

### Questions
Q1: For preference alignment dataset construction, the paper says that "we start with a base language model, sample responses from it, ..." (L85). I wanted to ask - is it truly a base model? As base models often exhibit low instruction-following skills / chat persona, I believe it is common to do SFT first on chat demonstrations, or few-shot chat elicitation (eg., https://arxiv.org/abs/2312.01552). Could the authors clarify exactly what they mean here?
Q2: L109: "producing respnoses in English that align with only 41% of human preferences". I had a hard time contextualizing this number - what exactly is meant here? What would a baseline be?
Q3: Was there any attempt to see if none of the responses represented the raters well, as with the apple/banana fruit example? If not, do the authors think that future work will need to try to address a coverage gap in preferences by eliciting this from raters, or would you expect that something like negatively-correlated sampling be sufficient for eliciting broad coverage?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines how LLMs can address diverse and conflicting human preferences across cultural and political contexts. A large-scale multilingual study (N=15,000) shows human preferences vary more than LLM outputs. The authors introduce negatively-correlated candidate sampling to better capture this diversity and release Community Alignment, a large multilingual dataset (~200,000 comparisons) designed to improve LLM alignment with global populations.

### Strengths
1. This paper introduces a new perspective for evaluating preference dataset diversity using Inglehart and Welzel (IW) dimensions, revealing that 21 LLMs exhibit an “algorithmic monoculture” and are poorly aligned with human preferences.
2. The authors propose negatively-correlated sampling, an efficient method to enhance the diversity of LLM-generated responses.
3. They also release Community Alignment, an open-source multilingual preference dataset built with negatively-correlated sampling, containing about 200,000 comparisons from annotators across five countries, with samples balanced by age, gender, and ethnicity in three of them.

### Weaknesses
1. The paper lacks a related work section, making it difficult for readers to understand its position within existing research. For instance, it is unclear whether the proposed negatively-correlated sampling method is novel or adapted from prior work.
2. I am concerned about using only four dimensions—secular-rational vs. traditional and self-expression vs. survival values—to measure preference diversity. How representative are these dimensions overall? The authors should elaborate on their rationale and limitations.
3. The paper provides no quality analysis of the collected dataset, and its description is too brief. As a key contribution, the dataset’s quality and utility should be discussed in greater depth. It is also unclear why the authors did not train or evaluate models using their collected dataset.

### Questions
1. L179-180: we generate and curate three model responses to vary along one of four known dimensions
of variation in individual values. How do you check the quality of generated responses here?
2. Table 1 takes up substantial space but provides limited information, could the authors present these results in a more concise or effective format?
3. Why did the authors not train or evaluate models using their collected dataset, given that it is one of the key contributions of the paper?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reveals algorithmic monoculture in LLMs through a large-scale study showing that 21 state-of-the-art models produce responses aligned with only 41% of human preferences, particularly favoring secular-rational and self-expression values over traditional and survival-oriented values. The authors demonstrate that this homogeneity combined with temperature sampling prevents standard alignment methods (prompt-steering, SFT, DPO, GRPO) from learning diverse preferences from existing datasets. They propose negatively-correlated (NC) sampling to generate more diverse candidate responses and show that models trained on NC-sampled preference data achieve substantially better alignment with diverse values. Finally, they introduce Community Alignment, a new dataset of 200,000 multilingual preference comparisons built using NC sampling.

### Strengths
- Valuable dataset contribution: opensourcing a large-scale multilingual preference dataset with unique features (prompt-level annotator overlap   and comparison-level natural language explanations)
- Comprehensive and rigorous experiments and evaluation: Nationally representative samples from five countries, professional translations, and systematic evaluation of 21 LLMs. Testing four different approaches (prompt-steering, SFT, DPO, GRPO) shows the problem is fundamental rather than method-specific.
- Well motivated; Clear problem identification with practical solution: Effectively demonstrates that temperature sampling from 21 models fails to generate sufficient diversity. NC sampling is simple yet effective.

### Weaknesses
- Limited analysis of downstream performance trade-offs: No measurement of NC sampling's impact on general task performance or helpfulness. Prior work suggests diversity might degrade fine-grained learning signals and overall model quality. Without evaluating performance on standard benchmarks, practical applicability remains unclear
- Insufficient clarity in experimental procedures: Critical details are relegated to appendices or described too briefly. Section 3 lacks clear explanation of win rate computation, response generation from different model variants, and sampling parameters. The paper mentions a "GPT-4o-based judge" but doesn't explain the evaluation setup adequately
- Potential overstatement of results: The claim of "significant heterogeneity" in preferences (line 197) feels unsupported given Figure 1's concentrated distribution. Figure 2 mentions 20-40% coverage for temperature sampling but doesn't provide explicit percentages for NC sampling, making improvement magnitude unclear. Quantitative measures should support such claims. The paper occasionally uses imprecise language (e.g., "significant" without statistical tests)

### Questions
- What is the impact of NC sampling on standard NLP benchmarks (e.g., MMLU, general helpfulness)?
- What do the different lines in Figure 1's leftmost plot represent? Please add clearer legends.
- Figure 2 shows different models respond differently to NC sampling (Llama-3.3-70B minimal change, Claude-3.7-sonnet substantial). Do you have any insights on what accounts for these differences?
- How sensitive is NC sampling to the specific diversity prompt used? Have you tested variations?

### Soundness
4

### Presentation
3

### Contribution
3
