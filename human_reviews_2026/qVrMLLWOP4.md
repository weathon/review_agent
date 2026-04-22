# Un-Doubling Diffusion: LLM-guided Disambiguation of Homonym Duplication

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Homonyms are words with identical spelling but distinct meanings, which pose challenges for many generative models. When a homonym appears in a prompt, diffusion models may generate multiple senses of the word simultaneously, which is known as homonym duplication. This issue is further complicated by an Anglocentric bias, which includes an additional translation step before the text-to-image model pipeline. As a result, even words that are not homonymous in the original language may become homonyms and lose their original meaning after translation into English. In this paper, we introduce a method for measuring duplication rates and conduct evaluations of different diffusion models using both automatic evaluation utilizing Vision-Language Models (VLM) and human evaluation. Additionally, we investigate methods to mitigate the homonym duplication problem through prompt expansion, demonstrating that this approach also effectively reduces duplication related to Anglocentric bias. The code for the automatic evaluation pipeline is publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the rate at which VLMs generate multiple senses of homonyms--identically spelled words with multiple distinct meanings--when prompted with just the single word. They curate a list of 171 frequently used homonyms in English to create a benchmark quantifying the duplication rate using both human crowdsourced evaluation as well as VLM as a Judge methods. The human evaluation showed 5% of images containing duplicates. Next, the authors investigate the Anglocentric bias in VLMs through this lens, since models tend to translate non-English prompts into English before generation, potentially resulting in images depicting the wrong meaning of the original word. By curating the list of 171 homonyms into distinct translations in Russian, they find that up to 50% of images generated depicted the wrong sense of the word. Using prompt expansion techniques in the original language proved to be an effective mitigation, dropping the rate to 22%.

### Strengths
- This paper is very well written and easy to follow. I especially appreciate the thorough related works and clear examples in Fig 1.
- The problem of the anglocentric bias of the homonym duplication problem is novel and important. The results show pretty definitively that this problem is widespread and a simple mitigation (prompt expansion) is fairly effective.
- The homonym list curation is very thorough and sound, including linguistic experts as well as experienced translators. 
- Eight diffusion VLMs were evaluated across various sizes and families
- I found the proper name bias analysis interesting

### Weaknesses
1. The automatic evaluation seemed to have pretty low agreement with humans, so I'm not really sure how reliable it is. The human evaluations also have only 90% instances with a consensus, it would strengthen the paper to address this. For example, the authors/additional annotators/expert annotators could make a determination on these cases. 
2. The single-word prompt into the models, while specially designed to be ambiguous for the sake of evaluation, are not realistic. It could be interesting to see if this phenomenon occurs in slightly more realistic prompt templates (e.g. "Generate an image of [word]" or variations thereof). This would also be closer to the expanded prompt scenarios.
    - On a related note, the prompt expansion mitigation didn't fully solve the problem (decreased by only half a percent on human eval). It would strengthen the paper to include an error analysis here. For example, did the cases in which duplicates were found for expanded prompts due to the expanded prompt also being ambiguous as to the meaning of the word? 
3. In the human evaluation, only about 5% of instances were judged as duplicates. How does this compare to prior works investigating this phenomenon? Even if it's not the same exact models, it would be good to have a sense of whether the frequency is in the same ballpark or not. 
    - In general, the paper lacks a discussion on how the findings fit into the related literature. Including this would greatly strengthen the paper.
4. All figures are low resolution and the text is extremely small (esp Figs 2-5), making them very hard to read. I'd suggest uploading them as PDFs in the LaTeX so they maintain resolution, as well as increase the text size to match the actual font of your paper.

### Questions
Please address the questions in the weaknesses section first. Below are more things I'm curious about.
- It would be cool to see if the Anglocentric bias issue occurs in any VLMs that are stronger multilingual models (ie don't rely on a translation). 

- Why do you use "VLM" in most of the paper, but in line 339 "VLLM"? Is this intentional?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the systematic failure mode of the diffusion-based text graph model when dealing with homonyms (for example, "palm" can refer to both the palm of the hand and the palm tree): the model tends to "play it safe" and draw all possible meanings in the same diagram, rather than picking the most appropriate semantics. The authors call this phenomenon "homonym duplication" . The authors point out that this phenomenon not only affects English prompts, but is also related to "Anglocentric bias": many generative systems translate non-English prompts into English before inference. As a result, words that were originally unambiguous in the source language are translated into English homographs, thereby introducing non-existent ambiguity and leading to incorrect or repeated generation (for example, the Russian word that clearly means "date/meeting" becomes "date" after translation into English, which may trigger two images of "date fruit + date on the calendar").

### Strengths
1. Systematic Benchmark + Large-Scale Empirical Research:
• 171 homographs (across English and Russian), 94,000+ generated samples, 430,000+ manual annotations, and coverage across 11 major open-source diffusion models. The scale and coverage are both high, sufficient to support the conclusions.
2. Pragmatic Metric Design:
• HDR directly addresses the user-perceivable problem of "multiple meanings appearing in a single image," rather than an abstract embedding score.
• PFFR explicitly distinguishes between "the model doesn't actually draw the image" and "the model draws the image correctly but without ambiguity," preventing the misjudgment of "low HDR = good."

### Weaknesses
1. The HDR metric itself relies on subjective judgment: Whether it's human annotators or the VLM, each is subject to personal interpretation or inherent model bias, resulting in non-objective and reproducible conclusions about whether an image exhibits polysemy.
2. The consistency between automated and human evaluation remains low: Although the authors report a moderate AUROC*, the overall correlation remains unstable, making it difficult to consider automated evaluation as a reliable alternative.
3. The proposed LLM expansion method did not significantly reduce HDR under human evaluation, only down by 0.5 points: The reduction in HDR using human annotators' metrics was limited, suggesting that the actual user benefits of this mitigation may be overestimated.

### Questions
In your results, (1) HDR determination is inherently highly subjective/model-biased, (2) the correlation between automated and human evaluations is low, and (3) LLM expansion does not significantly reduce HDR under human evaluation. Given these three points, how do you support the conclusion that "our approach is effective and can serve as a general mitigation solution/evaluation pipeline"? Specifically, can you provide: the significant reduction in HDR under human evaluation, and why we can trust automated evaluation when there is a discrepancy between automated and human evaluations?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies the homonym duplication problem in text-to-image (T2I) generative models, where a single ambiguous word is rendered with multiple meanings within the same generated image. To examine this issue, the paper constructs a list of homonyms and uses it to evaluate 11 T2I models. To alleviate the problem, it proposes a simple prompt expansion technique that uses a large language model (LLM) to rewrite single-word prompts into longer, disambiguated descriptions before passing them to the T2I models.

### Strengths
The paper tries to address the homonym duplication problem in T2I generative models, which is an under-explored topic in the current literature.

### Weaknesses
1. Problem formulation. 
   - The paper addresses the homonym duplication problem only for single-word prompts. This is an unrealistic setting for modern T2I applications, as real-world prompts are typically longer and more descriptive. The study should extend its scope to include general prompts rather than limiting itself to single-word cases.  
   - The negative impact of homonyms (or polysemous words) in T2I generation is not limited to duplication; it also includes the generation of undesired or incorrect concepts [1]. The paper does not clearly justify why it focuses exclusively on the duplication.

2. Limited novelty. 
   - The proposed evaluation metric is just a simple majority-voting, which limits its novelty.  
   - The proposed prompt expansion technique merely rewrites a single word prompt through textual expansion using an LLM. This approach lacks novelty and may unintentionally distort the meaning of the original prompt.  

3. Filtering criteria of the homonym list. 
   - To serve as a meaningful benchmark, the filtering criteria used to construct the homonym list should ensure that the selected homonyms are representative and unbiased. However, some criteria described in Section 3.2 appear to prioritize annotation convenience over representativeness or importance of homonyms.

[1] "Cross-Attention Head Position Patterns Can Align with Human Visual Concepts in Text-to-Image Generative Models." (ICLR 2025)

### Questions
Please refer to the Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1
