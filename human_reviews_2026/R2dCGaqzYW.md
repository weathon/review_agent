# What Do VLMs See? Benchmarking Vision-Language Models on Ambiguous Images

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Vision–language models (VLMs) have demonstrated remarkable capabilities in visual recognition and reasoning, in some cases even surpassing human performance on standard benchmarks. However, it remains largely unexplored whether VLMs possess higher-order aspects of human perception, such as abstract interpretation and the capacity to manage cognitive ambiguity, and to what extent. models with more human-aligned interpretive and reasoning abilities. In this paper, we introduce \textbf{AmbiBench}, a benchmark designed to systematically evaluate how VLMs perceive and reason about ambiguous images relative to human interpretations. AmbiBench comprises 2,238 ambiguous images spanning nine categories, including object-level, scene-level, and a newly introduced mixed-ambiguity class, paired with 2,687 carefully constructed visual question–answer pairs. Evaluation of 12 state-of-the-art VLMs reveals substantial limitations: in five categories, models achieve less than half of human accuracy, and on mixed-ambiguity images, most collapse to near-zero performance.
Our study shows that humans flexibly navigate multiple interpretations, shifting between global and local perspectives, whereas VLMs largely rely on dominant features and exhibit restricted perception and reasoning under ambiguity. We further probe the existence of perceptual-switch heads—attention mechanisms that may underlie cognitive ambiguity—using bistable images.
AmbiBench exposes critical gaps in current VLMs’ capacity to handle perceptual ambiguity and establishes a foundation for developing models with more human-aligned interpretive and reasoning abilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes AmbiBench, a benchmark for VLMs for understanding ambiguous images compared to human interpretations. Two main categories of ambiguity are defined that include several cases of ambiguity within an image. Based on these ambiguity cases, the benchmark dataset is collected by combining existing data and purposely generated images. The benchmark demonstrates that even advanced closed-source models have difficulty interpreting ambiguous images correctly.

### Strengths
- The paper provides a systematic categorization of ambiguity types.
- The evaluation is extensive and includes human baseline comparisons.
- The work incorporates mechanistic interpretability through attention similarity analysis of switch heads within VLMs.

### Weaknesses
**W1:** The primary novelty lies in the scale and combination of several ambiguity classes together, which represents an incremental contribution in my opinion (If I understood it correctly from the paper, there are already benchmarks for the specific ambiguity classes). 

**W2:** The quality of AI-generated images is not adequately addressed. If VLMs struggle to understand ambiguity, it raises questions about whether generative models can reliably design such ambiguous content.

**W3:** The dataset exhibits imbalance between categories, which may affect the validity of cross-category comparisons.

**W4:** The paper lacks comparison to existing benchmarks. Since most of these ambiguity types have been benchmarked individually in prior work, a comparison not only within this benchmark but also across existing benchmarks would provide valuable context and demonstrate added value.

**W5:** The results would benefit from more rigorous statistical analysis to make the findings more robust and meaningful.

Minor W6: Section 4.3 could be strengthened by incorporating additional mechanistic interpretability methods such as SAEs or classical saliency methods, which could visualize the switching of attention heads directly as heat maps overlaid on the images.

### Questions
See Weaknesses W1 to W5.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces AmbiBench, designed to evaluate vision-language models (VLMs) on ambiguous imagery. The benchmark covers three major ambiguity levels, object-level, scene-level, and mixed, and includes 2,687 carefully curated Q&A pairs across nine categories. There are 4 types of tasks: open-ended, multiple-choice, ambiguous localization, and local region description. Experimental results show large performance gaps between humans and VLMs. Furthermore, by comparing attention head activations before and after informing models that an image is bistable, the authors compute a switching score based on cosine similarity and identify a small subset of attention heads that act as perceptual-switch heads, suggesting an emergent internal mechanism for perceptual switching within VLMs.

### Strengths
- The paper presents a well-structured benchmark that systematically covers multiple levels of visual ambiguity and diverse task types.
- The authors evaluate a wide range of VLMs and conduct comparisons with human performance, offering insight into the gap between machine and human perceptual reasoning.

### Weaknesses
- While AmbiBench covers a broader range of ambiguity types, it remains unclear what makes it distinctively different from prior illusion-based benchmarks. For example, are there cases where models perform well on IllusionVQA but poorly on AmbiBench? If so, a discussion on what specific perceptual or reasoning abilities the models lack would strengthen the paper.
- The paper highlights that all evaluated VLMs perform significantly worse than humans, but provides no direction on how such performance gaps might be closed. Suggesting possible strategies or training paradigms would make the contribution more forward-looking rather than purely diagnostic.
- The paper repeatedly emphasizes that AmbiBench includes difficult tasks, yet does not articulate why succeeding on these tasks is meaningful. It would be valuable to discuss whether improvement on these ambiguous tasks correlates with broader visual reasoning or compositional understanding in other domains.
- The analysis in Sec. 4.3 relies solely on cosine similarity between attention activations before and after revealing bistability, which suggests correlation rather than causation. It remains unverified whether the model truly performs perceptual switching between interpretations, or if the observed activation change simply reflects prompt-induced attention shifts.

### Questions
- Since part of the dataset was collected from online sources, is there a possibility that some images were previously seen during pretraining, especially by frontier models such as GPT-5 or Gemini? If so, how was data contamination mitigated?
- IllusionVQA intentionally filters out images that GPT-4V already explains well to retain only challenging cases. Was any similar filtering or difficulty calibration applied in AmbiBench to ensure that the benchmark indeed focuses on ambiguous or failure-prone examples?

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
2

### Summary
The paper focuses on the evaluation gap in visual-language models regarding their ability to perceive and reason about ambiguous images. It innovatively proposes the AmbiBench benchmark, which fills a key void in systematically assessing VLMs’ higher-order human-like cognitive abilities, such as abstract interpretation and ambiguity resolution. The construction of the AmbiBench benchmark is methodologically rigorous. It includes 2,238 ambiguous images and 2,687 visual question–answer pairs, categorized into nine types of ambiguity. The paper clearly reveals the core limitations of VLMs in processing ambiguous images.

### Strengths
1. The paper identifies a key gap in current VLM evaluations. While VLMs perform well on standard benchmarks, their ability in higher-order cognition, such as abstract reasoning and ambiguity resolution, remains underexplored. 
2. AmbiBench includes 2,238 carefully collected and ambiguous images across nine categories (seven object-level, one scene-level, one mixed) and 2,687 curated QA pairs.
3. Four complementary tasks (open-ended QA, multiple-choice, object localization, and local region description) capture different aspects of VLM performance. They jointly assess global and local perception, providing a more comprehensive evaluation than earlier single-task benchmarks.

### Weaknesses
1.	The mixed ambiguity category, newly introduced as a core class, contains only 28 samples. Although these samples were manually curated, the small scale may affect the statistical stability of the evaluation results for this category. It is recommended to either include more mixed ambiguity samples or provide an explanation for the limited sample size and its potential impact on the results.
2.	The investigation of the perception-switching head mechanism focuses solely on bistable images, without further validation of its generalizability to other types of ambiguity (e.g., multi-scene or mixed ambiguity).

### Questions
See weaknesses.

1. The distinction from TET [1] can be further elaborated and discussed.
2. The experiments do not clearly specify the LLM judge’s criteria for verifying answers in open-ended and multiple-choice questions, particularly regarding the definition of all expected explanations. Providing additional details would enhance the reproducibility of the evaluation process.

[1] PIXELS, PATTERNS, BUT NO POETRY: TO SEE THE WORLD LIKE HUMANS, 2025.

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
3

### Summary
The paper introduces a new benchmark to investigate how VLMs perceive ambiguous images. They find that models perform worse than humans on all types of ambiguous images. They argue that humans can switch between local and global features, wheres VLMs are over-reliant on dominant features.

### Strengths
- *Originality*
The underlying question is interesting albeit a bit constrained.
- *Quality*
The idea behind the benchmark seems sensible, but I have doubts regarding the human experiment and reliability of the experiments.
- *Clarity*
The submission is clearly written. 
- *Significance*
The paper introduces a new benchmark for ambiguous images, however the scope of the investigation is somewhat limited.

### Weaknesses
While the general question is interesting, there are a number of related studies investigating how aligned VLMs are with humans on specific perceptual tasks. While this paper adds some new data to this line of work, I don't think the study offers enough novel insights into the differences between human and machine vision, and it also does not provide solutions to bridge these differences.

### Questions
**Main questions:**
- This is very related to the literature on the perceptions of illusions in VLMs. I think this could be fleshed out a bit more in the related works section, including citations to [1].

- Can you give more examples of the Hybrid category? The example in the Appendix is not very obvious to me and I can not seem to see the lion in it. In fact, the example wrong VLM answer with "face, circle" seems more fitting to me than the example correct response. 

- Human performance seems very low in the Hybrid and Multi-View experiments. In fact, it is so low that I think the task is too hard to really extract meaningful differences between models and humans. Additionally, you select a subset of 10 images per category, which is too little and you might end up selecting a subset that is a lot easier or harder on average compared to the full set. And then you only test 3 people on the 10 images if I understand correctly? That again is too little data to make a proper claim about human performance.

- If the number of humans tested and the number of images they are tested on really is as little as it seems, then I do not think the human performance reported here is reliable. Since parts of the stimuli in this paper are novel generated stimuli, we do not know how well they actually work as illusions (and looking at a few reported in this paper I am doubtful they are all meaningful). Human performance would be an important baseline to understand how well the stimuli work and if they are sensible illusions. Since this study does not offer a good measure of human performance it is very hard to put these stimuli and the model performance in context.

- I'd be really cautious about the take-aways from the Motion stimuli in the "Perception Catering to Human Preference" paragraph starting at line 425. Do you compare against negative examples here? As you hint yourself, models may have just learned that humans have this bias or they may be biased by the text prompt --- in any case, I would like to see a baseline here of what they answer for similar images that don't illicit this moving bias in humans. 

- While the perceptual switch head analysis is interesting, I wonder what I should take away from it. Is the point that only the activation of a small number of heads changes between the two different interpretations of a given image? To me it is not entirely unsurprising that many heads do not show a large change, as at least the input image (and therefore likely the majority of input tokens) is still the same for both interpretations. I think to really get at a causal mechanism ("are these heads what allow for perceptual switching in these models") you would have to ablate the a top number of heads and see if the model now can't produce the alernative interpretation of the image anymore. 

**Minor comments**:
- Caption for Figure 2: "categoreis" instead of categories
- Line 227 superfluous dot after subsection headline



[1] Ullman, Tomer. "The illusion-illusion: Vision language models see illusions where there are none." _arXiv preprint arXiv:2412.18613_ (2024).

### Soundness
2

### Presentation
3

### Contribution
2
