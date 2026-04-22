# Vision Language Models are Biased

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Large language models (LLMs) memorize a vast amount of prior knowledge from the Internet that helps them on downstream tasks but also may notoriously sway their outputs toward wrong or biased answers. In this work, we test how the knowledge of popular subjects hurts the accuracy of vision language models (VLMs) on standard, objective visual tasks of counting and identification. We find that state-of-the-art VLMs are strongly biased (e.g., unable to recognize that a 4th stripe has been added to a 3-stripe Adidas logo), scoring an average of 17.05% accuracy in counting (e.g., counting stripes in an Adidas-like logo) across 7 diverse domains spanning animals, logos, chess, game boards, optical illusions, and patterned grids. Removing image backgrounds nearly doubles accuracy (by 21.09 points), revealing that background visual cues trigger these biased responses. Further analysis of VLMs' reasoning patterns shows that counting accuracy initially rises with thinking tokens, reaching ~40%, before declining with model overthinking. Our work presents an interesting failure mode in VLMs and a human-supervised automated framework for testing VLM biases. Code and data are available at: vlmsarebiased.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work introduces VLMBias, a new benchmark that tests whether VLMs over-rely on memorized prior knowledge instead of actual visual evidence, by asking them to perform objective counting and identification on counterfactual images where subtle but crucial details (e.g., an Adidas logo with 4 instead of 3 stripes) have been altered. It shows that leading VLMs achieve only 17.05% mean accuracy on these counterfactual visual questions and default to biased, knowledge-aligned answers about 75% of the time, despite performing perfectly on unmodified versions of the same objects.

### Strengths
- Novel Task
- Easy to Follow
- Comprehensive Coverage
- Interesting Findings

### Weaknesses
- No Detailed Explanation
- Unclear Influence

### Questions
- The evaluation largely treats current model behavior as monolithic averages. It reports mean accuracies and bias rates but gives limited fine-grained error analysis/explanation by failure mode, which would help figure out why models fail beyond "they’re biased."
- Have you conducted any quantitative evaluation of the visual quality or realism of the edited images compared to the original ones, beyond manual inspection and filtering?
- While these findings are interesting, I find it difficult to determine how they and these benchmarks can realistically drive LLM research and applications.
- The benchmark intentionally tests minimal edits to expose perceptual bias, and does not explore performance under larger, more salient differences (like a six-stripe Adidas logo).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VLMBias, a benchmark designed to systematically measure visual biases in Vision-Language Models (VLMs) when objective visual facts conflict with common textual priors. The authors show that current state-of-the-art VLMs—including Gemini-2.5 Pro, GPT-4.1, o3, o4-mini, and Claude 3.7 Sonnet—perform very poorly on simple counting and identification tasks when the images contradict “common sense” visual facts (e.g., a four-striped Adidas logo or a three-legged bird).

## Key findings

1. Mean accuracy across seven domains (animals, logos, flags, chess, board games, optical illusions, and patterned grids) is only 17.05%, with 75.7% of answers biased toward prior knowledge.
2. Removing background context roughly doubles accuracy (+21 points).
3. Longer reasoning (more tokens) initially improves performance, but “overthinking” degrades it.
4. Bias persists even in the newest “thinking” models and tool-using agents, suggesting the bias originates in the language priors of the underlying LLM, not vision encoders.

### Strengths
1. **Clear empirical evidence**: strong empirical evidence through a broad and systematic evaluation across seven visual domains—ranging from naturalistic (animals, logos) to abstract (flags, chess boards, optical illusions, patterned grids)—and five major VLM families (Gemini-2.5 Pro, GPT-4.1, Claude 3.7 Sonnet, o3, o4-mini). 
2. **Novel benchmark framing**: benchmark framing is particularly novel in how it redefines the study of bias in multimodal models: rather than focusing on social, linguistic, or demographic biases—as most prior work does—it isolates visual prior bias by designing neutral, objective tasks that require models to rely solely on perceptual evidence rather than memorized world knowledge. 
3. **Investigating the source of the problem**: the authors train a logistic regression probe on frozen features from the SigLIP 400M vision encoder used in LLaVA-OneVision-S to predict leg counts (4 vs. 5). The probe achieves 95.26 % accuracy, while the full VLM using the same encoder performs at random chance (49.7 %) and outputs “4 legs” for 99 % of samples. This striking dissociation confirms that visual representations are accurate, but the language or fusion components override perceptual evidence, providing a clear mechanistic insight into where the bias emerges.

### Weaknesses
1. **Counterfactual design asymmetry**: The benchmark tests animal images modified to have one additional leg (e.g., 5-legged mammals, 3-legged birds) but does not include cases with one fewer leg. This one-sided manipulation limits interpretability of the bias direction—whether VLMs are simply drawn to the most familiar visual prior (e.g., “four legs”) or whether they exhibit asymmetric sensitivity to feature additions versus deletions. Including both types of counterfactuals would provide a more balanced and diagnostic test of how VLMs handle deviations from canonical visual knowledge and help disentangle whether bias strength depends on the magnitude or direction of the visual alteration.

### Minor Comment:

Please reduce the amount of model and task icons in the text, which—while visually distinctive—make the text visually cluttered and hard to parse. In several key pages, the abundance of small brand logos (e.g., for datasets or models) and overlapping color elements distracts from what's being discussed.

### Questions
1. **Extremal counterfactual realism**: How do models behave when the counterfactual modification becomes visually implausible, such as an animal with only one leg or no legs at all? Including such non-realistic or extreme manipulations could reveal whether VLMs fail gradually as realism decreases or exhibit a sharp threshold where visual absurdity triggers default, biased responses.
2. **Mitigation Experiments**: It remains unclear whether the authors have explored any debiasing interventions to evaluate how VLMBias might function beyond diagnostics—as a tool for improving model training. Given that the paper convincingly identifies where and how visual bias manifests, it would be valuable to test whether counterfactual data augmentation, explicit visual grounding objectives, or reweighting of chain-of-thought reasoning could reduce the bias measured by the benchmark.

### Soundness
4

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
4

### Summary
In this paper, the authors evaluate the degree to which vision-language models are biased towards prior knowledge learned during pre-training when encountering images that exhibit counterfactuals.  In particular, they focus on images where the count of a particular object (an animal’s legs, the stripes in a well-known logo, etc) are different than expected in the canonical case.  In addition to counting, they include modified optical illusions (though there are only a handful of these).  They find in all of these cases, state-of-the-art foundation models struggle to actually look at the images they’re presented, often defaulting to responses that reflect the canonical or dominant count / illusion.  Curiously, they find that dropping the background of counterfactual images dramatically reduces bias.  Finally, they show that while reasoning tokens can help, it results in a ceiling performance of ~40% after which point additional reasoning hurts.

### Strengths
The writing in this paper is very clear and the figures / results are presented in a way that makes their takeaways easy to grasp.  Overall, it is extremely polished in appearance.

I appreciate the authors’ rigor in using images generated from multiple different models, testing with multiple image resolutions (and, for their visual illusions, varying illusion strength) and testing with multiple runs.  

I also find the background removal & inverse scaling results quite interesting and a nice contribution to the existing body of research that explores common knowledge biases in VLMs.

### Weaknesses
### Conflation of knowledge prior and counting deficiencies 

The authors argue that existing commonsense knowledge benchmarks with synthetically generated counterfactuals (like Zhou et al. 2023 and Bitton-Guetta et al. 2023) exhibit language prompt biases which set VLMs up to fail.  While this seems plausible, it is also well documented that VLMs struggle with counting (see “Teaching CLIP to Count to Ten” by Paiss et al, ICCV 2023 or “Understanding the Limits of Vision Language Models through the Lens of the Binding Problem” by Campbell et al, Neurips 2024).  As model counting is a core prerequisite for the tasks the authors present, it seems important to disentangle counting deficiencies from the knowledge prior bias for the authors to make their claims.  I suspect that counting fails at a higher rate when there’s a counterfactual with a canonical count at play but can the authors show this?

### Narrow focus on counts

Nearly all the tasks the authors choose (with the exception of optical illusions, of which there are only 6 types) probe the reliance of VLMs on canonical counts they may have seen during pre-training.  While this is sufficient to demonstrate a knowledge bias in VLM numeracy for the class types they select (animals, logos, flags, game boards), this is quite narrow.  This work would be strengthened with an expansion to other kinds of knowledge biases models may exhibit (for example, color, shape, material, size, relative position, etc). 

### Novelty

As the authors note, prior work from 2023 (Zhou et al and Bitton-Guetta et al) have already explored synthetically generated counterfactuals for commonsense knowledge in a way that has broader coverage than the focus on numeracy in this work (as discussed above).  The authors argue that their work improves on this prior work as they have carefully constructed neutral statements to test model performance while prior work prompt models with questions where the answer according to the language prior is wrong.  Can the authors quantify the effect of this neutral framing on their results?  How much do their results change with non-neutral prompts?  How important is this – after all, the existing benchmarks are biased towards generating the wrong answer so to do well on them models really do need to “look at” their images (which is the goal after all?).  That doesn't seem like that much of a deficiency, if at all?

### Source of bias

The authors claim the source of the numeracy bias they observe is predominantly from textual pre-training (lines 25-31, 90-92, 287) but I think this merits stronger motivation through experimentation.  For example, the background removal results seem to suggest that the bias might be an artifact of the training of the vision encoder.  In line 363, they reference linear probing results on CLIP that they perform and while this does suggest that numerical information is expressed in CLIP’s vision embeddings, it does not mean that the bias is from the LLM.  For example, the issue might be in multimodal alignment (the numerical information is not properly mapped into the model’s text embedding space so the LLM is not ignoring the count information so much as extrapolating based off the coarse visual information it is receiving).  The authors might better tease these results apart by training similar probes on visual representations after mapping to the LLM’s space.  Additionally, it would be illuminating to 1) compare probe performance here on images with no canonical counts (to account for issues with counting in general in these models) and 2) see if a probe trained to count logo elements can generalize to animal legs or vice versa.

### Nitpicks

"Are VLMs biased?"  is quite broad/vague title that goes well beyond the experiments / claims in the paper which are more narrowly focused on biases in VLM numeracy.  I think the paper would benefit with a narrower / more specific title.

Additionally, “these tasks cover the most common subjects natural and manmade” reads like a bit of an overclaim.

### Questions
The pattern matching results reads differently than the other counting results – a bias towards in-context knowledge rather than learned knowledge.  I wonder if the authors can quantify the relative strengths of these somehow – perhaps through images with multiple animals with extra legs and a final animal with the expected number of legs (or vice versa).  

Can the authors expand on their background removal result at all?  As mentioned above, it seems at odds with their claims that the LLM is driving the numeracy bias given that the change is primarily in the visual encoding.  Are models less good at recognizing what the object is when the background is removed (and as such, can’t rely on knowledge in its weights to answer the counting question?).

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
3

### Summary
This paper investigates how prior knowledge biases vision-language models (VLMs) rather than visual evidence when performing objective counting and identification tasks. The authors propose VLMBias, a benchmark of 1,392 counterfactual (CF) images across seven domains (animals, logos, flags, chess pieces, game boards, optical illusions, patterned grids), where well-known objects are systematically modified by adding or removing signature elements. They evaluate 5 SOTA VLMs (Gemini-2.5 Pro, Sonnet-3.7, GPT-4.1, o3, o4-mini), showcasing that they achieve 100% accuracy on unmodified images but drop to 17.05% on CF images, and answers align with the predefined bias option 75.70% of the time. On yes/no identification, accuracy collapses from 100% on originals to 25.11% on CFs. The authors show further analyses, demonstrating that this bias is exacerbated by contextual cues in the image background and is not easily overcome by simple prompting strategies like few-shot examples or double-checking the answer, resulting in minimal improvements.

### Strengths
- The paper is well written and is easy to follow. The authors also report consistency in findings across 5 runs for select models. Do the numbers in the table show an average of 5 runs or just 1 of 5 runs?

- The papers show a comprehensive evaluation of 5 SOTA closed-source models (Gemini-2.5 Pro, Sonnet-3.7, GPT-4.1, o3, o4-min) in the main paper. In the appendix, they show 2 more recent versions of closed-source SOTA models (Grok and GPT-5) and 4 open-source models (Qwen, Pixtral, Molmo, and Moondream) to strengthen the generality of the findings

- This paper highlights a critical vulnerability: a tendency to "hallucinate" based on priors rather than perceive what is actually present. It also exhibits actionable insights, such as simple pre-processing of background removal, which reduces the bias.

### Weaknesses
- The paper's primary weakness is that while the benchmark dataset is novel, the core finding that large models rely on learned priors and can fail on out-of-distribution or counterfactual inputs is well known, as highlighted in related works. The title highlights general bias for VLMs, but the paper shows only for two specific tasks - counting and identification. Also, VLMs cannot count reliably is also being explored. It is a well-known bias that the VLMs fail at counting problems [A, B].

- The authors mention it is an automated pipeline for creating the benchmark, but there is a human evaluation at the end of data generation, where each data is either accepted or rejected. Seeing table 1, it seems that benchmark creation is fully automated, but it would be a human-in-the-loop system. Full automation could yield to bigger benchmark dataset than 1,392 images.

- The paper demonstrates that simple prompts are ineffective, which is useful. However, the strong performance of pointing VLMs (e.g., Moondream) suggests that forcing localization is key. The authors could have explored more descriptive prompting, chain-of-thought, or other prompts (e.g., "First, locate each stripe individually, count them one by one, and then state the final number") to encourage a more procedural approach.

- The current benchmark focuses on additive modifications, for example adding a leg to the animal (2 legs to 3 legs, 4 legs to 5 legs) and adding stripes (3 stripes to 4 stripes), which often result in visually implausible subjects as mammals don’t have 5 legs but the image is more likely to have 3 (or 2) (visibly) legged mammals due to occlusion photographic angles. How do the VLMs perform then?

**References:**

[A] Yin, Zhenfei, et al. "Lamm: Language-assisted multi-modal instruction-tuning dataset, framework, and benchmark." Advances in Neural Information Processing Systems 36 (2023): 26650-26685.

[B] Xu, Peng, et al. "Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models." IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

### Questions
- Unifying the benchmark leaderboard? The paper could be strengthened more by combining the results of all the models, including GPT-5/Grok and open-source models (Molmo, Moondream, Qwen, Pixtral) - Tables 2, 12, 13, and 16 into a single main-paper table. This would greatly improve readability and comparison, avoiding appendix hunting.

- Although linear probing the vision backbone of the VLMs helps to state that it contains information but it is still being trained (Section A.8). The paper can benefit from using interpretability methods like attention/attribution visualization from the (frozen) vision backbone to show whether models still attend to the edited regions, yet output wrong answers.

- The paper presents perplexing results. For instance, the zero or near-zero accuracy of Gemini-2.5 Pro and other models on several tasks (Table 2) is striking. Is this due to answer-formatting, decoding policy, or genuine content failure? A short diagnostic breakdown (format vs. content error) would help interpretability.

- How do the models perform when provided with both the original and counterfactual images together? Are the models able to identify when asked to compare the two images side by side?

- For original vs counterfactual experiments. Why does the original set contain 66 unmodified images (line 291) compared to a bigger set for modified images? How was this subset chosen? 

**Minor Comments:**

- Line 291 - Table 2 is wrongly referenced as Table 21

- Line 1430: performnace -> performance

### Soundness
3

### Presentation
3

### Contribution
2
