# Vision Language Models Cannot Reason About Physical Transformation

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Understanding physical transformations is fundamental for reasoning in dynamic, real-world environments. While Vision Language Models (VLMs) show promises in embodied applications grounded in the physical world, whether they genuinely understand physical transformations remains unclear. To address this gap, we introduce \textit{ConservationBench} to evaluate \textit{conservation}—whether physical quantities remain invariant under transformations despite appearance changes. Spanning four quantitative properties (number, length, volume, size), each task requires integrating visual evidence across time and includes counterfactuals where the targeted quantities are not conserved, forming paired conserving and non-conserving scenarios. With systematic variation in prompts, frame sampling methods, and task design, we generate 13,824 questions evaluating on 34 VLMs. Results reveal consistent failure: none demonstrates systematic conservation. Performance remains marginally above chance, with improvements on conservation tasks often accompanied by severe performance on counterfactual controls. This suggests a dependence on superficial patterns or shortcuts over genuine understanding and reasoning on conservation. Moreover, models show no benefit from higher temporal resolution or prompt design. Together, these findings indicate that current VLMs fail to reason about physical transformation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper explores whether vision-language models (VLMs) can reason about physical conservation - that the quantitative properties of objects, such as their number, length, volume, and size remain invariant over short time periods. A large benchmark of videos and extracted frame sequences is constructed, and used to study 34 VLMs. They find that no VLMs can robustly reason about conservation, at least not to the level of the average human. This is a surprising result given the established success of VLMs on large-scale general benchmarks and the triviality (for humans) of the conservation tasks being used.

### Strengths
This paper is methodologically sound, and is an incremental contribution to a growing literature exploring the physical reasoning capabilities of VLMs. Particular strengths are:
* The rigorous control conditions used throughout to test alternative explanations for VLM model performance.
* The large number of open-source models used.
* The use of a meaningful human baseline for comparison.

### Weaknesses
The paper has a number of weaknesses:
1. The hybrid evaluation is an interesting solution to the problem of evaluating complex outputs, but using LLM judges incurs significant overhead for the practitioner. Since the paper currently relies only on open-source models (as far as I can tell) and the benchmark uses multiple choice questions, the authors could simply use the log-probability of the choice label, conditional on the text-image input. This could be normalised across the possible outcomes too. This is quite standard in benchmark evaluation (see EleutherAI's lm-evaluation-harness).
2. Many new models are also capable of processing videos. I would suspect that the next generation of VLM will process video data effectively. Therefore, the longevity of the benchmark would be ensured if video-based evaluation was a central contribution. The authors could conduct a small study with, say, Qwen-2.5-VL-7B (an open source video-language model) to see if there is any difference between performance on video and performance with frame sequences.
3. Many of the tasks used here might be quite novel to the language model, so it's not clear whether these models *could* learn to reason about conservation if given the right training, or whether there is something deeper about architecture/pre-training that prevents them from doing so. Two further experiments are required to elucidate this. First, examining whether in-context learning can boost model performance. Here, examples of conservation/non-conservation with correct labels are given sequentially prior to the test question. Second, examining whether supervised fine-tuning on a subset of the tasks improves performance. There are three conditions of interest here: training on (a) a random subset of the tasks; (b) conservation problems; (c) non-conservation problems. I suspect, however, that fine-tuning would be just as brittle (see Schulze-Buschoff et al. 2025). This would make a more powerful point about VLMs. It's not just that current models off-the-shelf are incapable, but it's a feature of the architecture/large-scaled pre-training that prevents them from having these common-sense intuitions.
4. I don't really think the white-image condition is a good control. To me, the bias towards invariance is shown by the delta between the conservation and non-conservation conditions. I would not expect systematic deviations from chance with white images, because where would that systematicity come from, absent some visual input? Perhaps I am missing the logic here.
5. There is some missing literature, which I include below.
6. It's a semantic point, but I dispute that the main contribution of this paper is a benchmark. Rather, it's a careful series of experiments to test some hypotheses about MLLM capabilities. I don't think anyone will want to use this as a benchmark again, due to (a) the overhead of the LLM judges, and (b) the narrow scope of what the benchmark seeks to measure. I would recommend reframing it as an empirical investigation of the physical intuitions of MLLMs, rather than a useful dataset for the practitioner.
7. MLLM and VLM seem to be used interchangeably throughout, starting in the abstract.
8. In Figure 1, the question for the 'Number' task is the same as the 'Size' task (Is the size of the playdough in the first image the same as in the final image?). Shouldn't it be something to do with coins (given the details in the list in Section 3.1).

### Missing Literature

There are some further studies on visual reasoning that are not mentioned:

1. Balazadeh, V., Ataei, M., Cheong, H., Khasahmadi, A. H., & Krishnan, R. G. (2025). Physics context builders: A modular framework for physical reasoning in vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 7318-7328).

2. Campbell, D., Rane, S., Giallanza, T., De Sabbata, C. N., Ghods, K., Joshi, A., ... & Webb, T. (2024). Understanding the limits of vision language models through the lens of the binding problem. Advances in Neural Information Processing Systems, 37, 113436-113460.

3. Schulze Buschoff, L. M., Akata, E., Bethge, M., & Schulz, E. (2025). Visual cognition in multimodal large language models. Nature Machine Intelligence, 7(1), 96-106.

4. Schulze Buschoff, L. M., Voudouris, K., Akata, E., Bethge, M., Tenenbaum, J. B., & Schulz, E. (2025). Testing the limits of fine-tuning to improve reasoning in vision language models. arXiv preprint arXiv:2502.15678.

With respect to cognitive psychology, the line of work on the tunnel effect, object files, and object persistence should be discussed and mentioned:

1. Burke, L. 1952: On the tunnel effect. Quarterly Journal of Experimental Psychology, 4, 121 – 138.

2. Flombaum, J. I., & Scholl, B. J. (2006). A temporal same-object advantage in the tunnel effect: facilitated change detection for persisting objects. Journal of Experimental Psychology: Human Perception and Performance, 32(4), 840. 

3. Flombaum, J. I., Kundey, S. M., Santos, L. R., & Scholl, B. J. (2004). Dynamic object individuation in rhesus macaques: A study of the tunnel effect. Psychological science, 15(12), 795-800.

4. Mitroff, S. R., Scholl, B. J., & Wynn, K. (2004). Divide and conquer: How object files adapt when a persisting object splits into two. Psychological Science, 15(6), 420-425.

5. Noles, N. S., Scholl, B. J., & Mitroff, S. R. (2005). The persistence of object file representations. Perception & Psychophysics, 67(2), 324-334.

6. Scholl, B. J. (2007). Object persistence in philosophy and psychology. Mind & Language, 22(5), 563-591.

### Questions
* This work and others all point to the conclusion that VLMs really aren't that good at 'intuitive physics'. And yet, they seem to perform really well on standard large-scale benchmarks and users love to use them. I'm intrigued whether the authors think that an inability to reason about conservation is actually *problematic* for VLM use and deployment?

### Soundness
3

### Presentation
4

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
The paper introduces ConservationBench, a video-based benchmark to test whether VLMs can reason about physical transformations by judging conservation (vs. matched non-conserving counterfactuals) of four quantities—number, length, volume, size. The suite varies temporal sampling (3/8/16 frames), frame-selection strategies (uniform, model-based, human-picked), and prompt types, producing 13824 trials. Across 34 models, the authors report performance only marginally above chance on average; improvements on conserving items often invert on matched non-conserving cases, suggesting brittle heuristics rather than true transformation reasoning. Human accuracy (on a subset) is ~95%. The paper concludes that current VLMs cannot reason about physical transformation.

### Strengths
- This paper focuses on conservation under transformation with a counterfactual non-conserving item.

- This paper provides the results under different prompt styles, frame counts, and frame-selection methods (uniform / human / SEVILA-style)

### Weaknesses
- The current evaluation setup only provides models maximum 16 frames. It is questionable that is this enough even for human to understand the physical transformation happening in the video. Therefore, the claim like “VLMs cannot reason about physical transformation” are overstated if the inputs to the models does not contain enough information to solve the task.

- The human baseline details are missing. How did you evaluate the human performance exactly? 

- The paper does not evaluate state-of-the-art closed-source VLMs such as Openai, Claude, and Gemini models.

### Questions
- The question in Figure 1 for number case seems wrong. 

- Please fix VLM/MLLM naming consistency (in abstract) and remove duplicated paragraphs (line 180-186). 

- It would be helpful to include a full input prompt to the model including both image inputs and text inputs. 

- What is the performance of the state-of-the-art closed-source VLMs?

### Soundness
2

### Presentation
2

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
The paper introduces a benchmark to test if models understand that some physical properties remain invariant under transformations. They show that models can not keep track of conserved properties very well. While testing whether models can keep track of physical properties is a nice idea, I feel the scope of the paper and the insights it delivers are too limited to recommend acceptance.

### Strengths
**Originality** I'm not aware of any previous work investigating conservation in vision language models. 

**Quality** The benchmark is well constructed.

**Clarity** The paper follows a clear outline, although I feel the writing is a bit bloated and unaccessible at parts.

**Significance** The paper adds to a growing body of work highlighting the shortcomings of vision language models with basic visual processing. I like the split of the benchmark in conserving vs. non-conserving stimuli. Also, the result that models seem to default to either, always saying a property is conserved or always saying it has changed, regardless of the actual transformation, is interesting. However, there have been a large number of papers evaluating specific visual properties in vision language models with specific datasets/benchmarks.

### Weaknesses
In general, I feel like this paper does not provide a strong enough novel contribution to recommend acceptance. There is at this point a large growing body of evidence that vision language models fail at very basic visual processing. While this paper adds some novel data to this pile of findings, I find that the aspect of perception that it investigates is just too narrow. Also, the authors do not offer concrete ideas on how these problems could be overcome.

### Questions
**Main questions**
- How is the chance rate that is mentioned throughout at 33% if the models are always asked a binary question (does the property change versus does it not change)? Is this because you map the answer either to one of the two options or to "Fail" if it can't be parsed correctly? This is not the most obvious way of computing a baseline for me, if we accept that the models could output anything and LLM judges finally decide if the output maps to one of the two answer options, the chance level is not obvious to calculate. In any case, I think the reasoning for why you set it to 33.3% should be made more transparent in the text.
- It's interesting and a bit strange to me that the CoT prompting performs the worst. Could you maybe speculate a bit on why that is?
- For the human evaluation, you write "The aggregated human accuracy reaches 95.25%". Just to make sure, this is the accuracy given videos with all frames and in the "non-strict" evaluation, right? I think this could be made a bit more clear. 
- Initially I thought the number of frames (3, 8, 16) would be combined with the different methods of selection (Uniform, Human-selected, SeViLA), but here you seem to report them separately in Figure 3. For clarity, B shows uniformly sampled 3, 8, and 16 frames, right? And for C, is the number of frames fixed for all three selection methods and if so, what is it? Again, I may have missed these details in the text but feel they could be outlined more clearly, maybe even in the caption of the Figure.

**Minor comments**
- I think the abstract is too long and should be cut down.
- Line 43 "Yet it remains unclear whether VLMs possess a true understanding of physical principles or the capacity to operate reliably in embodied physical environments." there is previous work that shows VLMs do not understand physical principles [1, 2].
- Line 82 comes out of nowhere "Physical quantity refers to the measurable magnitude of objects along certain dimensions, while spatial transformation denotes the continuous processes through which objects change in appearance under perception." What is this in reference to? It seems a bit misplaced.
- Line 315 " 95. 25%", there's likely a space too much.
- Figure 2A the colors in the legends do not match the color of the bars in the plot?
- Figure 2 captions and titles are a not coherent, "Non-conserving" in the caption and "Non-conserve" in the title. 

[1] Schulze Buschoff, Luca M., et al. "Visual cognition in multimodal large language models." Nature Machine Intelligence 7.1 (2025): 96-106.

[2] Balazadeh, Vahid, et al. "Synthetic vision: Training vision-language models to understand physics." arXiv e-prints (2024): arXiv-2412.

### Soundness
3

### Presentation
2

### Contribution
2
