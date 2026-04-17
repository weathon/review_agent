# We'll Fix it in Post: Improving Text-to-Video Generation with Zero Training

- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Current text-to-video (T2V) generation models are increasingly popular due to their ability to produce coherent videos from textual prompts. However, these models often struggle to generate semantically and temporally consistent videos when dealing with longer, more complex prompts involving multiple objects or sequential events. Additionally, the high computational costs associated with training or fine-tuning make direct improvements impractical. To overcome these limitations, we introduce NeuS-E, a novel zero-training video refinement pipeline that leverages neuro-symbolic feedback to automatically enhance video generation, achieving superior alignment with the prompts. Our approach first derives the neuro-symbolic feedback by analyzing a formal video representation and pinpoints semantically inconsistent events, objects, and their corresponding frames. This feedback then guides targeted edits to the original video. Extensive empirical evaluations on both open-source and proprietary T2V models demonstrate that NeuS-E significantly enhances temporal and logical alignment across diverse prompts by almost 40%.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces NeuS-E, a refinement technique that improves semantic and temporal consistency in video generation models. This method leverages temporal logic to measure the presence of inconsistent events and objects and use such information to regenerate the affected sets of frames of the videos, in multiple iterations.

NeuS-E is designed to be applied both to open and closed source models, without any additional training, and in benchmarks it improves temporal and logical alignment by about 40% against the baselines.

### Strengths
The present work presents a very original result on an important problem in video generation.

The problem identified by the authors is that video models often generate semantic and temporal inconsistencies. The approach taken in this paper has two big strengths:

* it applies results from Temporal Logic together with a video automaton, which is very novel (particularly, the way in which the weakest proposition is found by assessing each partial specification is very interesting)

* this result allows to improve on existing video models, without source code or weight access, and without any additional training, just by applying iterative edits; and it is designed to continue to be useful for future models as well

The results are also impactful, providing high improvements on a variety of metrics (e.g. +23.3 points on temporal-logic recall, and temporal and logical alignment enhancements of almost 40%). But especially, the fact that this improvement is more pronounced in more challenging settings makes this result potentially more useful in production systems.

Finally, the exposition is very clear and well written.

### Weaknesses
There are a few weaknesses in this paper though I don't believe any of these are major:

* A discussion of latency or cost could be useful. Plotting the quality of existing systems and NeuS-E vs. their relative costs could be insightful.

  * In diffusion models, an iso-latency inference comparison could be useful (and could be obtained by adjusting the diffusion step count, for example)

* The paper proposes a baseline of sequential generation, `where complex prompts are carefully broken down and generated sequentially`. I wonder how it would compare with a baseline in which the same process is followed, but the best of N generations at every step is picked (like beam search). The value of N could be picked such that both NeuS-E and the baseline have the same cost or latency.

* As mentioned in the paper, although the temporal-logic recall shows improvements, a quality drop is measured.

  * I wonder whether the degraded quality after the refinement iterations is just caused by lower quality frame-to-frame interpolation in the video models (when compared to just text to video generation). One way to verify whether this is the case could be to compare the quality of the refinement iterations against the same amount of iterations of regenerating the same frames but while using the original prompt.

* The model name in the annotation tool is mentioned to users. I don't think this is necessary

### Questions
* Can you please explain a bit (in this comment section) the role of controlled noise to enhance numerical stability in the model-checking process?

* "We can identify the optimal threshold for the VLM by treating the above problem as either a single-class or multi-class classification problem. We opt to do the latter." - any thoughts on why this is chosen, and how significant this selection is?

### Soundness
3

### Presentation
4

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
The paper presents Neus-E, which bases on Neus-V to identify segments in the video that does not comply with the prompt in terms of handling complex or sequential events. The method first converts the text prompt into propositions using LLMs, and builds a video automaton to find the weakest proposition that does not align with the video. Based on the proposition, the method further identifies the video segment to be re-generated and prompts the video generation model with revised prompt from LLM to replace the original segment with better alignment.

### Strengths
1. The paper introduces an interesting idea of using neuro-symbolic feedback for identifying video segments that misalign to the given prompt, which is then is re-generated for better alignment. The overall motivation, and the method for identifying such segments - using temporal logic and video automatons, are very sound and convincing.

2. The method is designed to improve videos in a zero-shot manner, without fine-tuning the video for better alignment. Furthermore, the method is virtually model-agnostic asides from the part for accepting image inputs for stitching video segments.

### Weaknesses
1. The actual method for improving the problematic video segments however, relies on iterations of re-prompting the video generation model. While it would be reasonable for the model to better understand simple, short prompts from the atomic propositions, it would still depend on blind luck that the new segment would be better. Some study showing that the re-generated segments are much more aligned to the prompts would better support the method.

2. The evaluation mostly focuses in NeuS-V, where the methodology for identifying misaligned video segments, and the evaluation metric, shares a lot in common. If the method is basically identifying regions that bottlenecks the NeuS-V score, re-generating such segments would trivially result in better NeuS-V scores. A more thorough examination, such as evaluating with T2VCompBench[1], would be helpful.

3. The paper is lacking qualitative results, and no further results could be found in the appendix or the supplementary materials. This adds up with the concerns in the quantitative results, making it unclear how the much the proposed method can improve the original videos.

[1]Sun, Kaiyue, et al. "T2v-compbench: A comprehensive benchmark for compositional text-to-video generation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
1. Why would the step-by-step fall so short compared to the proposed method? Considering that both method rely on the point that smaller video segments generated from propositions are expected have better alignment, the best achievable alignment for both methods looks to be the same in theory. Could this imply that the "strong" segments, opposed to the edited segments from NeuS-E, to have benefitted from the overall context of the video and should be kept?

2. How often would a re-generated segment be selected again in the next iteration? This relates to the previous question, to rule out the method from simply benefitting from having more trials within the iterations. 

3. Considering that the method combines multiple models, involving LLMs and VLMs, would it be also possible to identify misaligned video segments in a more simplistic manner with modern large VLMs? For example, it would be possible to simply input the video and the prompt a LVLM and request the model to identify the misaligned frames. Demonstrating that such understanding is still difficult even with state-of-the-art LVLMs would be interesting, and would further strengthen the motivation of the work for introducing neuro-symbolic feedback.

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
This paper proposes a training-free video refinement pipeline that uses neuro-symbolic feedback to automatically enhance text-to-video generation by identifying temporally misaligned segments and editing to improve prompt alignment. The method works by decomposing text prompts into temporal logic specifications, constructing video automatons to identify weak propositions and their corresponding problematic frames, then iteratively regenerating only the misaligned portions until temporal consistency is achieved.

### Strengths
- The conceptual aspect of the proposed approach is thoroughly developed and impressive. Using neuro-symbolic feedback as a form of structured feedback for video improvement is intuitive and makes sense.
- Extensive experiments have been conducted corresponding to the key points of the proposed pipeline.
- Consistent performance improvements have been reported with NeuS-V score.

### Weaknesses
While the conceptual approach is interesting, there are worries about practical performance improvements given that video generative models are primarily considered black boxes, and the absence of enough qualitative results.

- The need for the proposed pipeline fundamentally arises because video generation models fail to properly understand and reflect user prompts. However, the main element improved by the proposed NeuS-E feedback is also text prompts. Although text-based image editing models are used via keyframes as intermediaries, the origin of performance improvement could be because off-the-shelf image models simply follow prompts better.
- While quantitative analysis is provided, the amount of qualitative results provided in the paper is extremely limited and video results were not provided, which makes the practical performance of the proposed method less convincing.

### Questions
The biggest concern is the absence of sufficient qualitative results. How are the qualitative results? It would be good if they were provided in the appendix or supplementary materials.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The author introduces a zero-training video refinement pipeline that leverages neuro-symbolic feedback to automatically enhance video generation.

### Strengths
The author introduces an interesting zero-training video refinement pipeline that leverages neuro-symbolic feedback to automatically enhance video generation.

### Weaknesses
1.The author uses video editing to correct incorrect video generation. However, video editing is a more difficult task than video generation, with lower accuracy and higher computational resource requirements. Therefore, I think this approach doesn't make sense in practice. Relying on video editing for correction is less effective than improving the success rate of video generation in the first place.


2. This pipeline is too idealistic and complex. If the video generation involves complex object changes, this pipeline is likely to fail.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2
