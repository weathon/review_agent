# VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
Safety evaluation of multimodal foundation models often treats vision and language inputs separately, missing risks from joint interpretation where benign content becomes harmful in combination. Existing approaches also fail to distinguish clearly unsafe content from borderline cases, leading to problematic over-blocking or under-refusal of genuinely harmful content. We present Vision Language Safety Understanding (VLSU), a comprehensive framework to systematically evaluate multimodal safety through fine-grained severity classification and combinatorial analysis across 17 distinct safety patterns. Using a multi-stage pipeline with real-world images and human annotation, we construct a large-scale benchmark of 8,187 samples spanning 15 harm categories. Our evaluation of eleven state-of-the-art models reveals systematic joint understanding failures: while models achieve 90\%+ accuracy on clear unimodal safety signals, performance degrades substantially to 20-55\% when joint image-text reasoning is required to determine the safety label. Most critically, 34\% of errors in joint image-text safety classification occur despite correct classification of the individual modalities, further demonstrating absent compositional reasoning capabilities. Additionally, we find that models struggle to balance refusing unsafe content while still responding to borderline cases that deserve engagement. For example, we find that instruction framing can reduce the over-blocking rate on borderline content from  62.4\% to 10.4\% in Gemini-1.5, but only at the cost of under-refusing on unsafe content with refusal rate dropping from 90.8\% to 53.9\%. Overall, our framework exposes weaknesses in joint image-text understanding and alignment gaps in current models, and provides a critical test bed to enable the next milestones in research on robust vision–language safety.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
VLSU proposes a multimodal safety framework with fine-grained severity across 17 patterns, creating an 8,187-sample, 15-category benchmark. Testing 11 state-of-the-art models shows >90% accuracy on unimodal signals but 20–55% on joint image–text reasoning; 34% of errors persist despite correct unimodal labels. Models also struggle to balance refusal and engagement (e.g., Gemini‑1.5: over‑blocking drops 62.4%→10.4%, but unsafe-content refusal falls 90.8%→53.9%). VLSU reveals gaps in joint understanding and alignment and provides a critical testbed for vision–language safety.

### Strengths
1. The authors designed a reasonable process to construct their dataset, carefully considering the combinations of text and image modalities. The process is also clearly described.
2. Compared to existing evaluation datasets, this paper’s dataset is more challenging and significantly larger.
3. Based on the dataset proposed in this paper, the authors conducted valuable analysis experiments to diagnose performance bottlenecks, which may help guide future work.

### Weaknesses
1. The authors claim in their contributions that “their dataset exposes failure points invisible to existing evaluations in Section 4.1.” However, in Section 4.1, they only mention that “existing multimodal safety benchmarks may not fully capture the challenges of joint vision-language understanding that our systematic approach exposes.” I did not see any specific descriptions of failure points that are invisible to existing evaluations. Are these systematic failure modes? If so, what are their characteristics and how common are they?

2. Considering that this paper was submitted to the datasets and benchmarks track, the number of tested models is rather limited—only 11 in total. Some popular SOTA model families, such as Claude and OpenAI’s o-series, were not even covered.

3. In addition, since combining different modalities may require good reasoning ability, it is necessary to supplement more results from reasoning models. Although the authors added some CoT experiments in Section 4.3, I think this is not sufficient.

4. Although this paper presents a larger and more challenging dataset for evaluating the risks of text-image combinations compared to existing work, the contribution is a bit incremental.

5. In Section 3, Stage 2, images need to be retrieved from a large-scale image repository. However, the authors did not specify which image repository was used. This could potentially lead to copyright, privacy, or safety issues.

### Questions
Could you please tell me what you think is the most fundamental difference between this work and existing works or datasets? 
For example, what kinds of joint multimodal risks can this work reveal that previous works or datasets could not? In which ways can it better help to address joint multimodal risk?

### Soundness
3

### Presentation
2

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
This paper introduces VLSU, a new multimodal safety benchmark. The dataset comprises 17 safety patterns obtained by permuting label categories across text-only, image-only, and joint modalities. The experiments show that current models struggle with joint image–text reasoning for accurate safety labeling.

### Strengths
* VLSU is comprehensive, with over 8,000 samples and 17 distinct safety patterns.

* The notions of borderline cases and the triplet safety pattern are useful and could inspire follow-up work.

* The experiments systematically expose the limitations of current models in multimodal safety understanding.

### Weaknesses
First, the borderline class may be subjective. The borderline class is defined as educational, informative, or discussion contexts. However, such contexts (e.g., the knowledge of making chemical weapons) can be subjective. How does VLSU ensure objectivity in this category? Including case studies of borderline cases would improve the clarity of the paper.

Second, the dataset generation pipeline is not sufficiently clear. For example:

* The image repository used in the retrieval process lacks descriptions and references.
* The two key instruction sets used in dataset generation (image-concept generation instructions and query generation instructions) are not provided.
  
Third, the authors should provide more details about language diversity. The dataset appears to be predominantly in English. If so, the authors are expected to evaluate safety performance across multiple languages or discuss this limitation.

There are also some smaller issues that merit clarification:

Since the authors state that "if the text modality is clearly unsafe, the joint label cannot be safe or borderline", why do the U-B-B and U-S-B patterns appear in Figure 3?

Why was Gemini-1.5 (and only Gemini-1.5) used for generating image concepts?

### Questions
1. How does the VLSU guarantee the objectivity of its borderline case? 

2. Could the authors provide more details about dataset construction and language diversity?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents Vision Language Safety Understanding (VLSU), a comprehensive framework and benchmark to systematically evaluate safety in multimodal foundation models. This framework addresses the problem of joint interpretation, where individually safe image and text inputs become harmful in combination. The papers evaluation of current VLMs reveals a gap joint understanding. More detailed analysis indicate a Systematic over-sensitivity to any unsafe component and high error rate when combining modalities despite correct assessments on each individual one.

### Strengths
- the problem is well motivated and formulated. The fact that safety assignments should consider the interplay of both text and image inputs is intuitive and easy to follow 
- introduces a comprehensive benchmark that will be a valuable contribution to the community
- safety taxonomy is grounded in prior work 
- I appreciate the more nuanced three class classification over prevalent binary safe/unsafe 
- Clear and significant findings on gaps in joint reasoning 
- detailed error analysis in sections 4 and 5 on lack of model nuance, text modality dominance and error distribution

### Weaknesses
# Major

The major weakness of this paper is a significant lack of details on the construction methodology which also limits reproducibility 
- further details on taxonomy guidelines (see below)
- **Stage 1 ** what is the exact setting for the "systematic parameterization"? what prompts where used and what were the inputs used in this parameterization?
- **Stage 2** There is no information provided on the image corpus used for retrieval. What is its size, origin, licensing? Is there some additional filters on image metadata wrt. resolution, aspect ratio, origins, etc? What exactly was the retrieval setting? 
- ** Stage 3** Again no details on the synthetization pipeline are provided. What were the prompts and inputs? 
- **Stage 4** no information on the makeup and setting of this human user study. Number of annotators, what makes them experts, compensation, instructions to annotators, measures to ensure that ethical guidelines where followed. What is the inter-annotator agreement? 
- In addition to overall statistics of the dataset what are the distributions over categories and do we have label balance within each category as well? 
-The paper should provide a datasheet for the newly introduced dataset (https://arxiv.org/abs/1803.09010) to adhere with standardized documentation practices, especially in safety related fields. 


## Minor comments

**Presentation**
The presentation of the paper could be improved in parts. For one all Figure and Table captions are rather short making it hard to grasp for readers only skimming the paper. Especially, Figure 3 and 4 feel hard to grasp and would benefit from some additional information or change in visual presentation. Page 8 is very convoluted and the results in Sections 4 and 5--while interesting---are presented somewhat disjointed resulting in a lack of a common thread throughout the paper 

**Human Oracle Topline**
Using a bootstrap of single annotators vs the majority consensus as human upper bound is more confusing to me then helpful. Since safety assessments are subjective to some extent deciding on a correct gold label per sample is challenging. Consequently, the human topline is much more an indicator of inter-annotator disagreement or the level of objectiveness then anything else. In my opinion the spread over inter-annotator agreement is a much more informative metric to plot and I would drop the human oracle. 

**Taxonomy**
While the taxonomy described in A.2 seems reasonable it is lacking some important details. Compared to prior works like LlamaGuard or LlavaGuard a detailed description per category on what content makes for safe, borderline and unsafe content respectively would be helpful. This also ties in with lack of information on the human annotator setup. Where they provided with additional guidelines on the taxonomy or did they have to figure those out themselves?

**Reliance on GPT-4o** 
The method uses GPT-4o to auto grade the severity of inputs. While hallucinations and inaccuracies here are mitigated by optimizing the system prompt agains a human annotated ground truth there might still be a selection bias here. However, since all final scores are human graded this is likely negligible, but the paper does not provide a qualitative assessment of GPTs accuracy on this pre-selection task. 

**lack of qualitative examples**
In the Appendix the paper provideds 5 examples per safety level of the dataset but overall there is a significant lack of qualitative examples. for example, a comprehensive set of per-category examples and outputs of different models is missing. 

**Limited Scope of Alignment Evaluation**
The safety alignment task (Section 4.4) only evaluates two models (Gemini-1.5 and Qwen2.5VL-32B) due to "compute constraints". While the findings are interesting (models trade off over-blocking for under-refusal based on prompts), this analysis feels incomplete. A broader evaluation across more models would be needed to claim this is a universal alignment gap.

### Questions
Please see the questions posed in the weakness section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents VLSU, a systematic vision–language safety framework and a benchmark of 8,187 real image–text pairs that captures cases where individually benign modalities become harmful when combined, and empirically demonstrates that models fail dramatically at compositional multimodal understanding. The authors introduce a Borderline class, the design of 17 combinatorial severity patterns, and a multi-stage pipeline that combines automatic filtering with human annotation.

### Strengths
1.	The inclusion of a borderline safety category is meaningful, and its necessity points to an appropriate direction for this field.

2.	The curation of 8,187 human-annotated real-image pairs, each categorized across harm types and joint safety patterns, is a significant step towards realistic, actionable safety evaluation.

### Weaknesses
1.	This paper does not contain several important related works that address multimodal safety evaluation benchmark, a key topic of this paper. A comparison with the following papers is necessary: ELITE [1], VLGuard [2], MLLMGuard [3], JailbreakV-28k [4]

2.	There is insufficient information about the human annotators. The paper states, "The image grade is labeled by one senior expert grader", which could introduce bias. Furthermore, the criteria for human annotators to judge "borderline" cases are ambiguous. Reliability could be improved by providing more specific guidelines and ensuring diversity among the annotators. This point is critically related to the overall reliability of the dataset.

3.	In lines 65-67, the issue of an image and text being individually safe but suggesting self-harm intent when combined has already been discussed in SIUO [5]. Although SIUO is cited in the related work section, it should also be cited in the Introduction.

4.	Adding qualitative examples for borderline cases would help understand the paper.

[1] Wonjun Lee, Doehyeon Lee, Eugene Choi, Sangyoon Yu, Ashkan Yousefpour, Haon Park, Bumsub Ham, and Suhyun Kim. ELITE: Enhanced language-image toxicity evaluation for safety. In Forty- second International Conference on Machine Learning, 2025.

[2] Zong, Y., Bohdal, O., Yu, T., Yang, Y., and Hospedales, T. Safety fine-tuning at (almost) no cost: A baseline for vision large language models. In Forty-first International Conference on Machine Learning , 2024. 

[3] Gu, T., Zhou, Z., Huang, K., Dandan, L., Wang, Y., Zhao, H., Yao, Y., xingge qiao, wang, K., Yang, Y., Teng, Y., Qiao, Y., and Wang, Y. MLLMGuard: A multi-dimensional safety evaluation suite for multimodal large language models. In The Thirty-eight Conference on Neural Infor- mation Processing Systems Datasets and Benchmarks Track, 2024.

[4] Luo, W., Ma, S., Liu, X., Guo, X., & Xiao, C. Jailbreakv: A benchmark for assessing the robustness of multimodal large language models against jailbreak attacks. First Conference on Language Modeling, 2024

[5] Wang, S., Ye, X., Cheng, Q., Duan, J., Li, S., Fu, J., ... & Huang, X. Safe Inputs but Unsafe Output: Benchmarking Cross-modality Safety Alignment of Large Vision-Language Model. Findings of the Association for Computational Linguistics: NAACL 2025

### Questions
1.	Could the authors provide a comparison with the other benchmarks mentioned (e.g., ELITE, VLGuard)? This would significantly help in establishing the reliability and contribution of VLSU.

2.	Could you provide the per-category inter-annotator agreement and Pearson correlation among the annotators?

### Soundness
1

### Presentation
2

### Contribution
1
