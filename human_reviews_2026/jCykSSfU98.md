# Sense it with your eyes: Sensation Generation and Understanding for Advertisements

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Effective advertisements might persuade the audience by evoking human sensations, yet current Text-to-Image (T2I) models struggle to generate persuasive visuals that convey implicit sensory experiences. We introduce the SensoryAd Generation task: given an advertisement message and a specific sensation related to the advertisement message, the goal is to generate advertisement images that both convey the message and evoke the sensation. To support this task, we build the SensoryAd dataset, consisting of human-designed and generated advertisements annotated with sensation categories, visual elements evoking the sensation, and human ratings. We further propose a novel evaluation method using contrastive and consistency losses across hierarchical sensation levels.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the novel task of SensoryAd Generation, which aims to create advertisement images that not only convey a given message but also evoke a specified human sensation (e.g., temperature, pain). To support this, the authors construct the SensoryAd dataset, comprising real and generated ads annotated with a hierarchical taxonomy of sensations, the associated visual elements, and human ratings of evocation strength. They further propose a sensation classification task to benchmark AI models and introduce EvoSense, a novel evaluation method using a fine-tuned LLM to measure how effectively an image evokes a target sensation. Finally, the paper benchmarks current Text-to-Image (T2I) models, revealing their significant limitations in successfully completing the proposed SensoryAd generation task, often failing to balance sensation evocation with message clarity.

### Strengths
1. The idea is nice, to explore the different human sensation evoked by visual elements.

2. The writing is readable and easy to follow.

### Weaknesses
1. The paper doesn’t clearly describe how the annotation was conducted — for example, how many annotators participated, whether each sample was labeled by multiple people, or how disagreements were resolved. In addition, the annotators’ backgrounds (e.g., expertise, age range, gender, or cultural factors) can strongly influence subjective annotations, but these aspects are not discussed. Providing such information is essential for understanding the reliability of the dataset.

2. The dataset seems extremely small — only about 670 real samples and 75 generated ones. With such a small amount of data, it’s difficult to believe that the trained models can yield results with strong statistical or practical significance. The authors should either expand the dataset or discuss the limitations more explicitly.

3. It would be valuable if annotators not only provided a label but also a brief explanation of why a given visual element evoked a certain feeling. This could help establish a more interpretable and causally grounded link between the visual content and human sensation.

4. Since this task is highly related to human perception and preference, incorporating some reinforcement learning or reasoning-based mechanism to align the model with human judgments could make the generated advertisements more convincing and appealing. As it stands, the approach feels too passive and purely descriptive.

5. The dataset is small and coarse, the annotation lacks depth, and there’s no real algorithmic innovation or a well-defined end-to-end framework for advertisement image generation. The experiments also seem quite superficial, a few MLLMs are used as baselines, but none of the analyses go deep enough. The paper gives the impression of touching on many ideas without fully developing any of them.

6. The main text references materials in the appendix but doesn’t include clickable links or clear cross-references. As a result, readers have to manually search through the appendix to find the relevant content. This makes the paper unnecessarily hard to navigate.

7. The text mentions only a few hundred images, but Table 2 claims there are 5,000 real and 5,000 generated images. This inconsistency must be clarified — is it a typo, or were additional samples collected later?

8. Using MLLM-generated image descriptions as inputs for LLM-based sensation classification seems fundamentally flawed. Different MLLMs emphasize different aspects in their captions, and without human verification or standardization, these descriptions introduce large uncontrolled biases. Without any demonstrations or quality checks, evaluating LLMs based on such synthetic inputs is, in my view, methodologically unsound.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new framework for understanding, evaluating and generating advertisement images that evoke human sensations, addressing the limitations of current Text-to-Image (T2I) models in this area. To accomplish this, the authors developed a hierarchical sensory taxonomy, which they used to construct the SensoryAd dataset, featuring detailed annotations on sensation type, intensity, and specific visual cues. They also introduce the SensoryAd Generation task, which requires generating an image that both conveys a given message and evokes a specified sensation. Critically, the paper proposes Evosense, a novel quantitative evaluation method that measures how effectively an image evokes a target sensation, utilizing a large language model (LLM) fine-tuned with a unique loss function that combines Contrastive Preference Optimization (CPO) with a Hierarchy Loss. Through several experiments, the results show that existing T2I models perform poorly on this specialized task. Meanwhile, EvoSense achieves a high agreement with human judgements, proving more reliable than other evaluation metrics like VQA-score.

### Strengths
# 1. Dataset and Evaluation Framework
This paper introduces a novel research direction that systematically explores the understanding and evaluation of advertisement images based on human sensations. It addresses the challenge of quantifying inherently abstract and subjective experiences by proposing a comprehensive framework. This framework includes a foundational dataset (SensoryAd) with detailed sensory annotations and an evaluation metric (EvoSense) designed to measure the intensity of evoked sensations.

# 2. Research Potential
The significance of this work lies in its systematic effort to model the intricate concept of human sensation within visual contexts. This contribution is expected to serve as a cornerstone for future research on generative models capable of understanding and reproducing implicit sensory experiences.

### Weaknesses
# 1. Validity of the Evaluation Metric
EvoSense first obtains an image description from an MLLM, then asks a text-only LLM to score the likelihood of the sensation term, reporting “average log-probability” as intensity. This setup may conflate lexical plausibility of captions with actual perceptual evocation in images. Moreover, the paper reports κ = 0.86 while the Pearson r is only 0.38 on real ads (Table 2), suggesting that categorical agreement might be inflated by frequent labels and does not convincingly validate the claimed intensity correlation.

# 2. Limitation of Generation
The SensoryAd Generation experiments rely entirely on textual prompting of existing models without any fine-tuning or conditioning on the SensoryAd dataset. Consequently, the results mostly reflect the models’ inherent prompt-following ability rather than learned sensation-aware generation. 

# 3. Suitability of the Evaluation
The task requires conveying both the Action-Reason (AR) message and the target sensation. However, quantitative results emphasize only EvoSense intensity (Table 3). The paper itself notes that models often exaggerate the sensation and overlook the message (Fig. 6; Appendix A.1). Without a complementary message-alignment metric, it is difficult to balance or interpret the trade-off between sensation and message fidelity.

# 4. Sensation vs. Persuasion
The paper positions sensation evocation as a persuasive strategy in advertising and implicitly assumes that stronger sensations lead to greater persuasive effect. Yet no behavioral or attitudinal measures of persuasion (e.g., message recall, emotional engagement) are collected, and evaluation remains limited to sensation-intensity scores. Including a human study or a message-alignment metric could help substantiate the claimed link between sensory evocation and persuasive impact.

# 5. Annotation Reliability is Under-reported
Although the paper outlines a multi-stage annotation process on Prolific, it does not specify the number of annotators per image, the inter-annotator agreement for either labels or intensity scores, or how disagreements and outliers were handled. Since EvoSense training converts these ratings into pairwise preferences, the lack of clarity around annotation reliability raises questions about the stability of the supervision signal.

# 6. Small Annotated Subset of Generations
For each T2I model, only 15 out of 75 generated images were annotated, which limits the reliability of the model-level comparisons in Table 3. Moreover, using only 10 generated images reduces the interpretability of the sensation-wise heatmaps in Figure 7. The small sample size makes it difficult to assess performance on less visual sensations mentioned in the paper and may lead to sampling noise rather than consistent model differences.

# 7. Need for Clearer Motivation of Sensation Classification
The Sensation Classification task is presented as a key component but lacks clear justification within the overall framework. While it measures whether models can recognize sensory concepts, its connection to the later evaluation and generation tasks is under-explained, leaving it somewhat disconnected rather than foundational.


# 8. Typos and Grammatical Errors
There are multiple errors: “a an” in the abstract (p. 1); “the an” and “Hierarchal” instead of “Hierarchical” (p. 2); “fo” instead of “of” (p. 3); “Classfication” in A.3 (p. 15); “different between” in A.2 (p. 14); and “coveys” instead of “conveys” in Table 10 (p. 20).

### Questions
# 1. Scope of Sensation Across Advertisement Types
Given that the dataset includes both commercial and public service advertisements, could you clarify whether sensation plays an equally important role across these different advertisement categories, or if its importance varies depending on the ad type?


# 2. Relationship Between Classification Accuracy and EvoSense Agreement
Since both the classification and EvoSense tasks rely on the same annotated data, could you explain whether higher classification accuracy for a given sensation is associated with stronger agreement between EvoSense and human judgments across models or sensation categories?


# 3. Human Baseline for Sensation Classification
Table 1 reports model performance for MLLMs and LLMs, but no human baseline is included. Could you provide information on how consistent human annotators are when predicting sensations using the same taxonomy prompts, to help contextualize the reported F1 and R_parent scores?


# 4. Sampling Strategy for EvoSense Pairwise Training
In Appendix A.3, EvoSense is trained on 50 images expanded into about 21k pairwise instances. Could you describe how the positive and negative pairs were sampled, and whether the sampling was random across sensations or constrained within the same hierarchy branch?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces SensoryAd, a benchmark and dataset for studying how advertisements evoke human sensations such as touch, taste, or temperature through visual cues. It defines two classification tasks for identifying sensations in ad images and a generation task aimed at producing ads that evoke specified sensations. To evaluate these, the authors propose EvoSense, a two-stage evaluation metric that uses a multimodal large language model to describe images and a fine-tuned language model (with contrastive and hierarchical losses) to estimate the intensity of evoked sensations. Experiments on 670 annotated ad images show that EvoSense correlates strongly with human judgments and reveals that current text-to-image models often exaggerate sensations while losing the intended advertising message.

### Strengths
1. The paper introduces a genuinely novel research angle: studying how visual advertisements evoke sensations beyond emotion or aesthetics. This focus on multi-sensory evocation represents a fresh and underexplored problem in multimodal learning.

2. The benchmark design is cohesive. The paper proposes a complete setup that includes a hierarchical taxonomy, well-defined sensation-classification tasks, and a generation task. Together, these form a usable foundation for further research on sensory perception in vision-language models.

3. The qualitative analyses are insightful. The examples showing “sensory exaggeration” highlight an important real-world failure mode of text-to-image models—overemphasizing sensory cues while neglecting message fidelity. This kind of observation can inform future model design and evaluation.

4. The paper’s overall presentation is clear and well-organized. The authors articulate the problem, dataset construction, and evaluation pipeline in a logical sequence, making the contributions understandable even to readers outside of advertising research.

### Weaknesses
1. The paper claims to develop a taxonomy of senses at different levels of granularity, where the first layer corresponds to the five fundamental sensory modalities.
   a. The description of annotation guidelines is vague. The statement that “each annotator was tested and approved based on completing a practice form after reading detailed instructions, definitions of sensations, and illustrative examples” is too high-level to assess reliability. It is unclear what exact criteria or examples were used, how consistency was ensured, or how ambiguous sensations were handled.
   b. It is also unclear how this sensory taxonomy meaningfully differs from the original PittAd dataset, which already contains message-based annotations (“I should do X because Y”) that capture intent and affect. The added sensory layer may be overlapping conceptually without sufficient justification for being treated as a distinct modality space.

2. The paper lacks comparisons with strong zero-shot and few-shot baselines such as GPT or large instruction-tuned multimodal models, as well as simple, transparent baselines like those used in Behavior-LLaVA. Without these comparisons, it is difficult to judge whether EvoSense or the proposed classification approach truly adds value beyond general-purpose reasoning models.

3. Dataset scale and diversity remain limited. The core corpus of 670 PittAd images (plus a small generated subset) is too small for training robust perception or evaluation models and risks domain bias. The reported 21k datapoints are augmented variations of this same source, limiting diversity rather than expanding it. Furthermore, there is a risk of evaluation circularity: EvoSense is fine-tuned on the same annotation distribution that it later evaluates. Even with held-out images, the taxonomy, task definitions, and metric are all co-designed, leaving open the possibility of domain overfitting or leakage from description-generation stages.

4. The claim that this is “the first investigation of how ads evoke the senses” is inaccurate. Prior work such as Behavior-LLaVA (ICLR) and related multimodal behavior modeling studies have already explored the link between generated content and human reactions, across a broader behavioral space—including Reddit upvotes, YouTube likes, replay rates, saliency, memorability, and comment patterns. Those efforts provide a more comprehensive foundation for connecting visual content to audience response.

5. A major conceptual weakness is that the paper equates sensation intensity with persuasive effectiveness without providing empirical behavioral validation. While the authors frame sensory evocation as “a persuasive strategy” and “a crucial element of persuasive advertising,” they offer no A/B or user studies (e.g., click-throughs, recall, purchase intent) to substantiate this claim. The link between sensory evocation and actual persuasion therefore remains hypothetical rather than experimentally demonstrated.

6. The T2I benchmarking is shallow. Only five models are tested, with approximately 75 images per model, fixed random seeds, and some quantization. This limited scope is insufficient to draw generalizable conclusions across a rapidly evolving T2I ecosystem, especially given the variation in prompt formats, diffusion steps, and stylistic priors across models.

7. The paper reports a κ = 0.86 “near-perfect” agreement between EvoSense and human annotators but does not provide confidence intervals, standard errors, or per-sensation breakdowns, which limits interpretability. Comparing κ across model types is statistically problematic because κ depends on class prevalence, which differs between LLM-only and MLLM-based setups. There is also no power or uncertainty analysis to demonstrate that these results are stable given the small dataset. All sensations are aggregated into a single κ/r score, even though some (e.g., “Glow”) are visually trivial while others (e.g., “Dryness”) are abstract and harder to ground. Without stratified or normalized results, it is unclear whether the model’s high κ reflects true generalization or dominance by a few easy classes. Consequently, the claim of “human-level” agreement is numerically strong but statistically under-supported.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces the SensoryAd task to generate an advertisement that evokes specific sensory cues. The authors build a build a sensation taxonomy, curate a dataset of 670 ads sampled from Pitts Ads data and collect crowdsourced annotations about sensory information evoked from these images. Two sensation classification tasks are designed – hierarchical and flat. Further, an evaluation method called EvoSense is presented. It measures how well a text-to-image model evokes a given sensation. Results indicate that the proposed EvoSense model shows better agreement with human annotations compared to existing metrics like ImageReward, CLIP score, Pick Score, etc. Finally, a benchmark of different text-to-image models on the sensory ad generation task is presented.

### Strengths
The task of sensory ad generation is well motivated for applications in advertising. This paper attempts to address a gap by studying this important problem in the context of text to image models.
The EvoSense evaluation pipeline is a key contribution. Compared to traditional image evaluation metrics (VQA score, ImageReward, CLIP score, Pick score), EvoSense demonstrates substantially better agreement with human ratings. 
The introduction of sensation classification tasks and benchmarking text to image models on sensory ad generation task are well-scoped problems useful to the community.

### Weaknesses
Conceptually, the goal of sensory marketing would be to enhance the appeal and persuasiveness of ads to a viewer. The paper does not discuss this connection about how invoking sensations impacts the persuasiveness of ad images. Further, the impact of the paper can be elevated if  the authors can provide evidence through a user study to show the above relation, if there exists any. If such a relation exists, the authors should discuss the ethical implications of sensory ad generation. 
The authors report that they trained EvoSense on only 50 images which is quite surprising to me. I wonder if there is enough signal in this data to help the model demonstrate generalisation. 
The authors sampled only 670 real ad images from the Pitts Ad dataset which is quite modest in size. It is not clear if the curate subset is representative enough and if there is any selection bias. The Pitts Ad data covers various ad types (eg: vintage vs modern) , industry sectors, etc. Can you please report the properties of the sampled version along these axes?
While the authors report the agreement of EvoSense with human ratings but details about inter-annotator agreement in the data seem missing. That would serve as a topline reference for interpreting the results of any evaluation method.
While EvoSense achieves high agreement with human ratings (κ = 0.86), however the Pearson correlation for this task is still a bit low (r = 0.38). Do these observations suggest that EvoSense might be good at picking the most relevant sensation but less so for actually predicting the intensity score of the sensations? I would like to see some more analysis being provided to explain this observation.

### Questions
Please see the weakness section and provide additional details as requested.

### Soundness
2

### Presentation
2

### Contribution
2
