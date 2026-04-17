# Video Scene Segmentation with Genre and Duration Signals

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Video scene segmentation aims to detect semantically coherent boundaries in long-form videos, bridging the gap between low-level visual signals and high-level narrative understanding.
However, existing methods primarily rely on visual similarity between adjacent shots, which makes it difficult to accurately identify scene boundaries, especially when semantic transitions do not align with visual changes.
In this paper, we propose a novel approach that incorporates production-level metadata, specifically genre conventions and shot duration patterns, into video scene segmentation.
Our main contributions are three-fold:
(1) we leverage textual genre definitions as semantic priors to guide shot-level representation learning during self-supervised pretraining, enabling better capture of narrative coherence;
(2) we introduce a duration-aware anchor selection strategy that prioritizes shorter shots based on empirical duration statistics, improving pseudo-boundary generation quality;
(3) we propose a test-time shot splitting strategy that subdivides long shots into segments for improved temporal modeling.
Experimental results demonstrate state-of-the-art performance on MovieNet-SSeg and BBC datasets.
We introduce MovieChat-SSeg, extending MovieChat-1K with manually annotated scene boundaries across 1,000 videos spanning movies, TV series, and documentaries.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a metadata-guided framework for video scene segmentation. Instead of relying only on visual similarity between adjacent shots, the authors incorporate genre conventions derived from IMDb textual definitions as semantic priors into a ViT-based shot encoder through affinity-based residual fusion, which functions similarly to cross-attention. They further introduce inverse-duration-weighted anchor sampling to improve pseudo-boundary generation during self-supervised pretraining and a test-time shot splitting strategy to handle long shots without retraining. In addition, they present a new benchmark, MovieChat-SSeg, with 1,000 manually annotated clips covering movies, TV series, and documentaries. Experiments on MovieNet-SSeg, BBC, and MovieChat-SSeg datasets show consistent improvements over previous methods such as BaSSL and TranS4mer. The authors also provide comprehensive ablations covering integration strategies, prompt designs, anchor sampling, and split thresholds.

### Strengths
1.The idea of incorporating genre metadata as soft priors is well motivated and practical.

2.The method is lightweight and can be easily applied to existing models.

3.Experimental results show consistent improvement across multiple benchmarks and good zero-shot generalization.

4.Ablation studies are detailed and analyze four aspects: genre integration, prompt strategy, anchor sampling, and shot split thresholds.

5.The proposed MovieChat-SSeg benchmark adds valuable evaluation data with diverse narrative structures.

### Weaknesses
1. The novelties of the proposed method could be highlighted by comparing with the most related methods, e.g. whether these genre and duration signals are newly proposed ? What are exactly advantages of these characteristics over the existing ones, e.g. video context.
[1] Video Scenes Segmentation Based on Multimodal Genre Prediction, Procedia Computer Science, 2020.

2. Robustness to noisy or conflicting genre priors is not explored. There is no experiment showing how the model behaves when the genre is wrong or uncertain.

3.Multi-genre scenarios are not addressed in detail, which may lead to bias or performance drop in real applications.

4.The efficiency impact of test-time shot splitting is not reported. This makes it difficult to assess its practicality for deployment.

5.Failure case analysis is weak and the qualitative examples focus mostly on positive results.

6. Although the state of the arts are compared in Table 9 of appendix, the comparison in Table 2 should include the methods published in recent two years to validate the superiority of the proposed algorithm.   

7. Typo errors, e.g. `a a scene` in line 162.

### Questions
1.The novelties and advantages of the proposed genre and duration signals could be highlighted.

2.How does the model behave if the genre prior is noisy or incorrect? Could you provide degradation curves or sensitivity analysis?

3.Is there statistical evidence that shorter shots are more likely to be boundaries? Could you provide quantitative analysis?

4.How do you handle multi-genre videos during training and inference?

5. What is the computational overhead of test-time shot splitting?

6.Could you show more failure case analysis, especially on non-narrative content such as news or user-generated videos?

7. Are there any related state of the arts that are published in recent two years for the comparison in Table 2 ?

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
The authors target the problem of scene segmentation for long form videos. They propose exploiting the genre information to enhance the semantic information to learn better shot representations during the pretraining stage. They do this by constructing genre-level textual prompts and encode them into embedding vectors and then use the relevance of each genre to the given visual token an affinity matrix and add it to the visual features. Furthermore the authors also propose anchor sampling scheme that gives more importance to shorter duration shots as against just using the sequence endpoints. At the test time, the authors further counter the high variability of the shot durations by just splitting the sho greater than certain threshold duration into three shots and use uniform sampling to extract the keyframes. Also, the authors propose a new manually annotated dataset which spans movies, TV series and documentaries and thus consists of sufficiently wide variety of inputs. For the pretraining stage the authors just use a standard contrastive loss annd also another cross entrop loss to distinguish a boundary shot from the non-boudary counterpart and use also the CE loss for finetuning. The authors finally reporrt the resuls on MovieNet SSeg for supervised training and evaluate the generalization on their datasets showing significant gains in Average precision for both and F1 metric for the first.

### Strengths
The overall proposed scheme for training including Genre guided shot representation and the inverse duration shot sampling seems to be sound semantically since in long videos the genre based prompts can help in the boundary detection. The test-time shot splitting strategy is standard and a sound way to handle the long videos considering variability in the shot duration. The results also sucessfully second that. The proposed dataset can also be used to drive the research further in this direction.

### Weaknesses
The proposed scheme doesn't seem to be very novel and can be more interpreted as standard way-out for solving this long-form video issue in the considered setup. The shot sampling although is a good finding but I am unsure of any novelty including the test time splitting. So method wise not a major contribution although results seem interesting. 

Initially I have given a rating 6 due to the impact solving the problem can make, but I might change it based on other reviews and the responses authors provide.

### Questions
When comparing with other baselines do the authors incorporate the textual information since they mentioned that they reimplent the BaSSL method. Or they just do it to avoid architectural differences? If they don't use the textual information while comparing then isn't it unfair to them?

### Soundness
3

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
This paper introduces a novel method for video scene segmentation, which extends beyond the traditional reliance on visual similarity between consecutive shots by leveraging production-level metadata in the form of genre conventions and shot duration patterns. Scene segmentation is designed to identify semantically consistent boundaries within long-form videos, thus linking low-level visual cues with broader narrative comprehension. The principal contribution of this work is the inclusion of narrative-oriented signals into a self-supervised learning framework, which  improves shot-level representation learning.

### Strengths
*  Integration of contextual movie level metadata: Inclusion of textual genre level definitions as semantic priors for guiding shot level representations.
* Duration based pseudo boundary generation: Inverse duration-based anchor sampling strategy for providing more weights to shorter shots as compared to fixed anchor approaches like BaSSL
* Introduction of a benchmark for visual scene segmentation called MovieChat-SSeg associated with domains like movies, TV series and documentaries.
* State of the art performance on MovieNet-SSeg dataset when compared to other state-of-the-art self-supervised methods like TranS4mer and BaSSL.

### Weaknesses
* Non-visual cues like background music and audio events help identify scene divisions but are not included.
* Genre based contextual information might be insufficient in dialogue heavy scenes with minimal visual variations.
* The final ablation studies do not indicate whether performance varies by genre class. For instance, including the definition of an action genre may affect results more than drama.
* Although the proposed method has been evaluated on movies, TV series, and documentaries, it can also be applied to other formats such as news broadcasts, vlogs, and educational videos.

### Questions
* What additional forms of production metadata—apart from genre, duration, shot scale, and shot angle—could be utilised as semantic priors, and how might their integration differ in terms of complexity and potential impact compared to primary signals such as genre and duration? Possible examples include camera movement classifications (e.g., panning vs. static), or color palettes associated with mood shifts or emotional cues associated with character.

* Do shorter video segments (avg. 7.4 min in MovieChat-SSeg) limit the model's ability to capture long-term narrative patterns in full-length content?

### Soundness
3

### Presentation
3

### Contribution
3
