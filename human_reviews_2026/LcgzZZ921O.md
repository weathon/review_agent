# VaseVQA-3D: Benchmarking 3D VLMs on Ancient Greek Pottery

- Decision: Accept (Poster)
- Scores: 4, 8, 2, 8, 4, 4

## Abstract
Vision-Language Models (VLMs) have achieved significant progress in multimodal understanding tasks, demonstrating strong capabilities particularly in general tasks such as image captioning and visual reasoning. However, when dealing with specialized cultural heritage domains like 3D vase artifacts, existing models face severe data scarcity issues and insufficient domain knowledge limitations. Due to the lack of targeted training data, current VLMs struggle to effectively handle such culturally significant specialized tasks. To address these challenges, we propose the VaseVQA-3D dataset, which serves as the first 3D visual question answering dataset for ancient Greek pottery analysis, collecting 664 ancient Greek vase 3D models with corresponding question-answer data and establishing a complete data construction pipeline. We further develop the VaseVLM model, enhancing model performance in vase artifact analysis through domain-adaptive training. Experimental results validate the effectiveness of our approach, where our VaseVLM-7B-RL achieves 12.8\% improvement in R@1 accuracy and 6.6\% improvement in lexical similarity compared to the strongest baselines on the VaseVQA-3D dataset, significantly improving the recognition and understanding of 3D vase artifacts, providing new technical pathways for digital heritage preservation research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the problem that existing Vision-Language Models (VLMs) lack the specialized data and domain knowledge to analyze 3D cultural artifacts like ancient Greek pottery, limiting their performance in this niche area. 
To overcome this, the authors created VaseVQA-3D, the first 3D visual question-answering dataset for this domain, by systematically filtering tens of thousands of 2D images, converting the best ones to high-fidelity 3D models, and then using this new dataset to train a specialized VaseVLM model. 
The paper then validates its approach by conducting a comprehensive evaluation of VaseVLM against a wide range of state-of-the-art 3D-specialized, closed-source, and open-source VLMs on the new benchmark. 
The superiority of this method is demonstrated through significant quantitative improvements, such as a 12.8% increase in R@1 accuracy and a 6.6% increase in lexical similarity over the previous best models, and is further confirmed by higher ratings from human archaeological experts.

### Strengths
1. The paper presents a well-structured and technically comprehensive pipeline for transforming 2D cultural heritage imagery into 3D visual question answering data, integrating filtering, reconstruction, and multimodal fine-tuning. This end-to-end design demonstrates methodological rigor and can serve as a reproducible framework for future domain-specific multimodal research.

2. The work introduces one of the first benchmarks focused on 3D VQA within the cultural heritage domain. By combining structured question–answer pairs with enhanced descriptive captions and expert-validated annotations, it provides a dataset resource for studying vision-language alignment in specialized, long-tail domains.

### Weaknesses
1. The abstract and contributions claim a "12.8% improvement on R@1 metrics and 6.6% on lexical similarity compared with previous state-of-the-art". However, the paper does not explicitly identify which specific model from Table 3 serves as this "previous state-of-the-art" baseline. While the R@1 improvement can be inferred by comparing VaseVLM-7B-RL (3.52%) to the highest non-proposed models (3.12%), the 6.6% claim for lexical similarity does not clearly correspond to any specific baseline in the table. It is recommended that the authors explicitly state the baseline model used for these headline claims.
2. The final VaseVQA-3D dataset consists of 664 3D models, with only 90 models in the test set. Conclusions drawn from a test set of this size may be subject to statistical noise and raise questions about the generalization capabilities of the models. 
3. The proposed Reinforcement Learning with Verifiable Rewards (RLVR) framework calculates rewards based on cosine similarity to ground truth content across six dimensions, using a threshold of τ=0.7. This reward design is inherently coupled with the final evaluation metrics, particularly "Lexical Sim.". This creates a risk that the model is not learning a deeper semantic understanding but is instead being optimized to "hack" the scoring function. The authors should discuss this potential for "overfitting to the metrics" and could strengthen their claims by providing evaluations on out-of-domain tasks or through more robust, task-based human evaluations that are orthogonal to the reward function.
4. The descriptive captions in the dataset were enhanced using GPT-4o. Since these enhanced texts serve as the ground truth for both model training and subsequent evaluation (e.g., retrieval and lexical similarity), there is a risk of information leakage. The evaluation may inadvertently reward models that are better at mimicking the linguistic style of GPT-4o rather than demonstrating a true understanding of the archaeological content. 
5. The selection of TripoSG over Hunyuan3D was based on a comparative analysis using the VaseEval validation set, which contains only 24 ground truth models. A conclusion based on such a small sample may not be generalizable to the full diversity of vase morphologies. The paper would be more convincing if this comparison were conducted on a larger and more varied set of real 3D assets to ensure the robustness of the chosen generation method.
6, A core methodological concern is the end-to-end process in which the dataset is self-generated, its captions are augmented by an LLM (GPT-4o), and the models are trained and evaluated against this same data ecosystem. This is compounded by the fact that the RLVR reward function directly mirrors the evaluation metrics. This tight coupling introduces a significant risk of bias, where the model's strong performance may stem from learning the artifacts and stylistic patterns of the dataset creation process itself, rather than achieving genuine "archaeological terminology understanding".

[Minor]
1. The paper positions itself as having significant social value for "digital heritage preservation". However, the results remain a "proof-of-concept" and do not address the practical feasibility, cost-effectiveness, or workflow for deploying this system in real-world scenarios such as museum curation, automated classification, or restoration assistance. For instance, the stated "13.5 days for 3D generation"  suggests significant computational cost. A discussion of these practical barriers would provide a more balanced perspective on the work's current impact.

### Questions
Please refer to the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, authors highlight the issue of state-of-the-art vision-language models being unable to perform well on specialized cultural heritage domains like 3D vase artifacts. They point out that existing models face severe data scarcity issues and insufficient domain knowledge limitations.
As a first step towards formalizing these limitations, the authors introduce one such domain: ancient Greek pottery. Authors demonstrate that existing models struggle to meaningfully answer questions about items in this new dataset.
The authors then provide a proof-of-concept approach to train a vision-language-model (VLM) to be specialized in this domain via "domain-adaptive training".

One of the distinguishing features of the present work is its 3-dimensional nature. Previously, the VaseVQA provided flattened 2-D images of vases from multiple aspects. In the present work, authors employ automated tiling methods to stitch these flattened images together to reconstruct a 3-D model to use in their ground-truth VQA data.

### Strengths
- Addressing an important, overlooked, application area for vision-language modeling via construction of an improved dataset.
- Providing proof-of-concept modeling approach that achieves improved performance according to certain metrics.

### Weaknesses
- I would consider making the pipeline diagrams more intuitive---currently they loop in and out in unintuitive ways and the arrows are hard to follow.

### Questions
- In what way are your specific metrics informative of how comparatively useful your dataset is vs. the previous one (VaseVQA-2D)?
- The authors should discuss how easily adaptable their specific methods (both for dataset construction and augmentation as well as modeling) are for other domains should researchers in other domains choose to construct similar pipelines for their visual artifact analysis. 
- It seems from Fig. 4 that even the preferred method, TripoSG, suffers from inconsistencies and mismatches with ground truth in its 3-D model reconstruction. Treating these as starting points for further model and benchmark development seems to sow seeds of data issues. What challenges will this benchmark face going forward?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a dataset and benchmark for 3D visual question answering on ancient Greek pottery. It develops a VLM model that demonstrated improved performance on vase artifact analysis.

### Strengths
- The paper addresses digital heritage presentation and proposes a dataset and benchmark for AI understanding on ancient potteries, which contributes to cultural protection.

### Weaknesses
- The method proposed was restricted to vase analysis, and does not generalize to other domains.
- There's very little novelty in the proposed approach -- it seems to be a great but standard engineering effort to piece together the pipeline, rather than a significant research effort.

### Questions
I think the contribution of the paper would be more significant if the proposed approach is demonstrated to be effective and efficient on several different domains rather than just one. Also I'm curious why choose visual question answering task on this dataset. The VQA task is designed to test an AI's ability to reason with both visual and text information, but the quality gap that motivated the paper just seemed like a regular "domain expertise" gap that can be covered with more data in the specific domain. Constructing a VQA task on a long-tail distribution data seems to defeat the purpose of the task.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces
1. VaseVQA-3D: A 3D Visual Question Answering dataset for ancient Greek pottery analysis. It aims to bridge the data scarcity and domain knowledge gaps for VLMs in cultural heritage. The dataset contains 664 high-quality 3D vase models generated from 2D images via a three-stage filtering and 2D-to-3D conversion pipeline, along with 4,460 enhanced question-answer pairs.
2. The work also proposes VaseVLM, a VLM fine-tuned for this domain, which achieves a significant improvement of 12.8% on recall at top 1 and 6.6% on lexical similarity over prior state-of-the-art models on the new benchmark (VaseVQA-3D).

### Strengths
## Originality and Significance
While there exist other 3D VQA datasets, this paper introduces the first benchmark to
1. Focus on 3D artifacts within a highly specialized, long-tail cultural heritage domain (ancient Greek pottery)
2. Require models to exhibit archaeological domain knowledge for answering questions (e.g.,identifying specific manufacturing techniques, dating periods, or decorative styles)

## Quality (Data Rigor)
The methodology for dataset construction is rigorous, featuring a three-stage filtering mechanism (ResNet-50, dual-CLIP) and a dedicated validation set (VaseEval - a small 3D models dataset) of real 3D models to ensure the quality and archaeological accuracy of the synthetic data.

## Quality (Model Performance)
The fine-tuned model, VaseVLM-7B-RL, demonstrates that specialization is highly effective for high-precision, domain-specific tasks. It shows
1. High-Precision Accuracy: The model shows a 60% relative improvement on R@1 over the powerful general-purpose Gemini 2.5 Flash, which is crucial for expert domains.
2. Domain Expertise: The significant lead in Lexical Similarity confirms that the fine-tuning successfully instilled the necessary specialized archaeological terminology and domain-specific knowledge.

## Clarity
The design of the full pipeline (Figure 5) and the experimental setup are presented clearly and logically.

### Weaknesses
## Data Scale and Synthetic Nature
The final dataset size of 664 unique 3D models is small. These 3D assets are also synthetic, meaning they were generated from 2D images. This limited, synthetic nature carries a risk. The model might end up overfitting to the specific style of the generated data. This could, in turn, limit how well the VaseVLM works on real-world archaeological artifacts.

## Trade-off in General Capability
The fine-tuned model excels on highly specialized metrics like $\text{R@1}$ and Lexical Similarity. However, the smaller VaseVLM-7B-RL still lags behind the massive, general-purpose Gemini 2.5 Flash. Specifically, it performs worse on broader retrieval metrics like R@5 and R@10. This shows a core limitation of fine-tuning a small model. Large, general foundation models still have an advantage because their sheer scale allows them to encompass a wider pool of plausible information.

### Questions
## Justification for 3D Generation Selection
The selection of TripoSG over Hunyuan3D is justified primarily based on achieving "more realistic results with better vase model quality" (visual assessment). However, the full paper already shows a trade-off in the quantitative geometric and visual metrics.  
**Question**: Given that the paper provides quantitative metrics, why was subjective visual quality prioritized over the composite quantitative scores in selecting the final 3D generation method?  
**Suggestion**: To provide a more robust defense of this subjective choice, the authors should quantify the visual assessment. This could be achieved by including a small-scale user study (e.g., 5-10 archaeologists or experts) to formally score the visual fidelity of the TripoSG and Hunyuan3D models on the VaseEval set. This would turn a subjective claim into a defensible metric.

## Generalizability and Overfitting Concerns
The final dataset is quite small and synthetic which raises a primary concern about the model's generalizability and risk of overfitting to the specific synthetic distribution.  
**Question**:
1. Could the authors provide additional experimental evidence to demonstrate the robustness of VaseVLM against overfitting?
2. Have the authors performed any zero-shot testing on a small set of real-world 3D vase models (outside the VaseEval set) to test generalization beyond the synthetic data?

# Suggestion
The general-purpose Gemini 2.5 Flash still holds a lead on broader retrieval metrics R@10 and overall generative quality (FID), even though the fine-tuned model shows significant gains in high-precision metrics R@1 and Lexical Similarity. The authors should reframe this discussion by emphasizing that in a highly specialized, expert domain like archaeology, precision R@1 is more critical than recall R@10. To fully validate the gain in R@1 as the true measure of success, the authors should include a qualitative error analysis in their rebuttal. This analysis should demonstrate a case where Gemini's top answer is plausible but turns out to be archaeologically inaccurate, and contrast it with a case where the specialized VaseVLM provides the exact, technical R@1 answer.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces dataset, VaseVQA-3D, a visual question answering (VQA) dataset for Greek pottery. The authors mention that there is a lack of datasets catering to cultural heritage artifacts. The paper makes the following contributions:
1. A data construction pipeline for transforming 2D images of vases into 3D models. The resulting dataset contains 664 3D models with over 4,000 associated question-answer pairs.
2. Fine tune a VLM (VaseVLM) for visual question answering in this domain.

### Strengths
The paper presents a well structured and technically sound pipeline for constructing 3D models out of 2D images and using them to fine tune a VLM. The end to end design could potentially be reproduced for other similar applications.

The paper expands the areas of application of language models by adding a dataset of 664 3D models of Greek pottery and associated question answer set.

### Weaknesses
The paper focuses exclusively on ancient Greek pottery and it is not clear why this domain is chosen over the others. While this is a valuable contribution, and brings AI to a new domain, maybe the authors could have provided a motivation as to why this over the other possibilities?

The dataset generation pipeline is composed of standard modules at each step. This seems to be a great software project, rather than a research project.

### Questions
.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper represents an interesting domain which is modeling 3D cultural heritage objects. The data construction pipline is clear and various data analysis prove the high quality of this dataset. The proposed baseline is interesting and its performance is impressive through extensive experiments.

### Strengths
- The author introduce a novel dataset about cultural heritage. This domain is interesting and has many cultural and historical motivation.
- Experimental results show that the proposed baselines is effective.

### Weaknesses
- The authors did not indicate what is the main characteristics of this domain that distinguish it from other domain in the same VQA task.
- The propose baseline is somewhat general. I cannot see which module or components are designed particularly for modeling the particular features of images in the specialized cultural heritage. Therefore the factors of giving VASEVLM performed better other VLMs are unclear.
- The authors only evaluated large models but not standard neural networks.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
