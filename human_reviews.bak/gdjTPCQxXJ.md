# From Scarcity to Efficiency: Improving CLIP Training via Visual-enriched Captions

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3

## Abstract
Web-crawled datasets are pivotal to the success of pre-training vision-language models, exemplified by CLIP. However, web-crawled AltTexts can be noisy and potentially irrelevant to images, thereby undermining the crucial image-text alignment. Existing methods for rewriting captions using large language models (LLMs) have shown promise on small, curated datasets like CC3M and CC12M. Nevertheless, their efficacy on massive web-captured captions is constrained by the inherent noise and randomness in such data. In this study, we address this limitation by focusing on two key aspects: data quality and data variety. Unlike recent LLM rewriting techniques, we emphasize exploiting visual concepts and their integration into the captions to improve data quality. For data variety, we propose a novel mixed training scheme that optimally leverages AltTexts alongside newly generated Visual-enriched Captions (VeC). We use CLIP as one example and adapt the method for CLIP training on large-scale web-crawled datasets, named VeCLIP.  We conduct a comprehensive evaluation of VeCLIP across small, medium, and large scales of raw data. Our results show significant advantages in image-text alignment and overall model performance,  underscoring the effectiveness of VeCLIP in improving CLIP training. For example, VeCLIP achieves a remarkable over 20\% improvement in COCO and Flickr30k retrieval tasks under the 12M setting. For data efficiency, we also achieve a notable over 3% improvement while using only 14\% of the data employed in the vanilla CLIP and 11% in ALIGN.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors proposed a pipeline (LLM-VeC) for generating descriptions for web images, aiming at improving the performance of Vision & Language models. LLM-VeC uses LLaVA to generate text describing the input image, and one LLM is then used to fuse the text and AltText to yield the final corresponding text. The data mixing strategy is used when training the VL models. Extensive experiments are conducted on the well-known datasets to show the advantages of the LLM-VeC.

### Strengths
The main strength is that the proposed is simple and efficient. Improvements are achieved on several datasets.

### Weaknesses
The proposed method only combines the result from LLaVA and LLM. It is very natural and straightforward. Thus, it seems to the reviewer that there is no scientific problem solved in this paper.

### Questions
The proposed method is very straightforward, and it is natural to achieve improvements with more data. The targeting scientific problem is needed to be described clearly. Otherwise, the contribution might be limited.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper designs a working pipeline to enrich visual caption data. Then, the generated data is used for training CLIP and significantly benefit the vision language retrieval performance. Experiments on COCO, flickr and other visual datasets show the superiority of the proposed method.

### Strengths
1. Data quality is important nowaday, since high-quality data always benefit the training quality. Therefore, enriching caption data is a valuable research topic.
2. The working pipeline is clear and simple to achieve the research goal.
3. Experimental results show the proposed method outperforms CLIP baseline with sufficient ablation and analysis. Different experimental settings are considered for empirical validation.

### Weaknesses
1. This method lacks of technical novelty. Simply using two rounds of prompt based on LLaVA and LLM to generate new caption are highly engineering operations. They actually work well based on the experiments in the draft, but I am concerning its technical contribution as a research work.
2. The experiments are mainly based on CLIP comparison, adding more backbones will further enrich this paper such as BLIP and I believe this method can easily improve performance of other backbones.

### Questions
Please refer to weakness for details.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The research in this paper has a reasonable starting point, focusing on noise in image-text pairs, especially the issue of noise in the text part. They combine existing tuned vision-language models and LLM to rephrase caption data. In terms of the approach, there don't to be any inherent issues, and they have validated their method on various databases, including cross-modal retrieval and classification data. The results demonstrate that their data refinement approach leads to improvements in the original model. However, there are still some aspects of this paper that require further refinement and exploration.

### Strengths
* The idea presented in this paper is reasonable because the training corpora for models like CLIP, such as LAION, are known to be noisy. Despite some data cleaning methods, the inherent noise in the data due to web scraping and its large-scale nature remains a significant issue.
*  The approach in this article, which combines alttexts, LLM prompts, and visual-enriched captions, is also rational because this fusion of methods can effectively address the noise problem in the original data.
* Furthermore, the experimental section conducted by the authors is comprehensive, with well-justified model and test data selection, and the corresponding experimental explanations are included.

### Weaknesses
However, there are still some issues with this article, and these issues are directly related to the article's motivation:

- In the section on the Potential Ethics of LLM and Failure Cases Processing, the authors' viewpoint is reasonable. However, the conclusion is somewhat peculiar. In theory, LLMs like GPT-4 might only encounter issues if provided with illegal or violent inputs. It would be interesting to understand the severity of this problem when using existing models. Even when rewriting only captions, I believe the issue persists. At the very least, GPT-4 would perform sensitivity checks on inputs. I would like to know if the authors conducted a detailed analysis of this issue.

- The paper employs a strategy to choose between raw text and refined text, but I did not find a comparison with a strategy involving further pretraining or training from scratch based on the existing CLIP model. For very large models like CLIP BigG, the improvements brought by refining captions in the later stages are uncertain, especially if the model has already been trained on a substantial amount of data. It is essential to discuss the value of this approach and the improvements it can bring in the context of these larger models.

- In the ablation study section, the authors mention two explanations related to VeC. Regarding LLM's style issues, it is crucial to address how to resolve this problem since VeC's data is already biased due to the model's inherent issues. The article should propose feasible methods to tackle this problem; otherwise, the contribution of using VeC might be limited.

- Furthermore, the paper should discuss the design of prompts given AltTexts, Visual Prompts, and LLM Prompts. There are numerous possible combinations, and it is essential to explore which combination is most suitable for generating better VeC. This aspect should also be validated through human evaluation to determine the best prompt combinations.

- Although GPT-3/4 is not the primary focus of this article, at least some comparative analysis should be provided. For instance, a comparison between the new captions obtained from open-source models and those generated by GPT-3/4 could be informative.

- The paper utilizes LLaVA, which is indeed a useful method. However, there are other image attribute detection methods available, such as Recognize Anything, which can provide various visual clues. These can also be integrated into caption refinement through improved prompt design. Moreover, this approach may lead to a more focused and cleaner dataset primarily centered on visuals.

### Questions
Most of the issues have been discussed in the Weaknesses section above. Authors can refer to the section on Weaknesses for detailed insights.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
