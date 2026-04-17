# OVid: Open Large-Scale Video Dataset as a Novel Source for Image-Text Data

- Decision: Reject
- Scores: 2, 8, 6, 2

## Abstract
We present OVid, a large open video dataset comprising _10 million hours_ of diverse content collected from CommonCrawl. To complement the raw data, we generate image captions for scene-changing frames and video-level captions for a 300M frame–caption subset. Using this subset, we train CLIP models at multiple scales and benchmark them against reference CLIP models trained on DataComp, Re-LAION and DataComp recaptioned with the same captioning pipeline. Observed scaling trends for classification and retrieval show evidence that OVid can be another valuable and scalable source of image-text data, in addition to image-text pairs from public webpages. OVid marks a significant step towards democratizing access to large-scale video data and fostering the development of open multimodal foundation models. To this end, all the data will be freely available to research institutions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a large-scale image-text dataset sourced from 1.3B online videos. From this collection, 300M frame-caption pairs and 12M video-level text summaries are extracted for training vision-language models. By training CLIP models on the newly introduced OVID dataset, the authors demonstrate superior performance on both text-to-video and video-to-text retrieval.

### Strengths
1. The OVID dataset collects image-text pairs from large-scale online videos, which is significantly different from previous image-text datasets.

2. Compared to existing open-ended video-language datasets, OVID contains a much larger number of publicly available videos.

### Weaknesses
1. I am confused by the statement of contributions in L100–106. It appears that all the contributions focus solely on the release of a large-scale image-text (or frame-text) dataset. In other words, I believe the contributions of this paper are limited, as it primarily presents a dataset.

2. Although OVID is significantly larger than previous datasets, it seems to sacrifice many details in its captions or summaries. As shown in Table 5, CLIP models trained on OVID perform almost the worst on ImageNet-1k, ImageNet-R, ImageNet-Sketch, and ImageNet-V2. The very simple prompt (“Provide a very coarse single line of caption”) used in OVID’s captioning pipeline likely omits too many details from video frames, resulting in unsatisfactory performance on zero-shot classification tasks.

3. In my opinion, the related work section includes many unrelated topics, such as multimodal LLMs and large multimodal models. What is the purpose of connecting these MLLMs to the OVID dataset? Are there any notable similarities or differences between them? The single sentence introducing large multimodal models in  L194–196 seems particularly out of place.

4. I suggest that this dataset paper be resubmitted to a dataset or benchmark track. Meanwhile, the authors may consider how to improve the overall quality of the captions while maintaining the scale of the dataset.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces OVID, a massive open video dataset sourced from Common Crawl, containing 1.3B video URLs (∼10M hours). From these, the authors generate 300M high-quality frame-caption pairs by extracting scene-change frames and captioning them with a efficient vision-language model (DeepSeek-VL2), followed by video-level summarization.

When used to train CLIP models, OVID demonstrates strong and scalable performance: it achieves state-of-the-art results on COCO image-text retrieval, outperforming comparable datasets like DataComp. However, a noted limitation is its weaker performance on zero-shot ImageNet classification, attributed to a domain gap from its synthetic captions.

The work provides a valuable, large-scale multimodal resource complementary to existing image-text collections.

### Strengths
This paper's main contribution is OVID, an extremely large-scale video-text dataset (1.3B URLs, 10M hours) that significantly dwarfs existing open collections. Its scale and diversity—multilingual and multi-topic—are its key strengths. The method of generating image-text pairs from scene-change frames is both novel and impactful.

The empirical validation is robust: CLIP models trained on OVID achieve state-of-the-art COCO retrieval performance across scales, proving the data's tangible value. This focus on large-scale, open multimodal data, combined with its commitment to reproducibility and open access, makes it a timely and valuable contribution to the community.

### Weaknesses
While the OVID dataset is a significant contribution, several limitations should be noted. 

1. Its reliance on synthetic captions introduces a domain gap; while this benefits retrieval tasks, it leads to a notable drop in zero-shot classification accuracy. 

2. The light data filtering strategy, while enabling scale, leaves potential concerns about noise, bias, and unsafe content unquantified in the current evaluation. 

3. The empirical study is thorough but narrowly focused on image-text tasks, leaving its value for video-language modeling an open question for future work.

### Questions
1. The paper notes a performance drop in ImageNet classification due to synthetic captions. It would be helpful to see a quantitative analysis of caption diversity (e.g., vocabulary size, sentence length) compared to human-written alt-text. Furthermore, could the authors comment on whether mixing OVID's data with smaller human-curated subsets might help bridge this domain gap and improve classification accuracy?

2. The reliance on platform-level moderation is noted. Have the authors conducted any evaluation for unsafe or biased content? Even a small-scale audit or analysis of potential demographic biases or NSFW frames would significantly strengthen the claims about dataset reliability and safety.

3. Could the authors elaborate on how they ensure the selected scene-change frames are semantically meaningful and not overly redundant? Would an adaptive, diversity-based sampling strategy be a worthwhile future direction to improve data efficiency?

4. Given the video origins of the data, has there been any consideration to benchmark OVID on temporal or video-language tasks? Demonstrating performance on benchmarks like MSR-VTT could further validate its value for video understanding, beyond image-text retrieval.

5. The dataset uses a research-only license. Could the authors clarify if there will be restrictions on creating derivative datasets or on fine-tuning commercial models? This clarity is important for the community to understand the full implications of the "open access" terms.

6. Figure 6 indicates different scaling trends for retrieval and classification. Could the authors provide more insight into why OVID exhibits stronger scaling for retrieval but weaker scaling for classification? Is this primarily attributed to caption style or domain bias?

7. Do the authors plan to extend OVID with audio or multimodal features? Integrating audio captions could make it an even more valuable resource for training unified vision-language-audio models.

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
2

### Summary
This paper introduces OVID, a massive open dataset comprising 10 million hours of video, positioning it as a scalable source for image-text data. The authors demonstrate that frames extracted from these videos, paired with machine-generated captions, constitute an image-text dataset with a distribution distinct from existing web-crawled image collections. They validate this by training CLIP models in image-text retrieval tasks.

### Strengths
- Unprecedented Scale: 10 million hours of video and 300 million frame-caption pairs.
- Notable Diversity: The dataset exhibits strong diversity across topics, languages, and video lengths, supporting a wide range of potential training scenarios.

### Weaknesses
1. Potential Circularity in Evaluation: The reliance on CLIP-based metrics (CLIPScore) for captioner selection and CLIP-based training for downstream validation may introduce a bias towards captions that align with the CLIP embedding space, rather than capturing broader semantic or human-aligned quality.
2. Insufficient Evidence for Distributional Difference: The central claim that video frames constitute a distributionally distinct and high-quality source of image-text data is only partially supported. The evidence rests heavily on the downstream performance of CLIP models, particularly on retrieval tasks. A more direct analysis—for instance, quantifying the domain shift between OVID frames and existing image corpora (e.g., using FID or other divergence measures)—would substantially strengthen this claim.
3. Limited Validation for Video Tasks: As acknowledged in the limitations, the dataset does not consider temporal or audio-visual aspects. While the frame-level data is validated for image-text tasks, the potential of OVID for video-language modeling (e.g., training ViCLIP or other video-text models) remains unexplored. This significantly reduces the demonstrated applicability of what is, fundamentally, a video dataset.

### Questions
- Could you provide a more quantitative or qualitative analysis of how the visual distribution of OVID frames differs from that of web-crawled image datasets?

### Soundness
2

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
The paper introduces OVID, an open large-scale video dataset designed to serve as a new source of high-quality image-text pairs. OVID comprises 10 million hours of video sourced from CommonCrawl, yielding 300 million captioned frames and 12 million video-level summaries generated through an automated captioning pipeline using the DeepSeek-VL2-tiny model. The authors train CLIP models on OVID and benchmark them against models trained on DataComp and Re-LAION, showing that OVID achieves state-of-the-art retrieval performance but slightly lower zero-shot classification accuracy on ImageNet. They argue that video-derived frames offer a novel and complementary data distribution to traditional web images. By releasing OVID and its curation pipeline openly, the work aims to democratize multimodal research and facilitate the development of open foundation models.

### Strengths
- This paper collected a new dataset with a large number of image and text pairs.

### Weaknesses
- The academic impact and the importance of the dataset are unclear. In Table 1, the authors only mention the number of samples. However, in terms of the number of samples, the advantage compared to the CoYo-300M dataset with 300M samples is unclear. The only difference is the source of the samples. However, the importance of the source is not discussed enough. A more detailed analysis of the diversity or bias to demonstrate the advantages of the proposed dataset is required.
-Overall, the advantage of the proposed dataset is unclear. In the experiment in Table 3, the authors only performed an engineering to find a favorable combination of existing models without any novel insight. In addition, the combinations covered in the experiment are not comprehensive enough.
- In Table 4, the experimental setting is questionable. The number of samples considered in the experiment is not consistent and not comprehensive. More importantly, CLIP models trained on OVID show weaker zero-shot classification performance, indicating limited fine-grained label alignment.
- Evaluation is restricted to retrieval and classification, offering limited evidence of general downstream utility. The authors should have considered other vision-language tasks such as image captioning.

### Questions
Please refer to the questions in the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
