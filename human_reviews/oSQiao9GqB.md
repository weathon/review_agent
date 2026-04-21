# LLaVA-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models

- Avg Score: 7.33
- Decision: Accept (Spotlight)
- Scores: 8, 8, 6

## Abstract
Visual instruction tuning has made considerable strides in enhancing the capabilities of Large Multimodal Models (LMMs). However, existing open LMMs largely focus on single-image tasks, their applications to multi-image scenarios remains less explored. Additionally, prior LMM research separately tackles different scenarios, leaving it impossible to generalize cross scenarios with new
emerging capabilities. To this end, we introduce LLaVA-Interleave, which simultaneously tackles Multi-image, Multi-frame (video), Multi-view (3D), and Multi-patch (single-image) scenarios in LMMs. To enable these capabilities, we regard the interleaved data format as a general template and compile the M4-Instruct dataset with 1,177.6k samples, spanning 4 primary domains with 14
tasks and 41 datasets. We also curate the LLaVA-Interleave Bench to comprehensively evaluate the multi-image performance of LMMs. Through extensive
experiments, LLaVA-Interleave achieves leading results in multi-image, video,
and 3D benchmarks, while maintaining the performance of single-image tasks.
Besides, our model also exhibits several emerging capabilities, e.g., transferring tasks across different settings and modalities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces LLaVA-NeXT-Interleave, a large multimodal model (LMM) designed to unify and enhance capabilities across multi-image, video, and 3D tasks while maintaining performance on single-image tasks. The authors propose using an interleaved image-text data format as a universal template to represent different computer vision scenarios. They compile a comprehensive dataset called M4-Instruct, consisting of over 1 million samples spanning four primary domains with 14 tasks and 41 datasets. Additionally, they create the LLaVA-Interleave Bench, a diverse set of benchmarks for evaluating multi-image performance. Extensive experiments demonstrate that LLaVA-NeXT-Interleave achieves state-of-the-art results across various benchmarks and exhibits emerging capabilities such as cross-task transfer.

### Strengths
Innovative Unification Framework: The paper presents a novel approach by using an interleaved image-text format to unify multi-image, video, and 3D tasks within a single LMM framework. This unification is both practical and innovative, addressing a significant gap in the current research.

Comprehensive Dataset and Benchmark Creation: The compilation of the M4-Instruct dataset and the LLaVA-Interleave Bench provides valuable resources for the community. These datasets are extensive and cover a wide range of tasks and scenarios, enhancing the reproducibility and applicability of the research.

State-of-the-Art Performance: The proposed model achieves leading results across various benchmarks in multi-image, video, and 3D tasks while maintaining strong performance on single-image tasks. This demonstrates the effectiveness of the approach.

Emerging Capabilities: The model exhibits emerging capabilities such as cross-task transfer and zero-shot task composition, indicating strong generalization and adaptability to new scenarios.

Detailed Experimental Analysis: The paper includes extensive experiments and ablation studies that thoroughly evaluate the model's performance and validate the proposed techniques.

### Weaknesses
Limited Details on Data Curation: While the paper introduces a large and diverse dataset, it provides limited information on the data curation process, quality control, and potential biases. More details on how data quality and diversity were ensured would strengthen the work.

Computational Efficiency Concerns: The paper does not thoroughly discuss the computational efficiency or resource requirements of the proposed model, especially when handling multiple modalities. Comparisons in terms of model complexity and inference speed with existing models would be beneficial.

Scope of Evaluation Metrics: The evaluation focuses mainly on quantitative performance metrics. Incorporating qualitative analyses or user studies could provide additional insights into the model's real-world applicability and limitations.

Scalability and Practical Deployment: Potential challenges related to scaling the model to larger datasets or deploying it in practical applications are not fully explored. Discussion on these aspects would enhance the paper's impact.

### Questions
Data Curation Process: Can the authors provide more details on the data collection and curation process for the M4-Instruct dataset? Specifically, how were data quality and diversity ensured, and what measures were taken to mitigate potential biases?

Computational Resources: How does LLaVA-NeXT-Interleave perform in terms of computational efficiency compared to other state-of-the-art models? Are there any optimizations implemented to handle the increased complexity of processing multiple modalities?

Scalability: Have the authors explored the scalability of the model when trained on even larger datasets or when applied to more complex tasks? Are there any observed limitations or degradation in performance?

Generalization to Unseen Tasks: While the model demonstrates emerging capabilities, how well does it generalize to completely unseen tasks or modalities not included in the training data?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposed llava-next-interleave, which unifies the muliple image, multiple frame (video), mulitple view (3D), and Multiple patches (single-image) scenarios in one MLLM, which is very interesting.
One dataset namely M4-Instruct dataset and the LLava-interleav Bench is proposed, which is very helpful for pushing the progresses of the MLLM research.
Detailed ablation studies are provided, where some useful and meaningful insights are provided.

### Strengths
The paper makes attempt to unify the visual format, such as  multiple images, multiple frames, multiple views, and mulitple patches into one signal MLLM, which can facilitate the application of MLLM and improve generalization ability of MLLM on processing different format visual signals. With such unification, different visual formats only need to be processed with the corresponding formate, which can thereby be learned and tuned on the MLLM.

The composed dataset and benchmark are very useful for the research of MLLM.

Some ablation studies are conducted which are very interesting, such as the mixed interleaved data formats during training, as well as combining differenet data scenarios improves individual task performance.

### Weaknesses
Some insights may be further studied. 

1. Mixed interleaved data formats can help improvign the perfromances. I am wondering the underlying reasons. One reason may be the different formats increases the diversity of the data. One reason may be the deficiency of the data, with mixed formats, the data are trained in two rounds (if I understand correctly), the performances are improved. The authors are highly suggested to performed detailed studies.

Please specify the experiment settings and also provide the results with the data with both formats. In this case, we can find with the data sufficiency matters for the performance gain.


2. The constructed data are trained in the continual training stage. I am wondering if we want to improve the ability of MLLM for handling different visual formats. Should we add more data (different types of visual formats) in the stage-i pretraining, besides the single image-text pair. Or we only perform the training in the continual training stage, the MLLM ablity on handling on multiple images, patches, frames, views can be well improved.

If possible, add the data in the pretraind stage. I am wondering whether there are still performance gain.

### Questions
Check detailed information in the Weakness part.

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
5

### Summary
The authors collect a interleave-formatted dataset from existing multi-image, video, multi-view, and image datasets. After that, they finetune a LMM from an existing single-image LMM to support multiple multi-image scenarios based on their collected dataset. Moreover, they curate a multi-scenario interleave benchmark to comprehensively evaluate LMM’s interleave capabilities.

### Strengths
In addition to the clear writing and logical flow of this work, I have outlined its strengths below:

1. This work presents a solid benchmark for comprehensively evaluating the interleaving capabilities of LMMs. This does help the LMM community.

2. The experiments in this work are thorough, providing an excellent baseline for future research in this field.

### Weaknesses
In my view, the technical innovation of this work is limited. I have outlined more detailed reasons below.

1. The proposed M4-Instruct dataset is merely a combination of several existing datasets, lacking sufficient technical innovation in its construction process.

2. The model proposed in this paper adopts the architecture and even checkpoints of previous methods, with only fine-tuning applied to the collected dataset. Technical innovation in the training process is also quite limited.

3. Although this work primarily follows a path of data collection and model fine-tuning, it only validates the data's effectiveness on a single model, LLaVA-Next. The authors should supplement their study by applying the M4-Instruct data to other 7B-sized single-image LMMs, such as InternVL-1.5, Cambrian, and MiniCPM-V2.5.

### Questions
See weakness above

### Soundness
3

### Presentation
3

### Contribution
2
