# Dragonfly: Multi-Resolution Zoom-In Encoding Enhances Vision-Language Models

- Decision: Reject
- Scores: 6, 5, 5

## Abstract
Recent advancements in vision-language models (VLMs) have highlighted the benefits of processing images at higher resolutions and leveraging multi-crop features to retain native resolution details. However, current vision transformers (ViTs) often struggle to capture fine-grained details from non-dominant objects, charts, and embedded text, limiting their effectiveness in certain tasks. In this paper, we push beyond the conventional high-resolution and multi-crop techniques by not only preserving but also zooming in past the native resolution of images. This enhancement allows our model to better extract fine-grained details, overcoming the limitations of current ViTs. To manage the increased token count and computational complexity, we show that a simple mean-pooling aggregation over tokens is effective. Our model, Dragonfly, achieves competitive performance on general tasks such as ScienceQA and AI2D, and excels in tasks requiring fine-grained image understanding, including TextVQA and ChartQA. On average, across ten general-domain benchmarks, Dragonfly ranks at the top, outperforming models that are significantly larger or trained on much larger datasets. Notably, Dragonfly sets new benchmarks on several biomedical tasks, achieving 91.6\% accuracy on the SLAKE (compared to 84.8\% for Med-Gemini) and a 67.1\% token F1 score on Path-VQA (compared to 62.7\% for Med-PaLM M). On biomedical image captioning tasks, Dragonfly attains state-of-the-art results majority of the performance metrics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript presents Dragonfly, a novel Vision-Language Model (VLM) that employs a multi-resolution zoom-in encoding strategy to enhance fine-grained visual understanding. Unlike conventional Vision Transformers (ViTs) that downsample images to fixed, low resolutions—thereby losing critical details—Dragonfly processes images at higher resolutions and employs a multi-crop technique that exceeds the native resolution. This approach allows the model to capture intricate details from non-dominant objects, charts, and embedded text, which are often challenging for existing ViTs. To address the computational complexity arising from the increased token count, Dragonfly utilizes a mean-pooling aggregation strategy. The model demonstrates competitive performance across ten general-domain benchmarks and sets new benchmarks in several biomedical tasks, outperforming larger models trained on significantly more extensive datasets.

### Strengths
1. Dragonfly introduces an innovative multi-resolution zoom-in encoding strategy that surpasses native image resolutions, enabling the capture of intricate details from non-dominant objects, charts, and embedded text.
2. It implements a simple yet effective mean-pooling aggregation method and achieves good performance across a diverse set of benchmarks.

### Weaknesses
1. Novelty. The proposed method builds upon existing multi-resolution and multi-crop techniques without offering substantial novel contributions. The idea of processing images at higher resolutions and using multi-crop strategies has been explored in prior works, and Dragonfly does not sufficiently differentiate itself beyond these established methods.
Model Comparison: Dragonfly is developed using the more advanced Llama3 model, whereas comparable methods utilize less capable language models, such as Llama2 and Qwen2. This discrepancy raises concerns about the fairness of comparisons. How does Dragonfly's performance measure up when evaluated against these models?
Data Influence: It is unclear whether the observed performance improvements with Dragonfly stem from the curated data or from the model's design. How does Dragonfly perform when tested with a commonly used dataset?

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
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces multi-crop techniques beyond the native resolution for high-resolution images. To handle the huge number of numbers, the authors employ the average pooling startegy on each crop. Except for general domain of benchmarks on fine-grained image understanding, this paper also introduce the contributioni on biomedical tasks. They also curate a SFT dataset including 2.4M general images and 1.4M biomedical images for training.

### Strengths
1. The ablation studies proves their proposed strategy.
2. Biomedical domain is considered by this paper.
3. The motivation is nature and easy to follow.
4. A SFT dataset with different domains and huge number of images is built.

### Weaknesses
1. The novelty is limited. The proposed strategy is only an extension of any-resolution technical. Compared to any-resolution which uses two levels of rosulotions, they only resize the image, crop more patches and use three levels of resolutions of image.
2. The visualizations only shows the response of the proposed Dragonfly, other MLLM's responses are encouraged to be listed for better comparision.
3. To balance the computational costs and performance, they use mean pooling within each crop. The paper lacks the dicussion about how to choose a proper compressing ratio for trade-off.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces DragonFly to enhance vision-language models. The main idea is to combine the multi-cropping with mean pooling, so that the VLM can use high-resolution image encoders and work on images' native resolution, while ensuring the efficiency of the model. The proposed enhanced VLM performs well in tasks needing finer details, such as biomedical tasks.

### Strengths
1. Dragonfly uses multi-crop techniques to process images at high resolutions, thus enhancing the model’s capability to capture fine details. Meanwhile, it uses a simple mean-pooling strategy to reduce visual tokens effectively, which retains the efficiency of the model.
2. The proposed method achieves competitive performance across multiple benchmarks and shows strong generalizability across both general and specialized biomedical tasks.

### Weaknesses
The paper uses a very simple strategy that combines multi-crop and mean-pooling to enhance VLM. However, the motivation for choosing mean-pooling instead of other compression techniques is not clearly stated. It's like an experimental report that simply states this method is effective. So why does mean-pooling outperform other strategies? Why do you choose such a pooling window? Will this direct pooling harm the extraction of the fine details? A more comprehensive analysis is expected.

### Questions
Please refer to the weaknesses part. Additionally, is there a comparison of the inference time or flops of different methods?

### Soundness
2

### Presentation
3

### Contribution
2
