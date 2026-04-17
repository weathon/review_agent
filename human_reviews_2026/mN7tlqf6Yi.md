# MV-RAG: Retrieval Augmented Multiview Diffusion

- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Text-to-3D generation approaches have advanced significantly, producing high-quality and 3D-consistent outputs. However, they often fail to produce out-of-domain (OOD) or rare concepts, yielding inconsistent or inaccurate results. To this end, we propose MV-RAG, a novel text-to-3D pipeline that first retrieves relevant 2D images from a large in-the-wild 2D database and then conditions a multiview diffusion model on these images to synthesize consistent and accurate multiview outputs. Training such a retrieval-conditioned model is achieved via a novel hybrid strategy bridging structured multiview data and diverse 2D image collections. This involves training on multiview data using augmented conditioning views that simulate retrieval variance for view-specific reconstruction, alongside training on sets of retrieved real-world 2D images using a distinctive held-out view prediction objective: the model predicts the held-out view from the other views to infer 3D consistency from 2D data. We also introduce a prior-guided fusion mechanism that dynamically balances retrieval signals with the model's prior. To facilitate a rigorous OOD evaluation, we introduce a new collection of challenging OOD prompts. Experiments against state-of-the-art text-to-3D, image-to-3D, and personalization baselines show that our approach significantly improves 3D consistency, photorealism, and text adherence for OOD/rare concepts, while maintaining competitive performance on standard benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes MV-RAG, a retrieval-augmented multiview diffusion framework for text-to-3D generation, addressing the limitations of existing methods when handling out-of-distribution (OOD) or rare concepts. MV-RAG first retrieves relevant 2D images from a large in-the-wild database based on text prompts, then uses a hybrid training scheme (combining 2D held-out view prediction and 3D multiview reconstruction) to train the diffusion model, with an adaptive fusion coefficient balancing the base model’s prior and retrieval signals. The authors also introduce the OOD-Eval benchmark for rigorous OOD evaluation. Experiments show MV-RAG improves 3D consistency, photorealism, and text alignment for OOD/rare concepts while maintaining competitive performance on in-domain benchmarks.

### Strengths
1. The introduction of the RAG for multiview diffusion models is interesting, which adds some external information for the diffusion model.
2. The proposed method shows some promising performance on OOD examples.

### Weaknesses
1. Motivation is not so clear. If the retrieved images are also OOD for the multiview diffusion model, then the method also cannot accurately generate the 3D shape. If the retrieved images fall in the training domain, then an alternative way is just to generate one image using the input texts, and then an image-to-3D model can handle this task. So theoretically, this paper does not well resolve the generalization problem.
2. The comparison is not so fair. For baseline methods like TRELLIS, they may be sensitive to the viewpoints of the input images (in the image-to-3D setting). The retrieved input images show diverse perspective distortions and uncommon viewpoints with incomplete foreground objects. I guess this is the main cause for the bad performance of TRELLIS on the retrieved images. I think this is also a good research problem to address the viewpoint problem, but this is deviated from the original claim of the paper.

### Questions
Refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents MV-RAG, a retrieval-augmented multiview diffusion framework for text-to-3D generation. The method retrieves relevant in-the-wild 2D images based on the input text and uses them to condition a multiview diffusion model, producing geometrically consistent and photorealistic multi-view outputs. A hybrid 2D–3D training strategy combines structured multiview supervision with unposed 2D data through a held-out-view prediction objective.

### Strengths
The application of retrieval-augmented generation to 2D–3D diffusion is novel and well-motivated.
    The proposed hybrid 2D–3D training scheme is technically sound and effectively integrates structured and unstructured supervision.
    The paper is thorough and well-organized, featuring extensive ablation studies, user evaluations, and clear architectural details.

### Weaknesses
Although qualitative results are strong, the quantitative improvements on standard metrics (e.g., PSNR, IS) are modest—noticeable but not substantial.
    The evaluation could benefit from comparisons with more diverse and standardized baselines such as Wonder3D or SyncDreamer.

### Questions
How sensitive is MV-RAG’s performance to retrieval quality or the domain coverage of the retrieval database?
    The paper mentions fine-tuning on a single GPU for three hours using a Stable Diffusion checkpoint. Were larger-scale experiments conducted? In particular, would increasing the dataset size or model capacity further improve performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
MV-RAG introduces a retrieval-augmented multi-view diffusion framework for text-to-3D generation, specifically addressing out-of-distribution (OOD) concepts. The approach conditions generation on relevant in-the-wild 2D images retrieved for each text prompt, employing a hybrid training scheme that combines 3D geometric consistency objectives with 2D held-out view prediction. During inference, a prior-guided attention mechanism dynamically balances the base model's internal knowledge with external retrieval signals. Experiments demonstrate MV-RAG's superior performance over SOTA baselines in 3D consistency, photorealism, and text alignment on OOD benchmarks while maintaining competitive in-domain results.

### Strengths
Quality & Originality: The primary strength of this paper lies in its exceptionally comprehensive and multi-faceted experimental evaluation, which leaves little room for doubt regarding the efficacy of the proposed MV-RAG framework. The experimental section is thorough and systematic, and the experiments are designed to systematically validate the method's performance across a wide range of scenarios and against numerous strong baselines. 

(1)	The paper includes detailed ablation experiments on a hybrid 2D/3D training scheme, the number of retrieved images(K), the retrieval method (BM 25 v.s. dense retrievers), noisy retrievals, and the importance of the prior-guided attention mechanism.

(2)	The author thoughtfully addressed the limitation of CLIP for evaluating OOD concepts and introduced a novel evaluation protocol (image-image similarity metrics as complementary).

(3)	The authors provided an exhaustive benchmarking against SOTA, including three different paradigms.

(4)	The author also provided an analysis of limitations and failure cases, honestly discussing the limitations and failure cases of their methods. 

(5)	To complement quantitative metrics, the paper also includes a user study that directly assesses the critical attributes of realism, text alignment and 3D consistency from human perspectives. 

Clarity: Beyond the technical contributions, the paper is well-written and structured. The clarity of exposition is supported by comprehensive appendices and supplementary materials. The complete OOD-Eval benchmark and code provided in the Supplementary Materials underscore a strong dedication to reproducibility.

(1)	Significance: The work has a significant impact, as evidenced by its SOTA performance on standardized benchmarks.

### Weaknesses
(1)	Ablating the distinct roles of 2D mode and 3D mode (Section 4.3)
The qualitative ablation in Figure 7 effectively illustrates the distinct roles of the 2D and 3D training modes. The accompanying text (lines 458-461) states that the 2D mode is crucial for separating objects from in-the-wild backgrounds, while the 3D mode is vital for correct shape rendering and background consistency.
However, the described effects that are specifically concerning background separation and shape correctness can sometimes be interdependent and challenging to distinguish purely from visual examples. To provide a more precise and quantifiable understanding for the reader, could the authors supplement the qualitative ablation with quantitative metrics designed to measure these specific aspects?

(2)	I appreciate the detailed description of the OOD-Eval benchmark construction in Appendix A.9 and the inclusion of the full dataset. The use of an LLM (GPT-4o) to curate "rare or unique" concepts is a practical approach.
However, I have a concern regarding potential category bias. The visualization in Figure 12 and a perusal of the supplementary data suggest a possible over-representation of certain categories, such as dogs and vehicles. I wonder if the LLM-based generation might inherently favor such semantically dense and well-documented categories.
This could potentially lead to an overestimation of the method's average performance on the true long tail, as these categories often have abundant, high-quality, multi-view images available for retrieval (which may make the retrieval task easier).

(3)	Computational Efficiency and Inference Latency
The paper rightly focuses on generation quality. However, for a complete understanding of the method's practical utility, its computational overhead needs to be addressed. The training efficiency is noted (∼3 hours on a single A100, line 848), which is commendable. Nevertheless, computational efficiency is mentioned briefly; a detailed comparison of end-to-end inference latency against baselines would help assess the method's practical utility.

### Questions
(1)	According to Weakness – (1), I kindly suggest the authors to provide the following experiment to quantify the effectiveness for 2D mode and 3D mode, which would be clearer to delineate the unique and complementary functionalities of each training mode.
a)	For object-background separation, a segmentation metric could be used to calculate the mIoU between generated objects and their backgrounds, assessing how well the model isolates the foreground.
b)	For shape correctness, the 3D consistency metrics already employed (e.g., the proposed re-rendered DINOv2 similarity) are a good proxy. A comparative analysis of this metric for models trained with and without the 3D mode could quantitatively underscore its contribution to geometric accuracy.

(2)	Concerning weakness - (2), to better characterize the benchmark and the method's performance across it, can the authors provide the following analyses:
a)	Category Distribution Analysis: Provide a statistical breakdown of the OOD-Eval concepts across broader semantic categories (e.g., animals, vehicles, furniture, artifacts, food). This would make the benchmark's composition transparent.
b)	Quantification of OOD: The paper states the texts were "far from any text... seen during training" (line 300). To substantiate this claim, could the authors quantify this distance? For instance, they could plot the distribution of CLIP text embedding distances between all OOD-Eval prompts and the training set prompts (or a large sample from LAION). Showing this distribution, and further segmenting it by the semantic categories from the first analysis, would provide a rigorous, quantitative measure of how "out-of-distribution" the benchmark truly is.

(3)	According to weakness – (3), can the authors provide the following results:
a)	A table reporting the average time from input text prompt to the output of 4 generated views for MV-RAG and its key competitors. This would allow readers to better assess the trade-off between the achieved quality gains and the associated computational cost.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes MV-RAG, a text-to-3D generation pipeline that addresses the possible failure of existing methods on out-of-domain (OOD) or rare concepts by first retrieving relevant 2D images from a large database, and then conditioning a multiview diffusion model on these images to generate consistent multiview outputs. The approach uses a hybrid 2-mode training strategy combining structured multiview data with augmented conditioning views and diverse 2D image collections with a held-out view prediction objective to infer 3D consistency, along with a prior-guided fusion mechanism that balances retrieval signals with the model's prior.

### Strengths
- The paper is very well written, organized, and easy to follow.
- The paper is well-motivated, tackling an important problem of OOD generation or "rare" concepts that were not sufficiently trained to diffusion models, therefore yielding suboptimal training results when applied to 3D generation, which is a practical and prevalent problem in multiview diffusion models. 
- The paper proposes a coherent and intuitive methodology that fits well to their problem at hand: they start from the MVDream architecture, which introduced multiview consistency, and propose to finetune it to accept extended attention from retrieved 2D images for highly detailed, prompt-adhering generation. Their 2-step approach of training a model at a multiview, 4-view generating setting (3D attention) with augmented retrieved images at random viewpoints, and a subsequent 2D image generation setting generating a prompt-adhering image solely from retrieved in-the-wild images makes sense and is well-designed, as well as being cost-efficient training-wise. 
- The experimental results show effective performance and are well-organized, demonstrating the effectiveness of this method.

### Weaknesses
- The paper could benefit from additional explanation on how it conducts semantic/geometric augmentation at a 3D training setting, and additional details regarding how much augmentation the model can take. For example, if the semantically augmented retrieval images deviate too much from the source 3D asset, the model training may suffer degradation rather than learning 3D consistency: please elaborate on this aspect of the method.
- One question that I have with this method is that 2D image generation is trained to generate the ground truth target image from retrieved images, relying on nothing but the text prompt embedding. However, in many cases, the text prompt may be insufficient in describing the target image, and the model may have difficulty generating the target image with a lack of any geometric / image structural guidance, or may resort to shortcuts such as simple memorization. This problem may be exacerbated as the model training is basically fine-tuning and not available for training on as large-scale data as the original model. Can the authors elaborate further on this issue?

### Questions
Please see the Weaknesses section.

### Soundness
4

### Presentation
3

### Contribution
3
