

# Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body

Zeqing Wang  
Sun Yat-sen University  
Guangzhou, China  
wangzq73@mail2.sysu.edu.cn

Qingyang Ma  
Sun Yat-sen University  
Guangzhou, China  
maqy23@mail2.sysu.edu.cn

Wentao Wan  
Sun Yat-sen University  
Guangzhou, China  
wanwt3@mail2.sysu.edu.cn

Haojie Li  
South China  
University of Technology  
Guangzhou, China  
12hjli4@gmail.com

Keze Wang  
Sun Yat-sen University  
Guangzhou, China  
Peng Cheng Laboratory  
Shenzhen, Guangdong, China  
kezewang@gmail.com

Yonghong Tian  
Peking University  
Beijing, China  
Peng Cheng Laboratory  
Shenzhen, Guangdong, China  
yhtianpk@edu.cn

## Abstract

*Recent improvements in visual synthesis have significantly enhanced the depiction of generated human photos, which are pivotal due to their wide applicability and demand. Nonetheless, the existing text-to-image or text-to-video models often generate low-quality human photos that might differ considerably from real-world body structures, referred to as “abnormal human bodies”. Such abnormalities, typically deemed unacceptable, pose considerable challenges in the detection and repair of them within human photos. These challenges require precise abnormality recognition capabilities, which entail pinpointing both the location and the abnormality type. Intuitively, Visual Language Models (VLMs) that have obtained remarkable performance on various visual tasks are quite suitable for this task. However, their performance on abnormality detection in human photos is quite poor. Hence, it is quite important to highlight this task for the research community. In this paper, we first introduce a simple yet challenging task, i.e., Fine-grained Human-body Abnormality Detection (FHAD), and construct two high-quality datasets for evaluation. Then, we propose a meticulous framework, named HumanCalibrator, which identifies and repairs abnormalities in human body structures while preserving the other content. Experiments indicate that our HumanCalibrator achieves high accuracy in abnormality detection and accomplishes an increase in visual comparisons while preserving the other visual content.*

## 1. Introduction

Visual content generation models have demonstrated the capacity to create highly realistic representations within human photos, which hold considerable importance across various downstream tasks such as virtual reality, augmented reality, and the entertainment industry. Recent developments in text-to-image [3, 46, 47, 50] and text-to-video models [25, 42, 62] have enhanced both the quality and the realistic of generated human photos. However, these models frequently struggle to accurately replicate human body structures as they exist in the real world, leading to human photos with abnormalities such as absent or redundant body parts. Compared to low-quality, this abnormality is more unacceptable because it is more noticeable and has a larger gap with the real-world human body structure.

Some methods try to solve this problem by adding additional constraints, such as Pose-ControlNet [66], HumanSD [22], and T2I-Adapter [38]. However, these methods are always hard to use due to their extra input or additional training. HumanRefiner [14], in another way, tackles the problem as a post-process method. It detects abnormalities in the generated human photos and then regenerates the whole content. However, this coarse-grained detection method can only detect which existing visual content is abnormal, ignoring the absent part. Furthermore, such a method can not preserve the background information of the original photo, which also limits its generalizability.

To detect abnormalities, it is intuitive to use a Vision-Language Model (VLM), which processes strong perception and reasoning capabilities and has been applied to various downstream perception tasks [10, 28, 43], as the back-

![Figure 1: Fine-Grained Human-body Abnormality Detection (FHAD). The image shows a guitar with a human hand playing it. A red box highlights a redundant hand on the guitar neck. Above the image, a 'Generated Human' icon is shown. To the left, three VLMs are listed with their analysis: GPT4V says '...there is no visible abnormality in the structure of the human hand playing the guitar...'; Claude 3.5 Sonnet says '...there don't appear to be any abnormalities...'; and Qwen2 VL 72B says '...there are no visible abnormalities in the structure of the human body in this image...'. A text box on the right of the image states 'There is a redundant hand.'](2173c0102e23a5ff29d90d4353fc0339_img.jpg)

Figure 1: Fine-Grained Human-body Abnormality Detection (FHAD). The image shows a guitar with a human hand playing it. A red box highlights a redundant hand on the guitar neck. Above the image, a 'Generated Human' icon is shown. To the left, three VLMs are listed with their analysis: GPT4V says '...there is no visible abnormality in the structure of the human hand playing the guitar...'; Claude 3.5 Sonnet says '...there don't appear to be any abnormalities...'; and Qwen2 VL 72B says '...there are no visible abnormalities in the structure of the human body in this image...'. A text box on the right of the image states 'There is a redundant hand.'

Figure 1. Fine-Grained Human-body Abnormality Detection (FHAD). Human body structures in AIGC often exhibit significant deviations from humans existing in the real world, making them easily recognizable as abnormal to human observers. However, current powerful VLMs, typically struggle with this abnormality perception despite excelling in various downstream perceptual tasks, which presents a challenge for fine-grained abnormality detection and motivates our research.

bone. However, when we test current powerful VLMs on our proposed Fine-grained Human-body Abnormality Detection (FHAD) task (Sec. 3), as shown in Figure 1, their performance is surprisingly poor, though the task is simple for humans. Quantitative analysis across multiple models in Figure 6 further demonstrates the lack of abnormality perception capabilities in existing VLMs.

With the observation, we review the capabilities required to detect abnormalities. For absent body parts, detection relies on existing body parts and the correlation among them to infer the absent bodies. In contrast, for redundant body parts, this detection depends more on the overall perception of visual contents, as such redundant abnormalities may appear anywhere and are unrelated to the existing body parts.

With the aforementioned observation, we train a detection model that leverages the correlation of body parts to detect absent abnormalities. Furthermore, to tackle the redundant body parts, we employ a diffusion-based model that boasts strong capabilities in comprehending overall visual content. By integrating these approaches, we develop “HumanCalibrator”, a fine-grained framework for the detection and repair of abnormalities in human body structure. Our HumanCalibrator precisely pinpoints the abnormalities within the human body structure and repairs the abnormal region while preserving other regions.

Our contributions can be summarized as follows: (i) To the best of our knowledge, we are the first to propose a simple yet challenging task, Fine-grained Human-body Abnormality Detection (FHAD) with two datasets across two main-stream domains (Sec. 3); (ii) Our extensive and comprehensive experiments demonstrate that it is a challenging task for the VLMs to understand abnormalities, despite being trained on large datasets and possessing strong ability

in multiple downstream tasks (Sec. 5). Then, leveraging the features of the human body structure, we propose a solution for absent and redundant body part abnormalities, which includes a detection model trained based on the correlation of human body structure (Sec. 4.1) and an approach focusing on the overall visual perception via diffusion-based model (Sec. 4.2); (iii) We further propose an effective framework, i.e., HumanCalibrator, which includes detecting and repairing the abnormalities in the body structure while preserving the rest of the visual content (Sec. 4.3).

## 2. Related Work

**Vision-Language Model:** Leveraging extensive pre-training datasets and benefiting from Large Language Models (LLMs) [1, 4, 5, 9, 21, 54, 56, 57, 61, 64] along with powerful Vision Encoders [13, 17, 29, 55], VLMs [8, 11, 17, 19, 29, 30, 33, 45, 55, 60] have achieved success in a range of visual tasks. These models excel at various perception and reasoning tasks [20, 23, 35, 52], prompting research that fine-tunes VLMs for specific applications [7, 48, 49, 67]. However, due to these VLMs being mainly based on text-image alignment training strategies, our experiments demonstrate that even the most powerful VLMs (such as OpenAI’s GPT4o) still fall short of detection abnormalities.

**Detection in AIGC:** Detection within AIGC comprises various tasks; some initiatives aim to discern AIGC products [6, 34, 36, 40, 44]. Recently, [14] tried to fix the abnormalities in the existing body parts, although it is limited to providing only coarse-grained results, leading to inconsistencies with the original visuals. In contrast, we introduce a novel task in abnormality detection, i.e., the FHAD. To the best of our knowledge, this is the first attempt at such Fine-Grained abnormality detection in AIGC products.

**Evaluation in AIGC:** Other methodologies interpret detection as a quality assessment exercise, evaluating AIGC products from diverse angles [27, 31, 41]. For instance, VideoPhy [2] assesses videos based on physical common sense, DEVIL [32] focuses on dynamic quality, and some studies assess the overall quality of AIGC videos with text-image alignment and video characteristics. Our proposed method, distinct from these, strives to ascertain whether the generated human body structure could occur in the real world. Figure 2 presents a comparative analysis of our task objective across three aspects: (i) distinguish AIGC-produced content by identifying AIGC products involves verifying whether the content originated from AIGC models; (ii) assess the AIGC content’s quality; (iii) detect the abnormality of AIGC content by spotting variances between generated outputs and real-world objects.

**Visual Content Generation:** The realm of visual content generation has undergone considerable evolution. Initial endeavors, such as GAN-based architectures [15, 37, 59,

![Figure 2: Fine-Grained Human-body Abnormality Detection (FHAD) plot. The plot shows various methods on a graph with 'Distinguish AIGC' on the x-axis and 'Abnormal Detection' on the y-axis. FHAD (Ours) is marked with a red star, indicating high performance in both detection and quality assessment. Other methods include StairReward[TCSV2023], GrainMat[CVPR2020], FNn[ECCV2020], UniDetectors[CVPR2023], LaR2[CVPR2024], What Matters[Arxiv2024], K-Sort Arena[Arxiv2024], DEVIL[Arxiv2024], and IPC[CVPRW2024].](ad04aa6cb9e7fb36bb7a91e817e2d314_img.jpg)

Figure 2: Fine-Grained Human-body Abnormality Detection (FHAD) plot. The plot shows various methods on a graph with 'Distinguish AIGC' on the x-axis and 'Abnormal Detection' on the y-axis. FHAD (Ours) is marked with a red star, indicating high performance in both detection and quality assessment. Other methods include StairReward[TCSV2023], GrainMat[CVPR2020], FNn[ECCV2020], UniDetectors[CVPR2023], LaR2[CVPR2024], What Matters[Arxiv2024], K-Sort Arena[Arxiv2024], DEVIL[Arxiv2024], and IPC[CVPRW2024].

Figure 2. Fine-Grained Human-body Abnormality Detection (FHAD) (★) is a novel task. It is distinct from AIGC detection and the assessment of AIGC product quality, as its objective is to identify the abnormality of content generated by AIGC methods in relation to the real world. Additionally, detection at a fine-grained level necessitates methods capable of providing detailed information about the abnormalities and their locations.

[65] and autoregressive methodologies [12, 46, 63], provide foundational breakthroughs but encountered issues like low resolution and stability concerns. Then, with the advent of diffusion models including text-to-image [39, 47, 50, 51] and text-to-video [18, 25, 42, 53, 62], the quality of visual content has been further developed. However, the content generated by these models is often quite different from the real world, especially in the structure of the human body which limits their widespread downstream use.

## 3. Task Definition

**Fine-grained Human-body Abnormality Detection (FHAD):** The goal of FHAD is to identify the differences in body structure from real-world humans in any given visual content that includes human photos. To achieve FHAD, the method needs to output the following two parts: (1) the semantic flag of abnormality  $S^a \subseteq \mathbf{A}$ , that is, what type of body part abnormality  $a$  exists. (2) the location of the abnormality  $a$ , output in the form of a bounding box  $B^a$ . For the input visual content within body part abnormalities  $\mathbf{X}$  and the pre-defined abnormalities set  $\mathbf{A}$ , we consider any detection method as  $M^d$ , the task can be formatted as:

$$[B^a, S^a] = M^d(X). \quad (1)$$

As shown in Figure 1, for a given human photo, the method needs to detect the redundant hand in the red bounding box.

**Human-body Abnormality Define:** After reviewing a large number of generated human photos, we conduct an analysis of body part abnormalities and identified 12 distinct body part abnormalities which often create a significant gap between the real-world human body structure. It contains two types, the absent and redundant body parts. For the class of body parts, we identify the following body

![Figure 3: Examples In AIGC Human-Aware 1K. The figure shows four examples of generated human photos with annotations. Top-left: A person with a missing hand, annotated 'An absent hand'. Top-right: A person with an extra arm, annotated 'A redundant arm'. Bottom-left: A person with a missing hand, annotated 'An absent hand' and 'A redundant arm'. Bottom-right: A person with all body parts, annotated 'All Well' with a thumbs-up icon.](1517552d07f4a8b389be86af8aff6424_img.jpg)

Figure 3: Examples In AIGC Human-Aware 1K. The figure shows four examples of generated human photos with annotations. Top-left: A person with a missing hand, annotated 'An absent hand'. Top-right: A person with an extra arm, annotated 'A redundant arm'. Bottom-left: A person with a missing hand, annotated 'An absent hand' and 'A redundant arm'. Bottom-right: A person with all body parts, annotated 'All Well' with a thumbs-up icon.

Figure 3. Examples in AIGC Human-Aware 1K. We manually annotate the abnormalities in frames from generated AIGC videos. Since the location of the abnormalities is ambiguous, we do not annotate the bounding box. Instead, we evaluate the accuracy of the bounding box location by assessing the repair quality.

parts, i.e., head, ear, hand, arm, leg, and foot for  $\mathbf{P}$ . Our following experiments are all conducted based on predefined abnormalities (e.g., *absent hand, redundant hand... \subseteq \mathbf{A}).*

**FHAD Dataset:** We propose two datasets for the proposed FHAD task, i.e., the COCO Human-Aware Val and the AIGC Human-Aware 1K. For **COCO Human-Aware Val**, we adopt an automated dataset construction method to construct images with pre-defined absent abnormalities from the full COCO Val split. We replace one body part with the background to make an absent body part abnormality. For **AIGC Human-Aware 1K**, it is a cross-domain, meticulously hand-labeled dataset with *inherent* body part abnormality. In order to enhance the generalization of the AIGC Human-Aware 1K. We build up this dataset based on the large generated video dataset VidProM [58]. To ensure the basic quality of the generated videos, we randomly choose the videos from the PIKA split with human photos. Then, we manually annotate 1K samples with the same body part classification in frame level, some cases are shown in Figure 3. Please note that compared to the COCO Human-Aware Val, AIGC Human-Aware 1K is a more challenging dataset for VLM as it comes from the AIGC domain and contains both absent and redundant body parts, which is the main evaluation dataset for our task. We provide a detailed dataset construction process in Appendix D, and we **highly recommend** reading this section, as it will be of great help in understanding our task. We also provide cases in COCO Human-Aware Val in Appendix F.

## 4. Methodology

### 4.1. Solution for Absent Body Part Detection

Within a detection pipeline, suppose the predefined human body classes of the whole body part as  $\mathbf{P}$  with the corresponding bounding box  $\mathbf{B}$  for the given visual content

within human photo  $X$ . The existing body parts are represented as a set  $\mathbf{P}^c \subseteq \mathbf{P}$  with their corresponding bounding boxes  $\mathbf{B}^c \subseteq \mathbf{B}$ . The evaluated method  $M^a$  should present the absent body part  $p^a$  and the absent area  $b^a$ . Note that,  $X$  also includes context for situations, e.g., obstructions, to avoid misjudgments of absent body parts. It can be represented as follows:

$$\{(p_i^a, b_i^a)\}_{i=0}^n = M^a(\mathbf{P}^c, \mathbf{B}^c; X), \quad (2)$$

where  $n$  denotes the number of absent body parts.

For this task, the VLM is an intuitive choice since it is trained on a vast dataset and exhibits strong capabilities across a wide range of downstream tasks. However, the results demonstrate that (Sec. 5), though VLMs have been trained with a large quantity of normal data like ours, they still lack awareness of the abnormality on COCO Human-Aware Val which is based on real-world images and only contain absent abnormalities. We discuss this phenomenon in Appendix C. To solve this problem, the simplest and most intuitive method is to manually annotate a large training dataset on real AIGC data like [2, 14]. However, for the tasks we propose, extensive manual annotation is unrealistic and extremely costly (we perform the detailed annotation process in Appendix D). Furthermore, data annotated based on a specific generative model will inevitably contain certain biases. Therefore, our goal shifted towards automating the construction of these training data from a real-world dataset, which is also more aligned with our target.

Inspired by the process of humans detecting absent abnormalities, we propose the following **body part correlation** training strategy. For the given visual content  $X$  with a normal human, we first ground all its body parts  $\{(p_i, b_i)\}_{i=0}^n$ , where  $n$  is the number of grounded body parts, based on our predefined set  $\mathbf{P}$ . For each body part representation  $(p_i, b_i)$ , we apply a mask operation to obtain:

$$X_i = \text{mask}(X, (p_i, b_i)), \quad (3)$$

where  $\text{mask}$  replaces the area  $b_i$  with the background of  $X$ . For each masked image, we maintain:

$$\{(X_i, (p_i^a, b_i^a))\}_{i=0}^n = \{(X_i, (p_i, b_i))\}_{i=0}^n, \quad (4)$$

to get the absent body part training sample  $(X_i, (p_i^a, b_i^a))$ . Similar to the current training objective of VLMs and LLMs, for each masked image  $X_i$  and its corresponding absent body part representation  $(p_i^a, b_i^a)$ , we optimize the following auto-regressive objective:

$$p((p_i^a, b_i^a) | X_i, I_a) = \prod_{j=1}^L p_\theta(x_j | X_i, I_{a,<j}, (p_i^a, b_i^a)_{,<j}), \quad (5)$$

where  $\theta$  represents the trainable parameters of the VLM,  $L$  is the length of the concatenated instruction  $I_a$  and the perception and position of the current absent body part  $(p_i^a, b_i^a)$ .

![Figure 4: Absent Human-body Detector (AHD) training strategy. The diagram shows an input image X of a person with a missing arm. A legend identifies: Body (red box), An arm at [0.43, 0.31, 0.67, 0.93] (blue box 1), An arm at [0.45, 0.12, 0.94, 0.33] (blue box 2), Bounding box 3 at [0.56, 0.75, 0.61, 0.93] (orange box), and A hand at [0.81, 0.13, 0.93, 0.27] (orange box 4). The input is processed by a VLM block with the question 'Is there any [red] absent in [green]?' and the instruction 'Based on X^1(1, 2, 3)'. The VLM outputs 'There should be a 3.', which is the target.](37a6ab1d23efb9dc00cfae09d353b1da_img.jpg)

Figure 4: Absent Human-body Detector (AHD) training strategy. The diagram shows an input image X of a person with a missing arm. A legend identifies: Body (red box), An arm at [0.43, 0.31, 0.67, 0.93] (blue box 1), An arm at [0.45, 0.12, 0.94, 0.33] (blue box 2), Bounding box 3 at [0.56, 0.75, 0.61, 0.93] (orange box), and A hand at [0.81, 0.13, 0.93, 0.27] (orange box 4). The input is processed by a VLM block with the question 'Is there any [red] absent in [green]?' and the instruction 'Based on X^1(1, 2, 3)'. The VLM outputs 'There should be a 3.', which is the target.

Figure 4. Absent Human-body Detector (AHD) training strategy. In the real world, many objects within the visual content are interconnected, meaning that based on the other objects, one can infer the presence of certain objects in specific locations. Our proposed training strategy leverages the correlation between body parts to facilitate this training process.

This training process sequentially replaces body parts in normal images with the background, as shown in Figure 4, allowing the VLM to learn the correlation between absent and existing body parts in terms of position and class based on the existing body parts.

Based on this training method, we develop a VLM named Absent Human-body Detector (**AHD**), represented as  $D$ , which is capable of detecting absent abnormalities of the body part. The AHD can infer from a given human photo whether there is any absent body, as well as identify the location and content of these absent body parts.

### 4.2. Solution For Redundant Body Part Detection

Compared to absent body part abnormality, the situation of redundant body part abnormality is more diverse, which is reflected in the following two parts: (1) The position of the redundant body parts, which can appear in any area of the given photo. (2) The number of redundant body parts, which can be arbitrary. These two factors make the judgment no longer dependent on the existing body parts  $\mathbf{P}^c$ . For each redundant body parts  $p^r$  with its corresponding area  $b^r$ , for given visual content  $X$  and the detection method  $M^r$  can be formalized as:

$$\{(p_i^r, b_i^r)\}_{i=0}^n = M^r(X), \quad (6)$$

where  $n$  is the number of redundant body parts. In contrast with Eq. 2, Eq. 6 indicates that addressing redundant bodies relies more on global visual information  $X$  rather than the existing body parts  $\mathbf{P}^c$  within  $X$ . Based on this, we utilize a diffusion-based inpainting model  $R$  with strong contextual understanding capabilities, combined with the grounding model  $G$ , to detect redundant body parts. In detail, for the given  $X$ , the model  $G$  can ground all body parts  $p_i^g \in \mathbf{P}$  with their locations  $b_i^g$ , as:

$$\{(p_i^g, b_i^g)\}_{i=0}^n = G(X, \mathbf{P}), \quad (7)$$

where  $n$  is the number of body parts in the given  $X$ . Then for each  $\langle p_i^g, b_i^g \rangle$ , we use the model  $R$  to re-generate the

![HumanCalibrator logo](bb08c83fc8939517c6803d65c69dd06b_img.jpg)

HumanCalibrator logo

#### HumanCalibrator: Perception&Re-generation Framework

![Figure 5: Illustration of the HumanCalibrator framework. The diagram is divided into two main sections: Perception Process and Re-generation. The Perception Process includes Redundant Body Parts Perception (Steps I, II, III) and Absent Body Parts Perception (AHD). The Re-generation section shows the final output and input image.](e394c2b5c61344f6a12397f430086072_img.jpg)

The diagram illustrates the HumanCalibrator framework, which consists of two main parts: Perception and Re-generation.

**Perception Process:**

- Redundant Body Parts Perception:** This section involves four steps:
  - Step I: Grounding** (indicated by a pink arrow), where the model identifies redundant body parts.
  - Step II: Inpainting** (indicated by a yellow arrow), where the model generates a version of the image without the redundant parts.
  - Step III: Comparison** (indicated by a green arrow), where the model compares the original image with the inpainted version to determine if the part is indeed redundant.
  - Comparison:** A box shows the result: "There is a redundant [arm] located at [0.181, 0.118, 0.356, 0.276]".
- Absent Body Parts Perception:** This section uses an Absent Human-body Detector (AHD) to identify missing parts.
  - A question is posed: "Question: Is any part of the person in the photo absent?"
  - The AHD output shows: "Yes, the [hand] is not present at [0.595, 0.072, 0.648, 0.120]".
  - The user is prompted to "Hint" or "Terminate".
  - If "No, ...", the process moves to Re-generation.
  - If "Yes, ...", a "New Round" is initiated.

**Re-generation:**

- This section shows the final output of the framework.
  - A box indicates: "There is a redundant [arm] located at [0.181, 0.118, 0.356, 0.276]".
  - Another box indicates: "Yes, the [hand] is not present at [0.195, 0.447, 0.299, 0.570]".
  - The final output shows the inpainted image with the redundant part removed and the absent part restored.

Figure 5: Illustration of the HumanCalibrator framework. The diagram is divided into two main sections: Perception Process and Re-generation. The Perception Process includes Redundant Body Parts Perception (Steps I, II, III) and Absent Body Parts Perception (AHD). The Re-generation section shows the final output and input image.

Figure 5. The illustration of our HumanCalibrator. The HumanCalibrator consists of two parts: perception and regeneration. In the perception stage, HumanCalibrator initially uses an inpainting model to re-generate various bodies based on its overall understanding of human body structure, determining the redundant bodies by comparing semantic differences before and after inpainting. Subsequently, relying on our Absent Human-body Detector (AHD) to assess the perception of absent abnormalities, our HumanCalibrator employs a cyclical strategy to identify absent bodies via AHD. Finally, by the results of the perception stage into the inpainting model as prompts, our HumanCalibrator can repair the detected abnormalities while preserving the visual content of the remaining areas.

content of  $b_i^g$  with the text-condition  $p_i^g$ , represented as  $p^R$ . Trained on large datasets of normal body structure without any redundant bodies, given the  $\langle p^r, b^r \rangle$  as mask location and text-condition for  $R$ , the semantic  $p^R$  of model  $R$ 's output content at  $b^r$  tend to exhibit a significant difference from the original  $p^r$ . For example, when a mask is applied to the location of a redundant arm,  $R$  is more likely to generate background in that area rather than the redundant arm itself. To determine if the original body part  $b_i^g$  is indeed the redundant body  $b^r$ , we compare the corresponding semantics  $p_i^g$  to  $p^R$ . If a significant semantic difference is detected (with the assistance of  $G$ ), it indicates that the body part  $\langle p_i^g, b_i^g \rangle$  is redundant.

This process can be formalized as:

$$\{(p_k^r, b_k^r)\}_{k=0}^n = \{G(R(p_i^g, b_i^g), p_i^g)\}_{i=0}^n < \tau, \quad (8)$$

where  $j$  is the number of the detected redundant body parts and  $\tau$  is the grounding threshold.

### 4.3. HumanCalibrator

By leveraging the proposed Absent Human-body Detector  $D$  with absent body part abnormality perception in Sec 4.1, and our proposed method for handling redundant body parts in Sec 4.2 we develop a comprehensive framework named HumanCalibrator, for the proposed Fine-Grained Human-body Abnormality Detection (FHAD) task. Furthermore, leveraging the ability of fine-grained detection, our HumanCalibrator can repair abnormalities of body parts while preserving other visual content unchanged. The details of HumanCalibrator are shown in Figure 5. In detail, our proposed framework can be divided into the following three

steps:

- Step I:** Detect redundant body parts in the given visual content  $X$ . The first step is to obtain the set of redundant body parts in  $X$  via Eq. 7, Eq. 8.i.e.,  $\{(p_k^r, b_k^r)\}_{k=0}^n$ , where  $k$  is the number of redundant body parts. Identifying redundant body parts first serves two purposes. (1) Provide a better image base for addressing absent body part abnormalities. (2) Self-refine during the process of addressing absent human body abnormalities. If it is found that the detected absenting body part is the same as the previously identified redundant bodies, it can be considered that this is a wrongly resolved redundant body, thereby improving the accuracy of the entire framework.
  - Step II:** Detect cyclically absent body parts in the given visual content  $X$ . We use  $D$  to detect the absent body part in the current  $X$  and obtain the detection result  $\langle p^a, b^a \rangle$ . Subsequently, we utilize the inpainting prompt template  $T$ , which is predefined according to  $P$ , to obtain the corresponding repair prompt  $T(p^a)$ . By combining  $p^a$  and  $b^a$ , we use the inpainting model  $R$  to obtain a new image  $X'$ . The resulting  $X'$  is then used as the image for the next detection cycle. This process continues until  $D$  determines that there are no new  $\langle p^a, b^a \rangle$  in the current  $X'$ , which can be formalized as:
- $$X_{t+1} = \begin{cases} R(X_t, T(p^a), b^a), & \text{if } \langle p^a, b^a \rangle = D(X_t) \neq \emptyset \\ X_t, & \text{otherwise.} \end{cases} \quad (9)$$
- Through this loop, we can obtain all the absent bodies in the image, denoted as  $\{(p_i^a, b_i^a)\}_{i=0}^n$ , where  $n$  is the number of detected absent body parts.
  - Step III:** Repair the abnormality detected above. After detecting the redundant and absent body parts separately,

![Figure 6: Bar chart comparing accuracy in COCO Human-aware Val. The chart shows accuracy for various models: Open-source VLM Baseline (InternVL2-2B: 4.84%, InternVL2-2.5B: 7.80%, InternVL2-2.4B: 1.84%, InternVL2-2-2.5B: 4.53%), Closed-source VLM Baseline (GPT-4o mini: 2.76%, GPT-4o: 4.20%), CLIP Baseline (absent body part detection: 76.91%), and Random Guess (11.11%).](431b8889a0e7f676f0eef40859590349_img.jpg)

| Model                                  | Accuracy (%) |
|----------------------------------------|--------------|
| InternVL2-2B                           | 4.84%        |
| InternVL2-2.5B                         | 7.80%        |
| InternVL2-2.4B                         | 1.84%        |
| InternVL2-2-2.5B                       | 4.53%        |
| GPT-4o mini                            | 2.76%        |
| GPT-4o                                 | 4.20%        |
| clip-with-absent-body-part-detection   | 23.32%       |
| clip-with-absent-body-part-detection-2 | 23.31%       |
| clip-with-absent-body-part-detection-3 | 30.21%       |
| Random Guess                           | 11.11%       |
| Absent Human-body Detector(AHD)        | 76.91%       |

Figure 6: Bar chart comparing accuracy in COCO Human-aware Val. The chart shows accuracy for various models: Open-source VLM Baseline (InternVL2-2B: 4.84%, InternVL2-2.5B: 7.80%, InternVL2-2.4B: 1.84%, InternVL2-2-2.5B: 4.53%), Closed-source VLM Baseline (GPT-4o mini: 2.76%, GPT-4o: 4.20%), CLIP Baseline (absent body part detection: 76.91%), and Random Guess (11.11%).

Figure 6. Comparison of accuracy in COCO Human-aware Val. We evaluate **Open-source** VLMs (InternVL2) and **Closed-source** VLMs (GPT-4o and GPT-4o mini), **CLIP** and our **Absent Human-body Detector (AHD)**. The powerful VLMs demonstrate significant limitations in perceiving abnormalities in body parts, with results often comparable to random guessing, even though the COCO Human-aware Val is built upon real-world images that are similar to their training data set.

we can repair the detected abnormalities through the Inpainting model  $R$ . We start with the original image  $X$  and repair these abnormalities one by one. We also use the pre-defined inpainting prompt template  $T$  to repair different types of abnormalities, which can be formalized as:

$$X_0 = X$$

$$X_{t+1} = R(X_t, T(p_t), b_t), \text{ for } t = 0, \dots, j + n \quad (10)$$

$$\text{where } \langle p_t, b_t \rangle = \begin{cases} \langle p_t^r, b_t^r \rangle, & \text{if } 0 \leq t \leq j \\ \langle p_{t-j}^a, b_{t-j}^a \rangle, & \text{if } j < t \leq j + n, \end{cases}$$

where  $j$  and  $n$  represent the number of redundant and absent body parts, respectively. The final  $X_{j+n}$  is the image after all abnormalities have been repaired.

## 5. Experiment

**Experiments on Exploring Current VLMs.** To objectively verify the ability of the existing VLM to detect the abnormalities of body structure, we employ the automatically generated COCO Human-Aware Val to test the VLMs. It is important to note that the COCO Human-Aware Val is automatically produced based on the entire COCO Val Split.

We compared different vision-language models (VLMs), including the state-of-the-art **Open-source VLM** InternVL2 and **Closed-source VLM** GPT4o and GPT4o-mini, along with contrastive learning-based models such as **CLIP**, as presented in Figure 6. The results demonstrate that despite their strong performance in many visual tasks, these models struggle with abnormal perceptions of body parts, displaying accuracies close to random guesses. Interestingly, even though humans and these VLMs are exposed to similar volumes of normal data, abnormality detection remains

markedly easier for human cognition. The intricacies of our comparative analysis are further discussed in Appendix C, and the specifics of the baseline models we test are detailed in Appendix B.

**Details of Human Calibrator:** For the absent body part detector, we finetune the LLaVA1.5 7B [33] on COCO Train Split via Eq. 5. The format of the training data is similar to the COCO Human-Aware Val, some cases are shown in Appendix F. All training runs on 4 NVIDIA A800 GPUs. It takes around 30 hours to fine-tune 2 epochs with a learning rate of  $2 \times 10^{-5}$ . We also test its ability on the COCO Human-aware Val, the results are shown in Figure 6. For the other pretrained models used in our HumanCalibrator, we list more details in the Appendix A.

**Body Part Abnormality Detection.** We evaluate the accuracy of our Human Calibrator’s perception ability of abnormalities on the AIGC Human-Aware 1K dataset. Similar to the assessment on the COCO Human-Aware Val, we also benchmark several recent powerful VLMs as our baseline, the results are shown in Table 1, and we provide the acc and false detection rate (FDR) for each specific category. Similar to the results on the COCO Human-Aware Val, existing VLMs have difficulty in accurately perceiving abnormalities. In the aspect of absent detection, the trained Absent Human Detector effectively detects the existing abnormalities while maintaining a low FDR. In terms of absences, even under training-free conditions, our method achieves better results compared to the baseline. We analyze these results in detail in Appendix B.

### 5.1. Body Part Abnormality Repair

**Metrics.** The goal of the task we proposed is to detect and locate the abnormality of body parts that make the human different from real-world humans, to comprehensively evaluate the effectiveness of our proposed HumanCalibrator in solving the task, we adopt six metrics for quantitative evaluation: (1) accuracy and FDR of abnormality detection, which calculate the detection results for each category of absent and redundant body parts. (2) Clip Score, the similarity between the image and the original prompt. (3) Human CLIP Score, the similarity between the image and the prompt which only contains only descriptions related to the human. (4) Human Concept Score, the similarity to the concept of ‘human’. (5) FID, we use the origin image as the real image and the repaired image as the generated image to assess the extent to which our repair method preserves the original information. (6) Latent Consistency, consistency between the original and repaired image in the latent space.

**Repair Quality** provides insight into the overall performance from two aspects. (1) The accuracy of the detected bounding boxes, due to the worse repair results caused by inaccurate re-generation to the detected areas compared to the original image. (2) Whether our repair makes the per-

| Type      | Method                | Human Body     |                  |                |                  |                |                  |                |                  |                |                  |                |                  |
|-----------|-----------------------|----------------|------------------|----------------|------------------|----------------|------------------|----------------|------------------|----------------|------------------|----------------|------------------|
|           |                       | hand           |                  | leg            |                  | ear            |                  | foot           |                  | arm            |                  | head           |                  |
|           |                       | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ |
| Absent    | LLaVA-34B             | 0.42%          | <b>0.92%</b>     | 15.00%         | <b>2.14%</b>     | 0.47%          | <b>0.00%</b>     | 0.00%          | –                | 6.00%          | 5.26%            | 36.00%         | 5.74%            |
|           | InternVL2-26B         | 2.95%          | 2.75%            | 10.00%         | 5.51%            | 5.69%          | 2.41%            | 3.03%          | 2.46%            | 42.53%         | 35.71%           | 52.00%         | 8.31%            |
|           | GPT-4o                | 8.02%          | 2.49%            | 0.00%          | –                | 0.47%          | 0.13%            | 10.61%         | <b>0.00%</b>     | 12.64%         | <b>3.83%</b>     | 20.00%         | <b>0.62%</b>     |
|           | CLIP-Large-14         | 14.35%         | 6.55%            | 15.00%         | 4.59%            | 17.06%         | 16.22%           | 19.70%         | 2.68%            | 44.33%         | 16.10%           | 44.00%         | 5.74%            |
|           | HumanCalibrator(Ours) | <b>79.75%</b>  | 12.19%           | <b>75.00%</b>  | 6.22%            | <b>79.15%</b>  | 10.14%           | <b>90.91%</b>  | 4.39%            | <b>75.14%</b>  | 12.92%           | <b>80.00%</b>  | <b>5.03%</b>     |
|           |                       | hand           |                  | leg            |                  | ear            |                  | foot           |                  | arm            |                  | head           |                  |
|           |                       | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ | Acc $\uparrow$ | FDR $\downarrow$ |
| Redundant | LLaVA-34B             | 3.16%          | <b>0.44%</b>     | <b>33.33%</b>  | 2.33%            | 0.00%          | –                | 0.00%          | –                | 48.57%         | 18.76%           | 14.29%         | 1.91%            |
|           | InternVL2-26B         | 2.11%          | 1.10%            | 0.00%          | –                | 16.67%         | <b>0.60%</b>     | 0.00%          | –                | 20.00%         | 5.70%            | 57.14%         | 0.70%            |
|           | GPT-4o                | 7.37%          | 0.88%            | 0.00%          | –                | 0.00%          | –                | 1.71%          | <b>0.20%</b>     | 25.71%         | 3.83%            | 14.29%         | <b>0.60%</b>     |
|           | CLIP-Large-14         | 21.05%         | 6.08%            | 25.00%         | 2.94%            | 33.33%         | 15.90%           | 0.00%          | –                | <b>57.14%</b>  | 17.41%           | 0.00%          | –                |
|           | HumanCalibrator(Ours) | <b>65.26%</b>  | 3.87%            | <b>33.33%</b>  | <b>0.81%</b>     | <b>83.33%</b>  | 2.31%            | <b>66.67%</b>  | 0.70%            | <b>45.71%</b>  | <b>2.49%</b>     | <b>57.14%</b>  | 5.03%            |

Table 1. Detection accuracy of body part abnormalities. We evaluate performance via acc and false discovery rate (FDR). The – implies that when the acc is zero, the FDR loses its statistical significance. The test is performed on open-source VLM (LLaVA-34B and InternVL2-26B), closed-source VLM (GPT4o), CLIP, and our HumanCalibrator. Compared to the best-performing existing VLM, our framework significantly improves accuracy across almost all abnormality types with low FDR. We provide a detailed evaluation process for the baseline and an analysis of the reasons for their poor performance in the Appendix B.

![Figure 7: Case study of the image repair quality of HumanCalibrator. The figure shows four rows (a-d) of images. Row (a) shows a woman with an 'Absent Hand' abnormality. Row (b) shows a man with an 'Absent Ear' abnormality. Row (c) shows a woman with a 'Redundant Arm' abnormality. Row (d) shows a person with a 'Redundant Arm' abnormality. Each row includes the original image, a detection result with a bounding box, a repair result with a bounding box, and a final repaired image with a smiley face icon.](1dba94e9a65ea5fbd805e44a5a2c8cd5_img.jpg)

Figure 7: Case study of the image repair quality of HumanCalibrator. The figure shows four rows (a-d) of images. Row (a) shows a woman with an 'Absent Hand' abnormality. Row (b) shows a man with an 'Absent Ear' abnormality. Row (c) shows a woman with a 'Redundant Arm' abnormality. Row (d) shows a person with a 'Redundant Arm' abnormality. Each row includes the original image, a detection result with a bounding box, a repair result with a bounding box, and a final repaired image with a smiley face icon.

Figure 7. Case study of the image repair quality of HumanCalibrator. In (a), the HumanCalibrator detects the absent hand of the woman and repairs it with a hand in the correct pose and shape. In (b), an absent ear is identified, and the HumanCalibrator regenerates the ear without altering the man’s face or expression. In (c), the redundant arm of the woman is detected and removed without affecting the background. In (d), the redundant arm of the person is corrected while preserving the rest of his body. Compared to the original images, our method achieves high-quality repair of the human body structure while preserving the remaining visual content.

son in the visual content more similar to a real-world human, which is the motivation of our proposed task. CLIP Score and Human CLIP Score are used to evaluate the quality of the repairs, with the Human CLIP Score focusing on the prompt describing the person, which is more closely related to our task. The Human Concept Score measures the distance between the human in the image and a ‘human’ in the real world. As shown in Table 2. Compared to the original visual content, the repaired images show a certain degree of improvement in various metrics. The extent of our

improvement is not significant, which is due to our good maintenance of consistency outside the abnormal parts of the image. To evaluate this consistency, we further measured the metrics on visual consistency.

**Visual Consistency.** A key advantage of our proposed fine-grained anomaly detection is its ability to repair only the abnormalities while maintaining the consistency of other information. We compare our method with the pose-condition-based abnormality repair method [14] in FID and Latent space, as shown in Table 2. It can be observed that

![Figure 8: Case study of the video repair quality of HumanCalibrator. The figure shows a sequence of five frames from an 'Origin Video' (Time: First Frame to Last Frame). The first row shows the original video with noticeable abnormalities (red boxes). The second row shows 'No Repaired Re-Generation' using a keyframe interpolation model, which preserves most visual information but still exhibits abnormalities. The third row shows 'Repaired Re-Generation' using HumanCalibrator, which efficiently addresses the abnormalities while preserving the remaining visual information. Insets on the right show close-ups of the repaired areas.](c834b9abb4ddf70e5d10641f87d5ff5b_img.jpg)

Figure 8: Case study of the video repair quality of HumanCalibrator. The figure shows a sequence of five frames from an 'Origin Video' (Time: First Frame to Last Frame). The first row shows the original video with noticeable abnormalities (red boxes). The second row shows 'No Repaired Re-Generation' using a keyframe interpolation model, which preserves most visual information but still exhibits abnormalities. The third row shows 'Repaired Re-Generation' using HumanCalibrator, which efficiently addresses the abnormalities while preserving the remaining visual information. Insets on the right show close-ups of the repaired areas.

Figure 8. Case study of the video repair quality of HumanCalibrator. The first row shows frames from the original video, which contain noticeable abnormalities; the second row shows the video generated by a keyframe interpolation model using the original first and last frames, which preserves most of the visual information but still exhibits abnormalities; the third row shows the video regenerated with repaired first and last frame by HumanCalibrator, efficiently addressing the abnormalities while preserving the remaining visual information.

| Metric                         | Original | Ours         |
|--------------------------------|----------|--------------|
| Human Concept Score $\uparrow$ | 22.59    | <b>22.77</b> |
| CLIP Score $\uparrow$          | 41.87    | <b>41.97</b> |
| Human CLIP Score $\uparrow$    | 26.36    | <b>26.42</b> |

  

| Metric                        | Pose Condition | Ours         |
|-------------------------------|----------------|--------------|
| FID $\downarrow$              | 98.86          | <b>16.55</b> |
| Latent Consistency $\uparrow$ | 0.668          | <b>0.964</b> |

Table 2. Repair quality and visual consistency evaluation. Repair quality is assessed via: (1) Human Concept Score, measuring the similarity between the person in the visual content with a real-world human. (2) CLIP Score, and (3) Human CLIP Score, focuses on the prompt describing the person, that is more relevant to our target. The repaired images show improvements over the original ones across these metrics. The modest improvement in the metrics is due to our effectiveness in preserving the remaining visual content. For visual consistency, we compare our method to a pose-conditioned method [14] via FID and Latent Consistency showing that our approach maintains strong visual consistency.

our HumanCalibrator maintains good visual consistency at both metrics. The details of the evaluation are shown in Appendix E.

**Case Study.** We present three levels of case studies: image-level, video-level, and generalization-level. (1) Image Case Study: As shown in Figure 7, owing to the fine-grained abnormality detection of our HumanCalibrator, it repairs the abnormalities in the human body structure within images while preserving other normal and unrelated information. (2) Video Case Study: As illustrated in Figure 8, due to the detecting and repairing ability of our HumanCalibrator, we can repair the first and last frames of a video and regenerate the intermediate frames with a keyframe interpolation model. It shows that with the repaired first and last

![Figure 9: Examples of HumanCalibrator applied to frames generated by other SOTA Video generation models. (a) CogVideoX: 'An absent ear' is detected and repaired. (b) PyramidFlow: 'An absent hand' is detected and repaired. (c) T2VZ: 'An absent hand' and 'An absent arm' are detected and repaired. (d) AnimatedDiff: 'A redundant hand' is detected and corrected. Each row shows the original frame with a red box around the abnormality, a blue arrow pointing to the repaired frame, and a green box highlighting the repaired area.](ec77a22da01e15158981baf4129a63b5_img.jpg)

Figure 9: Examples of HumanCalibrator applied to frames generated by other SOTA Video generation models. (a) CogVideoX: 'An absent ear' is detected and repaired. (b) PyramidFlow: 'An absent hand' is detected and repaired. (c) T2VZ: 'An absent hand' and 'An absent arm' are detected and repaired. (d) AnimatedDiff: 'A redundant hand' is detected and corrected. Each row shows the original frame with a red box around the abnormality, a blue arrow pointing to the repaired frame, and a green box highlighting the repaired area.

Figure 9. Examples of HumanCalibrator applied to frames generated by other SOTA Video generation models (frames in (a),(b),(c),(d) are produced by CogVideoX [62], PyramidFlow [26], T2VZ [24] and AnimatedDiff [16], respectively). Each model generates human photos with distinct abnormalities, which are subsequently addressed by our HumanCalibrator. In (a), the HumanCalibrator detects and repairs an absent ear on the singer. In (b), an absent hand is identified, with HumanCalibrator identifying the positions and completing the restoration. In (c), the HumanCalibrator locates and repairs an absent hand and arm. In (d), the girl’s redundant hand is detected and corrected.

frames and the original prompt, our HumanCalibrator can repair abnormalities while maintaining the other content of the video. (3) Generalization-level: Our HumanCalibrator also demonstrates strong performance on outputs generated

by other generative models, with the results shown in Figure 9. More cases are shown in the Appendix G.

## 6. Conclusion

In this paper, we propose HumanCalibrator, a fine-grained level abnormal detection and repair framework in AIGC visual content with two datasets across different domains. It can detect abnormal body parts, and repair the abnormality while maintaining the other visual content. However, there are still limitations to our proposed framework, e.g. the predefined abnormal human body class limits the generalizability. In the future, we plan to extend our method to support more visual categories and more types of abnormalities.

## References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. *arXiv preprint arXiv:2308.12966*, 2023. 2
- [2] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. *arXiv preprint arXiv:2406.03520*, 2024. 2, 4
- [3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassa, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. *OpenAI*, 2023. 1
- [4] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. *arXiv preprint arXiv:2401.02954*, 2024. 2
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. 2
- [6] Chirui Chang, Zhengzhe Liu, Xiaoyang Lyu, and Xiaojuan Qi. What matters in detecting ai-generated videos like sora? *arXiv preprint arXiv:2406.19568*, 2024. 2
- [7] Ling-Hao Chen, Shunlin Lu, Ailing Zeng, Hao Zhang, Benyou Wang, Ruimao Zhang, and Lei Zhang. Motionllm: Understanding human behaviors from human motions and videos. *arXiv preprint arXiv:2405.20340*, 2024. 2
- [8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *arXiv preprint arXiv:2404.16821*, 2024. 2, 1
- [9] Hyung Won Chung, Le Hou, Shayne Longpre, Barrett Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. *arXiv preprint arXiv:2210.11416*, 2022. 2
- [10] Dawei Dai, Yuanhui Zhang, Long Xu, Qianlan Yang, Xiaojing Shen, Shuyin Xia, and Guoyin Wang. Pa-llava: A large language-vision assistant for human pathology image understanding. *arXiv preprint arXiv:2408.09530*, 2024. 1
- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheung Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023. 2
- [12] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. *Advances in neural information processing systems*, 34:19822–19835, 2021. 3
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arxiv* 2020. *arXiv preprint arXiv:2010.11929*, 2010. 2
- [14] Guian Fang, Wenbiao Yan, Yuanfan Guo, Jianhua Han, Zutao Jiang, Hang Xu, Shengcai Liao, and Xiaodan Liang. Humaneffiner: Benchmarking abnormal human generation and refining with coarse-to-fine pose-reversible guidance. *arXiv preprint arXiv:2407.06937*, 2024. 1, 2, 4, 7, 8
- [15] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. *Advances in neural information processing systems*, 27, 2014. 2
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. *International Conference on Learning Representations*, 2024. 8
- [17] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yuezue Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. *arXiv preprint arXiv:2402.11530*, 2024. 2
- [18] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. *arXiv preprint arXiv:2210.02303*, 2022. 3
- [19] Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, and Zhuowen Tu. Bliva: A simple multimodal llm for better handling of text-rich visual questions. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 2256–2264, 2024. 2
- [20] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 6700–6709, 2019. 2
- [21] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las

- Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. *Mistral 7b*. *arXiv preprint arXiv:2310.06825*, 2023. [2](#)
- [22] Xuan Ju, Ailing Zeng, Chenchen Zhao, Jianan Wang, Lei Zhang, and Qiang Xu. Humansd: A native skeleton-guided diffusion model for human image generation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 15988–15998, 2023. [1](#)
- [23] Lei Ke, Wenjie Pei, Ruiyu Li, Xiaoyong Shen, and Yu-Wing Tai. Reflective decoding network for image captioning. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 8888–8897, 2019. [2](#)
- [24] Levon Khachatrian, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. *arXiv preprint arXiv:2303.13439*, 2023. [8](#)
- [25] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. [1, 3](#)
- [26] Jiarui Lei, Xiaohu Hu, Yue Wang, and Dong Liu. Pyramidflow: High-resolution defect contrastive localization using pyramid normalizing flow. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 14143–14152, 2023. [8](#)
- [27] Chunyi Li, Zicheng Zhang, Haoning Wu, Wei Sun, Xiongkuo Min, Xiaohong Liu, Guangtao Zhai, and Weisi Lin. Agica-3k: An open database for ai-generated image quality assessment. *IEEE Transactions on Circuits and Systems for Video Technology*, 2023. [2](#)
- [28] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. *Advances in Neural Information Processing Systems*, 36, 2024. [1](#)
- [29] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *arXiv preprint arXiv:2301.12597*, 2023. [2](#)
- [30] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. *arXiv:2403.18814*, 2023. [2](#)
- [31] Zhikai Li, Xuewen Liu, Dongrong Fu, Jianquan Li, Qingyi Gu, Kurt Keutzer, and Zhen Dong. K-sort arena: Efficient and reliable benchmarking for generative models via k-wise human preferences. *arXiv preprint arXiv:2408.14468*, 2024. [2](#)
- [32] Mingxiang Liao, Hannan Lu, Xinyu Zhang, Fang Wan, Tianyu Wang, Yuzhong Zhao, Wangmeng Zuo, Qixiang Ye, and Jingdong Wang. Evaluation of text-to-video generation models: A dynamics perspective. *arXiv preprint arXiv:2407.01094*, 2024. [2](#)
- [33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. 2023. [2, 6, 1](#)
- [34] Zhengzhe Liu, Xiaojuan Qi, and Philip HS Torr. Global texture enhancement for fake face detection in the wild. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8060–8069, 2020. [2](#)
- [35] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. *Advances in Neural Information Processing Systems*, 35:2507–2521, 2022. [2](#)
- [36] Yunpeng Luo, Junlong Du, Ke Yan, and Shouhong Ding. Lare<sup>2</sup>: Latent reconstruction error based method for diffusion-generated image detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 17006–17015, 2024. [2](#)
- [37] Elman Mansimov, Emilio Parisotto, Jimmy Lei Ba, and Ruslan Salakhutdinov. Generating images from captions with attention. *arXiv preprint arXiv:1511.02793*, 2015. [2](#)
- [38] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongqiang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 4296–4304, 2024. [1](#)
- [39] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. *arXiv preprint arXiv:2112.10741*, 2021. [3](#)
- [40] Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. Towards universal fake image detectors that generalize across generative models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 24480–24489, 2023. [2](#)
- [41] Fei Peng, Huiyuan Fu, Anlong Ming, Chuanming Wang, Huadong Ma, Shuai He, Zifei Dou, and Shu Chen. Aigc image quality assessment via image-prompt correspondence. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*, pages 6432–6441, 2024. [2](#)
- [42] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*, 2023. [1, 3](#)
- [43] Sanjita Prajapati, Tanu Singh, Chinnay Hegde, and Pranamesh Chakraborty. Evaluation and comparison of visual language models for transportation engineering problems. *arXiv preprint arXiv:2409.02278*, 2024. [1](#)
- [44] Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In *European conference on computer vision*, pages 86–103. Springer, 2020. [2](#)
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021. [2](#)
- [46] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In *International confer-*

- ence on machine learning, pages 8821–8831. Pmlr, 2021. 1, 3
- [47] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022. 1, 3
- [48] Hanoota Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. Glamm: Pixel grounding large multimodal model. *The IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024. 2
- [49] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. *arXiv, abs/2312.02051*, 2023. 2
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10684–10695, 2022. 1, 3
- [51] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. *Advances in neural information processing systems*, 35:36479–36494, 2022. 3
- [52] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. *arXiv*, 2022. 2
- [53] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without test-video data. *arXiv preprint arXiv:2209.14792*, 2022. 3
- [54] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. *arXiv preprint arXiv:2403.08295*, 2024. 2
- [55] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. 2024. 2
- [56] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023. 2
- [57] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023. 2
- [58] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. *arXiv preprint arXiv:2403.06098*, 2024. 3
- [59] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, XiaoLei Huang, and Xiaodong He. AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1316–1324, 2018. 2
- [60] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. *arXiv preprint arXiv:2408.08872*, 2024. 2
- [61] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024. 2
- [62] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. *arXiv preprint arXiv:2408.06072*, 2024. 1, 3, 8
- [63] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. *arXiv preprint arXiv:2206.10789*, 2(3):5, 2022. 3
- [64] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. *arXiv preprint arXiv:2210.02414*, 2022. 2
- [65] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, XiaoLei Huang, and Dimitris N Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In *Proceedings of the IEEE international conference on computer vision*, pages 5907–5915, 2017. 3
- [66] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 3836–3847, 2023. 1
- [67] Shanshan Zhong, Zhongzhan Huang, Shanghua Gao, Wushao Wen, Liang Lin, Marinka Zitnik, and Pan Zhou. Let’s think outside the box: Exploring leap-of-thought in large language models with creative humor generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13246–13257, 2024. 2

# Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body

## Supplementary Material

We highly recommend watching the supplementary video, as it comprehensively demonstrates our proposed task and the results of our proposed HumanCalibrator.

**Disclaimer:** The supplementary material includes images that may be unsettling or discomforting to some readers. We have removed all personal information from the cases and applied mosaic to some images that may cause discomfort.

### A. Details in HumanCalibrator

#### A.1. Model Usage

In addition to using LLaVA1.5-7B as the base model for the Absent Human-body Detector, the other models employed in the HumanCalibrator are as follows: (1) The inpainting model  $R$ , which utilizes StableDiffusion2-Inpainting<sup>1</sup>, (2) the grounding model  $G$ , which adopts GroundingDINO<sup>2</sup>, and (3) the video interpolation model, which employs CogVideoX-Interpolation<sup>3</sup> based on CogVideoX [62].

#### A.2. Other Implementation Details

Additional details in our HumanCalibrator are as follows: (1) To improve the repair quality of the overall human photo, we expand the bounding box of the abnormal region before applying inpainting. This ensures better visual quality between the inpainted and surrounding regions. (2) Since the inpainting model inevitably leads to a decline in overall image quality, we apply  $2\times$  super-resolution processing to the inpainted images. It is worth noting that, for a fair comparison, no super-resolution processing is applied in any of the comparisons in Table 2. (3) To better adapt the Absent Human-body Detector, trained on real-world COCO datasets, for application in AIGC, we perform semantic detection on each absent region identified by the detector using the Grounding Model  $G$ . If the same semantic content is detected, the result from this iteration of the Absent Human-body Detector is discarded.

<sup>1</sup><https://huggingface.co/stabilityai/stable-diffusion-2-inpainting>, Rombach, R. et al. (2022). High-Resolution Image Synthesis With Latent Diffusion Models. In Proc. CVPR2022.

<sup>2</sup><https://github.com/IDEA-Research/GroundingDINO>, Liu, S. et al. (2023). Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv

<sup>3</sup><https://huggingface.co/feizhengcong/CogvideoX-Interpolation>

### B. Details of Baselines and Analysis

#### B.1. Baseline for COCO Human-Aware Val

The COCO Human-Aware Val dataset only contains absent abnormalities caused by masking out body parts. Since it is derived from real-world images and includes only the “absent” category of abnormalities, our comparisons on this dataset primarily focus on two objectives: (1) demonstrating the deficiency of existing VLMs in abnormality perception and (2) evaluating the performance of our trained Absent Human-body Detector (AHD).

Evaluating the baseline of CLIP on COCO Human-Aware Val: Similar to other methods, we transform the different types of abnormalities into a classification problem. The CLIP model selects the text with the highest matching score to the image as its predicted answer. The specific text categories are as follows:

- “*The person in the picture has absent head.*”
- “*The person in the picture has absent ear.*”
- “*The person in the picture has absent arm.*”
- “*The person in the picture has absent hand.*”
- “*The person in the picture has absent foot.*”
- “*The person in the picture has absent leg.*”
- “*The person in the image has no abnormalities.*”

Evaluating the baseline of Generative VLMs, we use the following prompt to like VQA tasks [8, 33] to prompt the VLMs:

- “*Are there any absent body parts in the person shown in the image? If yes, please answer from ‘head’, ‘arm’, ‘leg’, ‘foot’, ‘hand’, or ‘ear’; otherwise, please answer ‘no’. Answer the question using a single word:*”

#### B.2. Baseline for AIGC Human-Aware 1K

Unlike the COCO Human-Aware Val dataset, the AIGC Human-Aware 1K dataset includes all categories of abnormalities. For CLIP, we directly add additional abnormal categories and use a similar classification approach to evaluate its performance. For Generative VLMs, we adopt a simpler method tailored to VLMs. Specifically, we separately ask whether there were abnormalities in the “redundant” category and the “absent” category. Additionally, since the abnormalities in AIGC Human-Aware 1K are diverse in number, we do not constrain the model’s responses to a single word, i.e., we do not use “answer the question using a single word”. After receiving the responses, we use an LLM for post-processing to produce formatted data suitable for accuracy calculation. Since these baseline VLMs perform

weakly on FHAD, we try various prompts per model to optimize performance in our experiments. The prompts that yielded the best performance are displayed below:

##### • For **LLaVA-34B**:

- In “Absent Abnormality Detection”: “*Are there any missing body parts in the person shown in the image? If so, please answer the precise part:*”
- In “Redundant Abnormality Detection”: “*Are there any extra body parts in the person shown in the image? If so, please answer the precise part:*”

##### • For **Intern VL2-26B**:

- In “Absent Abnormality Detection”: “*According to the human anatomical structure, are there any missing body parts in the person shown in the image? If so, please answer the precise part:*”
- In “Redundant Abnormality Detection”: “*According to the human anatomical structure, are there any extra body parts in the person shown in the image? If so, please answer the precise part:*”

##### • For **GPT-4o**:

- In “Absent Abnormality Detection”: “*It is a common sense that all human being has one head, two ears, two hands, two arms, two legs and two feet, are there any missing body parts which I discussed in the person shown in the image? If so, please answer the precise part:*”
- In “Redundant Abnormality Detection”: “*It is a common sense that all human being has one head, two ears, two hands, two arms, two legs and two feet, are there any extra body parts which I discussed in the person shown in the image? If so, please answer the precise part:*”

For the post process for the response of VLMs (Note that, we use the GPT4o-mini to post-process the response) as shown in Figure S1.

#### B.3. Baseline Analysis

Our work is based on a key assumption: that existing powerful VLMs fail to perform abnormality detection, a task that is exceptionally simple for humans. We provide a detailed case analysis of their poor performance. Specifically, there are two primary reasons for this under-performance: (1) a lack of understanding of human body structure, and (2) a misinterpretation of abnormalities. We present examples from **real test** in Figure S2.

#### B.4. Pose Condition

Since the code for HumanRefiner [14] is unavailable and our objective differs fundamentally, we only reproduce its step of using pose as an additional constraint to ensure no abnormalities in the number of body parts. Specifically,

![Figure S1: Prompt for post-processing the VLM output. The image shows a screenshot of a prompt box with instructions for analyzing VLM output regarding body parts. Below the instructions, there are three examples of input and output pairs. The first example shows a person with one visible arm, and the output identifies the missing arm. The second example shows the upper half of a person, and the output identifies missing legs and feet. The third example shows a person with an extra arm and an extra leg, and the output identifies these extra parts.](08e28a86c71221f509c8f3a828e89af2_img.jpg)

```

Please analyze the model's response about extra or missing body parts and output only a list of the specifically mentioned body parts that are confirmed as extra or missing. Return the result as a simple list of individual words wrapped in <output> tags (e.g. <output>[arm]</output>, <output>[leg, hand]</output>). If the response indicates uncertainty, normal body parts, or no abnormalities, return <output>[]</output>.

Input: "The image depicts a person with one visible arm. The other arm appears to be missing or obscured."
Output: <output>[arm]</output>

Input: "The image shows the upper half of a person. All visible body parts like the head, ears, arms, and hands seem present. Legs and feet are not visible in this image, so a determination about them cannot be made."
Output: <output>[leg, hand]</output>

Input: "The person in the image appears to have an extra hand"
Output: <output>[hand]</output>

Input: "The person in the image appears to have an extra arm and an extra leg"
Output: <output>[arm, leg]</output>

Input:
  
```

Figure S1: Prompt for post-processing the VLM output. The image shows a screenshot of a prompt box with instructions for analyzing VLM output regarding body parts. Below the instructions, there are three examples of input and output pairs. The first example shows a person with one visible arm, and the output identifies the missing arm. The second example shows the upper half of a person, and the output identifies missing legs and feet. The third example shows a person with an extra arm and an extra leg, and the output identifies these extra parts.

Figure S1. Prompt for post-processing the VLM output.

for the input human photo, we use MMPose<sup>4</sup> to extract the human pose and then employ Stable-Diffusion-v1.5<sup>5</sup> with t2iadapter.keypose<sup>6</sup> as a pose-conditioned method to regenerate the entire image.

### C. Why do current VLMs lack the ability to perceive abnormality?

Our extensive experiments demonstrate that existing VLMs are unable to perceive human abnormalities (some cases are shown in Figure S2), even though this task is very simple for humans, and both we humans and the models are trained on a large amount of normal data. We believe that the drawbacks arise from the simplistic image-text alignment approach of existing VLMs, which lacks perception of content and, consequently, an understanding of human body structure. Additionally, the existing VLMs underutilize the data and are undertrained, and the proportion of human subjects in the training data may not be substantial. In our work, we utilize the correlation among human body structures to train our absent human-body detector.

### D. AIGC Human-Aware 1K Annotation

The target of our proposed task, “Fine-grained Human Abnormality Detection”, is to detect whether the human photos in AIGC exhibit abnormalities that render them impossible to exist in the real world. This imposes two requirements on

<sup>4</sup>MMPose Contributors. (2020). OpenMMLab Pose Estimation Toolbox and Benchmark. Retrieved from <https://github.com/open-mmlab/mmpose>

<sup>5</sup><https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5>, Rombach, R. et al. (2022). High-Resolution Image Synthesis With Latent Diffusion Models. In Proc. CVPR2022.

<sup>6</sup><https://github.com/TencentARC/T2I-Adapter>, Mou, C. et al. (2023). T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv

#### Failure Cases In COCO Human-Aware Val

Question: Are there any absent body parts in the person shown in the image? If yes, please answer from 'head', 'arm', 'leg', 'foot', 'hand', or 'ear'; otherwise, please answer 'no'. Answer the question using a single word:

![Four examples of failure cases on COCO Human-Aware Val. Each example shows an image with a red box highlighting an abnormality, a prompt to replace that part with the background, and the GPT-4o response 'No'.](7832324609ad3cc688064e0341612b32_img.jpg)

Example 1: Image shows a person on a scooter. Red box highlights the hand. Prompt: Replace Hand with background. Answer: No. GPT-4o.

Example 2: Image shows a person walking. Red box highlights the foot. Prompt: Replace Foot with background. Answer: No. GPT-4o.

Example 3: Image shows a person's legs. Red box highlights the leg. Prompt: Replace Leg with background. Answer: No. GPT-4o.

Example 4: Image shows a person's head. Red box highlights the head. Prompt: Replace Head with background. Answer: No. GPT-4o.

Four examples of failure cases on COCO Human-Aware Val. Each example shows an image with a red box highlighting an abnormality, a prompt to replace that part with the background, and the GPT-4o response 'No'.

#### Failure Cases In AIGC Human-Aware 1K

Question: Are there any missing body parts in the person shown in the image? If so, please answer the precise part:

![Failure case on AIGC Human-Aware 1K. Image shows a person with an extra hand. Red box highlights the extra hand. Label for Absent: Absent hand. Answer: The person in the image appears to have all the body parts discussed: one head, two ears, two hands, two arms, two legs, and two feet. GPT-4o.](472b4b1952940f919fbd8584e83daa36_img.jpg)

Failure case on AIGC Human-Aware 1K. Image shows a person with an extra hand. Red box highlights the extra hand. Label for Absent: Absent hand. Answer: The person in the image appears to have all the body parts discussed: one head, two ears, two hands, two arms, two legs, and two feet. GPT-4o.

Question: Are there any extra body parts in the person shown in the image? If so, please answer the precise part:

![Failure case on AIGC Human-Aware 1K. Image shows a person with an extra hand. Red box highlights the extra hand. Label for Redundant: Redundant Leg. Answer: The person in the image does not appear to have any extra body parts. GPT-4o.](159f15d60777adb6b16c91d84b32c32f_img.jpg)

Failure case on AIGC Human-Aware 1K. Image shows a person with an extra hand. Red box highlights the extra hand. Label for Redundant: Redundant Leg. Answer: The person in the image does not appear to have any extra body parts. GPT-4o.

Figure S2. Failure cases of the powerful VLM (GPT-4o) on COCO Human-Aware Val and AIGC Human-Aware 1K. For COCO Human-Aware, it is observed that despite generating distinctly anomalous images, GPT-4o still responds with a definitive “No”. In the case of AIGC Human-Aware 1K, even though GPT-4o is aware of the components that constitute a normal human body, it fails to recognize or respond to abnormalities. Note that our prompt includes the category of abnormalities, which simplifies the task; however, GPT-4o still struggles to perform effectively, resulting in poor baseline performance.

our evaluation data: (1) The annotated abnormalities must be objective, avoiding controversial cases caused by ambiguity or other factors. (2) The human photos in the annotations must appear in real-world environments, which necessitates selecting realistic styles for annotation and excluding sci-fi or cartoon-style images. In Figure S3, we demonstrate examples of cases that are manually filtered out during the annotation process. After the initial annotation, to ensure data objectivity, we conduct multi-round and multi-reviewer checks on the data labels, removing any remaining controversial annotations. This process ensures the quality of our proposed AIGC Human-Aware 1K dataset. We provide statistics on the number of different annotation types in AIGC Human-Aware 1K, as shown in Table S3.

### E. Metric Details

It is essential to emphasize that a comprehensive evaluation of our proposed task requires the integration of multiple metrics. Specifically, we employ Accuracy (ACC) and False Discovery Rate (FDR) as detection metrics to ascertain the correct identification of existing abnormalities. Furthermore, we utilize perceptual metrics to evaluate the reasonableness of the identified abnormal locations and to assess the quality of the repairs to these abnormalities. Additionally, we use the Fréchet Inception Distance (FID) and Latent Consistency to examine the similarity between our repaired human photos and the original human photos,

demonstrating the granularity of our repairs; i.e., we only repair the abnormal areas while preserving the other content.

#### E.1. Human CLIP Scores

Since our task focuses on repairing the human body in a given human photo, directly using the original prompt to calculate the CLIP score is not ideal, as it includes substantial background and camera-related information. Instead, we utilize GPT4o-mini to extract prompts specifically related to the human body to evaluate whether our repair improves the correlation with human-related prompts, a metric we refer to as the Human CLIP Score. An example case is shown below:

- Original Prompt: “A girl with long hair is walking on the avenue in the forest, with a gentle breeze blowing her hair and falling leaves fluttering in the wind. The girl looks melancholy in the distance.”
- Processed Human Prompt: “A girl with long hair is walking on the avenue in the forest, looking melancholy into the distance.”

#### E.2. Human Concept Scores

Compared to the Human CLIP Score, which focuses more on the quality of the human body in the repaired human photo, the Human Concept Score emphasizes evaluating whether the repaired human conceptually aligns more closely with the distribution of “human” as understood by CLIP, trained on extensive real-world data. To verify this, we use a straightforward method: calculating the similarity between the human photo before and after repair and the prompt “an image contains human” to examine whether the repaired human better matches CLIP’s concept of a human existing in the real-world which learned from diverse real-world training data.

| Type   | Absent | Redundant | No Abnormality |
|--------|--------|-----------|----------------|
| Number | 649    | 158       | 343            |

Table S3. Statistics on the Number of Annotation Types in AIGC Human-Aware 1K

![Science fiction sample 1: A person in a yellow suit with a visor in a futuristic setting. Cartoon sample 1: A cartoon character with long hair. Too Low-quality sample 1: A blurry image of a person. Abnormality is not objective sample 1: A person in a dark, possibly NSFW setting. Horrible and other NSFW sample 1: A blurry image with a large pink blob. Science fiction sample 2: A person in a white suit in a futuristic setting. Cartoon sample 2: A cartoon character running through an archway. Too Low-quality sample 2: A blurry image of a person in a suit. Abnormality is not objective sample 2: A person in a dark, possibly NSFW setting. Horrible and other NSFW sample 2: A blurry image with a large pink blob.](5f2c99ae08864cf2d5c949947bac2b98_img.jpg)

**✗ Filtered samples**

| Science fiction | Cartoon | Too Low-quality | Abnormality is not objective | Horrible and other NSFW... |
|-----------------|---------|-----------------|------------------------------|----------------------------|
|                 |         |                 |                              |                            |
|                 |         |                 |                              |                            |

Science fiction sample 1: A person in a yellow suit with a visor in a futuristic setting. Cartoon sample 1: A cartoon character with long hair. Too Low-quality sample 1: A blurry image of a person. Abnormality is not objective sample 1: A person in a dark, possibly NSFW setting. Horrible and other NSFW sample 1: A blurry image with a large pink blob. Science fiction sample 2: A person in a white suit in a futuristic setting. Cartoon sample 2: A cartoon character running through an archway. Too Low-quality sample 2: A blurry image of a person in a suit. Abnormality is not objective sample 2: A person in a dark, possibly NSFW setting. Horrible and other NSFW sample 2: A blurry image with a large pink blob.

Figure S3. Categories and examples filtered out during the annotation process for AIGC Human-Aware 1K. The goal of our proposed task, “Fine-grained Human-body Abnormality Detection”, is to determine whether the body structure in a given human photo could exist in the real world. Thus our annotated data are grounded in real-world contexts, leading us to exclude images of genres such as science fiction and cartoons. Additionally, to enhance the dataset’s quality, we filter out samples where the specific abnormality cannot be ascertained or where the abnormality is controversial, labeled as “Too Low-quality” or “Abnormality is not objective”. All NSFW images have also been excluded, and *the displayed samples have been processed with mosaic*. These rigorous criteria not only ensure the quality of our AIGC Human-Aware 1K dataset but also explain why annotating a large number of data for the training process directly from AIGC is costly.

#### E.3. Visual Consistency

For the FID, we treat the repaired images as generated images and calculate the distributional discrepancy between them and the original images. For Latent Consistency, we encode the images into the latent space via the CLIP Visual Encoder and compute the cosine similarity between the original and repaired images.

### F. Cases in COCO Human-Aware Val

We also provide examples of the Absent Human-body Detector’s performance on the COCO Human-Aware Val, as shown in Figure S4. It shows that the trained Absent Human-body Detector accurately identifies the locations and the type of artificially created abnormalities.

### G. More Cases

In Figure S5 (a), we provide additional examples, including results for test cases with no abnormalities or in complex scenarios. Additionally, we present several failure cases, which primarily fall into two categories: the first involves incorrect abnormality identification by the HumanCalibrator, and the second involves inaccurate localization of abnormalities, leading to reduced repair quality. These are illustrated in Figure S5 (b).

#### Cases in COCO Human-Aware Val

![COCO Human-Aware Val sample 1: A person's head is missing, replaced by a brick wall pattern. A red box highlights the missing head area. COCO Human-Aware Val sample 2: A person's foot is missing, replaced by a brick wall pattern. A red box highlights the missing foot area. COCO Human-Aware Val sample 3: A person's foot is missing, replaced by a grass pattern. A red box highlights the missing foot area. COCO Human-Aware Val sample 4: A person's ear is missing, replaced by a blue pattern. A red box highlights the missing ear area.](349cffebeefaae56d9034d3fe65bf7c6_img.jpg)

|                                |                                |
|--------------------------------|--------------------------------|
|                                |                                |
| replace <head> with background | replace <foot> with background |
|                                |                                |
| replace <foot> with background | replace <ear> with background  |

COCO Human-Aware Val sample 1: A person's head is missing, replaced by a brick wall pattern. A red box highlights the missing head area. COCO Human-Aware Val sample 2: A person's foot is missing, replaced by a brick wall pattern. A red box highlights the missing foot area. COCO Human-Aware Val sample 3: A person's foot is missing, replaced by a grass pattern. A red box highlights the missing foot area. COCO Human-Aware Val sample 4: A person's ear is missing, replaced by a blue pattern. A red box highlights the missing ear area.

Figure S4. Examples of the AHD on COCO Human-Aware Val. The red boxes indicate the predictions made by AHD. It is observable that AHD, trained utilizing the correlation within human body structures, can accurately identify the location and type of artificially created abnormalities. Note that all personal information has been removed from the cases displayed. The training set created from the COCO Train Split is in a similar format.

(a) More Cases In HumanCalibrator

![Muscular man running Detect: Absent Ear Human Calibrator: Absent Ear (sad face) Repair: Absent Ear Close-up repair Woman with long hair Detect: Absent hand, Redundant Arm Human Calibrator: Absent hand, Redundant Arm (sad face) Repair: Absent hand, Redundant Arm Close-up repair Person running at sunset Detect: Absent Foot Human Calibrator: Absent Foot (sad face) Repair: Absent Foot Close-up repair Woman in a hood Detect: Redundant Ear Human Calibrator: Redundant Ear (sad face) Repair: Redundant Ear Close-up repair Woman in a purple dress Detect: Redundant Hand Human Calibrator: Redundant Hand (sad face) Repair: Redundant Hand Close-up repair Woman in a purple dress (different scene) Detect: Redundant Hand Human Calibrator: Redundant Hand (sad face) Repair: Redundant Hand Close-up repair Woman in a suit](662933fe00899933b36b5a214ac6095c_img.jpg)

| Origin Image | Detect                               | Human Calibrator | Repair |
|--------------|--------------------------------------|------------------|--------|
|              | → 'Absent Ear'                       | <br>             | →<br>  |
|              | → 'Absent hand'<br>→ 'Redundant Arm' | <br>             | →<br>  |
|              | → 'Absent Foot'                      | <br>             | →<br>  |
|              | → 'Redundant Ear'                    | <br>             | →<br>  |
|              | → 'Redundant Hand'                   | <br>             | →<br>  |
|              | → 'Redundant Hand'                   | <br>             | →<br>  |
|              | → 'No Abnormality'                   | ✓                |        |

Muscular man running Detect: Absent Ear Human Calibrator: Absent Ear (sad face) Repair: Absent Ear Close-up repair Woman with long hair Detect: Absent hand, Redundant Arm Human Calibrator: Absent hand, Redundant Arm (sad face) Repair: Absent hand, Redundant Arm Close-up repair Person running at sunset Detect: Absent Foot Human Calibrator: Absent Foot (sad face) Repair: Absent Foot Close-up repair Woman in a hood Detect: Redundant Ear Human Calibrator: Redundant Ear (sad face) Repair: Redundant Ear Close-up repair Woman in a purple dress Detect: Redundant Hand Human Calibrator: Redundant Hand (sad face) Repair: Redundant Hand Close-up repair Woman in a purple dress (different scene) Detect: Redundant Hand Human Calibrator: Redundant Hand (sad face) Repair: Redundant Hand Close-up repair Woman in a suit

(b) Failure Cases

![Failure case 1: Incorrect identification of a redundant arm as a redundant hand. The image shows a woman with a pink object on her head. The label is 'Redundant Arm' and the prediction is 'Redundant hand'. A red box highlights the arm, and a red X is on the prediction. A callout box says: 'Incorrectly identifying a redundant arm as a redundant hand.' Failure case 2: Inaccurate localization of abnormality. The image shows a woman with a pink object on her head. The label is 'Redundant Hand' and the prediction is 'Redundant hand'. A red box highlights the hand, and a red X is on the prediction. A callout box says: 'Inaccurate localization results to poor repair results.'](555df5c0300cb1fca5dc028fec5ec6be_img.jpg)

Incorrect abnormality identification

→ Label: 'Redundant Arm'  
Prediction: 'Redundant hand'

Inaccurate localization of abnormality

→ Label: 'Redundant Hand'  
Prediction: 'Redundant hand'

Failure case 1: Incorrect identification of a redundant arm as a redundant hand. The image shows a woman with a pink object on her head. The label is 'Redundant Arm' and the prediction is 'Redundant hand'. A red box highlights the arm, and a red X is on the prediction. A callout box says: 'Incorrectly identifying a redundant arm as a redundant hand.' Failure case 2: Inaccurate localization of abnormality. The image shows a woman with a pink object on her head. The label is 'Redundant Hand' and the prediction is 'Redundant hand'. A red box highlights the hand, and a red X is on the prediction. A callout box says: 'Inaccurate localization results to poor repair results.'

Figure S5. More Cases in HumanCalibrator and some failure cases.