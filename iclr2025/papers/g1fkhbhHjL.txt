

{0}------------------------------------------------

# BLACK SHEEP IN THE HERD: PLAYING WITH SPURIOUSLY CORRELATED ATTRIBUTES FOR VISION-LANGUAGE RECOGNITION

Xinyu Tian<sup>1</sup>, Shu Zou<sup>1</sup>, Zhaoyuan Yang<sup>2</sup>, Mengqi He<sup>1</sup>, Jing Zhang<sup>1</sup>

<sup>1</sup>Australian National University, <sup>2</sup>GE Research

## ABSTRACT

Few-shot adaptation for Vision-Language Models (VLMs) presents a dilemma: balancing in-distribution accuracy with out-of-distribution generalization. Recent research has utilized low-level concepts such as visual attributes to enhance generalization. However, this study reveals that VLMs overly rely on a small subset of attributes on decision-making, which co-occur with the category but are not inherently part of it, termed spuriously correlated attributes. This biased nature of VLMs results in poor generalization. To address this, 1) we first propose SPURIOUS ATTRIBUTE PROBING (SAP), identifying and filtering out these problematic attributes to significantly enhance the generalization of existing attribute-based methods; 2) We introduce SPURIOUS ATTRIBUTE SHIELDING (SAS), a plug-and-play module that mitigates the influence of these attributes, seamlessly integrating into various Parameter-Efficient Fine-Tuning (PEFT) methods. In experiments, SAP and SAS significantly enhance accuracy on distribution shifts across 11 datasets and 3 generalization tasks while preserving downstream performance, establishing a new state-of-the-art benchmark. The code will be available [here](#).

## 1 INTRODUCTION

The emergence of large-scale pre-trained Vision-Language Models (VLMs) (Radford et al., 2021; Li et al., 2022a) bridges the gap between images and texts. However, conventional fine-tuning of these models entails significant computational burdens, leading to Parameter-Efficient Fine-Tuning (PEFT), such as prompt tuning (Khattak et al., 2023a; Zhou et al., 2022a), adapters (Sung et al., 2022; Gao et al., 2024) and LoRA (Hu et al., 2021; Dettmers et al., 2024). With PEFT, requiring approximately 1% of model parameters, one may adeptly adapt to downstream tasks, achieving comparable or even superior performance to full fine-tuning (Liu et al., 2021). Yet, recent studies have revealed that in few-shot scenarios where observed samples are limited, PEFT struggles to generalize to out-of-distribution datasets and may compromise the VLMs’ strong zero-shot capability (Zhou et al., 2022b; Yao et al., 2023; Bulat & Tzimiropoulos, 2023). This creates a trade-off where individuals aim for strong performance on downstream tasks while endeavoring to maintain the ability of VLMs to handle distribution shifts.

In response to the above-mentioned challenges, various strategies have been proposed such as category conditioning (Zhou et al., 2022b; Yao et al., 2024), prompt regularization (Yao et al., 2023; Bulat & Tzimiropoulos, 2023; Khattak et al., 2023b) and training-free adaptation (Udandarao et al., 2022; Zhang et al., 2022b). Recently, it has been discovered that incorporating descriptors, also known as visual attributes during training can significantly improve the accuracy of adapted modules on out-of-distribution datasets (Zhang et al., 2024c; Liao et al., 2024; Tian et al., 2024; Ma et al., 2023; Liu et al., 2024b). The motivation behind these works is that attributes, as lower-level concepts, are more likely to establish connections to unseen categories compared to the high-level names (Zhang et al., 2024c; Tian et al., 2024). These methods can be divided into two types: one involves generating visual attributes for the target category using Large Language Models (LLMs) (Tian et al., 2024; Ma et al., 2023; Liu et al., 2024b), while the other entails searching for optimal attributes from a pre-defined vocabulary that maximizes semantic similarity (Zhang et al., 2024c) or training accuracy (Liao et al., 2024). Yet, a commonality among them is their reliance on the set of generated attributes, *i.e.*, the attribute pool.

{1}------------------------------------------------

![Figure 1: The phenomenon of Black Sheep in the Herd. The figure shows three panels: (a) prediction, (b) CLIP, and (c) CLIP + SAS. Each panel displays an image of a fireboat or park bench and a horizontal bar chart of attribute weights. Yellow bars indicate spurious attributes, and purple bars indicate core attributes. In (b), spurious attributes like 'grass', 'path', and 'backrest' are prominent for the bench. In (c), these are suppressed.](49ad3a646d84bcfeac02bdf2b3792a3e_img.jpg)

**(a) prediction**

**(b) CLIP**

**(c) CLIP + SAS**

Figure 1: The phenomenon of Black Sheep in the Herd. The figure shows three panels: (a) prediction, (b) CLIP, and (c) CLIP + SAS. Each panel displays an image of a fireboat or park bench and a horizontal bar chart of attribute weights. Yellow bars indicate spurious attributes, and purple bars indicate core attributes. In (b), spurious attributes like 'grass', 'path', and 'backrest' are prominent for the bench. In (c), these are suppressed.

**Figure 1: The phenomenon of Black Sheep in the Herd.** We rank attribute weights on VLM predictions using CBMs, with yellow and purple bars to denote spurious and core attributes respectively. In (b), we observe that for vanilla VLMs, 2 out of the top-3 are spurious attributes, heavily influencing decisions. In (c), SAS mitigates this by suppressing the influence of spurious attributes.

While promising, the limitations of this line of work have been underexplored. Initial suspicions emerge from Roth et al. (2023), which find that in certain cases, replacing attributes with random sequences does not lead to a notable performance decline. Subsequently, An et al. (2023) discover that VLMs sometimes disregard the presence of attributes, leading to minimal gains. This prompts us to inquire: *Are attributes truly dependable? If so, then whence do these failure cases arise?*

To tackle the aforementioned inquiries, we conduct a manual examination of the attribute pool generated by existing methods. We stumble upon an often overlooked fact: while most attributes accurately depict the intrinsic characteristics of the target category, there exists a small subset of attributes that co-occur with the category but are not part of it, leading to strong spurious correlations. For instance, when querying LLM with what does a mountain bike look like we receive attributes like wheels, handle, and basket, yet unexpected attributes such as trees and road also emerge. This phenomenon is also observed in vocabulary-based methods, where attributes are chosen based on in-distribution samples. Inspired by Singla & Feizi (2021), we refer to the former as *core attributes* and the latter as *spuriously correlated attributes*<sup>1</sup>.

Building upon the motivation to enhance generalization, a natural idea is to strive for a “pure” attribute pool that accurately reflects the true characteristics of the category. Therefore, we conduct a simple experimental study based on existing methods (Zhang et al., 2024c; Tian et al., 2024) where we manually identify attributes that might lead to spurious correlations with the target category and remove them. Despite the small proportion of these attributes ( $< 7\%$ ), we observe a significant improvement in out-of-distribution generalization accuracy. To gain deeper insights into how spurious attributes affect VLMs, we employ concept bottleneck models (CBMs) (Koh et al., 2020; Yang et al., 2023), a well-established method for interpreting and ranking attribute weights in decision-making processes. Our analysis reveals that despite their small representation in the overall pool, spurious attributes exert a significant influence, particularly among the top-3 attributes on decision-making as depicted in Fig. 1. We term this phenomenon *Black Sheep in the Herd* since 1) spurious attributes act as the “black sheep” within the pool, constituting a small fraction; 2) nevertheless, this small fraction significantly impacts the generalization ability of VLMs.

We could utilize the aforementioned manual inspection to assist existing attribute-based methods. However, this remains a mere fancy dream since manually identifying spurious attributes in the pool is prohibitively expensive. This motivates us to devise a new method for generating a pure pool, one that contains only those core attributes belonging to the category. Hence, we propose **SPURIOUS ATTRIBUTE PROBING (SAP)**, an approach to derive an attribute pool where core and spurious attributes are clearly separated. SAP integrates Multi-modal Large Language Models (MLLMs) and Concept Bottleneck Models (CBMs) to tackle this challenge. Leveraging MLLMs, SAP initially distinguishes core attributes from non-core counterparts, and then CBMs prioritize the latter by selecting those with a significant impact on model decisions as spurious attributes. SAP complements existing attribute-based methods and, to the best of our knowledge, is the first approach to identifying spurious attributes in open-world settings without explicit human supervision.

Despite the effectiveness of SAP, it faces limitations: It may prevent the presence of spurious attributes in the language branch, yet it cannot stop the model from learning spurious features. This ex-

<sup>1</sup>We refer spuriously correlated attributes to spurious attributes for brevity.

{2}------------------------------------------------

tends the scope beyond attribute-based methods, and will result in poor generalization across PEFT family. Therefore, we propose SPURIOUS ATTRIBUTE SHIELDING (SAS), a module to mitigate the influence of spurious features which can be seamlessly integrated into arbitrary PEFT methods. Specifically, SAS introduces a subsidiary task by creating a set of pseudo categories defined by spurious attributes alongside the real ones, allowing VLMs to distinguish between them. For instance, if streetlight is considered a spurious attribute for the target category vehicle, we establish a separate pseudo category exclusively for streetlight and discern between the two, thus decreasing the dependency on streetlight for identifying vehicle. The experiments show that by combining SAS into existing PEFT approaches, the accuracy under distribution shifts is significantly improved, reaching a new state of the art.

In summary, our main contributions are as follows:

- Despite the promise of visual attributes in various applications, we discover a group of *black sheep*, i.e., spurious attributes, on which VLMs inherently heavily rely, thereby leading to poor generalization and robustness.
- We introduce SPURIOUS ATTRIBUTE PROBING (SAP), aiming to identify and eliminate these problematic attributes, thereby substantially improving the generalization of current attribute-based methods.
- We present SPURIOUS ATTRIBUTE SHIELDING (SAS), a plug-and-play module seamlessly integrating into various PEFT methods to mitigate the influence of spurious attributes on predictions.

## 2 RELATED WORK

**Vision-Language Models.** Recently, it has been discovered that associating text and images for pre-training, instead of using images alone, enables powerful zero-shot capability, leading to VLMs. Initially, simple dual-tower structures are employed, where the representations of the two modalities are modeled by separate encoders and connected via contrastive learning, i.e., CLIP (Radford et al., 2021). Subsequently, more works have been built upon this foundation. For instance, Li et al. (2022a) bridge two encoders by fusion for better cross-modality interactions, Li et al. (2023b) employ masked image modeling to achieve a trade-off between accuracy and training time, and Li et al. (2022b) incorporate visual detection and grounding in pre-training for object-level reasoning. For more information, we refer to Zhang et al. (2024a) for a detailed survey of recent VLMs.

**Parameter-Efficient Fine-Tuning.** As pre-trained models grow larger, traditional fine-tuning demands significant resources, leading to PEFT (Hu et al., 2021; Liu et al., 2021; Lester et al., 2021; Houlsby et al., 2019). However, PEFT is a double-edged sword: while it adeptly adjusts to downstream tasks, it also brings poor generalization to the open world, inspiring various current remedies. For instance, category conditioning (Zhou et al., 2022b; Yao et al., 2024) infuses category-aware knowledge for discriminative and generalizable learning, prompt regularization (Yao et al., 2023; Bulat & Tzimiropoulos, 2023) confines learnable prompts to corresponding textual features, and training-free adaptation (Udandarao et al., 2022; Zhang et al., 2022b) eschews gradient-based optimization to prevent overfitting. Recently, another line of work leveraging visual attributes has shown promising results, achieving state-of-the-art performance in various generalization tasks.

**Visual Attributes for Recognition.** The initial exploration of attributes for recognition begins in zero-shot settings (Menon & Vondrick, 2023; Pratt et al., 2023), where individuals utilize attributes generated by LLMs to offer more expressive and accurate descriptions. Subsequently, it is observed that training VLMs to grasp fundamental concepts like visual attributes aids in generalizing to unseen data, prompting a surge in attribute-based methods. For instance, Tian et al. (2024) appends attributes to category names, Liao et al. (2024) initializes learnable tokens as attribute embeddings, and Ma et al. (2023) adopts a more aggressive approach by replacing category names entirely with attributes. Additionally, Wei et al. (2019) utilize adversarial training to learn attribute-object composition, while Huang et al. (2024) and Wang et al. (2015) improve the model’s fine-grained understanding by building multi-granularity and hierarchical attributes. Nonetheless, recent studies have noted a decline in attribute effectiveness in certain scenarios (An et al., 2023), sometimes reducing to a mere ensembling effect (Roth et al., 2023). This paper delves into the issue, attributing it to spurious attributes, and proposes two plug-and-play approaches to complement existing methods.

{3}------------------------------------------------

**Spurious Attribute Identification.** Spurious attributes arise from model debiasing (Seth et al., 2023; Chuang et al., 2023; Berg et al., 2022), defined as *those likely to co-occur with the object but not part of it* (Singla & Feizi, 2021). Although a well-known term, it remains underexplored due to the difficulty in identification. The initial endeavor by Singla & Feizi (2021) involves manually labeling to identify spurious attributes. Similarly, Wong et al. (2021) integrates human supervision with sparse linear layers to mitigate labor expenses. Others identify spurious attributes by analyzing their properties. For instance, Wu et al. (2023) observe that spurious attributes exhibit instability across data environments and introduce concept sensitivity for identification. Conversely, Teotia et al. (2022) train an attribute probing network to predict spurious attributes. Recently, the work most akin to ours, Adila et al. (2023), utilize LLMs to derive harmful insight representations by comparing differences between concepts. However, the inference complexity of this method escalates exponentially with the number of concepts, restricting its application to small-scale datasets. In contrast, our proposed SAP 1) necessitates neither human labeling nor a training process, rendering it extremely cost-effective; and 2) is scalable to any large-scale dataset, e.g., ImageNet.

**Spurious Correlation Mitigation.** Current spurious mitigation methods can be mainly categorized into two types. The first assumes that spurious attributes within the dataset are either unknown or complex, employing various proxies to mitigate spurious correlations. For instance, Xu et al. (2020); Yao et al. (2022); Han et al. (2022) introduce augmentation via domain mix-up to learn invariant features, while Li et al. (2022c); Zhang et al. (2022a); Utama et al. (2020) advocate for instance reweighting to emphasize hard samples. Others calibrate biased representation through contrastive learning (You et al., 2024; Zhang & Ré, 2022). The second type explicitly assumes that spurious correlations arise from known attributes (Chuang et al., 2023; Berg et al., 2022). For instance, Wu et al. (2023) balance training data by swapping spurious concepts among categories, whereas Adila et al. (2023) calibrate embeddings by removing spurious representations. In contrast, SAS belongs to the latter, where the attribute prior is known, and thanks to the effectiveness of SAP, it may accurately mitigate spurious correlations caused by identified spurious attributes. For further details, a quantitative comparison between SAS and related works is provided in Supp. Mat. B.

## 3 METHOD

### 3.1 PROBLEM SETUP

We assume the training set of pairs  $\mathcal{D} = \{(x, c)\}$ , where  $x \in \mathcal{X}$  and  $c \in \mathcal{C}$  represent the image and ground truth label, respectively. The attribute-based methods aim to construct a category-wise prompt  $t_c = f(c, \mathcal{P})$  such that the conditional distribution of the prediction  $y$  given  $x$  is modeled as

$$P(y|x) = \frac{\exp(s(\phi_I(x), \phi_L(t_y))/\tau)}{\sum_{c \in \mathcal{C}} \exp(s(\phi_I(x), \phi_L(t_c))/\tau)}, \quad (1)$$

where  $\phi_I$  and  $\phi_L$  represent the vision and language encoder, respectively,  $s(\cdot, \cdot)$  indicates the similarity function, and  $\tau$  is the temperature scaler.  $\mathcal{P} = \{\mathcal{A}_c \mid c \in \mathcal{C}\}$  is an attribute pool generated by  $\mathcal{A}_c = \mathcal{U}(\mathcal{H}(c))$  such that  $\mathcal{A}_c = \{a_c^1, a_c^2, \dots, a_c^d\}$ . Depending on previous work,  $\mathcal{U}$  could be a LLM, thus  $\mathcal{H}(c)$  is a set of LLM prompts incorporating the category name of  $c$ .  $\mathcal{U}$  could also be a large vocabulary, such that  $\mathcal{H}(c)$  becomes a key to search for the semantically related attributes. Therefore,  $f(\cdot, \cdot)$  is denoted as a concatenation function to integrate the category name and corresponding attributes together. The optimization objective is typically a cross-entropy loss  $\mathcal{L}_{ce}$ .

It's important to mention that in this work, we refrain from specifying particular learnable parameters; they could encompass learnable prompts, adapters, or LoRA. Our goal is to ensure the versatility of our plug-and-play method across various PEFT approaches.

### 3.2 MOTIVATION

We present the motivation of this work by revealing an overlooked fact: the attribute pool in current methods are not purely aligned with the intrinsic semantics of categories. Some attributes are spurious, co-occurring with categories but not inherently linked to them. To investigate the impact of these "black sheep", we conduct a simple experimental study. Specifically, we manually traverse the attribute pool  $\mathcal{P}$  across various methods and identify spurious attributes within. We use a conventional method following Singla & Feizi (2021) with a simplistic version. Given the category  $c$ ,

{4}------------------------------------------------

| Method | FGVCAircraft |  |  | SUN397 |  |  | Flowers102 |  |  | DTD |  |  | Average |  |  |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|  | Base | New | SR | Base | New | SR | Base | New | SR | Base | New | SR | Base | New | SR |
| CPL | 42.27 | 38.85 | 5.43 | 81.88 | 79.65 | 6.61 | 98.07 | 80.43 | 5.71 | 80.92 | 62.27 | 5.13 | 75.79 | 65.30 | 5.72 |
| CPL - SA | <b>42.62</b> | 41.33 | — | 82.14 | <b>82.36</b> | — | 98.35 | <b>82.16</b> | — | <b>81.62</b> | 64.77 | — | <b>76.18</b> | 67.66 | — |
| ArGue | 41.29 | 38.80 | 5.13 | 81.89 | 80.48 | 6.45 | 98.62 | 77.96 | 6.69 | 80.33 | 67.03 | 5.97 | 75.53 | 66.07 | 6.06 |
| ArGue* | 41.34 | 39.34 | 4.86 | 81.97 | 80.62 | 6.11 | 98.58 | 78.11 | 6.44 | 80.41 | 67.26 | 5.62 | 75.58 | 66.33 | 5.76 |
| ArGue - SA | 41.55 | <b>41.60</b> | — | <b>82.33</b> | 81.94 | — | <b>98.73</b> | 78.75 | — | 80.79 | <b>68.47</b> | — | 75.85 | <b>67.69</b> | — |

Table 1: **The results on base-to-new generalization before and after removing spurious attributes (SA) from the pool.** We report accuracy on base and new categories, and spurious rate (SR), which refers to the proportion of spurious attributes to the entire pool.

we randomly sample 5 images from the shots and visualize the heatmap. For specific attribute  $a_c^k$ , we determine whether it is a part of the main object, or separate objects in the background based on the sampled images with the heatmap activations.

Upon identifying these spurious attributes, we remove them from the pool and compare the changes in their generalization capability before and after elimination. The experiment is evaluated on base-to-new generalization, following the outlined settings in Section 4. As baselines, we select CPL (Zhang et al., 2024c) and ArGue (Tian et al., 2024), representing vocabulary-based and LLM-assisted methods, respectively. Additionally, we consider a variant, ArGue\*, where we modify the LLM prompts to reduce the likelihood of spurious attribute occurrence. For instance, we append an additional instruction focus on mountain bike itself to the original prompt. Further details are provided in Supp. Mat. A.

**Removing spurious attributes significantly enhances generalization.** While most attributes contribute positively to generalization, spurious attributes are exceptions to this trend. Removing these exceptions leads to a notable increase in accuracy on the new category on average (65.30%  $\rightarrow$  67.66% for CPL and 66.07%  $\rightarrow$  67.69% for ArGue), without compromising accuracy on the base category. This phenomenon is aptly described as Black Sheep in the Herd since 1) spurious attributes constitute only a small portion of the pool ( $< 7\%$ ); 2) yet this small portion significantly impacts the generalization ability of VLMs.

**VLMs heavily rely on spurious attributes for predictions.** To deepen our understanding of this phenomenon, we use concept bottleneck models (CBMs) (Koh et al., 2020) to determine attribute weights in model decision-making. In Fig. 1, attributes are ranked by weight from high to low. Among the top-3 attributes influencing predictions, spurious attributes occupy two positions. For instance, in predicting fireboat, VLMs heavily rely on sea and lake as crucial concepts, while for park bench, grass and path are primary indicators. This implies that 1) VLMs may exhibit insensitivity to the presence of core attributes; 2) directly adapting to downstream tasks may heavily rely on spurious features for predictions. In fact, this also aligns with findings from concurrent work (Wang et al., 2024), which indicates that VLMs are more susceptible to spurious features compared to unimodal architectures.

In addition to the above observations, this prompts us to consider several questions.

**Where do spurious attributes come from?** In the case of LLMs, this phenomenon may be attributed to statistical bias in their large-scale training data. In practical scenarios, when describing a complex object, there may be a tendency to focus more on its accompanying scenes and associated elements rather than its core components. However, these accompanying elements may not contribute to a VLM’s generalizable understanding of a specific category. Conversely, regarding vocabulary-based methods, their attribute selection heavily relies on in-distribution samples, and this preference for attributes may be also detrimental to generalization.

**Is there a better way to identify spurious attributes?** While manually purifying the attribute pool may enhance generalization, it faces two primary challenges: 1) scalability issues as the dataset size grows, and 2) it is a simple solution lacking quantitative assessment of the spurious correlation of each attribute, potentially leading to false positives, *i.e.*, attributes that co-occur with the category merely by chance. Hence, we also experiment with an LLM-assisted variant called ArGue\*, which adjusts the LLM prompts to encourage a stronger focus on the category itself. However, as demonstrated empirically in Table 1, the reduction in the spurious rate is modest (6.06%  $\rightarrow$  5.76%), resulting in only marginal gains (66.07%  $\rightarrow$  66.33%).

{5}------------------------------------------------

### 3.3 SPURIOUS ATTRIBUTE PROBING

Motivated by the above considerations, we introduce SPURIOUS ATTRIBUTE PROBING (SAP), an approach to creating a comprehensive attribute pool where spurious and core attributes are distinctly separated. Initially, SAP utilizes Multi-modal Large Language Models (MLLMs) to differentiate attributes belonging to the target category, distinguishing core attributes from non-core counterparts. To determine if the coexistence of the latter with the category is coincidental or correlated, Concept Bottleneck Models (CBMs) gauge their impact on VLMs' decision-making, with high-influence ones being identified as spurious attributes. By leveraging SAP, a pure and robust attribute pool is achieved, significantly improving the generalization of existing attribute-based methods.

**Prompting MLLMs.** Here we assume  $\mathcal{U}$  as a MLLM, differing from prior methods that only accept textual prompts.  $\mathcal{U}$  concurrently processes both prompts and images as input such that  $\mathcal{A}_c = \mathcal{U}(\mathcal{X}_c, \mathcal{H}(c))$ , where  $\mathcal{X}_c$  represents training images labeled with  $c$ , and  $\mathcal{H}(c)$  is a set of chain-of-thought prompts probing two aspects: core attributes and non-core counterparts. Specifically, we design three question formats:

- Q1:** List all the visual cues you see in the photo:
- Q2:** Are the objects you list a part of \_\_\_?
- Q3:** Describe \_\_\_ in the photo in details:

Combining Q1 and Q2 helps identify non-core attributes in the images, while Q3 provides detailed core attributes belonging to the category. Empirically, we'll use multiple templates for each question to ensure thoroughness. Upon reformulation, we derive  $\mathcal{A}_c = \tilde{\mathcal{A}}_c^- \cup \mathcal{A}_c^+$ , where  $\tilde{\mathcal{A}}_c^-$  denotes non-core attributes, and  $\mathcal{A}_c^+$  represents the core ones.

**Finding spurious attributes.** Given non-core attributes  $\tilde{\mathcal{A}}_c^-$ , we use their weights on model predictions as a proxy to indicate the extent of spurious correlations. We use a CBM (Koh et al., 2020) to achieve this goal. Specifically, we construct a bottleneck embedding  $\mathcal{E} \in \mathbb{R}^{N \times d}$  against the attribute pool  $\mathcal{P}$ , where  $N = |\mathcal{C}| \times J$  indicates the total number of attributes in the current pool, each row  $\mathcal{E}_i \in \mathbb{R}^d$  indicates the feature of the corresponding attribute, and  $d$  is the feature dimension. In other words,  $\mathcal{E}$  is a feature matrix that concatenates attributes across all the categories together. The procedure of CBMs is to combine two functions to make the prediction:  $\hat{c} = h(g(\phi_I(x), \mathcal{E}))$ , where  $g: \mathbb{R}^d \times \mathbb{R}^{N \times d} \rightarrow \mathbb{R}^N$  measures the score between the image feature and every attribute in the bottleneck,  $h: \mathbb{R}^N \rightarrow \mathcal{C}$  produces the final prediction based on the score vector. Following Yang et al. (2023), we set the score vector  $g$  as the dot product of the features between two modalities  $g(\phi_I(x), \mathcal{E}) = \phi_I(x) \cdot \mathcal{E}^\top$ , and  $h$  as the linear projection with a learnable weight matrix  $\mathcal{W} \in \mathbb{R}^{|\mathcal{C}| \times N}$  such that  $h(g; \mathcal{W}) = \text{softmax}(g \cdot \mathcal{W}^\top)$ . Intuitively,  $\mathcal{W}_{ij}$  indicates the impact factor of  $j^{\text{th}}$  attribute, i.e.,  $a_i^j$  on the prediction  $i$ . To learn the weights of attributes on predictions, a cross-entropy loss is typically employed.

For non-core attributes, a high weight indicates a strong correlation with the category, whereas a low weight suggests that its presence might be coincidental. Thus, a natural idea to confirm spurious attributes is by thresholding  $\gamma$ . Formally, for a specific prediction  $i$ , the spurious attributes are defined as

$$\mathcal{A}_i^- = \{a_i^j \in \tilde{\mathcal{A}}_i^- \mid \mathcal{W}_{ij} \geq \gamma\}. \quad (2)$$

There is a trade-off in choosing  $\gamma$ . If  $\gamma$  is too large, some spurious attributes may be overlooked. Conversely, if it is too small, a large number of false positives may be introduced. Additionally, we observe significant variability in attribute weight distributions among different categories, posing challenges in identifying spurious attributes with a uniform threshold. Creating a manual threshold for each category is prohibitively expensive. Hence, we introduce an adaptive strategy. Given a prediction  $c$ , we select  $\gamma_c$  as the lowest weight of  $\mathcal{A}_c^+$  such that non-core attributes with weights higher than any of the core attributes are considered spurious. This ensures flexible selection of spurious attributes, greatly aiding SAS introduced in Section 3.4 to be discussed next.

### 3.4 SPURIOUS ATTRIBUTE SHIELDING

SAP complements existing attribute-based methods by screening out spurious attributes from the pool, while it does not prevent the PEFT family from learning spurious features in the training im-

{6}------------------------------------------------

![Figure 2: The overview of SAS. (a) Spurious attribute identification: A mountain bike image is processed by an MLLM to identify attributes like 'wheel', 'tree', 'helmet', 'basket', 'handle', 'road'. (b) Pseudo category construction: A photo of a 'spurious attr' (e.g., 'trees', 'road', 'mountain') is processed by LAION or SD to generate 'constructed images'. (c) Subsidiary task: A training pipeline showing 'Adapt' modules, 'CLIP's Visual Encoder', and 'CLIP's Textual Encoder' for 'primary objective' (target categories: truck, bus, scooter) and 'secondary objective' (pseudo categories: trees, road, mountain).](1956f44611abd5c3c41049836aa78ad8_img.jpg)

Figure 2: The overview of SAS. (a) Spurious attribute identification: A mountain bike image is processed by an MLLM to identify attributes like 'wheel', 'tree', 'helmet', 'basket', 'handle', 'road'. (b) Pseudo category construction: A photo of a 'spurious attr' (e.g., 'trees', 'road', 'mountain') is processed by LAION or SD to generate 'constructed images'. (c) Subsidiary task: A training pipeline showing 'Adapt' modules, 'CLIP's Visual Encoder', and 'CLIP's Textual Encoder' for 'primary objective' (target categories: truck, bus, scooter) and 'secondary objective' (pseudo categories: trees, road, mountain).

Figure 2: The overview of **SAS**. In (a), we generate and identify spurious attributes with **SAP**. In (b), we construct pseudo categories by synthetic data (**SD**) or retrieval (**LAION**). In (c), apart from the main objective (i), e.g., cross-entropy loss, we introduce an auxiliary subsidiary task (ii) for learning robust features.

ages. Hence, we propose **SPURIOUS ATTRIBUTE SHIELDING (SAS)**, a plug-and-play module to be seamlessly integrated into arbitrary **PEFT** methods by mitigating the influence of spurious features. Building upon the spurious attributes detected by **SAP**, **SAS** introduces a subsidiary task by constructing a set of pseudo categories alongside the real one and let VLMs differentiate among them. This auxiliary learning objective effectively prompts VLMs to learn robust features rather than ones referred by these spurious attributes. For instance, if streetlight is a spurious attribute for the category vehicle impacting decision-making significantly, we introduce a pseudo category specifically for streetlight and differentiate between the two, thereby reducing the reliance of streetlight when identifying vehicle. Fig. 2 demonstrates the overall pipeline of **SAS**.

Formally, given a category  $c$ , we establish a set of pseudo categories  $\mathcal{J}_c$  with constructed images  $\{\tilde{\mathcal{X}}_j \mid j \in \mathcal{J}_c\}$ . Thus we define a subsidiary dataset  $\mathcal{D}_c = \{(x, y) \mid x \in \tilde{\mathcal{X}} \cup \mathcal{X}_c \text{ and } y \in \mathcal{J}_c \cup \{c\}\}$ . We aim to optimize the following

$$\mathcal{L}_{pse} = - \sum_{c \in \mathcal{C}} \mathbb{E}_{(x, y) \in \mathcal{D}_c} \log \frac{\exp(s(\phi_I(x), \phi_L(t_y)) / \tau)}{\sum_{j \in \mathcal{J}_c \cup \{c\}} \exp(s(\phi_I(x), \phi_L(t_j)) / \tau)}. \quad (3)$$

That is, we introduce an additional cross-entropy loss for classifying between each target category and its corresponding pseudo categories, which are defined by spurious attributes. This can be viewed as a subsidiary task, aimed at reducing reliance on spurious attributes while achieving correct classification in the downstream task. When integrated with existing methods, we introduce a scalar  $\lambda$  to balance the importance of  $\mathcal{L}_{pse}$ :  $\mathcal{L}_{tot} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{pse}$ .

A natural question to ask is: how to construct  $\tilde{\mathcal{X}}$  such that the adapted modules may effectively distinguish spurious attributes from the target categories? In this work, we introduce two approaches.

**Synthetic Generation.** We create pseudo categories using synthetic data by leveraging the text-to-image model Stable Diffusion (SD) (Rombach et al., 2022). We consider two key factors: 1) diversity: Our goal is for pseudo categories to fully represent the features of spurious attributes. To achieve this, we use LLMs to generate various prompts, which are then used as inputs to SD to produce a range of images. 2) purity: If the constructed images contain not only spurious attributes but also unexpected elements, i.e., noise attributes, these noise attributes may create new shortcuts, affecting the effectiveness of **SAS**. Empirically, selecting the top-k images that are most similar to the corresponding spurious attribute can help reduce the presence of noise attributes. Further details are in Supp. Mat. A.

**Pretraining Retrieval.** An alternative way is to gather image samples from pre-training data such as LAION-5B (Schuhmann et al., 2022), a publicly available subset of CLIP’s pre-training datasets. We use captions as a proxy to efficiently determine semantic similarity between pre-training images and spurious attributes. Finally, we select the top-k matches to the spurious attributes to create the pseudo categories.

{7}------------------------------------------------

![Figure 3: Three line plots showing accuracy vs. ID accuracy for different generalization tasks. (a) base-to-new generalization: ID accuracy from 70% to 74%, New accuracy from 72% to 80%. (b) cross-dataset transfer: ID accuracy from 71% to 74%, OOD accuracy from 55% to 70%. (c) domain generalization: ID accuracy from 71% to 74%, OOD accuracy from 59% to 62.5%. A horizontal dashed line at ~70% represents zero-shot capability. Methods include CoCoOp, KgCoOp, PromptSRC, MaPLe, LASP, TCP, CPL, ArGue, MAP, CLIP-Adapter, Tip-Adapter, Zero-shot CLIP, Baseline + SAP, and Baseline + SAS.](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

Figure 3: Three line plots showing accuracy vs. ID accuracy for different generalization tasks. (a) base-to-new generalization: ID accuracy from 70% to 74%, New accuracy from 72% to 80%. (b) cross-dataset transfer: ID accuracy from 71% to 74%, OOD accuracy from 55% to 70%. (c) domain generalization: ID accuracy from 71% to 74%, OOD accuracy from 59% to 62.5%. A horizontal dashed line at ~70% represents zero-shot capability. Methods include CoCoOp, KgCoOp, PromptSRC, MaPLe, LASP, TCP, CPL, ArGue, MAP, CLIP-Adapter, Tip-Adapter, Zero-shot CLIP, Baseline + SAP, and Baseline + SAS.

Figure 3: **The average results of three generalization tasks over 11 datasets.** The x-axis and y-axis represent in-distribution/base accuracy and out-of-distribution/new accuracy, respectively. We present the out-of-distribution accuracy of vanilla CLIP as a horizontal bar to represent the zero-shot capability. The detailed numerical results are provided in Supp. Mat. E.

![Figure 4: Example samples from test set and counter group. (a) training set: penguins on ice. (b) test set: a penguin on a beach. (c) counter group: an airplane on a runway.](efca2dce0095c9dc2a68e9af6b2bfd40_img.jpg)

Figure 4: Example samples from test set and counter group. (a) training set: penguins on ice. (b) test set: a penguin on a beach. (c) counter group: an airplane on a runway.

Figure 4: **Example samples from test set and counter group.** The samples from counter group do not contain spurious attributes, *e.g.*, ice or sky.

| Method | ImageNet |  | OxfordPets |  | FGVCAircraft |  |
|-|-|-|-|-|-|-|
|  | Test | Counter | Test | Counter | Test | Counter |
| PSRC | 72.46 | 63.06 | 87.51 | 74.08 | <b>39.72</b> | 21.35 |
| PSRC + SAS | <b>73.82</b> | <b>68.55</b> | <b>88.27</b> | <b>77.17</b> | <b>39.54</b> | <b>26.43</b> |
| TCP | 71.54 | 60.17 | 89.11 | 72.84 | 38.30 | 23.61 |
| TCP + SAS | <b>72.81</b> | <b>64.50</b> | <b>89.86</b> | <b>76.14</b> | <b>39.23</b> | <b>28.44</b> |
| CPL | 69.08 | 64.75 | <b>90.32</b> | 79.91 | 40.43 | 27.65 |
| CPL + SAS | <b>70.15</b> | <b>67.98</b> | <b>90.12</b> | <b>83.45</b> | <b>40.64</b> | <b>32.12</b> |

Table 2: **The results for standard few-shot classification on test set and counter group, respectively.** Essentially, counter group is a subset of test set where spurious attributes are removed.

## 4 EXPERIMENT

**Task Setting.** Following previous work (Khattak et al., 2023a; Zhou et al., 2022b; Khattak et al., 2023b), the experiment is conducted on base-to-new generalization, cross-dataset transfer and domain generalization. For base-to-new generalization, the datasets are equally divided into base and new categories, where the model is trained on base categories and evaluated on unseen ones. For cross-dataset transfer, the model will be trained on a large-scale dataset, and generalized across various other datasets. For domain generalization, the model will be transferred from an in-distribution dataset to several variants.

**Datasets.** For base-to-new generalization, we employ 11 datasets, including ImageNet (Deng et al., 2009), Caltech101 (Fei-Fei et al., 2004), OxfordPets (Parkhi et al., 2012), StanfordCars (Krause et al., 2013), Flowers102 (Nilsback & Zisserman, 2008), Food101 (Bossard et al., 2014), FGVCAircraft (Maji et al., 2013), SUN397 (Xiao et al., 2010), UCF101 (Soomro et al., 2012), DTD (Cimpoi et al., 2014) and EuroSAT (Helber et al., 2019). For cross-dataset transfer, we train models on ImageNet (Deng et al., 2009), and evaluate on the remaining datasets mentioned above. For domain generalization, we designate ImageNet as the in-distribution dataset, with four out-of-distribution variants encompassing ImageNetV2 (Recht et al., 2019), ImageNet-Sketch (Wang et al., 2019), ImageNet-A (Hendrycks et al., 2021b) and ImageNet-R (Hendrycks et al., 2021a). The experiments are carried out in the few-shot setting, where we randomly sample 16 shots for each category to compose the training set.

**Baselines.** We consider various PEFT approaches. Specifically, for prompt tuning, we consider category conditioning including CoCoOp (Zhou et al., 2022b) and TCP (Yao et al., 2024), regularization techniques encompassing KgCoOp (Yao et al., 2023), LASP (Bulat & Tzimiropoulos, 2023) and PromptSRC (Khattak et al., 2023b), attribute-based methods such as CPL (Zhang et al., 2024c), ArGue (Tian et al., 2024) and MAP (Liu et al., 2024b). We also consider multi-modal prompt tuning, *i.e.*, MaPLe (Khattak et al., 2023a). Besides, CLIP-Adapter (Gao et al., 2024) and its training-free

{8}------------------------------------------------

![Figure 5: Saliency maps for VLMs with and without SAS. The figure is divided into three sections: (a) chocolate cake, (b) personal laptop, and (c) street sign. Each section shows three images: 'Original', 'CLIP', and '+SAS'. The saliency maps in the '+SAS' column are more focused on the target object compared to the 'CLIP' column.](d4e9f8f6bf5d7853ecae9c9633900af1_img.jpg)

Figure 5: Saliency maps for VLMs with and without SAS. The figure is divided into three sections: (a) chocolate cake, (b) personal laptop, and (c) street sign. Each section shows three images: 'Original', 'CLIP', and '+SAS'. The saliency maps in the '+SAS' column are more focused on the target object compared to the 'CLIP' column.

Figure 5: **The saliency map of VLMs with and without SAS.** From left to right we show three example categories, including chocolate cake, personal laptop, and street sign.

| #p | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-|-|-|-|-|-|-|-|
| Base | 83.10 | 83.47 | 83.41 | 83.59 | 83.64 | 83.51 | <b>83.68</b> |
| New | 75.05 | 75.89 | 76.91 | 77.19 | 77.36 | <b>77.41</b> | 77.23 |
| HM | 78.87 | 79.50 | 80.03 | 80.26 | <b>80.38</b> | 80.34 | 80.32 |

Table 3: **Varying number of SD prompts on base-to-new generalization.** The results are averaged across 11 datasets and 11 baselines.

| $\gamma$ | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | Ada. |
|-|-|-|-|-|-|-|-|
| Base | 82.95 | 83.16 | 83.53 | 83.43 | 83.28 | 83.23 | <b>83.64</b> |
| New | 76.21 | 76.60 | 76.41 | 76.01 | 75.30 | 74.82 | <b>77.36</b> |
| HM | 79.44 | 79.75 | 79.81 | 79.55 | 79.09 | 78.80 | <b>80.38</b> |

Table 4: **Varying  $\gamma$  on base-to-new generalization.** We experiment with fixed values, juxtaposing them against the suggested adaptive strategy.

| Method | Epoch | Time $\downarrow$ | Accuracy $\uparrow$ | Gain $\uparrow$ |
|-|-|-|-|-|
| ZSCLIP | — | — | 70.22 | — |
| CoCoOp | 10 | 4h37m | 73.10 | — |
| + SAS | 10 | 6h18m | 74.21 | +1.11 |
| + selective trick | 10 | 4h51m | 74.02 | +0.92 |
| PromptSRC | 50 | 3h29m | 74.01 | — |
| + SAS | 50 | 4h56m | 75.46 | +1.45 |
| + selective trick | 50 | 3h38m | 75.20 | +1.19 |

Table 5: **The efficiency of SAS with and without selective trick.** We evaluate in terms of training time and accuracy gains given the same number of epochs. We opt for two time-intensive baselines, *i.e.*, CoCoOp and PromptSRC, and train both on ImageNet under base-to-new generalization task.

version Tip-Adapter (Zhang et al., 2022b) are involved. All results are averaged over three runs with distinct initialization.

**Implementation Details.** Unless specified otherwise, we use pre-trained CLIP (Radford et al., 2021) and ViT-B16 as the visual backbone for fair comparison. Since our proposed method is a plug-and-play module, we strictly adhere to the settings of existing works, including optimizers, batch size, learning rate, and other strategies. This indicates that for different baselines, we may use distinct hyperparameters specified in their respective papers. For SAP, we use GPT-4V Turbo (Achiam et al., 2023) as the MLLM, with a temperature scaler of 0.7 and an image understanding level set to *high*. For SAS, by default, we use ChatGPT to generate 5 prompts for each spurious attribute, which are then fed into Stable Diffusion (Rombach et al., 2022) to create pseudo categories. More details, such as the effect of choices of MLLMs and comparison between pseudo category construction with synthesized and pre-training data, are provided in Supp. Mat. B. Each pseudo category contains 16 shots, matching the number in the target category. All experiments are conducted on a single NVIDIA 4090 GPU.

### 4.1 MAIN RESULTS

**Our method is complementary to PEFT approaches.** Fig. 3 depicts the results of baselines and their integration with SAP and SAS. We observe an upward trend in accuracy, indicating improvements in out-of-distribution accuracy without compromising downstream task performance. For conventional methods, *e.g.*, CoCoOp, SAS helps achieve zero-shot capability on distribution shifts. For strong baselines, *e.g.*, CPL, the incorporation of SAP and SAS enables them to reach a new state-of-the-art benchmark. Overall, applying our method leads to an average improvement of over 2% in most baselines. These promising results align with the observation of the biased nature of VLMs, as demonstrated in Section 3.2.

**Our method is effective on counter test samples.** To further highlight the effectiveness of SAS, we conduct standard few-shot classification in an adversarial evaluation manner. This involves selecting a subset from the original test set to create a counter group for evaluating the VLM. For each category, we filter out images from the test set that bear high semantic similarity to the identified spurious attributes, retaining only images free of such attributes. This counter group is significantly

{9}------------------------------------------------

more challenging to predict using spurious attributes compared to the entire test set. Fig. 4 displays example images from the test set and counter group. Table 2 presents the improvement in accuracy of *SAS* over baselines, both for the test set and counter group. We notice that 1) for the counter group, its accuracy is much lower than that of the test set; 2) *SAS* effectively bridges this gap, with improvement on the counter group far exceeding that on the test set, up to approximately 6%.

### 4.2 ABLATION STUDY

**Diverse construction is beneficial for learning robust features.** During training, we aim for the constructed pseudo-categories to possess similar semantics to their target counterparts, thereby creating a strong contrast, while also maintaining high diversity to comprehensively represent spurious features. This trade-off is achieved by varying the number of SD prompts. As shown in Table 3, the effectiveness of *SAS* in assisting the baselines becomes more evident with an increasing number of prompts. This underscores the importance of the quality of pseudo-categories, which should thoroughly reflect the corresponding spurious attributes.

**Selecting appropriate spurious attributes matters.** The core principle of *SAS* is to introduce auxiliary categories to be trained alongside the main task, preventing the model from achieving high accuracy through spurious features. A natural concern is whether the model’s gains are due to the introduction of additional data rather than an increase in robustness. In other words, does the model genuinely learn to distinguish spurious attributes from pseudo categories? We investigate this by adjusting the threshold  $\gamma$  to control the presence of spurious attributes in pseudo categories. A higher  $\gamma$  indicates a shortage of identified spurious attributes, while a lower  $\gamma$  may introduce false positives. In Table 4, we observe that performance significantly drops when  $\gamma$  is either too high or too low. This indicates that 1) spurious attributes play a crucial role in the contribution of *SAS*, and 2) the introduction of noisy attributes actually impairs the model’s robustness. Additionally, the suggested adaptive strategy, which allows for flexible selection of spurious attributes, outperforms the pre-defined  $\gamma$ .

### 4.3 FURTHER ANALYSIS

***SAS* corrects the preference of VLMs on spurious attributes.** To qualitatively assess *SAS*’s impact on VLMs, we present the saliency maps of VLMs with and without *SAS* in Fig. 5. Common spurious correlations can be observed, such as (a) utensils appearing alongside chocolate cake, and (b) a mouse typically appearing with a laptop. In critical applications, such as autonomous driving, (c) road tends to act as confounders for street sign. *SAS* can effectively shift attention from these spurious attributes to the corresponding main objects. While we revisit Fig. 1(c), this also aligns with the interpretation of CBMs that *SAS* suppresses the influence of spurious attributes on predictions.

**Balancing the trade-off between efficiency and effectiveness.** One potential concern with *SAS* is its impact on training efficiency. Applying a distinct loss to each category can be computationally demanding. To address this, we introduce a selective optimization trick. Rather than targeting all categories, we only optimize ones that heavily rely on spurious attributes for predictions. Details of this approach are outlined in Supp. Mat. C. In Table 5, we demonstrate the effectiveness of this selective strategy by optimizing only 10% of the categories, showing the training time and accuracy. This approach significantly reduces *SAS*’s training time while preserving most of its accuracy gains.

## 5 CONCLUSION

This paper is motivated by an often-overlooked fact: VLMs tend to favor spurious attributes in their predictions, leading to decreased accuracy on out-of-distribution datasets. To tackle this issue, we first introduce *SPURIOUS ATTRIBUTE PROBING (SAP)*, which identifies and filters out these problematic attributes, significantly improving the generalization of existing attribute-based methods. Furthermore, to alleviate the biased nature of VLMs, we introduce *SPURIOUS ATTRIBUTE SHIELDING (SAS)*, a plug-and-play module that reduces the influence of these attributes on predictions and complements various *PEFT* approaches. Both solutions significantly enhance accuracy in handling distribution shifts without compromising performance on downstream tasks, achieving a new state-of-the-art level.

 Rest of paper (reference and Appendix) is removed.