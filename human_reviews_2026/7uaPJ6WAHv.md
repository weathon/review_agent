# Big-Layers: Enabling end-to-end training

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2

## Abstract
Training deep neural networks on extremely large inputs—such as gigapixel Whole Slide Images (WSIs) in digital pathology—poses significant challenges due to GPU memory constraints. Multiple Instance Learning (MIL) circumvents this limitation by processing patches from a WSI. However, the encoder used to get patch embeddings is usually a generic pre-trained deep neural network model. In this paper, we propose a training strategy that enables training the encoder by dynamically offloading intermediate activations of a layer to CPU RAM, allowing the layer to process inputs that do not fit in the GPU memory. We demonstrate the effectiveness of our approach on PANDA and CAMELYON datasets using popular MIL approaches. Experimental results indicate that our method improves the Quadratic Weighted Kappa (QWK) metric, on PANDA, by 7–15 percentage points compared to ResNet-18 baselines where encoders are kept frozen. Evaluations on external test sets further suggest better generalisation, and in some configurations, our models even outperform foundation-model encoders on TCGA-PRAD. The code will be made publicly available upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an approach for enabling end-to-end training on Whole Slide Images by offloading intermediate comptuations to CPU. Their experiments, performed on the Resnet-18 backbone, reveals consistent improvement over a frozen backbone. They propose layer-specific approaches for offsetting GPU memory requirements. Unfortunately, their experiments are solely performed on the CNN backbone, while current SOTA is dominated by vision transformers. For instance, their approach with ABMIL achives a quadratic weighted kappa of 84.60, while using the frozen UNI encoder achieves a performance of ~94 with ABMIL. The generalization of their approach to the self-attention mechanism of vision transformers is not discussed nor intuitive from their proposed algorithm, substantially limiting the utility of this approach.

### Strengths
- The ability to train models end-to-end is of substantial interest to the computational pathology community. 
- The model is evaluated on three MIL methods, indicating that performance is not specific to a specific mechanism. 
- The model is evaluated on a sufficient number of tasks.

### Weaknesses
- Performance is only shown on the Resnet-18 encoder, which is substantially behind the state of the art vision encoders (e.g UNI-2, Virchow-2, Conch 1.5). Consequently, even the improved end-to-end model lags behind frozen vision encoders by a wide margin. 
- The way to apply this method to self-attention is not clear from the text, further limiting the relevance of these results to current state of the art. 
- There are no benchmarks against other end-to-end approaches in CPath. There is consequently no evidence that this method provides benefit over existing approaches. 
- There is substantial room in the paper for additional details. An improved description of the method should be provided. 

As it stands, the work is not framed in terms of the state-of-the-art, and I believe there is not sufficient time to include such state-of-the-art. I am recommending reject as a result. However, I would be willing to update my rating if my comments are sufficiently addressed.

### Questions
- The code should be provided to help us further evaluate its quality and usability.
- The benchmark protocol does not need to be in bullet points. The extra space should be used to describe the method in greater detail.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a new approach for making predictions on gigapixel histopathology images. Instead of the classical pipeline: (1) tiling slides into patches, (2) embedding patches with pretrained encoders, and (3) aggregating for slide-level prediction; the authors address GPU memory constraints by offloading intermediate activations to CPU RAM when GPU memory is insufficient. This enables joint training of the patch encoder and the slide-level predictor. Concretely, they apply this only to the first layers of a ResNet-18, stopping once sufficient downsampling allows the remainder to fit on the GPU.

### Strengths
The paper tackles an important limitation of Multiple Instance Learning (MIL) methods in computational pathology that tile whole-slide images into instances. It might be desirable to perform end-to-end training, which usually is not done with current workflows.

### Weaknesses
1. The paper is not well motivated within the existing literature. The community’s reliance on pretrained encoders stems not only from memory limits but also from limited labeled data, where pretraining provides substantial gains. The work cites a 2023 pipeline for MIL training, while the GPU memory bottleneck citations are from 2020–2021; the related work feels uneven and insufficiently up to date. 
2. The primary encoder, ResNet-18, is dated for this domain. 
3. The main comparison pits a fully trained ResNet-18 against a frozen ImageNet-pretrained ResNet-18. A fairer baseline would include contemporary frozen encoders pretrained on histopathology patches (e.g., UNI, Virchow, H-Optimus) to assess whether encoder finetuning is truly necessary and beneficial. 
4. The approach is stated to be incompatible with ViTs, which are common encoders in this area. It would be important to evaluate compatibility and performance with larger CNNs (e.g., ResNet-50) and, if possible, discuss extensions to transformer-based models. 
5. The runtime comparison is debatable. Since the baseline reportedly does not benefit from data augmentation; the time to be compared with should take into account the fact that the embeddings can be stored.

### Questions
1. Is the method practical with larger backbones such as ResNet-50? 
2. How does the approach compare against strong frozen pretrained encoders (UNI, Virchow-2, H-Optimus)?
3. Given that the current methods not only respond to memory limits but also to limited labeled data, could the authors specify and demonstrate the use case where their method improves on current SOTA ?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose an approach to train the patch encoders on gigapixel WSI data with label supervision. This requires backpropagating gradients from the WSI label to the patch encoder layers, for which they propose dynamically offloading intermediate activations of a layer to CPU RAM, allowing the layer to process inputs that do not fit in the GPU memory. The authors show improved performance on PANDA, TCGA-PRAD and CAMELYON in terms of quadratic weighted kappa metric, compared to an ImageNet pre-trained encoder baseline, at a cost in training time.

### Strengths
Significance: The method allows backpropagating supervised signal back to the patch encoders of WSI for potentially significant gains in performance on WSI classification tasks. This type of strategy is interesting and could potentially help design very strong encoders in the future.

Clarity: The paper is generally clear.

### Weaknesses
The main weaknesses currently have to do with the evaluation of the proposed method. Reinforcing the evaluation either during the rebuttal phase or after resubmission would change my assessment of the paper. 

1. The single baseline is a ResNet-18 encoder pre-trained on ImageNet, which artificially inflates the gain in performance from the proposed approach. The field has largely moved to task-specific encoders trained with self-supervised learning (DINO, MoCov3, etc.), or to encoders based on foundation models (UNI, Virchow, GigaPath, etc.) trained on large numbers of WSIs from various histopathology datasets. These encoders should be used as baselines instead. 

2. I am not convinced by the fairness of the comparison in terms of runtime. The authors report 45 minutes for one epoch for the baseline on PANDA vs. 200 minutes for the proposed approach. But my understanding is that this is when computing patch "embeddings on the fly on GPU" also for the baseline, whereas patch embeddings are usually computed offline and saved once and for all in the standard approach (as no backpropagation is required). Hence training the MIL stage from pre-computed patch embeddings for the baseline could be significantly faster.

3. A nice addition would be to also demonstrate the method on a ViT encoder and on a ResNet-50, to get a sense of the scalability and flexibility of the approach. Experiments are so far limited to ResNet-18.

### Questions
The major questions that could be answered to change my initial rating have to do with the weaknesses listed above.

Additional questions:
- Could the authors also report AUCs when relevant for direct comparison with the literature?

### Soundness
2

### Presentation
3

### Contribution
3
