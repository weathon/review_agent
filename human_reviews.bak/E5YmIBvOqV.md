# Large Convolutional Model Tuning via Filter Subspace

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Efficient fine-tuning methods are critical to address the high computational and parameter complexity while adapting large pre-trained models to downstream tasks.
Our study is inspired by prior research that represents each convolution filter as a linear combination of a small set of filter subspace elements, referred to as filter atoms. In this paper, we propose to fine-tune pre-trained models by adjusting only filter atoms, which are responsible for spatial-only convolution, while preserving spatially-invariant channel combination knowledge in atom coefficients.
In this way, we bring a new filter subspace view for model tuning. 
Furthermore, each filter atom can be recursively decomposed as a combination of another set of atoms, which naturally expands the number of tunable parameters in the filter subspace.
By only adapting filter atoms constructed by a small number of parameters, while maintaining the rest of model parameters constant, the proposed approach is highly parameter-efficient. It effectively preserves the capabilities of pre-trained models and prevents overfitting to downstream tasks. 
Extensive experiments show that such a simple scheme surpasses previous tuning baselines for both discriminate and generative tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposes a PEFT technique for convolution by decomposing the convolutional kernel into spatial and channel components and only fine-tuning the spatial components. Furthermore, the authors introduce a second-order decomposition technique to allow for the training of more parameters. The author validate the effectiveness of this method on various backbone models, such as ResNet50, ConvNeXt, and Stable Diffusion.

### Strengths
- The idea of decomposing the convolutional kernel and only fine-tuning the spatial convolution part is interesting, providing options for fine-tuning convolution layers. 

- The explanation of the methods and the mathematical formulas are clear.

### Weaknesses
- The paper requires additional sparse coding at the start of training to decompose convolutional atoms and coefficient atoms. Due to the need to solve optimization problems, I express concern about its efficiency. The computational cost and time delay associated with this part need to be provided. 

- The benchmarks compared in Tables 1 and Table 4 are not up-to-date. LoRA was proposed in 2021, but it is now 2024. To my knowledge, a series of related tasks have been continuously proposed in discriminative tasks in recnet years, such as SSF, FacT, Adapter, Compactor, BinaryAdapter, etc. The authors are encouraged to include the latest methods to demostrate the effectiveness of the proposed method. 

- The evaluation metrics for the generation task seem non-standard. It appears that the authors only compared results under one theme image, i.e., the castle. As far as I know, exsting common experimental setups for evaluating subject-driven generation tasks 750 prompt-image pairs, such as in OFT. The experimental setup in this paper only take one subject image, makeing it difficult to prove the effectiveness of the method, especially considering the inherent randomness of diffusion. In addition, I also suggest adding OFT and COFT in the compared methods, which are important and widely used baselines in diffusion model fine-tuning, and are included in the HuggingFace's PEFT library.

### Questions
- Besides comparing the number of parameters, what is the GPU memory footprint during fine-tuning for the proposed method? Considering that there is already work indicating that PEFT methods are generally not memory-efficient. 

- The idea of decomposing the convolutional kernel and only fine-tuning filter atoms is interesting. However, the experiments in this paper on various tasks do not solid to support the effectiveness of the method. It is necessary to further increase the comparison methods and improve experimental settings. Condidering all the factors, I tend to give a rating of below the acceptance level.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to fine-tune large pre-trained models over the filter subspace by only adjusting filter atoms and keeping atom coefficients unchanged for parameter-efficient fine-tuning. To adapt to more complex tasks, the number of tunable parameters in filter subspace is increased to construct an overcomplete set of filter atoms by recursively decomposing each filter atom over another set of filter atoms. Experiments on multiple CNN network architectures across discriminative and generative tasks show the effectiveness of the proposed method.

### Strengths
1.Clear motivation to fine-tune the spatial-only filter atoms for PEFT.
2.An interesting idea is to use the overcomplete filter atoms to improve performance.
3.Comprehensive experiments to evaluate the effectiveness of the proposed method.

### Weaknesses
1. Spatial-only convolution and cross-channel mixing are similar to group convolution and point-wise convolution. What is the difference when using group convolution and point-wise convolution as filter atoms and coefficients?

2. The authors mainly consider the parameter usage by only fine-tuning filter atoms. I think memory usage and computation are important for PEFT, which should be discussed in this paper for further evaluating the effectiveness of the proposed method. In addition, how to obtain the total parameters of fine-tuning across different networks should be analyzed to improve the readability
3.There are multiple important hyper-parameters (e.g., $m, m_1, k_c$), which significantly affect the final performance. How to set these hyper-parameters.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a new way to decompose convolutional layers and experimented a new way to fine-tune large models with those layers by adjusting a small number of parameters based on the decomposition. In particular, the observation that maintaining fixed atom coefficients leads to better results is showed based on the experimental results. Experimental results were compared with other PEFT methods such as LoRA and LoHa and showed interesting results in the provided examples.

### Strengths
1. This paper presents an interesting parameter decomposition method to split parameters in large convolutional models. 
2. In some situations as shared in the paper, the proposed method can achieve comparable or better results by fine-tuning an even smaller amount of parameters.

### Weaknesses
While the proposed decomposition and fine-tuning method is different, this method adjusts parameters in the big model. Comparatively, LoRA serves as a plug-in, which reduces the chance to hurt the capacity of pre-train models.

### Questions
Parameter fine-tuning often involves one large pre-trained model and many small tasks. Multiple LoRA's can be plug-in to one model even though there could be conflicts, to solve that scenario. How could this method achieve that?

### Soundness
3

### Presentation
3

### Contribution
3
