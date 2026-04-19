# Understanding deep neural networks through the lens of their non-linearity

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
The remarkable success of deep neural networks (DNN) is often attributed to their high expressive power and their ability to approximate functions of arbitrary complexity. Indeed, DNNs are highly non-linear models, and activation functions introduced into them are largely responsible for this. While many works studied the expressive power of DNNs through the lens of their approximation capabilities, quantifying the non-linearity of DNNs or of individual activation functions remains an open problem. In this paper, we propose the first theoretically sound solution to track non-linearity propagation in deep neural networks with a specific focus on computer vision applications. Our proposed affinity score allows us to gain insights into the inner workings of a wide range of different architectures and learning paradigms. We provide extensive experimental results that highlight the practical utility of the proposed affinity score and its potential for long-reaching applications.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new metric to measure the non-linearity of activation functions in deep neural networks. Extensive comparative studies have been done for the analysis of different network architectures. The proposed metric is also somehow relevant to the performance of a DNN.

### Strengths
+ The topic of analyzing the non-linearity of DNNs is interesting.
+ It seems novel (to me, not an expert) to leverage OT to measure the extent of non-linearity.

### Weaknesses
- Def 3.1 only focuses on pointwise non-linear activation functions to measure non-linearity. However, many functions other than the ACT family also introduces non-linearity to the DNN, such as max pooling. How to measure the non-linearity of these functions?
- Def 3.1 considers each activation function independently. How about combining two or more activation functions? Intuitively, by stacking the non-linearities, the overall non-linearity will grow exponentially.
- How to interpret Figure 1? It seems that the non-linearity of sigmoid and gelu are better than relu. Then why does relu become the most widely used activation function?
- About the result in Figure 9. How to interpret the negative correlation between the maximum affinity score and accuracy. It seems contradict with the NAS models results in Figure 20.
- Could the proposed metric bring any new insights (or serve as a regularization term) to help us build better DNNs or design better activation functions?

### Questions
see above

### Soundness
2 fair

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a theoretically sound solution to track non-linearity propagation in deep neural networks. Specifically, this method measures the nonlinearity of a given transformation using optimal transport (OT) theory. More critically, the authors investigate the practical utility of the proposed affinity score and apply the proposed affinity score to a wide range of popular DNNs.

### Strengths
1. This paper develops a new method to track non-linearity propagation in deep neural networks.
2. This paper proposes the affinity score to evaluate the non-linearity and apply it to diverse architectures.

### Weaknesses
1. The authors mention that "consider transformer architectures with a
specific focus on the non-linearity present in their MLP blocks". How about the non-linear operation inside the attention block, e.g., softmax?

2. In Figure 2, the authors highlight the robustness of the affinity score. Nevertheless, it is still unclear why the robustness of this score is important.

3. It is unclear how accurate the affinity score is to evalute the non-linearity. More details are required towards this.

4. It seems that the non-linearity does not vecessarily correlate with the performance. From this point of view, how do we understand DNNs based on this metric?

### Questions
Please refer to the weakness part.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies various different NN architectures through a a unique metric of non-linearity. It shows how different networks throughout history have leveraged non-linearity and how they improved. It also shows some theoretical guarantees.

### Strengths
The paper is quite interesting! I think it goes through various architectures and shows some compelling results.

### Weaknesses
The presentation can be greatly improved in my opinion. The figures are not clear, colormaps are missing, many effects in the plots are not thoroughly explained (e.g. the effects in most figures except the ViTs are not very clear). I think this paper has the potential to be much better if the writing were to be improved, along with the presentation, and more thorough explanations.

### Questions
Minor comments:
1. Plots are not clear, they do not have a color bar, and the colors are not clear (e.g. Figure 4, 5, ...)

Questions:
1. Have the authors tried to check how the non-linearity metric corresponds with intermediate layer performance (e.g. the linear eval on the layer)? This could be interesting to check to improve the paper.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a formulation, named 'affinity score,' to measure the non-linearity of deep neural networks. The authors then use the proposed affinity score to evaluate the non-linearity signatures of popular neural networks and illustrate the affinity scores for every layer within these networks. The authors claim that these non-linearity scores will bring insights into the understanding of neural networks.

### Strengths
1. The topic of understanding non-linearity in DNNs is intriguing, and the author proposes a novel formulation to evaluate it.
2. The writing style is engaging, and the presentation of the material is well-executed.

### Weaknesses
1. In the computation of affinity scores, which are tied to the activations within a neural network, there is an inherent dependence on both the input data and the network's parameters. My understanding is that affinity scores are influenced by the architecture and the parameters (the trained weights) of the network. However, it seems that in the evaluation of their experiments, the authors have placed a predominant focus on the architecture while possibly overlooking the significance of the network parameters. It is important to consider that the parameters, which are shaped by the training process, are crucial for the network's performance and ultimately for the validity of the affinity scores. An in-depth analysis that includes the impact of these parameters could provide a more comprehensive understanding of the network's behavior and the experimental outcomes.
2. The author claims "Despite being almost 20 times smaller than VGG16, the accuracy of Googlenet on Imagenet remains comparable, suggesting that increasing and varying the linearity is a way to have high accuracy with a limited computational complexity compared to predecessors.”
However, this assertion seems to overlook the fact that a significant portion of VGG16's parameters are concentrated in the final fully connected layers, which consist of two 4096-dimensional layers. Empirical evidence suggests that reducing the parameter count of these fully connected layers does not drastically diminish performance. Thus, concluding that 'increasing and varying the linearity is a way to have high accuracy with a limited computational complexity' may not be entirely justified.

### Questions
Please refer to the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
