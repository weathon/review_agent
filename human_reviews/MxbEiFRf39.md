# NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
We introduce NNsight and NDIF, technologies that work in tandem to enable scientific study of the representations and computations learned by very large neural networks. NNsight is an open-source system that extends PyTorch to introduce deferred remote execution. The National Deep Inference Fabric (NDIF) is a scalable inference service that executes NNsight requests, allowing users to share GPU resources and pretrained models. These technologies are enabled by the Intervention Graph, an architecture developed to decouple experimental design from model runtime. Together, this framework provides transparent and efficient access to the internals of deep neural networks such as very large language models (LLMs) without imposing the cost or complexity of hosting customized models individually. We conduct a quantitative survey of the machine learning literature that reveals a growing gap in the study of the internals of large-scale AI. We demonstrate the design and use of our framework to address this gap by enabling a range of research methods on huge models. Finally, we conduct benchmarks to compare performance with previous approaches.

Code, documentation, and tutorials are available at https://nnsight.net/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces NNsight and NDIF, a framework and a cloud inference service designed to facilitate research on large-scale AI models, and enable study of the representations and computations learned by large neural networks. This work proposes a method for organizing experiments on very large models, reducing engineering burden, enhancing reproducibility, and enabling low-cost communication with remote models. The NNsight is an open-source implementation of the intervention graph architecture that extends PyTorch to support transparent model interventions without requiring local storage or management of model parameters. And the NDIF is an open-source cloud inference service that supports the NNsight by providing behind-the-scenes user sharing of model instances to reduce the costs of large-scale AI research.

### Strengths
1. The intervention graph architecture and the NNsight system provide a practical and efficient way to decouple experimental and engineering code, making it easier to conduct complex experiments on studying the intermediate representations and gradients of large models.
2. The paper includes a thorough performance evaluation comparing NNsight and NDIF with HPC and Petals, demonstrating the efficiency and effectiveness of the proposed solutions.
3. The performance comparisons between NNsight and other frameworks like baukit, pyvene, TransformerLens show the similar efficiency of NNsight.
4. The remote execution of NNIF provides the easy usage for users.

### Weaknesses
1. While the developed tools of NNsight and NDIF are very useful in scientific study of AI models, there is no founded insights of this work. I'm not sure whether this contribution is enough for the acceptance.
2. Exploiting the DAG representation is not novel. As indicated by the authors, [1] has proposed the DAG concepts and use it to accelerate computation. Besides, the Torch Profiler [2] and torch.jit.trace [3] can help to dissect the model. I do not see any obvious technical challenges to implement the NNsight. Could you elaborate more on such challenges and illustrate how you address them? And how the NNsight is different from the Torch Profiler and torch.jit.trace?
3. For the practical usage like shown in Figure 4, I do not see obvious advantages of NNsight than using hooks of Torch. The required number of code lines of them are similar. Could you also elaborate more about this?

[1] Theano: A python framework for fast computation of mathematical expressions. 2016
[2] PyTorch Profiler. https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html.
[3] Torch.jit.trace. https://pytorch.org/docs/stable/generated/torch.jit.trace.html.

### Questions
Please refer to the weaknesses and provide more clarifications.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a new PyTorch-compatible library for remotely accessing the internal structures of deep-neural-network-based models such as large language models (LLMs) and altering the way they operate. Using this library, it is possible to produce intervention code which is hooked into the original model to read or replace original model parameters and activations. This intervention code is translated into a directed acyclic intervention graph that augments the computational graph of the original model.  Others have proposed similar, but less flexible mechanisms. For instance, pyvene (Wu et al., 2024) supports dictionary-based intervention definitions instead of the code-based interventions proposed by this work, which can also be transformed into graph-based representations and further optimised, e.g., by TorchScript. Petals (Borzunov et al., 2023), on the other hand, does not support remote execution of the intervention code, requiring it to be executed locally, which limits virtualisation opportunities and leads to additional communication overheads.

### Strengths
- The proposed approach involves some new mechanisms that complement the related work.
- By enabling remote execution of the intervention code the authors have demonstrated a superior performance compared to Petals.

### Weaknesses
An interesting real-life application of the infrastructure built is missing. For instance, the authors could consider demonstrating their system on a large-scale interpretability study or model editing task that would be infeasible without their infrastructure.

### Questions
It looks like the main technical contribution of this work is the enablement of efficient/virtualised remote interventions on LLMs. Could the authors elaborate on why they believe this contribution is relevant and impactful for the ICLR community?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces NNsight and NDIF, two open systems that provide efficient, transparent access to the internals of large neural networks for research purposes. NNsight extends PyTorch to offer deferred remote execution of intervention graphs, while NDIF serves as a scalable inference service that executes these intervention requests, enabling resource sharing among users. The work addresses challenges like limited access to state-of-the-art models and the significant resource demands of large-scale AI research by sharing resources.

### Strengths
* The intervention graph extends the model computational graph and decouples experimental design from model runtime, reducing engineering complexity.
* NDIF effectively shares GPU resources among researchers (co-tenancy), reducing cost and enabling large-scale experiments
* This is a novel idea with great potential benefit to the research community.

### Weaknesses
* The evaluation section is limited to the end-to-end performance with little detail on the system optimizations. Specifically, more information is needed to assess the performance and scalability of this system using an increasing number of users.
* ~~While NDIF addresses large-scale experiments on open models, it does not cover closed, proprietary models hosted proprietarily.~~
* Lack of discussion with relevant co-serving systems like S-LoRA (Sheng et al, MLSys 2024) and dLoRA (Wu et al, OSDI 2024).

### Questions
Could you please share more evaluation results on the performance and scalability of the proposed systems?

### Soundness
2

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
The paper introduces two systems NNsight and NDIF that collectively aim to reduce the developer and hardware costs of analyzing and modifying the inference behavior of open-source models. NNsight is an instrumentation framework for PyTorch models, while NDIF is an inference service that enables deferred execution of NNsight instrumentations on a remote shared model deployment. The paper provides evaluation comparing to inference baselines of shared (Petal) and non-shared HPC deployments.

### Strengths
- This paper is focusing on an important problem since tools that enable introspection of model internals are extremely valuable for ML research and applications. 
- Opting for Pytorch-native tools is a great design choice as that significantly reduces development and integration burden for practitioners. 
- Enabling resource sharing via NDIF service is very useful to democratizing access to SOTA models.

### Weaknesses
- The paper seems to overclaim, particularly in the title and abstract, that the work applies to foundation models in general, when in reality it only applies to open-source models, since model internals knowledge is required to create interventions. The authors should more carefully scope the claims. 
- The evaluation does not study co-tenancy scenarios where the NDIF is servicing multiple NNsight requests. This is a significant oversight considering that the resource sharing benefits of NDIF is a major contribution of the work. The authors should include results showing not just the performance implications of servicing the multiple NNsight requests but also validating the correctness of the co-tenancy features.

### Questions
1. Do users submit a pair of NNsight request and an input prompt? Or how is the input prompt for exercising an intervention generated?
2. When multiple NNsight requests are submitted, are all the interventions applied to a single model instance for inference, or is a separate model instance created for each request?
3. If a single model instance is instrumented with multiple NNsight requests, how does NDIF ensure that a request that modifies model parameters does not affect other requests?
4. How is KV-cache managed for multiple NNsight requests?

### Soundness
3

### Presentation
3

### Contribution
3
