# Infinite-parameter Large Language Model

- Decision: Reject
- Scores: 3, 3, 1, 1

## Abstract
In the standard transformer architecture, increasing model parameters leads to linear growth in computational cost and activation memory. To address this issue, we propose a novel Infinite Parameter Large Language Model (IP-LLM) architecture that decouples model size from computational cost and device memory. Existing large language models are all fixed-parameter models, while human knowledge is infinite and expands daily. Finite parameters are inherently limited in their capacity to accommodate this boundless knowledge. Our IP-LLM architecture can potentially accommodate infinite knowledge, resolving this issue and laying the foundation for realizing a truly omniscient and omnipotent artificial general intelligence in the future.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper presents IP-LLM inspired from MoE which use routing mechanism to enable continual learning of multi-domain tasks and memory efficient inference. The authors use segmented pre-training strategy to train a base block to acquire general linguistic skills and then train router and domain-specific modules.
The authors evaluate IP-LLM's performance on 4 different tasks.

### Strengths
The authors propose a novel pre-training strategy to parse the general linguistic comprehension skills in the base model and later train individual experts on top on different domains.

### Weaknesses
1. Evaluation

- I think the biggest issue in current manuscript is evaluation of the model. While the current evaluation only includes monolithic architectures without MoE strategy, wouldn't it be fairer to include the models using MoE's in terms of both performance and memory/compute efficiency? 

- The authors claim in the list of contributions that the new approach allows higher routing accuracy but I cannot find the explicit result for this.

- Also the authors claim the memory and training efficiency but having a explicit numerical comparison and what exact 'training cost' is meant here. 

2. Related works on lifelong learning / continual learning using MOE

- I believe there are already few literatures on lifelong learning of LLM using MoE e.g Chen et al (2023) https://arxiv.org/pdf/2305.12281. I suggest authors to incorporate more relevant literatures and what is the novelty of their method. 

3. Paper presentation should be improved

- There are many typos and inconsistent citing notations which makes readability very low. For example, I found Section 2 related work very hard to parse the included citations (spacing, parentheses etc). I highly recommend authors to do careful proofreading of the entire manuscript. 

- Section 4 training strategy can be improved with adding a schema for better delivery.

### Questions
I believe my comments about the possible improvements and weaknesses incorporate my questions.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper proposes an "Infinite-Parameter Large Language Model" with the idea to accommodate the increasing amount of information generated in the world that it hypothesizes models with a fixed number of parameters will eventually not be able to contain. The implementation involves training the model as a Mixture of Experts.

### Strengths
The paper tackles an important question.

### Weaknesses
- The paper does not make a distinction between the proposed approach from MoE training. And if there is indeed a difference, please include the MoE baseline. 
- **Major issue**: The paper compares their model trained on downstream tasks with other pre-trained models, zero-shot on the downstream tasks. Therefore, it is not making an apples-to-apples comparison
- It would be important to add in a couple of baselines to showcase the benefit of the proposed method over others
  - A single model trained on all the data that the base, router, and individual experts are trained on
  - MoE baseline trained on the data used to train the IP-LM. 
- Why are there no entries in the table for some models on C-Eval?
- There are not many details provided on training, model architecture, and dataset. Unclear what data is additionally used to train the base model. What is the architecture of the model? 
- The writing in the paper is clear but not precise. For example, the abstract or intro does not tell you anything concrete about what the paper builds, it only goes so far as to specify the problem and motivate it.

### Questions
Please take a look at the section on weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
This paper proposes a language model based on a router that selects domain-specific parameters. A base model parses an input, a router classifies the input into a category corresponding to a domain, then inference is done with parameters corresponding to the selected domain.

Each domain-specific parameter set is implemented using the last transformer layer of the base model replicated four times, and then trained with a "defined proportion" of domain-specific data and general data. The router has a similar parameterization, and is trained using classes corresponding to the domain-specific data.

The paper reports metrics on MMLU, C-Eval, GSM8K, and MATH, and also reports metrics from Llama2, Mistral, and Qwen1.5.

### Strengths
The idea of routing to domain-specific parameters is interesting (though more discussion of related work is needed, e.g. [1][2]). 


[1] Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models, Li et al 2022

[2] Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM, Sukhbaatar et al COLM 2024

### Weaknesses
The paper appears to be in an early stage. For example, key details of the method and experiment are not described or justified, and only 1 experiment (not fully described) has been performed. There is also missing discussion of key related work (e.g. [1], [2]). Here are some specific examples:

- The datasets have not been described. The data can substantially impact the downstream tasks that are evaluated in the experiment. Similarly, the number of tokens trained on is important and has not been reported.

- The evaluation is done on a proprietary evaluation pipeline, making reproducibility difficult.

- The experiments need a controlled comparison of the method against alternatives. Currently it is difficult to draw conclusions from the experiment provided. For example, IPLLM-24B and Qwen1.5-32B are not comparable since IPLLM has been trained on additional domain-specific data (which has not been specified, and may be relevant for the experimental comparison). One example comparison could be finetuning Qwen1.5-32B on the union of corpora that IPLLM finetunes on.

- Several claims made in the introduction and conclusion have not been justified. For example:
    - "Significant advantages in terms of reduced device memory requirements for both training and inference": this has not been justified. For example, the proposed method requires two forward passes at inference time, and an additional 4 x (number-of-domains + 1) layers to train.
    - "Enabling the model to learn new knowledge without catastrophic forgetting". This has not been justified experimentally.

- Ablations on key design decisions have not been done. For example, the routing strategy, number of layers, and the base model.

- Regarding novelty, Branch-Train-Merge [1] proposed to train different parts of the model independently on different subsets of the data (each subset corresponding to a domain, such as scientific or legal text). They also have a domain posterior that models the probability of a sequence belong to each domain (akin to the functionality of the proposed router). BTM and related follow-up work such as Branch-Train-MiX [2] should be discussed and compared with.

I would encourage the authors to continue improving the work since their idea has potential, but I believe the current manuscript is not yet ready for ICLR. 

[1] Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models, Li et al 2022
[2] Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM, Sukhbaatar et al COLM 2024

### Questions
Please address the points discussed in the Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
2

### Summary
This paper proposes to divide the LLMs into two step paradigm: (1) a routing step to have the transformer output the category of tasks; (2) for each category of tasks, a delegated transformer (base transformer + category special layers)  is applied to generate the outputs. Experiments are described in the paper to demonstrate the performance is comparable to existing models but not better.

### Strengths
• An interesting attempt to look for continuous scalable architecture for LLMs.

### Weaknesses
• It is not clearly to me how the proposal can achieve infinite-parameter models just by classifying the input to into different classes and training/using different networks for different classes.

### Questions
1. How the routing network (Equation 6) can be generalized to unseen categories so as to be generalized to infinite many categories? Is the routing token set fixed or open? 
2. Is section 5 not completed in this version? It seems that a significant amount of texts are truncated between line 270 and line 282.

### Soundness
1

### Presentation
1

### Contribution
1
