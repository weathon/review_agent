## Human Reviewer 1

### Summary
This paper introduces a long context, visual needle in a haystack benchmark which composed of 1k yes/no questions changeling the model to reasoning and find the target object in the images. It evaluated on both open-source and closed-source LMMs and reveal several critical findings such as susceptibility to visual distractors, difficulty in multi-image reasoning, and a bias in image positioning. It introduces a new baseline called MIRAGE (Multi-Image Retrieval Augmented Generation) for better handling of VH tasks.

### Strengths
1. This paper introduced a new visual needle in a haystack benchmark which composed of 1k yes/no questions. 
2. Evaluated on both open-source and close-source models and gained three insightful findings. 
3. Introduced a new baseline called MIRAGE for better handling of visual haystack tasks.

### Weaknesses
1. The questions are only limited to yes/no questions. 
2. The question template are very limited, seems only three. 
3. MIRAGE has a significant performance drop in 4 out of 7 general VQA tasks. 
4. The approach of MIRAGE, deselecting unrelated (distracting) images somehow circumvents the VH challenge, as the this challenge lies in how model can reasoning in long context.  
5. The task of finding a target object seems still not simulating a real world scenario of long context visual reasoning task.

### Questions
1. I'm confused about the difference between the MIRAGE model in Table 1 and the Q-former Model in Table. Doesn't MIRAGE utilize Q-former. 
2. See above.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper addresses the limitations of Large Multimodal Models (LMMs) in multi-image question answering, where handling large visual contexts does not ensure effective retrieval and reasoning across images. Current benchmarks reveal biases and challenges in MIQA, such as poor cross-image reasoning and sensitivity to information placement. To overcome these, the authors propose "Visual Haystacks (VHs)," a vision-centric benchmark that tests retrieval and reasoning over multiple images, highlighting models' struggles with visual distractors and multi-image reasoning. They also introduce MIRAGE, an open-source Multi-Image Retrieval Augmented Generation framework capable of handling up to 10,000 images on a single GPU, achieving significant improvements over existing models and setting new standards in MIQA benchmarks like RetVQA. Key contributions include VHs, systematic LMM evaluation, and MIRAGE's scalable MIQA capabilities.

### Strengths
- I generally feel the direction is important to our community where design meaningful Visual Haystack benchmark for evaluating VLM. 
- Some interesting points are discovered when evaluating models on the proposed benchmark. Since random guess could achieve 50% accuracy in the proposed benchmark, some open-sourced VLMs performance significantly drop even the Haystack size is very small. However, those models maintain high scores in some public evaluation-datasets. 
- Some detailed experiments are conducted such as needle position and running time. 
- The proposed benchmark are made publicly available under MIT license, which is good for community.

### Weaknesses
- Benchmark construction is still mainly centered around recognition tasks, based on benchmark design principles listed in Line129~138. Basically, it requires a strong recognition among all the input images, rather than true visual reasoning. 
- Based on the Figure 2 and 3, certain models, such as Gemini, GPT and the proposed MIRAGE, consistently perform better on the proposed multi-needle challenges compared to single-needle tasks. However, the multi-needle challenges are intentionally designed to be more difficult, as they demand additional reasoning across multiple images. Does this mean failure in designing the benchmark?
- Since the benchmark is constructed in way of examining recognition, therefore the proposed method contain ad-hoc modules, such as "a retriever module then calculates relevance scores, ensuring that only the most relevant images are passed to the LLM for final reasoning." Does this design hold for general visual reasoning tasks? For example, many of the tested single image dataset used in this paper, do not need this retriever module at all. 
- The proposed framework achieved not-very-good performance on some of the tested datasets. Also, there are many datasets that not being tested such as SEED, MME, and CHAIR.

### Questions
- Could you please address the points raised in the above weakness?
- Could you please add some randomly sampled failure cases made by GPT or Gemini? Sometimes failure cases can tell more than good cases. 
- Could you please address the ethics concerns around the code license?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
The authors presents Visual Haystacks (VHs), a new vision centric benchmark designed to assess the performance of Large Multimodal Models (LMMs) in the multi-image question answering (QA) task. In addition, the author proposes a new visaul-RAG framework, MIRAGE, to enhance the task performance.

### Strengths
* Novel Multi-Image QA Benchmark: The authors introduce an interesting multi-image QA benchmark, Visual Haystacks, designed around a vision-centric "needle-in-a-haystack" scenario, providing a fresh and challenging setting for the LMM evaluation.

* Comprehensive Model Evaluation:  The paper conducts a thorough evaluation of LMMs on the VHs benchmark, uncovering important insights into current models, such as vulnerability to visual distractors, challenges with multi-image understanding, and tendencies toward positional visual bias.

* Novel Visual RAG Framework: The authors introduce a novel visual RAG framework that combines a compressor and a retriever. The compressor efficiently processes up to 10,000 images on a single 40GB A100 GPU, while the retriever identifies the top-k most relevant images for a given question, enhancing the framework’s scalability and efficiency.

### Weaknesses
* Limited Object Diversity: The authors constructed the VHs benchmark using objects from the COCO dataset, which contains only 80 object categories. This limited selection may restrict the diversity and comprehensiveness of the benchmark, potentially affecting its ability to evaluate models across a broader range of visual scenarios.

* Restricted Question Diversity: The authors appear to rely on a few simple templates to generate questions, which may restrict the variety of question types in the benchmark.

* More like Object Detection than QA Reasoning: Many questions in the benchmark (e.g., "For the image with a truck, is there a dog?") seem to primarily assess the model’s object detection abilities rather than its visual QA reasoning skills. It is questionable if the benchmark requires the advanced visual QA reasoning skills from the models.

* Missing Related Work: The paper does not reference several recent multi-image QA benchmarks, for example: 
 1. CompBench: A Comparative Reasoning Benchmark for Multimodal LLMs
 2. MANTIS: Interleaved Multi-Image Instruction Tuning
 3. MUIRBENCH: A Comprehensive Benchmark for Robust Multi-Image Understanding. 

Additionally, a similar multi-image retrieval approach was introduced in "ColPali: Efficient Document Retrieval with Vision Language Models", but this work was also not cited.

### Questions
Please see the weakeness. In addition,
* How many templates were used to generate questions?
* What advantages does the VHs benchmark offer compared to recent multi-image QA benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4