# Chain of Ideas: Revolutionizing Research in Idea Development with LLM Agents

- Avg Score: 5.75
- Decision: Reject
- Scores: 5, 5, 8, 5

## Abstract
Effective research ideation is a critical step for scientific research. However, the exponential increase in scientific literature makes it challenging for researchers to stay current with recent advances and identify meaningful research directions. Recent developments in large language models~(LLMs) suggest a promising avenue for automating the generation of novel research ideas. However, existing methods for idea generation either trivially prompt LLMs or directly expose LLMs to extensive literature without indicating useful information.  Inspired by the research process of human researchers, we propose a Chain-of-Ideas (CoI) agent, an LLM-based agent that organizes relevant literature in a chain structure to effectively mirror the progressive development in a research domain. This organization facilitates LLMs to capture the current advancements in research, thereby enhancing their ideation capabilities. Furthermore, we propose Idea Arena, an evaluation protocol that can comprehensively evaluate idea generation methods from different perspectives, aligning closely with the preferences of human researchers. Experimental results indicate that the CoI agent consistently outperforms other methods and shows comparable quality as humans in research idea generation. Moreover, our CoI agent is budget-friendly, with a minimum cost of \$0.50 to generate a candidate idea and its corresponding experimental design.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper aims to address the increasing difficulty researchers face in generating new ideas due to the exponential growth of scientific literature. It introduces a framework called Chain-of-Ideas (CoI), which mirrors the way human researchers process research by organizing literature in a progressive chain. The CoI framework builds chains of relevant research literature, allowing LLMs to better understand the current landscape and generate novel research ideas. This method organizes the literature by focusing on both the progression of ideas (forward and backward from an anchor paper) and the development of trends. Through rigorous experiments, the CoI agent consistently outperforms existing methods, such as retrieval-augmented generation (RAG), and produces ideas comparable to human creativity.

### Strengths
1. The CoI framework is innovative in organizing research literature systematically. It demonstrates superior performance compared to existing methods (RAG, GPT-Researcher, AI-Scientist), particularly regarding novelty, significance, and clarity of generated ideas.
2. The framework is budget-friendly and capable of producing ideas and experiment designs at a low cost.
3. The introduction of Idea Arena as an evaluation system adds robustness to the assessment of idea-generation methods, ensuring alignment with human researchers' preferences.

### Weaknesses
1. For the method section, I highly recommend (1) labeling the corresponding letters of the formula (such as $I_{0}$) and (2) captioning more (in your paper, the caption of Figure 2 is only "Our proposed CoI agent framework") in Figure 2; otherwise, it may cause significant confusion for the readers.
2. In the field of research on idea generation using LLMs, the metrics are the most important. The authors evaluate the novelty of ideas depending on LLMs (ICML2020 review guidelines) and human researchers. There are two questions: 
(1) ICML review guidelines are used for papers, not ideas, thus do you have more operations on this step?
(2) I would like to know more details about human researchers. As 10 human researchers have different research interests in AI, how you ensure they can finish these evaluations? I would like to provide [1] for you to think more about this question and complete your work.
Please provide more details.
3. Currently your work focuses on AI research. On the one hand, this area is too small in scientific research, which makes an overclaimed contribution. On the other hand, if your framework is solely used for idea generation, I believe its application may be limited, i.e., I hope you can achieve broader applications, such as AI Scientist, which aims at writing a paper.
4. I believe the most critical issue lies in the comparison between the experiments and the baseline, especially when comparing with the AI Scientist. On the one hand, the details of the comparison are not fully explained. In your paper, you use the description "We extract the components related to idea generation and experiment design to serve as our baseline", which I think is too simple to believe how you re-implement. On the other hand, I find this comparison to be unfair to some extent. As you mentioned in the paper, the purpose of the AI Scientist is to complete an entire paper, whereas your research focuses on generating novel ideas. Maybe you could provide more details.

[1] Si, Chenglei, Diyi Yang, and Tatsunori Hashimoto. "Can LLMs Generate Novel Research Ideas?." arXiv preprint arXiv:2409.04109 (2024).

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduces a novel framework called the Chain-of-Ideas (CoI) agent, designed to enhance the capability of LLMs in generating novel research ideas. The CoI agent simulates the progressive development of research domains by organizing relevant literature into a chain structure, allowing LLMs to better understand current advancements and produce more innovative ideas. Experimental results demonstrate that the CoI agent consistently outperforms other methods and produces ideas comparable to those generated by humans, with minimal costs involved in generating candidate ideas and corresponding experimental designs.

### Strengths
**Provides a Solid Idea Generation Method**: The CoI agent framework effectively enhances AI's capability in generating research ideas by simulating human researchers' practices. By organizing literature into a chain structure, the model can capture the evolving trends within a research domain, leading to more innovative and contextually relevant ideas.

**Comprehensive Evaluation Protocol**: The introduction of the Idea Arena provides a robust method for evaluating idea generation systems. By incorporating multiple criteria and aligning with human preferences, it offers a valuable tool for future research in assessing the quality of AI-generated ideas.

**Demonstrated Effectiveness Through Experiments**: The experimental results show that the CoI agent not only outperforms baseline methods but also approaches the quality of human-generated ideas. This indicates the method's potential impact in advancing AI-assisted research ideation.

### Weaknesses
**Lack of Discussion on Idea Safety and Ethical Issues**: The paper does not address potential safety concerns and ethical issues associated with AI-generated ideas. Without safeguards or discussions on these topics, there is a risk of the model generating unsafe or unethical research proposals.

**Questionable Reliability of Evaluation Metrics**: From Table 1, the agreement between human judges and GPT evaluators is not very high, raising doubts about the reliability of the evaluation metrics used. This discrepancy needs further investigation to ensure the validity of the results.

**Similarity to Chain-of-Thought Methods**: The proposed method resembles the Chain-of-Thought (CoT) approach, appearing more like an application of CoT in the context of research idea generation. This raises questions about the novelty of the method itself.

**Insufficient Evaluation of Hallucination Issues**: The potential for hallucinations in the generated idea chains, inherent in prompt-based methods, is not adequately addressed or evaluated. This could impact the reliability and factual accuracy of the generated ideas.

### Questions
**What is the Specific Background of the Human Annotators?** Understanding the expertise and diversity of the human judges is crucial for assessing the evaluation results' validity. Could the authors provide details on the annotators' backgrounds, areas of expertise, and how they were selected?

**How Does the CoI Agent Handle Interdisciplinary or Emerging Fields?** In areas where literature is sparse or the progression of ideas is not well-defined, how effective is the CoI agent in constructing meaningful idea chains and generating valuable ideas?

**How Are the Literature and Idea Chains Quality-Controlled?** What measures are in place to ensure that the selected literature and the constructed idea chains are of high quality and relevance? Incorrect or low-quality inputs could negatively impact the generated ideas.

**More details of the milestone paper choosing?** As I understand, you use citations over 1000 to choose the milestone paper; however, these papers sometimes may be too broad as a topic(survey paper) or too fundamental (GPT-3) and hardly provide any valuable insights. Have you considered this issue?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors show a way how LLMs can be utilized to generate novel paper directions together with experimental design etc. They call it Chain-of-Ideas (CoI) since it utilizes the trend of ideas in the papers to come up with new ideas. They show that their method seems to have some correlation to real papers.

### Strengths
- The paper shows that CoI performs well in their benchmarks when using LLMs as judges
- It is supposed to be cheap, which is important due to use of closed source LLMs
- It does seem to even find some ideas that are there in the new papers, following the ones they provided in their RAG

### Weaknesses
- I'm concerned with the evaluation metrics. Utilizing LLMs as judges for ideas should be better motivated.
- Similar with human evaluators. One rarely can guess the outcome from ideas described in a few paragraphs before starting to implement them.

### Questions
- Could you better motivate using LLMs as judges for novel ideas?
- Could there be some contamination of new papers (even the ones as recent as a few weeks) when generating novel ideas with LLMs?
- Why is the 70\% agreement rate between humans and LLMs show a strong indication of agreement? as said in line 353?

I'm open to revising my score if I get convincing answers to my questions.

Post-rebuttal update: All my concerns are addressed by the author's rebuttal.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a Chain-of-Ideas (CoI) agent, designed to enhance research idea generation using large language models (LLMs). While recent similar approaches ((Lu et al. 2024) and (Besta et al. 2024) for example) just mix up papers into the prompt without effective organisation, the Col agent organises relevant literature in a chain structure emulating the progressive development in a particular research topic, thus facilitating the LLMs to generate new ideas by identifying current advancements. In addition, the authors propose an evaluation framework, Idea Arena, to assess idea generation from multiple perspectives akin to human researchers. The experimental results demonstrate that the CoI agent outperforms some existing methods and is also cost-effective, with its idea generation quality being comparable to human-generated ideas.

### Strengths
- **Originality:**
    - The CoI agent introduces an innovative way of organizing literature in a chain-of-ideas structure to improve idea generation, which is a notable shift from traditional RAG approaches or simple prompting.
    - Idea Arena, with its tournament-style evaluation, presents a structured and potentially more accurate method to assess the quality of LLM-generated ideas.
  
- **Quality:**
    - The research is thoroughly backed by comprehensive experiments and robust results, which demonstrate the superiority of the CoI over baseline methods such as simple RAG, AI Scientist (Lu et al. 2024), GPT-researcher (https://github.com/assafelovic/gpt-researcher), and ResearchAgent (Baek et al. 2024). 

- **Clarity:**
    - The paper is well presented and clear, with descriptions of methodology, evaluation criteria (LLM-as-a-Judge and Human evaluators), and some results that substantiate the claims.

- **Significance:**
    - By enhancing the ideation capabilities of LLMs, the paper contributes significantly to automating and streamlining the process of research ideation, a critical step toward innovation and advancement in scientific domains.

### Weaknesses
- **Overall weakness:**
    - While the chain structure is useful, it may be seen as a slight modification or straightforward adaptation of known LLM prompting strategies, such as chain-of-thought or retrieval-based prompting.

- **Limited analysis and results:**
    - The authors should do more analysis on the generated ideas. How are LLM-based generation different from human researchers?
    - Is there an assessment of accuracy of generated chains/trends?
    - (Apologies if I missed...) Did the authors include the entire results (with all generated ideas and chains) anywhere in the appendix or supplementary?

- **Limited evaluations:**
    - Is one method of evaluation of generation ideas sufficient? There could be more different ways of evaluations.  
    - Although Idea Arena is novel, its reliance on ELO scores and pairwise rankings might not fully capture qualitative differences in idea originality or real-world feasibility.

- **Lack of Limitation section:**
    - The authors should discuss more on the limitations of the proposed method. Clarify what are achieved and what are not. What else do we need to improve idea generation to achieve the human-level.
    - The CoI’s performance heavily depends on the capability and quality of the underlying LLM, which may vary significantly based on the LLM's training and data scope. The authors may test the same task on local LLMs (i.e. weaker LLMs).
    - Although marketed as budget-friendly ($0.50/idea), detailed cost analysis, including potential computing resources and any scaling issues, would provide more transparency into real-world applicability.
    - While the focus on AI topics for evaluation demonstrates success, the applicability of the CoI approach in other diverse domains remains unexplored and would benefit from broader validation.

- **Comparison with SciAgents:**
    - The concurrent work SciAgents (https://arxiv.org/abs/2409.05556) incorporates graph-based reasoning for idea generation. The authors should compare the proposed method with SciAgents.

### Questions
- How does the CoI agent handle domains or topics with limited literature or less-established research progression? Does it hallucinate?
  
- The current work only considers linear chains. Have you considered more complex chains (multi-path or networked)? Could the CoI framework be adapted to incorporate more dynamic literature, such as rapidly evolving fields where foundational works are continuously supplanted?
  
- What metrics or thresholds determine the construction of a CoI and its length, and how do these parameters affect the quality and novelty of generated ideas?
  
- Is there a systematic way for the CoI agent to identify and mitigate biases in the literature selection that could potentially skew the ideation process?

- Where are the entire results including all the generated ideas?

- Have you considered using graph-based RAG or Knowledge-Graph based approaches?

- Some good services like connected papers can handle a network-based literature survey. Have you considered using their APIs?

- Have you optimised the prompts?

- Have you considered optimising multi-agent systems using ADAS (https://arxiv.org/abs/2408.08435) or AFlow (https://arxiv.org/abs/2410.10762)?

### Soundness
2

### Presentation
2

### Contribution
1
