# FHE-Coder: Benchmarking Secure Agentic Code Generation for Fully Homomorphic Encryption

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Fully Homomorphic Encryption (FHE) is a foundational technology for confidential computing, yet its practical adoption remains limited by the need for specialized cryptographic expertise and error-prone parameter configuration. To lower this barrier, we investigate whether Large Language Model (LLM) agents can reliably generate secure FHE code from natural-language specifications. We present FHE-Coder, a three-phase agentic framework that addresses the key failure modes of FHE code generation: semantic ambiguity, API misuse, and cryptographic insecurity. The framework integrates (1) a Prompt Formalizer that structures user intent and enforces secure parameterization, (2) a specialized retrieval-augmented generation (RAG) module that supplies scheme-specific API and documentation knowledge, and (3) an automated Security Verifier that performs iterative validation and feedback to detect and correct cryptographic flaws. We evaluate FHE-Coder across four leading LLMs on a benchmark of ten FHE programming tasks spanning increasing functional and security complexity. While baseline agents frequently produce code that compiles and passes functional tests, they often violate security constraints or misuse cryptographic parameters. In contrast, FHE-Coder consistently generates solutions that are compilable, functionally correct, and verifiably secure across schemes including TFHE and CKKS. Our work establishes a systematic methodology and benchmark for agentic FHE code generation, providing a practical step toward democratizing secure computation without compromising cryptographic guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TFHE-CODER, a three-stage agentic framework for secure Fully Homomorphic Encryption (FHE) programming. Unlike conventional code generation approaches that rely solely on post-hoc human inspection, TFHE-CODER enables large language models (LLMs) to proactively integrate security constraints during the generation process. The framework comprises three key components: FHE Prompt Formalizer, FHE API RAG Retriever, and FHE Security Verifier. To systematically evaluate the framework’s effectiveness, the authors constructed the TFHE-CODER Benchmark, consisting of ten FHE programming tasks. Experimental results demonstrate that large language models can autonomously generate FHE programs that are not only functionally correct and compilable but also verifiably secure.

### Strengths
1. The authors propose an end-to-end agentic framework explicitly designed around security. By integrating prompt formalization, retrieval, and automated security verification into a closed feedback loop, the system treats security correctness as a primary objective rather than a post-generation check.
2. The framework’s security verifier can automatically identify multiple classes of critical vulnerabilities and generate structured Formal Error Reports, enabling the agent to iteratively refine and correct insecure code.
3. The paper introduces a novel metric specifically tailored for security-critical code generation. This metric complements traditional functional correctness measures and prevents misleading conclusions based solely on task completion accuracy.

### Weaknesses
1. The RAG dictionary requires offline preparation and manual expert involvement, which limits the framework’s degree of automation and scalability.
2. The paper does not provide detailed measurements of computational overhead, runtime latency, or resource consumption introduced by the multi-stage verification process, making it difficult to assess practical efficiency.
3. While the framework performs well on simple FHE operations, its success rate drops significantly on more complex tasks.

### Questions
Please refer to my comments on weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets an important problem of automatically generating secure API calls and parameters for Fully Homomorphic Encryption over Torus (TFHE). It proposes to leverage the power of agentic large language models (LLMs) to generate TFHE API calls and parameters, with the assistance of Retrieval-Augmented Generation (RAG) and post validation techniques. Evaluation on a set of tasks shows that the method outperforms vanilla prompting and Chain-of-Thought (CoT) prompting.

### Strengths
- Addresses an important and practical problem of generating secure TFHE API calls and parameters.
- Clear writing and well-structured presentation.

### Weaknesses
- The evaluation tasks are limited in complexity, diversity, and reproducibility.
- No guarantee on the security and functional correctness of the generated programs.
- Limited novelty in the technical contributions, where RAG and post validation are not new.

### Questions
Generating secure API calls and parameters for TFHE is an important problem, as developers often lack cryptographic expertise in programming TFHE applications. The proposed method leverages agentic LLMs with enhancements like RAG and post validation to improve the generation accuracy. However, there are several concerns over the empirical evaluation and technical contributions.

The evaluation of the proposed method is limited. The benchmark tasks are simple Machine Learning (ML) tasks, which are as simple as vector/matrix multiplications and basic ML models like MLP or CNN. These tasks are often too simple to demonstrate the effectiveness of the proposed method, as human developers can also easily implement these tasks. It would be more convincing to show the effectiveness of the method on more complex and diverse tasks, including natural language processing models, transformer-based models, and other non-ML tasks. Additionally, the reproducibility of the evaluation is not well demonstrated. Although the paper mentions that the seeds are fixed, the randomness in LLM generation can still lead to variance in results. It would be helpful to repeat each experiment multiple times and report the average performance with confidence intervals.

There is limited guarantee on the security and functional correctness of the generated TFHE programs. While the post validation step is a safe-guard to filter out incorrect programs, the paper cannot ensure that the validation conditions are correct and comprehensive. For instance, how do we know that the verification conditions in Dafny code are sufficient to eliminate all security vulnerabilities? Also, it is not clear that how the proposed method can ensure that the generated conditions are aligned with the developer's intent. More discussion on these aspects would be helpful.

The technical contributions of the paper appear incremental. The use of RAG and post validation are not new techniques in the context of LLMs. This work is a simple application of such techniques to the TFHE domain. The contributions would be stronger if such a technique can also extend to other cryptographic domains other than TFHE, such as MPC, partial HE, or ZK proofs.

### Soundness
2

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
This paper investigates the potential of Large Language Model (LLM) agents to automate the generation of secure Fully Homomorphic Encryption over the Torus (TFHE) code from natural language. (Fig 2 of the paper gives a giid outline of the method.) The role of LLMs is to respond to user intent, configure secure parameters,  adapt the API, and leverage an automated security verifier that provides iterative feedback to correct cryptographic flaws. The system produces code that is compilable, functionally correct, and verifiably secure.  The focus on TFHE, as opposed to other FHE schemes, is based on the  efficient gate bootstrapping and functional bootstrapping, which allow computation of arbitrary functions while refreshing noise.

### Strengths
This seems to be a good case study of using LLMs, in combination with structured workflow and other tools, to solve a specific programming problem with modest scope.  The focus on TFHE will be a plus for specific audience.

### Weaknesses
From an application perspective, the focus on TFHE is specialized, limiting the apparent audience for this work.  How many people need TFHE code and how much variety is there among possible users? 
From a scientific perspective, it is not clear how much of the difficulty, or how much of the apparent success of the approach is due to particular characteristics of TFHE.   
Further, it is hard to see from the conference-length writeup how broad the solution is (how different are different requests for TFHE code?) and how compelling the verification is.   A little more information on how fully the code is verified would be helpfui.

### Questions
What would be needed to generalize this work beyond TFHE?  There are likely other programming tasks where LLMs could have relevant knowledge about parameters and there could be appropriate verification techniques.  If someone provided versions of FHE Prompt Formalizer, FHE API RAG Retriever and FHE Security Verifier. for another problem, would hte method be likely to work?  Why or why not?

### Soundness
3

### Presentation
3

### Contribution
3
