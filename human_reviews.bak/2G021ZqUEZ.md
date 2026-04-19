# From Commands to Prompts: LLM-based Semantic File System for AIOS

- Decision: Accept (Poster)
- Scores: 8, 3, 5, 5

## Abstract
Large language models (LLMs) have demonstrated significant potential in the development of intelligent LLM-based agents. However, when users use these agent applications to perform file operations, their interaction with the file system still remains the traditional paradigm: reliant on manual navigation through precise commands. This paradigm poses a bottleneck to the usability of these systems as users are required to navigate complex folder hierarchies and remember cryptic file names. To address this limitation, we propose an LLM-based Semantic File System (LSFS) for prompt-driven file management in LLM Agent Operating System (AIOS). Unlike conventional approaches, LSFS incorporates LLMs to enable users or agents to interact with files through natural language prompts, facilitating
semantic file management. At the macro-level, we develop a comprehensive API set to achieve semantic file management functionalities, such as semantic file retrieval, file update summarization, and semantic file rollback). At the micro-level, we store files by constructing semantic indexes for them, design and implement syscalls of different semantic operations, e.g., CRUD (create, read, update, delete),
group by, join. Our experiments show that LSFS can achieve at least 15% retrieval accuracy improvement with 2.1× higher retrieval speed in the semantic file retrieval task compared with the traditional file system. In the traditional keyword-based file retrieval task (i.e., retrieving by string-matching), LSFS also performs stably well, i.e., over 89% F1-score with improved usability, especially when the keyword conditions become more complex. Additionally, LSFS supports more advanced file management operations, i.e., semantic file rollback and file sharing and achieves 100% success rates in these tasks, further suggesting the capability of LSFS . The code is available at https://github.com/agiresearch/AIOS-LSFS.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper represents a problem in the current scenario of semantic file matching algorithms; currently, we use the traditional way of semantic matching algorithms based on file name, size, and timestamps. This involved remembering syntax and filenames. This fails in scenarios where two files have similar text; here its hard to distinguish files based on pure string matching. The paper introduces LLM with traditional file systems to do LLM based semantic file management.

LSFS extracts semantic features from the file content and generates corresponding embedding vectors. LSFS incorporates semantic information into its file operations. In Linux, if we need to change a file, i.e., replace a file with another, we need to remember the path, but with LSFS  the users don't need to remember the file name and can talk to LLM to make the changes for them. They have introduced a LLM-based Semantic File System (LSFS), an LSFS parser, and safety insurance mechanisms to the traditional file matching algorithms. The paper has done a great job at explaining the traditional way and modifications done with NLP. They have elaborately explained the API changes they have made over traditional architecture and given diagrams to explain the architecture. Also, they have demonstrated how components of LSFS interact with each other to achieve different functionalities.

Evaluations are carried out based on success, performance, and performance on non-semantic based tasks like file sharing over sample data/files.

### Strengths
The paper touches on an existing problem that exists in the day-to-day lives of developers and Mac OS users of remembering the file names and directory where the files are present and need modification. There is no way to solve this problem at the present. Even while using LLMs, sometimes developers have to hard code the file path for retrieval. LLM based file retrival system is new and useful for anyone who is fed up of the traditional based systems.They did pretty well work on describing the APIs to be used in the new framework and a commendable job in comparing the APIs to the traditional ones. The quality of the paper was good and the presentation with diagrams were very useful to get the context of the paper.
The architecture of the new framework was explained in detail and they have done a good job in explaining how each component in the architecture is integrated with LLMs. Evaluations are carried out based on success, performance, and performance on non-semantic based tasks like file sharing over sample data/files and are pretty easy to follow.

### Weaknesses
The paper could have used an example to walkthrough the implementation. Each component description could have been presented with design diagrams or a flowchart that is easy to understand; visual representation always helps! More evaluations to prove their architecture is better than the traditional ones based on performance, latency, operational burden, and cost. The paper didn't touch on any security concerns while using the LLMS. Are there guardrails in place to restrict the LLMs to scanning through the personal data? One more thing the paper lacked was elaborating on the use cases where this architecture can be used.

### Questions
1) Are there guardrails in place to restrict the LLM to not scan through personal data?
2) How much cost are we saving with the new architecture?
3) Are there any security concerns for using this architecture?
4) How much of an operational overhead is this architecture based on traditional architecture ?
5) What are the other use cases of this architecture in real life scenarios?
6) This seems like an ongoing problem that needs to be resolved; are there any similar existing architectures? Have you looked at those papers?
7) Is there an Andon Cord mechanism to stop the LLM to give out hallucinations and wonky results to the user?
8) While scanning through the files, is the data saved in memory? Does the data contain PII information (ppersonal information about the user)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose, implement and describe an LLM-based semantic file system, where commands are replaced by prompts. They describe APIs and in several experiments compare how this filesystem is used and performs, compared to a conventional file system.

### Strengths
- very interesting and original setup
- includes interesting examples how to use the file system

### Weaknesses
- weak evaluation, based on examples and not on more extensive use cases and human evaluation, evaluation setup not described in detail / cannot be reproduced
- unclear for which users this could be helpful
- unclear how robust the system is when translating NLP input into actual actions
- unclear how the new API maps and extends conventional file handling APIs, and why setting up a new API set is superior to adding some APIs to a conventional file system

### Questions
1. Your systems seems to have advantages for searching and retrieving files based on keyword or semantic search. This could be implemented on top of a conventional file system, why implement a new API for that?
2. Is the accuracy of the LSFS parser of about 90% enough for meaningful work? That means 10% incorrect results. How did you collect the NLP commands for this evaluation?
3. How exactly did you perform the evaluation summarized in table 2?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In this article, the authors based their efforts on the hypothesis that Large language models (LLMs) have the potential to improve file management systems by enabling interactions through natural language rather than traditional manual commands. Following this idea, they proposed LLM-based Semantic File System (LSFS) to address some of the current File System limitations (to the users), by allowing typically semantic file management through natural language prompts. LSFS has through some APIs for semantic file operations, achieves better retrieval accuracy and speed compared to traditional systems or the use of standalone LLMs respectively. It supports complex tasks like semantic file retrieval, rollback, and sharing with high success rates.

### Strengths
+ Well motivated, especially through the Introduction.
+ Clearly written
+ Very nice figures
+ Hot topic nowadays, with many LLM-based applications reshaping the ways we interact with machines

### Weaknesses
### General Remarks
- Even though the Introduction is clear, I'd have liked a more concrete / detailed example, maybe having a finer-grained figure 1 would help.
- The balance between §3 Vs. §4 is unexpected, I would have imagined a more detailed architecture section (§3), explaining the various design choices. Instead, the authors motivated their architecture. In addition to not being referenced, this motivation -to me- would have been better positioned directly in the Introduction. Similarly, the overview of the architecture and the description of Figure 2 would have benefit also the introduction of an example, especially since Fig2.a doesn't contain precise information, but rather e.g. a list of blocks entitled API.
- Overall, for §5, it would have been very interesting and convincing to see an experiment involving users' usage and performances, using a TFS and the presented LSFS. The authors could have reviewed the success rate and the time efficiency of users in both settings together with collecting feedback from them, following the traditional user studies.
- Very disappointed to have the future directions (as Appendix H) not included at all in the main article, but instead, pushed as optional reading material at the very very end of the .pdf…
- Finally, the overall article, from §3 onwards especially, reads a bit like a technical report and lacks -to me- a step-back to better highlight the novelties and the perspectives while guiding the reading with more examples / intuitions directly in the text.

### Minor Comments:
- Abstract "agent applications **TO** perform file operations"
- Introduction (line 95), "a LLM-based" should be **an**
- Table 1 (line 235), typo on "Hybrid retrieval" better to put everything lower-case as the rest of the table
- In §4.1, in Composite Syscall of LSFS, it would be better if the authors could make explicit the composition of atomic calls, i.e. for each entry, adding a generic formula (or examples) of how the composite call is practically chaining the atomic ones.
- In Figure 4, "Please summary all paper from AAA University about LLM" there's a typo in the second word: **summarize**.
- Similarly, still in Figure 4, "Please use file A update the content of file B" misses the word **to** before 'update'.
- In §5.2, (line 450), typo: "vary the the number of rollback versions" remove one **the**.
- In §5.3, (line 478), "Therefore, we make two enhanced versions, named as TFS-grep and TFS-grep* to make the comparison" I would be great to tell there differences in a line instead of relying on the Appendix, so to make the article (before page 10) self-contained.

### Questions
1. Intro (line 57) "The community still lacks a more general LSFS to serve as a common foundation that can be used by various agents on the application-level." Do you have some references backing up this lack? I mean, have people expressed somewhere they'd like/need a FS organized by an intelligent agent, i.e., an LLM here?
2. Related Work (line 133) "Besides, it integrates comprehensive semantic information across all aspects of the file system–from storage and file operations to practical applications" This sentence is a bit vague, could you name some of the semantic information here, so to guide the reader?
3. Could you add a positioning sentence in §2.3 to explain clearly where LSFS delineates from this research axis?
4. In this stage many operations from traditional FS aren't there, from what I understood, this is typically the case for right modification of files or group affiliation… These would be particularly helpful so to "propagate" these rights to any retrievers, preventing typically to see the private/exclusive file of someone else appearing in _my_ search results, wouldn't it?
5. In §4,2 (line 294), the authors mentioned that supervisor updates "periodically" what is the period between each check and therefore how expensive resource-wise is it? Did the authors check various values for this, searching for the sweet-spot between resource-consumption and freshness of the LFSF data? Also how does it scale in terms of file number and disk footprint?
6. Overall, §4.4 seems to be more or less an NL2CSV tool, filling fields of a JSON, right? In such case, this is something that the community has been exploring a lot these past two years, so maybe adding some pointers wouldn't hurt. This goes also for the §5.1 associated with RQ1.
7. Are authors considering releasing their test data for §5.1? Also, it would be good to have some examples in the body of the article.
8. In §5.2 why no QWen or Gemma in the experimental run for Table 2, and no Gemma in Figure 6?
9. In §5.2 still, what about very large number of files?
10. Ibid., same for the number of versions?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces an LLM-based Semantic File System (LSFS), designed to improve file management through natural language prompts, rather than traditional command-based interactions. LSFS integrates large language models (LLMs) to facilitate semantic file operations like retrieval, summarization, and rollback. At its core, LSFS uses a vector database to create semantic indexes for files, enabling high-level file operations that consider the content and context of files. It also includes a comprehensive set of APIs that allow complex operations, such as CRUD, grouping, and semantic retrieval, to be executed through natural language prompts. Experimental results show that LSFS outperforms traditional systems in retrieval accuracy (with a 15% improvement) and speed (2.1x faster), proving especially effective for semantic file tasks that go beyond conventional keyword searches.

### Strengths
S1. Semantic file systems enhance file management by incorporating content context, enabling more intuitive and effective operations, which is an important direction.

S2. LSFS simplifies interactions with file system, making file management more accessible and user-friendly.

S3. Integrating LLMs in system-level tasks expands functionality, enabling intelligent, responsive, and user-focused file extraction.

### Weaknesses
W1. The motivation is not concretely convincing, especially the first challenge mentioned in Introduction.

In the Intro section, the authors mentioned that "For instance, if two files have similar content–such as different versions of the same document–traditional file systems lack the capability to organize or retrieve these files based on their content similarity." Why the files should be organized by the similarity of their content? What are the benefits and what are the practical application scenarios? It would be better to at least add one short discussion or a few examples. 

W2. This paper does not point out what key problem they want to solve. Compared to a research paper, it seems more like a technical report.

W3. The experimental setting is questionable. No baselines and introduction of datasets.

W4. The experimental results need more explanation. The traditional file system needs to execute commands to extract files, which should be faster than calling one LLM, even though these LLMs are light-weight. The authors should also list the inference time of LLMs, which should be also counted as the interaction time between users and the file system. Then the authors can also list the time that users manually write commands, which should be a good point to prove the point that -- LSFS can not only speed up file retrieving accuracy and speed, but also can reduce the interaction time between users and file system.

W5. Safety insurance mechanisms is pointed out as one contribution, however, there is no description of this mechanism and no experimental comparison between the performance of LSFS with and without the safety insurance mechanisms.

### Questions
W1 - W5

### Soundness
2

### Presentation
3

### Contribution
2
