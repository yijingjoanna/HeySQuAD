# HeySQuAD: A Spoken Question Answering Dataset

https://arxiv.org/abs/2304.13689

Spoken question answering (SQA) systems are critical for digital assistants and other realworld use cases, but evaluating their performance is a challenge due to the importance of human-spoken questions. This study presents a new large-scale community-shared SQA dataset called HeySQuAD, which includes 76k human-spoken questions, 97k machine-generated questions, and their corresponding textual answers from the SQuAD QA dataset. Our goal is to measure the ability of machines to accurately understand noisy spoken questions and provide reliable answers. Through extensive testing, we demonstrate that training with transcribed human-spoken and original SQuAD questions leads to a significant improvement (12.51%) in answering humanspoken questions compared to training with only the original SQuAD textual questions. Moreover, evaluating with a higher-quality transcription can lead to a further improvement of 2.03%. This research has significant implications for the development of SQA systems and their ability to meet the needs of users in realworld scenarios.


Audio Dataset:

human-spoken: https://huggingface.co/datasets/yijingwu/HeySQuAD_human

machine-generated: https://huggingface.co/datasets/yijingwu/HeySQuAD_machine

We also include json dataset in the SQuAD 1.1 format in this repository.

