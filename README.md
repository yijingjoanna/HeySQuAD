# HeySQuAD: A Spoken Question Answering Dataset

https://arxiv.org/abs/2304.13689

Spoken question answering (SQA) systems are critical for digital assistants and other realworld use cases, but evaluating their performance is a challenge due to the importance of human-spoken questions. This study presents a new large-scale community-shared SQA dataset called HeySQuAD, which includes 76k human-spoken questions, 97k machine-generated questions, and their corresponding textual answers from the SQuAD QA dataset. Our goal is to measure the ability of machines to accurately understand noisy spoken questions and provide reliable answers. Through extensive testing, we demonstrate that training with transcribed human-spoken and original SQuAD questions leads to a significant improvement (12.51%) in answering humanspoken questions compared to training with only the original SQuAD textual questions. Moreover, evaluating with a higher-quality transcription can lead to a further improvement of 2.03%. This research has significant implications for the development of SQA systems and their ability to meet the needs of users in realworld scenarios.


## Audio Dataset:

human-spoken: https://huggingface.co/datasets/yijingwu/HeySQuAD_human

machine-generated: https://huggingface.co/datasets/yijingwu/HeySQuAD_machine

```python

from datasets import load_dataset

dataset = load_dataset("yijingwu/HeySQuAD_human")
dataset = load_dataset("yijingwu/HeySQuAD_machine")

```

## Json Dataset
We also include json dataset in the SQuAD 1.1 format in this repository for fine-tuning and evaluating Question-Answering model using trainscribed questions.

An example of fine-tuning roberta-large model using human-transcribed dataset:

```python
python /transformers/examples/legacy/question-answering/run_squad.py \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --do_eval \
  --do_train \
  --do_lower_case \
  --train_file /HeySQuAD_train/train-common-human-transcribed-48849.json \
  --predict_file /HeySQuAD_test/dev-common-human-transcribed-1002.json \
  --per_gpu_train_batch_size=4 \
  --per_gpu_eval_batch_size=4 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --data_dir /HeySQuAD_json/ \
  --output_dir roberta-large-human
```



## Citation:
If you use HeySQuAD, please cite the following paper:
```
@misc{wu2023heysquad,
      title={HeySQuAD: A Spoken Question Answering Dataset}, 
      author={Yijing Wu and SaiKrishna Rallabandi and Ravisutha Srinivasamurthy and Parag Pravin Dakle and Alolika Gon and Preethi Raghavan},
      year={2023},
      eprint={2304.13689},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

