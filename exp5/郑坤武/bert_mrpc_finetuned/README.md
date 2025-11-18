本实验在本地目录 `bert_mrpc_finetuned` 中成功保存了微调后的 BERT 模型，其中包含 `config.json、tokenizer_config.json、vocab.txt、special_tokens_map.json ` 以及权重文件 `model.safetensors`.
由于 `model.safetensors` 体积较大，不适合作为作业附件上传，故提交时仅保留源代码和实验报告，并在文中附上本地模型文件夹结构的截图以证明模型已成功保存。

如需复现，只需在相同环境中运行所附代码即可重新得到该权重文件。