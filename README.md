# LangGPS
The repository for <a href="https://arxiv.org/pdf/2511.10229" target="_blank">"LangGPS: Language Separability Guided Data Pre-Selection for Joint
Multilingual Instruction Tuning"</a> (AAAI 2026)


<p align="center">
  <img src="Assets/poster.png" width="750px" >
</p>

## Environment Setup

```
1. conda create -n langgps python=3.9
2. conda activate langgps
3. pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
4. cd ./LLaMA_Factory  # Optional: If training is needed under our repo (or you can perform training in your own env)
5. pip install -e ".[torch,metrics,deepspeed]"  # Optional: If training is needed under our repo (or you can perform training in your own env)
6. pip install jsonlines
```

## LangGPS Data Pre-selection

```
# Note: please make sure your vanilla data follows the format of './data/full.json'
1. python 1_get_vectors.py --model_path {path of model under selection} --data_path {path of where your vanilla data stores} --save_path {path of targeted folder for saving}
2. python 2_scoring.py --save_path {same save_path in step 1}
3. python 3_select.py --data_path {path of where your vanilla data stores} --save_path {same save_path in step 1} --percent {pre-selection ratio}
```

## Citation
If you find our work useful, please cite the following paper~
```
@article{ye2025langgps,
  title={LangGPS: Language Separability Guided Data Pre-Selection for Joint Multilingual Instruction Tuning},
  author={Ye, Yangfan and Feng, Xiaocheng and Feng, Xiachong and Huang, Lei and Ma, Weitao and Hong, Qichen and Lu, Yunfei and Tang, Duyu and Tu, Dandan and Qin, Bing},
  journal={arXiv preprint arXiv:2511.10229},
  year={2025}
}
```


