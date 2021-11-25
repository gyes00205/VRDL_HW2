# VRDL HW2

## 1. inference notebook and dependencies
The [inference.ipynb](https://colab.research.google.com/drive/1Mni7VdotHETDFupKyUomC5I4f_2Dheoq?usp=sharing) show the instructions that can reproduce inference time and generate answer.json in **Colab**.
In the inference.ipynb, you will install the required dependencies like **TF-object-detection-API**. Please follow the instructions in inference.ipynb.

![](https://i.imgur.com/4r2XT6w.png)


## 2. Training code
In this homework, I use **TF-object-detection-API** to train EfficientDet D2 model.
Before training, you should download [efficientdet d2](https://drive.google.com/drive/folders/1evO_zwJSFwO7iAgsJ6fvfZRXg0Xhb52h?usp=sharing) weights from TF API and [training data](https://drive.google.com/drive/folders/1JPQYq7o-51E0aP0yfz-85UW8m9M09JQ-?usp=sharing) that converted to tfrecord format.
**model_main_tf2.py** is provided by TF API.
```
!python model_main_tf2.py \
--model_dir=training_configs/efficientdet_d2 \
--pipeline_config_path=training_configs/efficientdet_d2/pipeline.config
```

```
VRDL_HW2
├───data
│   ├───coco_train.record
│   ├───coco_val.record
│   └───label_map.pbtxt
└───pre-trained-models
    └───efficientdet_d2_coco17_tpu-32
        ├───checkpoint
        ├───saved_model
        └───pipeline.config
```

## 3. Export model
**export_main_v2.py** is provided by TF API.
```
!python exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path training_configs/efficientdet_d2/pipeline.config \
--trained_checkpoint_dir training_configs/efficientdet_d2/  \
--output_directory efficientdet_d2_ckpt202
```

```
VRDL_HW2
└───efficientdet_d2_ckpt202
    ├───checkpoint
    ├───saved_model
    └───pipeline.config
```
## 4. Evaluation
```
!python model_main_tf2.py \
--model_dir=training_configs/efficientdet_d2 \
--pipeline_config_path=training_configs/efficientdet_d2/pipeline.config \
--checkpoint_dir=training_configs/efficientdet_d2
```
After evaluation, we can see the results.
![](https://i.imgur.com/hEhLXF6.png)


## 5. Pre-trained models
You can follow the [inference.ipynb](https://colab.research.google.com/drive/1Mni7VdotHETDFupKyUomC5I4f_2Dheoq?usp=sharing) to download the pre-trained model and reproduce result.
```
!python detect.py \
--saved_model_path=efficientdet_d2_ckpt202/ \
--test_path=test \
--output_path=output_image \
--min_score_thresh=0.0 \
--label_map=data/label_map.pbtxt
```

## Reference
1. [TF-object-detection-API](https://github.com/tensorflow/models)
2. [mmdetection](https://mmdetection.readthedocs.io/en/v2.18.1/2_new_data_model.html)
3. [Access SVHN data in Python](https://www.vitaarca.net/post/tech/access_svhn_data_in_python/)