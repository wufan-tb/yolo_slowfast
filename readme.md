# Yolov5+SlowFast: Realtime Action Detection

### A realtime action detection frame work based on PytorchVideo. 

#### Here are some details about our modification:

- we choose yolov5 as an object detector instead of detectron2, it is faster and more convenient
- we use a tracker(deepsort) to allocate action labels to all objects(with same ids) in different frames
- our processing speed reached 24.2 FPS at 30 inference barch size (on a single RTX 2080Ti GPU)

> Relevant infomation: [FAIR/PytorchVideo](https://github.com/facebookresearch/pytorchvideo); [Ultralytics/Yolov5](https://github.com/ultralytics/yolov5)

#### Demo comparison betwween original(<-left) and ours(->right).

<img src="./demo/ava_slowfast.gif" style="zoom: 50%;" /><img src="./demo/yolov5+slowfast.gif" style="zoom:60%;" />

## Installation

1. create a new python environment (*optional*):

   ```
   conda create -n env_name python=3.7.11
   ```

2. install requiments:

   ```
   pip install -r requirements.txt
   ```

3. download weights file(ckpt.t7) from [[deepsort]](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) to this folder:

   ```
   ./deep_sort/deep_sort/deep/checkpoint/
   ```

4. test on your video:

   ```
   python yolo_slowfast.py --input {path to your video}
   ```

   The first time execute this command may take some times to download the yolov5 code and it's weights file from torch.hub, keep your network connection.

## References

Thanks for these great works:

[1] [Ultralytics/Yolov5](https://github.com/ultralytics/yolov5)

[2] [ZQPei/deepsort](https://github.com/ZQPei/deep_sort_pytorch) 

[3] [FAIR/PytorchVideo](https://github.com/facebookresearch/pytorchvideo)

[2] AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions. [paper](https://arxiv.org/pdf/1705.08421.pdf)

[3] SlowFast Networks for Video Recognition. [paper](https://arxiv.org/pdf/1812.03982.pdf)

## Citation

If you find our work useful, please cite as follow:

```
{   yolo_slowfast,
    author = {Wu Fan},
    title = { A realtime action detection frame work based on PytorchVideo},
    year = {2021},
    url = {\url{https://github.com/wufan-tb/gmm_dae}}
}
```





