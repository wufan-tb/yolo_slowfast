import numpy as np
import os,cv2,time,torch,natsort,random,pytorchvideo,subprocess,warnings,argparse
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort


def ava_inference_transform(clip, boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None,thickness=1,fontsize=0.5,fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def deepsort_update(Tracker,pred,xywh,np_img):
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs

def visualize_yolopreds(yolo_preds,id_to_ava_labels,color_map,save_folder):
    for i, (im, pred) in enumerate(zip(yolo_preds.imgs, yolo_preds.pred)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    ava_label = ''
                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box,im,color,text)
        cv2.imwrite(os.path.join(save_folder,yolo_preds.files[i]),im)
        
def extract_video(video_path,img_folder):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            index = int(cap.get(1))
            cv2.imwrite(os.path.join(img_folder,f'{index}.jpg'),frame)
        else:
            break
    cap.release()
    
def clean_folder(folder_name):
    sys_cmd = "rm -rf {}".format(folder_name)
    child = subprocess.Popen(sys_cmd,shell=True)
    child.wait()

def main(config):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 200
    if config.classes:
        model.classes = config.classes
    device = config.device
    imsize = config.imsize
    video_model = slowfast_r50_detection(True).eval().to(device)
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    video_path = config.input
    video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)

    img_path="temp1"
    clean_folder(img_path)
    os.makedirs(img_path,exist_ok=True)
    print("extracting video...")
    extract_video(video_path,img_path)
    imgnames=natsort.natsorted(os.listdir(img_path))

    save_path="temp2"
    clean_folder(save_path)
    os.makedirs(save_path,exist_ok=True)

    process_batch_size = config.process_batch_size    # 10 ~ 30 should be fine, the bigger, the faster
    video_clip_length = config.video_clip_length   # set 0.8 or 1 or 1.2
    frames_per_second = config.frames_per_second     # usually set 25 or 30
    print("processing...")
    a=time.time()
    for i in range(0,len(imgnames),process_batch_size):
        imgs=[os.path.join(img_path,name) for name in imgnames[i:i+process_batch_size]]
        yolo_preds=model(imgs, size=imsize)
        mid=(i+process_batch_size/2)/frames_per_second
        video_clips=video.get_clip(mid - video_clip_length/2, mid + video_clip_length/2 - 0.04)
        video_clips=video_clips['video']
        if video_clips is None:
            continue
        print(i/frames_per_second,video_clips.shape,len(imgs))
        deepsort_outputs=[]
        for i in range(len(yolo_preds.pred)):
            temp=deepsort_update(deepsort_tracker,yolo_preds.pred[i].cpu(),yolo_preds.xywh[i][:,0:4].cpu(),yolo_preds.imgs[i])
            if len(temp)==0:
                temp=np.ones((0,8))
            deepsort_outputs.append(temp.astype(np.float32))
        yolo_preds.pred=deepsort_outputs
        id_to_ava_labels={}
        if yolo_preds.pred[len(imgs)//2].shape[0]:
            inputs,inp_boxes,_=ava_inference_transform(video_clips,yolo_preds.pred[len(imgs)//2][:,0:4],crop_size=imsize)
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)
            with torch.no_grad():
                slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                slowfaster_preds = slowfaster_preds.cpu()
            for tid,avalabel in zip(yolo_preds.pred[len(imgs)//2][:,5].tolist(),np.argmax(slowfaster_preds,axis=1).tolist()):
                id_to_ava_labels[tid]=ava_labelnames[avalabel+1]
        visualize_yolopreds(yolo_preds,id_to_ava_labels,coco_color_map,save_path)
    print("total cost: {:.3f}s, video clips length: {}s".format(time.time()-a,len(imgnames)/frames_per_second))
        
    vide_save_path = config.output
    img_list=natsort.natsorted(os.listdir(save_path))
    im=cv2.imread(os.path.join(save_path,img_list[0]))
    height, width = im.shape[0], im.shape[1]
    video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))

    for im_name in img_list:
        img = cv2.imread(os.path.join(save_path,im_name))
        video.write(img)
    video.release()

    clean_folder(img_path)
    clean_folder(save_path)
    print('saved video to:', vide_save_path)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/wufan/images/video/vad.mp4", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="output.mp4", help='folder to save result imgs, can not use input folder')
    # yolo config
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # slowfast config
    parser.add_argument('--process_batch_size', type=int, default=25, help='10 ~ 30 should be fine, the bigger, the faster')
    parser.add_argument('--video_clip_length', type=float, default=1.2, help='# set 0.8 or 1 or 1.2')
    parser.add_argument('--frames_per_second', type=int, default=25, help='usually set 25 or 30')
    config = parser.parse_args()
    
    print(config)
    main(config)
