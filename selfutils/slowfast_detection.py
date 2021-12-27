import warnings
warnings.filterwarnings("ignore",category=UserWarning)

import os,sys,torch,cv2,pytorchvideo,time

from functools import partial
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection # Another option is slowfast_r50_detection, slow_r50_detection

from visualization import VideoVisualizer

# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    with torch.no_grad():
        predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    # print(predictions._fields.keys())
    # print("ROI info:",boxes.tensor.shape,scores.shape,predictions.pred_classes.shape)
    # for i in range(predictions.pred_classes.shape[0]):
    #     conf,pred=predictions.onehot_labels[i].max(-1)
    #     print(predictions.pred_classes[i],conf,pred)
    predicted_boxes = boxes[np.logical_and(classes!=-1, scores>0.5 )].tensor.cpu() # only person
    return predicted_boxes

# ## Define the transformations for the input required by the model
def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):

    boxes = np.array(boxes)
    roi_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )
    
    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )
    
    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )
    
    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def main(args):
    # ## load slow faster model
    device = args.device # or 'cpu'
    video_model = slowfast_r50_detection(True) # Another option is slowfast_r50_detection
    video_model = video_model.eval().to(device)

    # ## Load an off-the-shelf Detectron2 object detector
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # Create an id to label name mapping
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('selfutils/ava_action_list.pbtxt')
    # Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
    video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.7)

    # Load the video
    encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(args.input)
    print('Completed loading encoded video.')

    # Video predictions are generated at an internal of 1 sec from 90 seconds to 100 seconds in the video.
    time_stamp_range = range(0,int(encoded_vid.duration//1),1) # time stamps in video for which clip is sampled. 
    clip_duration = 1 # Duration of clip used for each inference step.
    gif_imgs = []
    a=time.time()
    for time_stamp in time_stamp_range:    
        print("processing for {}th sec".format(time_stamp))
        
        # Generate clip around the designated time stamps
        inp_imgs = encoded_vid.get_clip(
            time_stamp , # start second
            time_stamp + clip_duration # end second
        )
        inp_imgs = inp_imgs['video']
        # print("clips shape for slowfaster:",inp_imgs.shape)
        # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
        # We use the the middle image in each clip to generate the bounding boxes.
        inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
        inp_img = inp_img.permute(1,2,0)
        # print("img shape for faster rcnn:",inp_img.shape)
        
        # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
        predicted_boxes = get_person_bboxes(inp_img, predictor) 
#         print("ROI boxes (only person):",predicted_boxes)
        if len(predicted_boxes) == 0: 
            print("no detected at time stamp: ", time_stamp)
            continue
            
        # Preprocess clip and bounding boxes for video action recognition.
#         print(predicted_boxes)
        inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy(), crop_size=args.imsize)
        # Prepend data sample id for each bounding box. 
        # For more details refere to the RoIAlign in Detectron2
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
        
        # Generate actions predictions for the bounding boxes in the clip.
        # The model here takes in the pre-processed video clip and the detected bounding boxes.
        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0).to(device)
        # print("slowfaster's inputs shape:",len(inputs),inputs[0].shape,inputs[1].shape)
        with torch.no_grad():
            preds = video_model(inputs, inp_boxes.to(device))

        preds = preds.to('cpu')
        # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
        preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
        
        # Plot predictions on the video and save for later visualization.
        inp_imgs = inp_imgs.permute(1,2,3,0)
        
        inp_imgs = inp_imgs/255.0
        # print("pred shapes:",preds.shape,predicted_boxes.shape)
        out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes,repeat_frame=1)
        gif_imgs += out_img_pred

    print("Finished generating predictions.")
    print("total cost: {:.3f}s, video clips length: {}s".format(time.time()-a,len(time_stamp_range)))

    # ## Save predictions as video
    height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]

    vide_save_path = os.path.join(args.output,'output.mp4')
    video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))

    for image in gif_imgs:
        img = (255*image).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()

    print('video saved to:', vide_save_path)
    
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data/VAD/SHTech/testing/videos/01_0015.mp4')
    parser.add_argument('--output', type=str, default='videos')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--imsize', type=int, default=384)
    args = parser.parse_args()
    
    main(args)




