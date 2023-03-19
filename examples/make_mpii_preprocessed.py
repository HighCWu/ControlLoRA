import os
import cv2
import glob
import math
import torch
import datasets
import scipy.io
import jsonlines
import numpy as np

from tqdm import tqdm
from PIL import Image
from clip_interrogator import Config, Interrogator
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict


MAIN_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')

img_dir = f'{MAIN_DIR}/data/MPII/cut/images'
guide_dir = f'{MAIN_DIR}/data/MPII/cut/guides'
os.makedirs(img_dir, exist_ok=True)
os.makedirs(guide_dir, exist_ok=True)

assert os.path.exists(f'{MAIN_DIR}/data/MPII/images')
assert os.path.exists(f'{MAIN_DIR}/data/MPII/mpii_human_pose_v1_u12_1.mat')

jlname = f"{MAIN_DIR}/data/mpii_preprocessed.jsonl"
def make_prompt():
    imgs = sorted(glob.glob(f'{MAIN_DIR}/data/MPII/images/*.jpg'))
    if not os.path.exists(jlname + '.done'):
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        with jsonlines.open(jlname, 'w', flush=True) as w:
            for img in tqdm(imgs):
                basename = os.path.basename(img)
                with torch.no_grad():
                    image = Image.open(img).convert('RGB')
                    prompt = ci.interrogate_fast(image)
                    w.write({
                        "image": basename, 
                        "text": prompt}
                    )
        with open(jlname + '.done', 'r+', encoding='utf-8') as f:
            f.write('Done.')

# mpii
# 0 - r ankle, 
# 1 - r knee, 
# 2 - r hip,
# 3 - l hip,
# 4 - l knee, 
# 5 - l ankle, 
# 6 - mid hip， 
# 7 - lower neck，
# 8 - upper neck, 
# 9 - head top,
# 10 - r wrist,
# 11 - r elbow, 
# 12 - r shoulder, 
# 13 - l shoulder,
# 14 - l elbow, 
# 15 - l wrist

# coco
# { 
#     0: "nose", 
#     1: "left_eye", 
#     2: "right_eye", 
#     3: "left_ear", 
#     4: "right_ear", 
#     5: "left_shoulder", 
#     6: "right_shoulder", 
#     7: "left_elbow", 
#     8: "right_elbow", 
#     9: "left_wrist", 
#     10: "right_wrist", 
#     11: "left_hip", 
#     12: "right_hip", 
#     13: "left_knee", 
#     14: "right_knee", 
#     15: "left_ankle", 
#     16: "right_ankle" 
# }

# body 18
# {0,  "Nose"}, 0 , mpii: (8~upper neck + 9~head top) * 0.5
# {1,  "Neck"}, (RShoulder + LShoulder) * 0.5, mpii: 7
# {2,  "RShoulder"}, 6, mpii: 12
# {3,  "RElbow"}, 8, mpii: 11
# {4,  "RWrist"}, 10, mpii: 10
# {5,  "LShoulder"}, 5, mpii: 13
# {6,  "LElbow"}, 7, mpii: 14
# {7,  "LWrist"}, 9, mpii: 15
# {8,  "RHip"}, 12, mpii: 2
# {9, "RKnee"}, 14, mpii: 1
# {10, "RAnkle"}, 16, mpii: 0
# {11, "LHip"}, 11, mpii: 3
# {12, "LKnee"}, 13, mpii: 4
# {13, "LAnkle"}, 15, mpii: 5
# {14, "REye"}, 2, mpii: (8~ * 0.4 + 9~ * 0.6), ~x * 0.8 + 12~x * 0.2
# {15, "LEye"}, 1, mpii: (8~ * 0.4 + 9~ * 0.6), ~x * 0.8 + 13~x * 0.2
# {16, "REar"}, 4, mpii: (8~ * 0.45 + 9~ * 0.55), ~x * 0.7 + 12~x * 0.3
# {17, "LEar"}, 3 # end of coco, mpii: (8~ * 0.45 + 9~ * 0.55), ~x * 0.7 + 13~x * 0.3

# body 25
# {0,  "Nose"}, 0
# {1,  "Neck"}, # (RShoulder + LShoulder) * 0.5
# {2,  "RShoulder"}, 6
# {3,  "RElbow"}, 8
# {4,  "RWrist"}, 10
# {5,  "LShoulder"}, 5
# {6,  "LElbow"}, 7
# {7,  "LWrist"}, 9
# {8,  "MidHip"},  (RHip + LHip) * 0.5
# {9,  "RHip"}, 12
# {10, "RKnee"}, 14
# {11, "RAnkle"}, 16
# {12, "LHip"}, 11
# {13, "LKnee"}, 13
# {14, "LAnkle"}, 15
# {15, "REye"}, 2
# {16, "LEye"}, 1
# {17, "REar"}, 4
# {18, "LEar"}, 3 # end of coco
# {19, "LBigToe"},
# {20, "LSmallToe"},
# {21, "LHeel"},
# {22, "RBigToe"},
# {23, "RSmallToe"},
# {24, "RHeel"},
# {25, "Background"}

# draw the body keypoint and lims
def draw_bodypose(img, poses, model_type = 'coco'):
    if len(poses[0]) == 16:
        model_type = 'coco'
        new_poses = []
        for n in range(len(poses)):
            pose = poses[n]
            new_pose = pose[[0,7,12,11,10,13,14,15,2,1,0,3,4,5,0,0,0,0]].copy()
            new_pose[0] = (pose[8] + pose[9]) * 0.5
            new_pose[14] = pose[8] * 0.4 + pose[9] * 0.6
            new_pose[15] = pose[8] * 0.4 + pose[9] * 0.6
            new_pose[16] = pose[8] * 0.45 + pose[9] * 0.55
            new_pose[17] = pose[8] * 0.45 + pose[9] * 0.55
            new_pose[14,0] = new_pose[14,0] * 0.8 + pose[12,0] * 0.2
            new_pose[15,0] = new_pose[15,0] * 0.8 + pose[13,0] * 0.2
            new_pose[16,0] = new_pose[16,0] * 0.65 + pose[12,0] * 0.35
            new_pose[17,0] = new_pose[17,0] * 0.65 + pose[13,0] * 0.35
            new_pose = [p if p[2] > 0.9 else [-1,-1,-1] for p in new_pose]
            new_poses.append(new_pose)
        poses = new_poses
    
    if len(poses[0]) == 17:
        model_type = 'coco'
        new_poses = []
        for n in range(len(poses)):
            pose = poses[n]
            new_pose = pose[[0,0,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]]
            new_pose[1] = (pose[5] + pose[6]) * 0.5
            new_poses.append(new_pose)
        poses = new_poses

    stickwidth = 4
    
    limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
                   [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], \
                   [0, 15], [15, 17]]
    njoint = 18
    if model_type == 'body_25':    
        limbSeq = [[1,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],\
                            [10,11],[8,12],[12,13],[13,14],[0,15],[0,16],[15,17],[16,18],\
                                [11,24],[11,22],[14,21],[14,19],[22,23],[19,20]]
        njoint = 25

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],\
                  [255,255,255],[170,255,255],[85,255,255],[0,255,255]]
    for i in range(njoint):
        for n in range(len(poses)):
            pose = poses[n][i]
            if pose[2] <= 0:
                continue
            x, y = pose[:2]
            cv2.circle(img, (int(x), int(y)), 4, colors[i], thickness=-1)
    
    for pose in poses:
        for limb,color in zip(limbSeq,colors):
            p1 = pose[limb[0]]
            p2 = pose[limb[1]]
            if p1[2] <= 0 or p2[2] <= 0:
                continue
            cur_canvas = img.copy()
            X = [p1[1],p2[1]]
            Y = [p1[0],p2[0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
   
    return img

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def get_all_pose():
    pose_dict = {}
    # index to joint name conversion
    mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                    6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 11: 'relb', 10: 'rwri', 9: 'head',
                    12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

    # This template will then be updated as and when we read ground truth
    mpii_template = dict([(mpii_idx_to_jnt[i], []) for i in range(16)])

    # Load the mat file.
    matlab_mpii = scipy.io.loadmat(f'{MAIN_DIR}/data/MPII/mpii_human_pose_v1_u12_1.mat', struct_as_record=False)['RELEASE'][0, 0]
    num_images = annotation_mpii = matlab_mpii.__dict__['annolist'][0].shape[0]

    # Load images and GT in batches of 200
    initial_index = 0
    batch = 1e9 # 200

    # Initialize empty placeholder
    img_dict = {'mpii': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': []}}
    
    # Iterate over each image
    for img_idx in tqdm(range(initial_index, min(initial_index + batch, num_images))):
        annotation_mpii = matlab_mpii.__dict__['annolist'][0, img_idx]
        train_test_mpii = matlab_mpii.__dict__['img_train'][0, img_idx].flatten()[0]
        person_id = matlab_mpii.__dict__['single_person'][img_idx][0].flatten()

        # Load the individual image. Throw an exception if image corresponding to filename not available.
        img_name = annotation_mpii.__dict__['image'][0, 0].__dict__['name'][0]
        # Flag is set to true if atleast one person exists in the image with joint annotations.
        # If Flag == True, then the image and GT is considered for visualization, else skip
        annotated_person_flag = False
        poses = []
        
        # Iterate over persons
        for person in (person_id - 1):
            try:
                annopoints_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0]
                num_joints = annopoints_img_mpii.__dict__['point'][0].shape[0]

                # Iterate over present joints
                pose = {id_: [0,0,-1] for id_ in range(16)}
                for i in range(num_joints):
                    x = annopoints_img_mpii.__dict__['point'][0, i].__dict__['x'].flatten()[0]
                    y = annopoints_img_mpii.__dict__['point'][0, i].__dict__['y'].flatten()[0]
                    id_ = annopoints_img_mpii.__dict__['point'][0, i].__dict__['id'][0][0]
                    vis = annopoints_img_mpii.__dict__['point'][0, i].__dict__['is_visible'].flatten()
                    pose[id_] = [x,y,1]
                pose = sorted(pose.items())
                pose = [v for _, v in pose]
                poses.append(pose)

                annotated_person_flag = True if num_joints > 0 else False
            except KeyError:
                # Person 'x' could not have annotated joints, hence move to person 'y'
                continue
        
        if annotated_person_flag:
            poses = np.asarray(poses)
            
        pose_dict[img_name] = poses

    return pose_dict

def image_map(examples):
    image = examples['image']
    basename = os.path.basename(image)
    img_path = os.path.join(f'{MAIN_DIR}/data/MPII/images', basename)
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    ratio = 512 / h
    w, h = int(round(w * ratio)), int(round(h * ratio))
    img = img.resize([w, h], Image.LANCZOS)
    canvas = np.zeros_like(np.asarray(img))
    poses = np.asarray(pose_dict[basename])
    if len(poses) > 0:
        poses[:,:,:2] = poses[:,:,:2] * ratio
        
        xs = poses[:,:,0]
        minx = int(max(np.min(xs) - h//16, 0))
        maxx = int(min(np.max(xs) + h//16, w))
        if maxx - minx < h:
            cenx = (minx + maxx) / 2
            minx = max(0, int(cenx - h // 2))
            maxx = minx + h
            if maxx > w:
                maxx = w
                minx = w - h
        poses[:,:,0] -= minx
        img = img.crop((minx, 0, maxx, h))
        canvas = canvas[:,minx:maxx]
        canvas = draw_bodypose(canvas, poses)
    guide = Image.fromarray(canvas)

    img_path = f'{img_dir}/{basename}'
    guide_path = f'{guide_dir}/{basename}'
    img.save(img_path)
    guide.save(guide_path)

    return { 'image': img_path, 'guide': guide_path }


make_prompt()
pose_dict = get_all_pose()

dataset = Dataset.from_json(jlname)
dataset = dataset.map(image_map)
dataset = dataset.cast_column("image", datasets.Image(decode=True))
dataset = dataset.cast_column("guide", datasets.Image(decode=True))
dataset = DatasetDict(train=dataset)
dataset.save_to_disk(os.path.join(
    MAIN_DIR, 
    'data/mpii-preprocessed'))
