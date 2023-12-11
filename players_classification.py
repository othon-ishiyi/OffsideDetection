#Add detectron2 directory to path
import os, sys
sys.path.insert(0, os.path.abspath('./detectron2'))

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from collections import Counter
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class players_classification:

    def __init__(self):
        '''
        Initializes the panoptic and keypoints model from Detectron2
        '''
        # Using panoptic segmentation to get field area
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.panoptic_pred = DefaultPredictor(cfg)
        self.cfg = cfg

        # keypoint detection model
        cfg_keypoint = get_cfg()   # get a fresh new config
        cfg_keypoint.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg_keypoint.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg_keypoint.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.keypoint_pred = DefaultPredictor(cfg_keypoint)

    def get_image_info(self, file_name):
        '''
        file_name: image filename
        return: image_info object that contains information about player keypoints and teams
        '''
        image_info = {
            'players': get_people_keypoints(cv2.imread(file_name), self.cfg, self.panoptic_pred, self.keypoint_pred),
            'file_name': file_name
        }

        field_players_dict, field_players_id, other_players_id = classify_teams(image_info)

        for id in field_players_id:
            image_info['players'][id]['team'] = f'Team{field_players_dict[id]+1}'
        for id in other_players_id:
            image_info['players'][id]['team'] = 'GK'

        return image_info
    
    def get_image(self, image_info):
        '''
        image_info: image_info dictionary obtained by get_image_info()

        plots the image with all keypoints and teams classified
        Blue: Team1
        Green: Team2
        Others (GK category): Red
        '''
        im = cv2.imread(image_info['file_name'])
        for player in image_info['players']:
            if player['team'] == 'Team1':
                color = (255, 0, 0) # Blue
            elif player['team'] == 'Team2':
                color = (0, 255, 0) # Green
            else:
                color = (0, 0, 255) # Red
            for keypoint in player['keypoints']:
                x, y = keypoint
                x = round(x)
                y = round(y)
                im = cv2.circle(im, center=(x,y), color=color, radius=5, thickness=-1)

        return im

def remove_blobs(mask):
    '''
    mask: imagem com um canal de cor. O valor desse canal é:
        255 se for região de campo
        0 se não for campo

    A função retorna uma mask modificada removendo blobs indesejados detectados originalmente como campo
    '''
    mask_with_border = mask.copy()
    height, width = mask.shape

    #Create a board for the image
    for y in range(height):
        mask_with_border[y,0] = 0
        mask_with_border[y,-1] = 0
    for x in range(width):
        mask_with_border[0,x] = 0
        mask_with_border[-1,x] = 0
    
    contours, _ = cv2.findContours(mask_with_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate areas of blobs
    areas = [cv2.contourArea(cnt) for cnt in contours]

    # Find the index of the largest blob
    max_index = np.argmax(areas)

    # Create a binary mask
    result = np.zeros_like(mask)
    cv2.drawContours(result, contours, max_index, 255, thickness=cv2.FILLED)

    return result

def isolate_field(image, panoptic_cfg, panoptic_predictor):
    '''
    image: cv2 image object
    panoptic_cfg: panoptic_predictor cfg object 
    panoptic_predictor: a panoptic predictor model from detectron2

    This function returns a image without the game audience
    '''
    panoptic_seg, segments_info = panoptic_predictor(image)["panoptic_seg"]
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(panoptic_cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    max_person_area = 0

    for info in segments_info:
        if(info['category_id'] == 18): # playing field
            field_id = info['id']

        if(info['category_id'] == 0):  # person
            max_person_area = max(max_person_area, info['area'])

    dilate_size = round(np.sqrt(max_person_area)*1.5)
    erode_size = round(np.sqrt(max_person_area)/2)

    dilate_box = np.ones((dilate_size, dilate_size))
    erode_box = np.ones((erode_size, erode_size))


    field_mask = np.where(np.array(panoptic_seg.cpu()) == field_id, 255, 0)
    field_mask = np.array(field_mask, np.uint8)
    field_mask = cv2.erode(field_mask, erode_box)
    field_mask = cv2.dilate(field_mask, dilate_box)
    field_mask = remove_blobs(field_mask)

    edited_image = image.copy()
    edited_image[field_mask == 0] = 0

    return edited_image

def get_people_keypoints(image, panoptic_cfg, panoptic_predictor, keypoint_predictor):
    '''
    image: cv2 image object
    panoptic_cfg: panoptic_predictor cfg object 
    panoptic_predictor: a panoptic predictor model from detectron2
    keypoint_predictor: a person keypoint predictor model from detectron2

    This function returns the keypoints of all the players in the field
    '''

    outputs = keypoint_predictor(isolate_field(image, panoptic_cfg, panoptic_predictor))

    people = []

    # Iterate through keypoints and add them to list
    keypoints = outputs["instances"].pred_keypoints.cpu()
    for i, instance_keypoints in enumerate(keypoints):
        person_info = dict()
        person_info['keypoints'] = np.concatenate((instance_keypoints[0:1,:2], instance_keypoints[5:6,:2],
                                            instance_keypoints[11:12,:2], instance_keypoints[13:14,:2],
                                            instance_keypoints[15:,:2], instance_keypoints[14:15,:2],
                                            instance_keypoints[12:13,:2],instance_keypoints[6:7,:2]), axis=0).tolist()
        person_info['player_id'] = i
        people.append(person_info)

    return people

def get_mean_color(image, points: np.array):
    '''
    image: cv2 image object
    points: (N,2) array of points

    returns the mean color in the polygonal region delimited by the points
    '''
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.fillPoly(mask, np.int32([points]), 255)
    mean_color = cv2.mean(image, mask=mask)[:-1]
    return mean_color

def get_color_list(image, points):
    pool_size = 10
    result = []

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.fillPoly(mask, np.int32([points]), 255)

    # Iterate over the image and mask using the pool size
    for y in range(0, image.shape[0], pool_size):
        for x in range(0, image.shape[1], pool_size):
            # Check if the current region in the mask is non-zero (masked)
            masked_pixels = np.sum(mask[y:y + pool_size, x:x + pool_size])
            if masked_pixels > 0:
                # Calculate the mean color in the region
                sum_color = np.sum(image[y:y + pool_size, x:x + pool_size], axis=(0, 1))
                mean_color = sum_color/masked_pixels
                result.append(mean_color)

    return result

def color_amplify(color_array):
    '''
    color_array: array of BGR colors to be transformed

    returns a modified array of BGR colors.
    The modification involves amplifying the difference between the bigger and the lower BGR component
    '''

    sum_colors = np.sum(color_array, axis = 1)[:, None]
    normalized = color_array/sum_colors
    transformer = np.exp(2*normalized) - 1
    transformed = normalized*transformer
    normalized_transformed = transformed/np.sum(transformed, axis = 1)[:, None]

    return np.clip(normalized_transformed * sum_colors, 0, 255)

def cluster_teams(image_info, eps = 55):
    '''
    image_info: dictionary in image_info format
    eps: eps for the gaussian model
    return: list of all classifications

    Uses the DBSCAN clustering model to segregate field players (team1 and 2) and
    other persons (GK, referees and possibly unidentified people)
    '''
    image = cv2.imread(image_info['file_name'])
    player_colors = []
    average_colors = []
    for player in image_info['players']:
        keypoints = player['keypoints']
        points = np.array([keypoints[1], keypoints[2], keypoints[7], keypoints[8]])
        player_colors += get_color_list(image, points)
        average_colors.append(get_mean_color(image, points))

    #amplified = color_amplify(np.array(player_colors))
    db = DBSCAN(eps=eps, min_samples=1)
    db.fit(color_amplify(np.array(player_colors + average_colors)))
    return list(db.labels_[-len(average_colors):])

def get_teams(labels):
    '''
    labels: list from clustes_teams function

    returns the labels corresponding to team1 and team2
    '''
    counter = Counter(labels)
    team1 = max(counter, key=counter.get)
    counter.pop(team1)
    team2 = max(counter, key=counter.get)

    return team1, team2

def segregate_field_players(image_info):
    '''
    image_info: dictionary in image_info format

    returns two lists:
    1) list of all field players
    2) list of all other persons (GK, referees and unidentified persons)
    '''
    clusters = cluster_teams(image_info)
    team1, team2 = get_teams(clusters)

    return [player for player in image_info['players'] if clusters[player['player_id']] in (team1, team2)], [player for player in image_info['players'] if clusters[player['player_id']] not in (team1, team2)]

def lab_transform(color_array):
    '''
    transform an array of BGR colors to an array of colors in LAB format
    '''
    reshaped_array = np.uint8(color_array.reshape(-1, 1, 3))
    return cv2.cvtColor(reshaped_array, cv2.COLOR_BGR2LAB).reshape(-1, 3)

def classify_teams(image_info):
    '''
    image_info: dictionary in image_info format

    returns a dictionary and two lists:
    1) dictionary associating the field player id and its team
    2) id of all field players
    3) id of all other persons (GK, referees and unidentified persons)

    The id of a person is its index in image_info['players']
    '''
    field_players, other_players = segregate_field_players(image_info)
    n_samples = len(field_players)
    image = cv2.imread(image_info['file_name'])
    field_player_colors = []
    for player in field_players:
        keypoints = player['keypoints']
        points = np.array([keypoints[1], keypoints[2], keypoints[7], keypoints[8]])
        field_player_colors.append(get_mean_color(image, points))

    field_lab_colors = lab_transform(np.array(field_player_colors))
    gaussian_model = GaussianMixture(n_components=2, random_state=0)
    gaussian_model.fit(field_lab_colors)
    field_players_result = gaussian_model.predict(field_lab_colors)

    field_players_id = [player['player_id'] for player in field_players]
    other_players_id = [player['player_id'] for player in other_players]

    field_players_dict = dict(zip(field_players_id, field_players_result))

    return field_players_dict, field_players_id, other_players_id


if __name__ == '__main__':
    model = players_classification()
    filename = './teste3.png'
    im_info = model.get_image_info(filename)
    im = model.get_image(im_info)
    cv2.imwrite('./output/output.png', im)




