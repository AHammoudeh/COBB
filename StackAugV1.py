from pycocotools.coco import COCO
import cv2
import json
import tqdm
import glob
import pylab
import random
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.decomposition import PCA
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity
from scipy.ndimage import binary_dilation, convolve
from matplotlib.patches import Polygon as MatplotlibPolygon


def remove_crowded(annotations):
  annotations_after = []
  for ann in annotations:
    if ann['iscrowd'] == 0:
      annotations_after.append(ann)
  return annotations_after

def coco_bbox_to_coords(bbox):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    return x_min, y_min, x_max, y_max

def Is_within_boundaries(ann, img_info, margin_ratio=0.1):
  img_height = img_info['height']
  img_width = img_info['width']
  x_min, y_min, x_max, y_max = coco_bbox_to_coords(ann['bbox'])
  c1 = x_min> (margin_ratio)*img_width
  c2 = y_min> (margin_ratio)*img_height
  c3 = x_max< (1-margin_ratio)*img_width
  c4 = y_max< (1-margin_ratio)*img_height
  within_bounaries = (c1 and c2) and (c3 and c4)
  return within_bounaries

def remove_beyond_boudaries(annotations,img_info, margin_ratio=0.1):
  annotations_after = []
  for ann in annotations:
    if Is_within_boundaries(ann, img_info, margin_ratio):
      annotations_after.append(ann)
  return annotations_after

def get_segmentation_mask(anns,img_info, coco):
    image_size = (img_info['height'],  img_info['width'])
    # Initialize an empty mask
    Full_mask = np.zeros(image_size, dtype=np.uint8)
    for ann in anns:
        if 'segmentation' in ann:
            # Convert segmentation to mask
            mask_1obj = coco.annToMask(ann)
            #hull, image_width, image_height, object_name, mask_1obj = annotation_items(ann, coco)
            Full_mask = Full_mask + mask_1obj  # Combine masks
    return Full_mask

def compute_inContact_ratio(object_mask, all_objects_mask):
    """
    Computes the overlap ratio of the perimeter of object1 with other objects.
    Parameters:
    - object_mask: 2D numpy array (binary) representing the single object mask.
    - all_objects_mask: 2D numpy array (binary) representing all objects in the scene.
    Returns:
    - Overlap ratio (float)
    """
    # Define a 3x3 structuring element for detecting the perimeter
    struct_elem = np.ones([7,7])
    # Compute the perimeter of object1
    dilated_object = binary_dilation(object_mask, structure=struct_elem)
    perimeter = dilated_object & ~object_mask
    # Compute the overlapping pixels by checking if perimeter pixels touch other objects
    inContact_pixels = perimeter & (all_objects_mask & ~object_mask)
    # Compute overlap ratio
    perimeter_count = np.sum(perimeter)
    inContact_count = np.sum(inContact_pixels)
    return inContact_count / perimeter_count if perimeter_count > 0 else 0

def remove_inContact(coco, annotations, Full_mask, inContact_ratio_limit = 0.15):
  annotations_cleaned = []
  for ann in annotations:
    if 'segmentation' in ann:
      object_mask = coco.annToMask(ann)
      inContact = compute_inContact_ratio(object_mask, Full_mask)
      ann['inContact'] = inContact
      if (inContact<inContact_ratio_limit) or (inContact>0.95):
        annotations_cleaned.append(ann)
  return annotations_cleaned

def apply_transformation(image,mask, angle=0, scale=1, tx=0, ty=0):
    height, width = image.shape[:2]
    # Compute the transformation matrix
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    # Apply transformation
    transformed_image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    transformed_mask = cv2.warpAffine(mask, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed_image,transformed_mask, M

def compute_coco_bbox(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # No object found after transformation
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

def compute_VisibleArea(original_mask, transformed_mask, scale):
    original_pixels = np.sum(original_mask > 0)
    transformed_pixels = np.sum(transformed_mask > 0)
    if original_pixels == 0:
        return 0  # Avoid division by zero if the original mask has no object
    Area_loss = transformed_pixels/((scale**2)*original_pixels)
    return int(Area_loss*100)

def mask_to_coco_annotation(binary_mask):
    binary_mask = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4: # coco needs at least 3 points
            segmentation.append(contour)
    if not segmentation:
        return None # return none if no valid contour is found.
    x, y, w, h = cv2.boundingRect(binary_mask)
    bbox = [x, y, w, h]
    area = int(cv2.contourArea(contours[0])) #calculate area.
    return segmentation, bbox, area

def copy_paste_augmentation(coco, annotations_source_cleaned,I_source,I_destination, img_id_destination ):
  h0,w0,d0 = I_destination.shape
  Total_number_objects = len(annotations_source_cleaned)
  if Total_number_objects>0:
    selected_object = np.random.randint(Total_number_objects)
    ann_1obj = annotations_source_cleaned[selected_object]#random.choice(annotations_source_cleaned)
    mask_1obj = coco.annToMask(ann_1obj)
    Img_1obj = I_source * mask_1obj[:, :, np.newaxis]
    xmin, ymin, w, h = ann_1obj['bbox']
    if h>h0 or w>w0:
      #scale image and mask if the size of the object is larger than that of the
      Img_1obj = cv2.resize(Img_1obj, (w0//2, h0//2))
      mask_1obj = (np.sum(Img_1obj,axis=-1)>0).astype(np.uint8)
      xmin, ymin, w, h =compute_coco_bbox(mask_1obj)
    h = int(h)
    w = int(w)
    ymin_src = max(int(ymin), 0)
    xmin_scr = max(int(xmin),0)
    ymax_scr = ymin_src + h
    xmax_src = xmin_scr + w
    ymin_dist = max(int((h0-h)//2), 0)
    xmin_dist = max(int((w0-w)//2),0)
    ymax_dist = min(ymin_dist + h,h0)
    xmax_dist = min(xmin_dist + w, w0)
    # place object in the center
    Img_cenetred = np.zeros([h0,w0,3])#np.zeros_like(I_destination)
    Mask_cenetred = np.zeros([h0,w0])
    Img_cenetred[ymin_dist:ymax_dist, xmin_dist:xmax_dist, :] = Img_1obj[ymin_src:ymax_scr, xmin_scr:xmax_src,:].astype(np.uint8)
    Mask_cenetred[ymin_dist:ymax_dist, xmin_dist:xmax_dist] = mask_1obj[ymin_src:ymax_scr, xmin_scr:xmax_src]
    # Augment the image
    angle = 0#np.random.uniform(-20, 20)
    scale = np.random.uniform(0.4, 1.25)
    ty = np.random.uniform(-0.4, 0.4) * h0
    tx = np.random.uniform(-0.4, 0.4) * w0
    augmented_image,augmented_mask,M = apply_transformation(Img_cenetred,Mask_cenetred, angle, scale, tx, ty)
    # create new annotation
    visible_area_image = compute_VisibleArea(Mask_cenetred, augmented_mask, scale)
    #print(visible_area_image)
    segmentation, bbox, area = mask_to_coco_annotation(augmented_mask)
    New_annotation = {'segmentation':segmentation,
        'iscrowd':ann_1obj['iscrowd'],
        'image_id': img_id_destination, # the negative sign to distingush it from the origional image
        'category_id':ann_1obj['category_id'], 'id':ann_1obj['id'], #'inContact':ann_1obj['inContact'],
        'source_image_id':ann_1obj['image_id'], 'isAugmented':1,
        'area': area,'bbox': bbox,'Visible_due2_cut':visible_area_image, }
  else:
    print('No objects to augment')
  return augmented_image,augmented_mask, New_annotation

def calculate_visibility(Current_mask,Overlaying_mask):
    Current_mask = Current_mask>0
    Overlaying_mask = Overlaying_mask>0
    Overlap = np.multiply(Overlaying_mask, Current_mask)>0
    Visibility = 100*np.round(1 - (np.sum(Overlap)/np.sum(Current_mask)),2) if np.sum(Current_mask)>0 else 0
    return Visibility

def Check_bboxes_intersect(bbox1, bbox2):
    """
    Check if two COCO-format bounding boxes intersect.
    COCO bounding box format: [x, y, width, height]
    Args:
        bbox1 (list or tuple): [x1, y1, w1, h1] for the first bbox
        bbox2 (list or tuple): [x2, y2, w2, h2] for the second bbox
    Returns:
        bool: True if the bounding boxes intersect, False otherwise.
    """
    x1_min, y1_min, w1, h1 = bbox1
    x2_min, y2_min, w2, h2 = bbox2
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2
    # Check if there is an overlap
    if x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min:
        return True
    return False

# Get N source images
def generate_overlayed_frames(coco, I_destination,img_id_destination, N_source_images=5, margin_ratio=0.05, inContact_ratio_limit=0.1 ):
  #(coco, annotations_source_cleaned,I_source,I_destination, img_id_destination )
  #N_source_images =5
  #margin_ratio=0.05,
  #inContact_ratio_limit=0.1
  h0,w0,d0 = I_destination.shape
  Img_id_sources = random.sample(coco.getImgIds(), N_source_images)
  Image_layers = np.zeros([N_source_images+1, h0,w0, d0])
  Mask_layers = np.zeros([N_source_images+1, h0, w0])
  Added_annotations ={}
  for i in range(N_source_images):
    # Get a source image
    #i = 0
    Img_id_source = Img_id_sources[i]
    img_info_source = coco.loadImgs(Img_id_source)[0]
    img_path_source = img_info_source['coco_url']
    I_source = io.imread(img_path_source)
    annotations_source = coco.loadAnns(coco.getAnnIds(imgIds=Img_id_source))
    # Filter the annotations: remove crowded objects
    annotations_source = remove_crowded(annotations_source)
    # Filter the annotations: remove objects in contact with other objects
    Full_mask = get_segmentation_mask(annotations_source,img_info_source, coco)
    annotations_source_cleaned1 = remove_inContact(coco, annotations_source, Full_mask, inContact_ratio_limit)
    # Filter the annotations: remove objects beyond boundaries
    annotations_source_cleaned = remove_beyond_boudaries(annotations_source_cleaned1,img_info_source, margin_ratio)
    # Augment the selected object
    if len(annotations_source_cleaned)> 0:
      augmented_image,augmented_mask, New_annotation = copy_paste_augmentation(coco,annotations_source_cleaned,I_source,I_destination,img_id_destination )
      Image_layers[i] = augmented_image
      Mask_layers[i] = augmented_mask.astype(np.uint8)
      New_annotation['Layer']=i #This is to indicate the closeness of the object to the camera [bigger N -> closer to camera: overlayed above other objects]
      Added_annotations[i]= New_annotation
    else:
      Added_annotations[i]= {}
      #print(f'Caution:no enough objects in source image {i}')
  return Image_layers,Mask_layers, Added_annotations

#annotations_source_cleaned1 = remove_overlapped(annotations_source, Full_mask, overlap_ratio_limit = 0.1)
#print(len(Added_annotations))
#print(Image_layers,Mask_layers, New_annotation )

def overlay(coco, I_destination,annotations_destination, Image_layers,Mask_layers, Added_annotations, N_source_images=5 ):
  # Overlay images and calculate the visibility of new objects
  projected = I_destination.astype(np.uint8).copy()  # Initialize with zeros
  Image_layers.astype('float64')
  Visibility={}
  for j in range(N_source_images):
      projected[Mask_layers[j] > 0] = 0  # Zero out pixels where the new object will be placed
      projected = np.add(projected, Image_layers[j])  # Place the new object
      projected = projected.astype(np.uint8)
      Overlaying_mask = np.sum(Mask_layers[j+1:],axis=0)
      Added_annotations[j]['Layer'] = j
      Visibility[j] = calculate_visibility(Current_mask=Mask_layers[j], Overlaying_mask = Overlaying_mask)
      Added_annotations[j]['Visible_due2_overlay'] = min(Visibility[j], 100)
      if 'Visible_due2_cut' in Added_annotations[j].keys():
        Added_annotations[j]['Visible'] = Visibility[j]*Added_annotations[j]['Visible_due2_cut']/100
      else:
        Added_annotations[j]['Visible'] = Visibility[j]
      #print(Visibility[j])

  #measure the visbility of the objects in the destination image due to projections
  Overlaying_mask_of_added_objects = (np.sum(Mask_layers,axis=0)>0).astype(np.uint8)
  for dest_ann in annotations_destination:
    dest_1obj_mask = coco.annToMask(dest_ann)#get_segmentation_mask(anns,img_info, coco)
    Visibility = calculate_visibility(dest_1obj_mask, Overlaying_mask = Overlaying_mask_of_added_objects)
    dest_ann['Visible_due2_overlay']=min(Visibility,100)
    dest_ann['Layer']=0

  # Merge annotations
  Added_annotations_list = list(Added_annotations.values())
  Added_annotations_list
  All_annotations_after_pasteAug = annotations_destination+Added_annotations_list

  #print('number of origional objects:',len(annotations_destination))
  #print('number of pasted objects:',len(Added_annotations_list))
  #print('number of all objects:',len(All_annotations_after_pasteAug))
  return projected, All_annotations_after_pasteAug, Added_annotations_list, annotations_destination


def visible_annotations_with_captions(annotations):
  visible_annotations=[]
  for ann in annotations:
    if 'category_id' in ann:
      if 'Visible_due2_overlay' in ann:
        if ann['Visible_due2_overlay']>1:
          visible_annotations.append(ann)
  return visible_annotations
