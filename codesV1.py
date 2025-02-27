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
from matplotlib.patches import Polygon as MatplotlibPolygon
#from google.colab.patches import cv2_imshow
#from PIL import Image
pylab.rcParams['figure.figsize'] = (4.0, 6.0)


# Function to calculate rotated rectangles using rotating calipers
def rotating_calipers_rectangles(hull):
    rectangles = []
    hull_points = hull[:, 0, :]
    n = len(hull_points)
    max_length = 0
    for i in range(n):
        # Get the edge formed by two consecutive points
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n]
        # Compute the edge vector and angle of rotation
        edge_vector = p2 - p1
        edge_length = np.linalg.norm(edge_vector)
        edge_angle = np.arctan2(edge_vector[1], edge_vector[0])
        # Rotate the hull points to align the edge with the x-axis
        rotation_matrix = np.array([[-np.cos(edge_angle), np.sin(edge_angle)],
                                     [-np.sin(edge_angle),  -np.cos(edge_angle)]])
        rotated_points = np.dot(hull_points - p1, rotation_matrix.T)
        # Get the bounding box of the rotated points
        x_min, y_min = np.min(rotated_points, axis=0)
        x_max, y_max = np.max(rotated_points, axis=0)
        # Calculate the rectangle corners in the original coordinate system
        rectangle = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
        rectangle = np.dot(rectangle, rotation_matrix) + p1
        rectangles.append(rectangle)
        if edge_length>max_length:
          max_length = edge_length
          rect_aligned_with_maxEdge = rectangle
    return rectangles, rect_aligned_with_maxEdge

def intersection_area(rect1: np.ndarray, rect2: np.ndarray, plot=False) -> float:
    """
    Calculate the intersection area of two rectangles (possibly rotated).
    :param rect1: numpy array of shape (4, 2) representing the first rectangle
    :param rect2: numpy array of shape (4, 2) representing the second rectangle
    :return: The intersection area (float). Returns 0 if no intersection.
    """
    def validate_polygon(coords):
        """Validates and fixes a polygon if necessary."""
        polygon = Polygon(coords)
        if not polygon.is_valid:
            print(f"Invalid polygon: {explain_validity(polygon)}")
            # Attempt to fix self-intersection by using the convex hull
            polygon = polygon.convex_hull
            print(f"Polygon fixed using convex hull: {polygon.wkt}")
        # Ensure the polygon is oriented counterclockwise
        return orient(polygon, sign=1.0)
    # Validate and fix input polygons
    poly1 = validate_polygon(rect1)
    poly2 = validate_polygon(rect2)
    # Find the intersection of the two polygons
    intersection = poly1.intersection(poly2)
    # Plotting
    if plot:
        fig, ax = plt.subplots(figsize=(3,3))
        # Add rectangles
        rect_patch1 = MatplotlibPolygon(np.array(poly1.exterior.coords), closed=True, edgecolor='blue', facecolor='blue', alpha=0.4)#, label="p1")
        rect_patch2 = MatplotlibPolygon(np.array(poly2.exterior.coords), closed=True, edgecolor='green', facecolor='green', alpha=0.4)#, label="p2")
        ax.add_patch(rect_patch1)
        ax.add_patch(rect_patch2)
        # Add intersection if it exists
        '''if not intersection.is_empty and isinstance(intersection, ShapelyPolygon):
            x, y = intersection.exterior.xy
            ax.fill(x, y, color='red', alpha=0.5, label="Intersection")'''
        # Adjust plot limits
        all_points = np.vstack((rect1, rect2))
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_aspect('equal', adjustable='box')
        # Add legend and labels
        ax.legend()
        #ax.set_title("Rectangle Intersection")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.grid(True)
        plt.show()
    # Calculate the area of the intersection
    if intersection.is_empty:
        return 0.0  # No intersection
    return intersection.area


def rle_to_mask(rle):
    """
    Converts Run-Length Encoding (RLE) to a binary segmentation mask.
    Args:
        rle (dict): A dictionary containing:
            - 'counts': The RLE as a list or string (COCO format).
            - 'size': A list or tuple with the dimensions [height, width].
        shape (tuple): The shape of the mask (height, width).
    Returns:
        np.ndarray: A binary mask of the given shape (height, width).
    """
    # Extract RLE counts and mask size
    counts = rle['counts']
    height, width = rle['size']
    # If counts are in string format, split and convert to integers
    if isinstance(counts, str):
        counts = list(map(int, counts.split()))
    # Flatten the RLE counts into a binary array
    mask = np.zeros(height * width, dtype=np.uint8)
    start = 0  # Start position in the flattened mask
    for i, length in enumerate(counts):
        if i % 2 == 1:  # Foreground (object) pixels
            mask[start:start + length] = 1
        # Move the starting point forward
        start += length
    # Reshape the flat mask into the specified 2D shape
    return mask.reshape((width, height))


def find_main_axis(mask):
    # Get the coordinates of the object (where mask == 1)
    y, x = np.where(mask == 1)
    points = np.column_stack((x, y))
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    # Principal axis (first component)
    center = np.mean(points, axis=0)
    direction = pca.components_[0]
    return center, direction

def oriented_bbox(mask, direction_vector, center):
    rectangles = []
    hull_points = np.column_stack(np.where(mask > 0))#hull[:, 0, :]
    angle = np.arctan(direction_vector[1]/(direction_vector[0]+0.00000001))
    # Rotate the hull points to align the edge with the x-axis
    rotation_matrix = np.array([[-np.cos(angle), np.sin(angle)],
                                  [-np.sin(angle),  -np.cos(angle)]])
    rotated_points = np.dot(hull_points - center, rotation_matrix.T)
    # Get the bounding box of the rotated points
    x_min, y_min = np.min(rotated_points, axis=0)
    x_max, y_max = np.max(rotated_points, axis=0)
    # Calculate the rectangle corners in the original coordinate system
    rectangle = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])
    rectangle = np.dot(rectangle, rotation_matrix) + center
    return rectangle

def annotation_2_category(annotation):
  category_id = annotation['category_id']
  categories = coco.loadCats(category_id)
  if categories:
      return categories[0]['name']  # Return the name of the category
  else:
      return "NA"

def annotation_items(annotation):
  image_id = annotation['image_id']
  image_info = coco.loadImgs(image_id)[0]
  image_height = image_info['height']
  image_width = image_info['width']
  object_name = annotation_2_category(annotation)
  segmnt = annotation['segmentation']
  if annotation['iscrowd']== 0:
    polygons_lists = segmnt
    flattened_list = [item for sublist in polygons_lists for item in sublist]
    hull = np.array(flattened_list).reshape([-1,1,2])
    #hull = cv2.convexHull(hull.astype(np.int32))
    mask = np.zeros([image_height, image_width])
    cv2.fillPoly(mask, [hull.astype(np.int32)],1)
    mask = mask.T
  else:
    mask = rle_to_mask(segmnt)
    # Find all non-zero points in the mask
    segmented_points = np.column_stack(np.where(mask > 0))
    if len(segmented_points) == 0:
        raise ValueError("No object found in the segmentation mask.")
    # Compute the convex hull of all points
    hull = cv2.convexHull(segmented_points)
    #mask = mask
  return hull, image_width, image_height, object_name, mask

def get_OBB(annotation, method='regular', plot=False):
  #method = ['regular', 'PCA', 'rotating_calipers','logest_edge']
  #flattened_list = [item for sublist in polygons_lists for item in sublist]
  #hull = np.array(flattened_list).reshape([-1,1,2])
  hull, image_width, image_height, object_name, mask = annotation_items(annotation)
  rectangles, rect_aligned_with_maxEdge = rotating_calipers_rectangles(hull)

  if  method == 'rotating_calipers':
    image_rectangle = np.array([[0,0],[image_width,0],[image_width, image_height],[0,image_height]])
    #image_rectangle = np.array([[0,0],[image_height,0],[image_height, image_width],[0,image_width]])
    min_area = np.inf
    for i, rect in enumerate(rectangles):
      area = intersection_area(image_rectangle, rect,plot=plot)
      if area<min_area:
        min_area = area
        selected_box = rect
  elif  method == 'regular':
    x_min, y_min, width, height = annotation['bbox']
    x_max, y_max = x_min + width, y_min + height
    selected_box = np.array([[x_min, y_min],
                              [x_max, y_min],
                              [x_max, y_max],
                              [x_min, y_max]])
  elif  method == 'logest_edge':
    selected_box = rect_aligned_with_maxEdge
  elif method == 'PCA':
    center, direction = find_main_axis(mask)
    selected_box = oriented_bbox(mask, direction, center)
    annotation['center']= center
    annotation['direction']= direction
  return selected_box

def BB_criterion(annotations, standard_method='regular',
            floating_objects=[], Standing_objects=[],objects_with_axis=[]):
  #All_objects = floating_objects+Standing_objects+objects_with_axis
  if standard_method != 'None':
    for annotation in annotations:
      category = annotation_2_category(annotation)
      #print(category)
      #method = ['None', 'regular', 'PCA', 'rotating_calipers','logest_edge']
      if category in Standing_objects:
        method = 'regular'#'regular'
      elif category in objects_with_axis:
        method = 'PCA'
      elif category in floating_objects:
        method = 'rotating_calipers'#'rotating_calipers'
      else:
        method = standard_method
      selected_box=get_OBB(annotation, method=method)
      annotation['obbox'] = list(np.round(selected_box,1).reshape(-1))
  return annotations

def show_obbox(I_plot, annotations, title, with_segment=False, with_arrow=True, with_category=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(I_plot)
    ax.axis('off')
    ax.set_title(title)
    if with_segment:
        plt.sca(ax)  # Set the current axis to the provided subplot axis
        coco.showAnns(annotations, draw_bbox=False)
    for annotation in annotations:
        coords = annotation['obbox']
        polygon_coords = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        mpl_patch = MatplotlibPolygon(polygon_coords, closed=True, 
                                      edgecolor=[random.random() for _ in range(3)], 
                                      facecolor="none", linewidth=2)
        ax.add_patch(mpl_patch)
        if with_arrow and 'center' in annotation:
            center = annotation['center']
            direction = annotation['direction']
            ax.arrow(center[1], center[0], direction[1]*50, direction[0]*50, 
                     color='blue', head_width=5)
        if with_category:
            category_name = annotation_2_category(annotation)
            bbox_x, bbox_y = min(polygon_coords, key=lambda point: point[1])#np.max(polygon_coords, axis=0)  # Use first coordinate as reference
            ax.text(bbox_x, bbox_y, category_name, color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))


