from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import json
import tqdm
import glob
import pylab
import random
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity
from matplotlib.patches import Polygon as MatplotlibPolygon


# Function to calculate rotated rectangles using rotating calipers
def rotating_calipers_rectangles(hull):
    rectangles = []
    hull_points = hull[:, 0, :]
    n = len(hull_points)
    for i in range(n):
        # Get the edge formed by two consecutive points
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n]
        # Compute the edge vector and angle of rotation
        edge_vector = p2 - p1
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
    return rectangles

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
    return mask.reshape((height, width))


def annotation_items(annotation, coco):
  segmnt = annotation['segmentation']
  if  annotation['iscrowd']== 0:
    polygons_lists = segmnt
    flattened_list = [item for sublist in polygons_lists for item in sublist]
    hull = np.array(flattened_list).reshape([-1,1,2])
  else:
    mask = rle_to_mask(segmnt)
    # Find all non-zero points in the mask
    segmented_points = np.column_stack(np.where(mask > 0))
    if len(segmented_points) == 0:
        raise ValueError("No object found in the segmentation mask.")
    # Compute the convex hull of all points
    hull = cv2.convexHull(segmented_points)
  image_id = annotation['image_id']
  image_info = coco.loadImgs(image_id)[0]
  image_height = image_info['height']
  image_width = image_info['width']
  return hull, image_width, image_height

def get_OBB(hull, width, height):
  #flattened_list = [item for sublist in polygons_lists for item in sublist]
  #hull = np.array(flattened_list).reshape([-1,1,2])
  rectangles = rotating_calipers_rectangles(hull)
  image_rectangle = np.array([[0,0],[width,0],[width, height],[0,height]])
  min_area = np.inf
  for i, rect in enumerate(rectangles):
    area = intersection_area(image_rectangle, rect)
    if area<min_area:
      min_area = area
      selected_box =  rect
  return selected_box


def show_obbox(I_plot, coco, obboxes=[[]],  anns=None, with_obb=True, with_segment = False, with_regula_bb = False):
  plt.imshow(I_plot); plt.axis('off')
  # Add polygons to the existing plot
  if with_segment:
    coco.showAnns(anns, draw_bbox=with_regula_bb)
  if with_obb:
    ax = plt.gca()  # Get the current axis
    for coords in obboxes:
        # Convert the coordinates to a format compatible with Matplotlib patches
        polygon_coords = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        # Create and add a Matplotlib polygon
        mpl_patch = MatplotlibPolygon(polygon_coords, closed=True, edgecolor=[random.random() for _ in range(3)], facecolor="none", linewidth=2)
        ax.add_patch(mpl_patch)
    plt.show()
