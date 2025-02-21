import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import random
import imageio
import json
import math
import cv2
import os
import multiprocessing
from functools import partial


def load_profiles(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data["profiles"]


def plot_grid(grid: np.ndarray, grid_size, filename, i, save_pdf = True):
    # Load JSON file containing annotations
    json_path = filename.format(fileformat='json')
    with open(json_path, 'r') as file:
        data = json.load(file)

    annotations = data["annotations"]
    categories = {category["id"]: category["name"] for category in data["categories"]}

    # Create figure and axis for plotting the grid
    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.imshow(grid, cmap="gray", origin="upper")

    # Overlay annotations
    for annotation in annotations:
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        category_name = categories.get(category_id, "Unknown")

        # Draw the bounding box
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=1,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        # Process segmentation data
        segmentation = annotation["segmentation"]
        if segmentation:
            # Determine if segmentation is a list of segments or a single segment
            segments = segmentation if isinstance(segmentation[0], list) else [segmentation]
            for segment in segments:
                points = [(float(segment[i]), float(segment[i + 1]))
                          for i in range(0, len(segment), 2)]
                polygon = Polygon(points, linewidth=1, edgecolor="blue", facecolor="none")
                ax.add_patch(polygon)

        # Add the category label text above the bounding box
        ax.text(
            bbox[0] + 7.5,
            bbox[1] + bbox[3] + 22,
            category_name,
            color="red",
            fontsize=8,
            bbox=dict(facecolor="none", edgecolor="red"),
            ha='left',
            va='top'
        )

    # Save the grid as a binary JPG image
    binary_image = ((grid > 0).astype(np.uint8)) * 255
    imageio.imwrite(filename.format(fileformat='jpg'), binary_image)

    # Show the plot
    plt.show()

    # Save the plot as a PDF if enabled
    if save_pdf:
        pdf_filename = f"data/pdf/grid_with_mask_{i:04d}.pdf"
        fig.savefig(pdf_filename, format="pdf", bbox_inches="tight", pad_inches=0)

    plt.close(fig)


def add_noise(grid, noise_level, cluster_count, max_cluster_size):
    if not (0.0 <= noise_level <= 1.0):
        raise ValueError("Noise level must be between 0.0 and 1.0.")

    # Calculate the total number of cells to be affected by noise
    total_cells = grid.size
    num_noise_cells = int(total_cells * noise_level)

    # Create a copy of the grid to modify
    noisy_grid = grid.copy()

    # Generate random clusters
    cells_affected = 0
    rows, cols = grid.shape

    for _ in range(cluster_count):
        if cells_affected >= num_noise_cells:
            break

        # Random cluster center
        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)

        # Random cluster bounding box size
        cluster_width = np.random.randint(1, max_cluster_size + 1)
        cluster_height = np.random.randint(1, max_cluster_size + 1)

        # Randomly toggle cells within the bounding box
        for x in range(center_x, center_x + cluster_width):
            for y in range(center_y, center_y + cluster_height):
                if 0 <= x < cols and 0 <= y < rows:
                    if np.random.rand() < 0.5:  # Random chance to toggle this cell
                        noisy_grid[y, x] = 1 - noisy_grid[y, x]  # Flip color
                        cells_affected += 1
                        if cells_affected >= num_noise_cells:
                            break
            if cells_affected >= num_noise_cells:
                break

    return noisy_grid


def check_no_intersection(existing_walls, start_x, start_y, end_x, end_y, cols, rows):
    for horizontal_y in existing_walls["horizontal"]:
        if start_y < horizontal_y < end_y or start_y > horizontal_y > end_y:  # Check horizontal lines
            x_intersect = start_x + (horizontal_y - start_y) * (end_x - start_x) / (end_y - start_y)
            if 0 <= x_intersect <= cols:  # Ensure the intersection point is within grid bounds
                return False

    for vertical_x in existing_walls["vertical"]:
        if start_x < vertical_x < end_x or start_x > vertical_x > end_x:  # Check vertical lines
            y_intersect = start_y + (vertical_x - start_x) * (end_y - start_y) / (end_x - start_x)
            if 0 <= y_intersect <= rows:  # Ensure the intersection point is within grid bounds
                return False

    return True


def is_point_in_polygon(point, polygon):
    global xinters
    x, y = point
    n = len(polygon)
    inside = False

    px, py = polygon[0]
    for i in range(1, n + 1):
        nx, ny = polygon[i % n]
        if y > min(py, ny):
            if y <= max(py, ny):
                if x <= max(px, nx):
                    if py != ny:
                        xinters = (y - py) * (nx - px) / (ny - py) + px
                    if px == nx or x <= xinters:
                        inside = not inside
        px, py = nx, ny

    return inside


def create_wall(ax, grid, start_x, start_y, length, width, orientation=None, color="black", diagonal=False):
    global rect
    if diagonal:
        # Normalize the length to ensure correct direction
        direction_x = 1 if length > 0 else -1
        direction_y = 1 if length > 0 else -1
        length = abs(length)

        # Calculate the four corners of the diagonal wall
        x0, y0 = start_x, start_y
        x1, y1 = start_x + length * direction_x, start_y + length * direction_y
        x2, y2 = x1 - direction_y * width, y1 + direction_x * width
        x3, y3 = x0 - direction_y * width, y0 + direction_x * width

        # Update the grid along the wall polygon
        for x in range(int(min(x0, x1, x2, x3)), int(max(x0, x1, x2, x3)) + 1):
            for y in range(int(min(y0, y1, y2, y3)), int(max(y0, y1, y2, y3)) + 1):
                if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                    # Check if the point (x, y) is inside the polygon
                    if is_point_in_polygon((x, y), [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]):
                        grid[y, x] = 0

        # Draw the wall (only if ax is provided)
        if ax is not None:
            polygon = plt.Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], closed=True, color=color)
            ax.add_patch(polygon)

    else:
        # Horizontal or vertical wall
        if orientation == 0:  # Horizontal wall
            grid[int(start_y):int(start_y + width), int(start_x):int(start_x + length)] = 0
        elif orientation == 1:  # Vertical wall
            grid[int(start_y):int(start_y + length), int(start_x):int(start_x + width)] = 0

        # Draw the wall (only if ax is provided)
        if ax is not None:
            if orientation == 0:
                rect = plt.Rectangle((start_x, start_y), length, width, color=color)
            elif orientation == 1:
                rect = plt.Rectangle((start_x, start_y), width, length, color=color)
            ax.add_patch(rect)


def add_random_walls_with_variable_width(grid_size, wall_length_range, wall_width_range,
                                         min_distance,num_walls_range):
    rows, cols = grid_size
    grid = np.ones((rows, cols))  # Initialize grid with ones (white cells)

    # List to store all wall information
    existing_wall_positions = []

    # Add boundary walls
    selected_walls = random.sample(["top", "bottom", "left", "right"], random.randint(0, 4))

    if "top" in selected_walls:
        boundary_wall_thickness_top = random.randint(*wall_width_range)
        create_wall(None, grid, 0, 0, cols, boundary_wall_thickness_top, orientation=0)
        existing_wall_positions.append({
            "type": "horizontal",
            "start": (0, 0),
            "length": cols,
            "width": boundary_wall_thickness_top
        })

    if "bottom" in selected_walls:
        boundary_wall_thickness_bottom = random.randint(*wall_width_range)
        create_wall(None, grid, 0, rows - boundary_wall_thickness_bottom, cols, boundary_wall_thickness_bottom, orientation=0)
        existing_wall_positions.append({
            "type": "horizontal",
            "start": (0, rows - boundary_wall_thickness_bottom),
            "length": cols,
            "width": boundary_wall_thickness_bottom
        })

    if "left" in selected_walls:
        boundary_wall_thickness_left = random.randint(*wall_width_range)
        create_wall(None, grid, 0, 0, rows, boundary_wall_thickness_left, orientation=1)
        existing_wall_positions.append({
            "type": "vertical",
            "start": (0, 0),
            "length": rows,
            "width": boundary_wall_thickness_left
        })

    if "right" in selected_walls:
        boundary_wall_thickness_right = random.randint(*wall_width_range)
        create_wall(None, grid, cols - boundary_wall_thickness_right, 0, rows, boundary_wall_thickness_right, orientation=1)
        existing_wall_positions.append({
            "type": "vertical",
            "start": (cols - boundary_wall_thickness_right, 0),
            "length": rows,
            "width": boundary_wall_thickness_right
        })

    # Generate random straight walls
    num_walls = np.random.randint(*num_walls_range)

    for _ in range(num_walls):
        orientation = np.random.choice([0, 1])  # 0 for horizontal, 1 for vertical

        wall_created = False
        counter = 0
        if orientation == 0:  # Horizontal wall
            while not wall_created:
                row = np.random.randint(0, rows - wall_width_range[1])
                start_col = np.random.randint(0, cols - wall_length_range[0])
                max_length = min(cols - start_col, wall_length_range[1])
                if (max_length >= wall_length_range[0]) and (all(abs(row - wall["start"][1]) >= min_distance for wall in existing_wall_positions if wall["type"] == "horizontal")):
                        wall_length = np.random.randint(wall_length_range[0], max_length + 1)
                        wall_width = random.randint(*wall_width_range)
                        create_wall(None, grid, start_col, row, wall_length, wall_width, orientation=0)
                        existing_wall_positions.append({
                            "type": "horizontal",
                            "start": (start_col, row),
                            "length": wall_length,
                            "width": wall_width
                        })
                        wall_created = True
                else:
                    counter += 1
                    if counter > 10000:
                        wall_created = True
                        print('Not able to place the requested number of walls.')


        else:  # Vertical wall
            while not wall_created:
                col = np.random.randint(0, cols - wall_width_range[1])
                start_row = np.random.randint(0, rows - wall_length_range[0])
                max_length = min(rows - start_row, wall_length_range[1])
                if (max_length >= wall_length_range[0]) and (all(abs(col - wall["start"][0]) >= min_distance for wall in existing_wall_positions if wall["type"] == "vertical")):
                        wall_length = np.random.randint(wall_length_range[0], max_length + 1)
                        wall_width = random.randint(*wall_width_range)
                        create_wall(None, grid, col, start_row, wall_length, wall_width, orientation=1)
                        existing_wall_positions.append({
                            "type": "vertical",
                            "start": (col, start_row),
                            "length": wall_length,
                            "width": wall_width
                        })
                        wall_created = True
                else:
                    counter += 1
                    if counter > 10000:
                        wall_created = True
                        print('Not able to place the requested number of walls.')

    return grid, existing_wall_positions


def add_columns(grid, size_a_range, size_b_range, number_range, filename, rotation_probability, steel_profiles_file):
    rows, cols = grid.shape
    column_positions = []
    number = random.randint(number_range[0], number_range[1])

    # Load steel profiles if provided.
    steel_profiles = load_profiles(steel_profiles_file)

    # Define COCO categories.
    categories = [
        {"id": 0, "name": "rect"},
        {"id": 1, "name": "round"}
    ]
    if steel_profiles:
        categories.append({"id": 2, "name": "steel"})

    coco = {
        "images": [
            {
                "id": 1,
                "width": cols,
                "height": rows,
                "file_name": filename.format(fileformat="jpg")
            }
        ],
        "annotations": [],
        "categories": categories
    }

    annotation_id = 1
    steel_count = 0  # Counter for steel columns

    for _ in range(number):
        # Randomly choose shape from the three options.
        shape = np.random.choice(["rect", "round", "steel"])
        if shape == "steel" and steel_profiles is None:
            shape = np.random.choice(["rect", "round"])

        # Define corner regions.
        corners = {
            "top-left": (random.randint(0, cols // 4), random.randint(0, rows // 4)),
            "top-right": (random.randint(3 * cols // 4, cols), random.randint(0, rows // 4)),
            "bottom-left": (random.randint(0, cols // 4), random.randint(3 * rows // 4, rows)),
            "bottom-right": (random.randint(3 * cols // 4, cols), random.randint(3 * rows // 4, rows))
        }

        if shape in ["rect", "round"]:
            # Generate random sizes.
            size_A = random.randint(size_a_range[0], size_a_range[1])
            size_B = random.randint(size_b_range[0], size_b_range[1])

            place_in_corner = random.choice([True, False])
            if place_in_corner:
                corner_key = random.choice(list(corners.keys()))
                x_center, y_center = corners[corner_key]
            else:
                x_center = random.randint(size_A, cols - size_A)
                y_center = random.randint(size_B, rows - size_B)

            if shape == "rect":
                angle = random.uniform(-90, 90) if random.random() < 0.5 else 0
                half_A, half_B = size_A // 2, size_B // 2
                rectangle = np.array([
                    [x_center - half_A, y_center - half_B],
                    [x_center + half_A, y_center - half_B],
                    [x_center + half_A, y_center + half_B],
                    [x_center - half_A, y_center + half_B]
                ], dtype=np.float32)

                rotation_matrix = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)
                rotated_rectangle = cv2.transform(np.array([rectangle]), rotation_matrix)[0]
                cv2.fillPoly(grid, [rotated_rectangle.astype(np.int32)], 0)

                x_min = int(rotated_rectangle[:, 0].min())
                y_min = int(rotated_rectangle[:, 1].min())
                x_max = int(rotated_rectangle[:, 0].max())
                y_max = int(rotated_rectangle[:, 1].max())
                segmentation = rotated_rectangle.flatten().tolist()
                width_val = x_max - x_min
                height_val = y_max - y_min

            elif shape == "round":
                r = math.floor(size_A / 2)
                # Ensure the circle fits inside the grid.
                x_center = random.randint(r, cols - r)
                y_center = random.randint(r, rows - r)
                for y in range(max(0, y_center - r), min(rows, y_center + r)):
                    for x in range(max(0, x_center - r), min(cols, x_center + r)):
                        if (x - x_center) ** 2 + (y - y_center) ** 2 <= r ** 2:
                            grid[y, x] = 0
                width_val = height_val = 2 * r
                x_min = x_center - r
                y_min = y_center - r
                num_points = 100
                segmentation = []
                for i in range(num_points):
                    theta = 2 * np.pi * i / num_points
                    segmentation.extend([x_center + (size_A / 2) * np.cos(theta),
                                         y_center + (size_A / 2) * np.sin(theta)])

            column_positions.append((x_center, y_center, width_val, height_val, shape))
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": 1,
                "category_id": 0 if shape == "rect" else 1,
                "bbox": [x_min, y_min, width_val, height_val],
                "segmentation": segmentation,
                "area": width_val * height_val,
                "iscrowd": 0
            })
            annotation_id += 1


        elif shape == "steel":
            # Select a random steel profile.
            profile = random.choice(steel_profiles)
            profile_h = profile["h"]
            profile_b = profile["b"]
            web_thickness = profile["s"]
            flange_thickness = profile["t"]
            place_in_corner = random.choice([True, False])

            if place_in_corner:
                corner_key = random.choice(list(corners.keys()))
                x_center, y_center = corners[corner_key]

            else:
                x_center = random.randint(int(profile_b // 2), cols - int(profile_b // 2))
                y_center = random.randint(int(profile_h // 2), rows - int(profile_h // 2))
            half_h = profile_h / 2
            half_b = profile_b / 2
            half_web = web_thickness / 2

            # Create polygon for the I-beam cross-section.
            polygon = np.array([
                [x_center - half_b, y_center - half_h],
                [x_center + half_b, y_center - half_h],
                [x_center + half_b, y_center - half_h + flange_thickness],
                [x_center + half_web, y_center - half_h + flange_thickness],
                [x_center + half_web, y_center + half_h - flange_thickness],
                [x_center + half_b, y_center + half_h - flange_thickness],
                [x_center + half_b, y_center + half_h],
                [x_center - half_b, y_center + half_h],
                [x_center - half_b, y_center + half_h - flange_thickness],
                [x_center - half_web, y_center + half_h - flange_thickness],
                [x_center - half_web, y_center - half_h + flange_thickness],
                [x_center - half_b, y_center - half_h + flange_thickness]
            ], dtype=np.float32)

            # Rotate the steel profile
            angle = random.uniform(-90, 90) if random.random() < rotation_probability else 0
            rotation_matrix = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)
            rotated_polygon = cv2.transform(np.array([polygon]), rotation_matrix)[0]
            cv2.fillPoly(grid, [rotated_polygon.astype(np.int32)], 0)

            # Bounding box and segmentation
            x_min = int(rotated_polygon[:, 0].min())
            y_min = int(rotated_polygon[:, 1].min())
            x_max = int(rotated_polygon[:, 0].max())
            y_max = int(rotated_polygon[:, 1].max())
            width_val = x_max - x_min
            height_val = y_max - y_min
            segmentation = rotated_polygon.flatten().tolist()

            # Append position and COCO annotations
            column_positions.append((x_center, y_center, width_val, height_val, "steel"))

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": 1,
                "category_id": 2,
                "bbox": [x_min, y_min, width_val, height_val],
                "segmentation": segmentation,
                "area": width_val * height_val,
                "iscrowd": 0
            })
            annotation_id += 1
            steel_count += 1

    with open(filename.format(fileformat='json'), "w") as f:
        json.dump(coco, f, indent=4)

    print(f"COCO annotations saved to {filename.format(fileformat='json')}")
    print("Column positions:", column_positions)
    print(f"Total steel columns created: {steel_count}")
    return grid, column_positions




def generate_sample(i, output_dir, grid_size, wall_length_range, wall_width_range, min_distance,
                    num_walls_range, num_diagonal_walls_range, diagonal_wall_length_range, noise_level,
                    column_size_range, number_of_columns, rotation_probability, steel_profiles_file):
    filename = f"{output_dir}grid_with_mask_{i:04d}.{{fileformat}}"

    # Add random walls to the grid
    grid_with_walls, _ = add_random_walls_with_variable_width(grid_size, wall_length_range, wall_width_range,
                                                              min_distance, num_walls_range)

    # Add noise into the grid
    grid_with_walls = add_noise(grid_with_walls, noise_level, cluster_count=150, max_cluster_size=5)

    # Add columns and get COCO annotations
    grid_with_columns, _ = add_columns(grid_with_walls, column_size_range, column_size_range, number_of_columns,
                                       filename, rotation_probability, steel_profiles_file)

    # Save the grid as image
    plot_grid(grid_with_columns, grid_size, filename, i, save_pdf=False)

    return filename.format(fileformat="json")


def main():
    os.makedirs("data", exist_ok=True)
    num_samples = 2000  # Number of grids to generate
    output_dir = "data/"
    master_json_path = output_dir + "master_annotations.json"
    steel_profiles_file = "steel_profiles.json"

    # Dataset parameters
    grid_size = (512, 512)
    wall_length_range = (150, 512)
    wall_width_range = (2, 13)
    min_distance = 50
    num_walls_range = (15, 35)
    num_diagonal_walls_range = (0, 2)
    diagonal_wall_length_range = (50, 1024)
    noise_level = 0.05  # Noise level must be between 0.0 and 1.0
    column_size_range = (10, 35)
    number_of_columns = (1, 5)
    rotation_probability = 0.35


    num_processors = multiprocessing.cpu_count()
    use_multiprocessing = num_processors > 1

    if use_multiprocessing:
        with multiprocessing.Pool(processes=num_processors) as pool:
            json_files = pool.map(partial(generate_sample, output_dir=output_dir, grid_size=grid_size,
                                          wall_length_range=wall_length_range, wall_width_range=wall_width_range,
                                          min_distance=min_distance, num_walls_range=num_walls_range,
                                          num_diagonal_walls_range=num_diagonal_walls_range,
                                          diagonal_wall_length_range=diagonal_wall_length_range,
                                          noise_level=noise_level,
                                          column_size_range=column_size_range, number_of_columns=number_of_columns,
                                          rotation_probability=rotation_probability,
                                          steel_profiles_file=steel_profiles_file), range(num_samples))
    else:
        json_files = [generate_sample(i, output_dir, grid_size, wall_length_range, wall_width_range, min_distance,
                                      num_walls_range, num_diagonal_walls_range, diagonal_wall_length_range,
                                      noise_level,
                                      column_size_range, number_of_columns, rotation_probability, steel_profiles_file)
                      for i in range(num_samples)]

    # Merge all JSON annotations into a master file
    master_coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "rect"},
            {"id": 1, "name": "round"},
            {"id": 2, "name": "steel"}
        ]
    }

    annotation_id = 1
    image_id = 1
    for json_file in json_files:
        with open(json_file, "r") as f:
            coco_data = json.load(f)

        for image in coco_data["images"]:
            image["id"] = image_id
            master_coco["images"].append(image)

        for annotation in coco_data["annotations"]:
            annotation["id"] = annotation_id
            annotation["image_id"] = image_id
            master_coco["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    with open(master_json_path, "w") as master_json_file:
        json.dump(master_coco, master_json_file, indent=4)

    print(f"Dataset created with {num_samples} samples. Annotations saved to {master_json_path}.")

if __name__ == "__main__":
    main()
