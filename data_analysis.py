import os
import argparse
from collections import defaultdict

def analyze_kitti_labels(base_path):
    """
    Analyze KITTI format labels to count empty labels and distribution of label counts.
    
    Args:
        base_path (str): Base path containing image_2 and label_2 folders
    
    Returns:
        tuple: (empty_count, total_count, label_distribution)
    """
    # Paths to image and label directories
    image_dir = os.path.join(base_path, 'image_2')
    label_dir = os.path.join(base_path, 'label_2')
    
    # Check if directories exist
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise ValueError("Image or label directory not found!")
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    total_count = len(image_files)
    
    if total_count == 0:
        raise ValueError("No image files found!")
    
    # Initialize counters
    empty_count = 0
    label_distribution = defaultdict(int)
    
    # Process each image file
    for img_file in image_files:
        # Get corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_file}")
            continue
        
        # Count objects in label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
            num_objects = len(lines)
            
            # Update counters
            if num_objects == 0:
                empty_count += 1
            
            # Group into categories (0, 1, 2, 3, 4, >4)
            if num_objects > 4:
                label_distribution['>4'] += 1
            else:
                label_distribution[str(num_objects)] += 1
    
    return empty_count, total_count, dict(label_distribution)

def print_analysis(empty_count, total_count, label_distribution):
    """
    Print the analysis results in a formatted way.
    """
    print("\nDataset Analysis Results")
    print("-" * 30)
    
    # Empty labels analysis
    empty_percentage = (empty_count / total_count) * 100
    print(f"\nEmpty Labels Analysis:")
    print(f"Total samples: {total_count}")
    print(f"Empty labels: {empty_count}")
    print(f"Percentage of empty labels: {empty_percentage:.2f}%")
    
    # Distribution analysis
    print(f"\nLabel Count Distribution:")
    for n in ['0', '1', '2', '3', '4', '>4']:
        count = label_distribution.get(n, 0)
        percentage = (count / total_count) * 100
        print(f"Images with {n} labels: {count} ({percentage:.2f}%)")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze KITTI dataset labels.')
    parser.add_argument('--dataset_path', 
                        type=str, 
                        required=True,
                        help='Path to the KITTI dataset (or similar) containing image_2 and label_2 folders')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    try:
        empty_count, total_count, label_distribution = analyze_kitti_labels(args.dataset_path)
        print_analysis(empty_count, total_count, label_distribution)
    except Exception as e:
        print(f"Error: {str(e)}")