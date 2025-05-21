# Creating Multi-Modal Datasets for YOLO5D+GPS

This guide explains how to create and use real multi-modal datasets with the YOLO5D+GPS model. The repository includes both synthetic data support (for easy testing) and the ability to use real RGB, depth, thermal, and GPS data.

## Dataset Structure

Create your dataset in the following structure:

```
my_dataset/
├── rgb/               # RGB images (RGB, 3 channels)
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── depth/             # Depth maps (grayscale, 1 channel)
│   ├── 000001.png     # 16-bit PNG recommended for depth
│   ├── 000002.png
│   └── ...
├── thermal/           # Thermal images (grayscale, 1 channel)
│   ├── 000001.png     # Thermal images, normalized values
│   ├── 000002.png
│   └── ...
├── annotations/       # Pascal VOC style XML annotations
│   ├── 000001.xml
│   ├── 000002.xml
│   └── ...
├── gps_coords.csv     # CSV file with GPS coordinates
├── train.txt          # List of image IDs for training (optional)
└── val.txt            # List of image IDs for validation (optional)
```

## File Requirements

### 1. RGB Images
- Standard JPG or PNG format
- 3 channels (RGB)
- Any resolution (will be resized during training)

### 2. Depth Maps
- PNG format (16-bit recommended for depth data)
- Single channel grayscale
- Values normalized to 0-1 range or in raw depth units
- Same filename as corresponding RGB image

### 3. Thermal Images
- PNG format
- Single channel grayscale
- Values normalized to 0-1 range
- Can be lower resolution than RGB (model upsamples as needed)
- Same filename as corresponding RGB image

### 4. Annotations
Standard Pascal VOC format XML files:

```xml
<annotation>
  <folder>my_dataset</folder>
  <filename>000001.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>car</name>
    <bndbox>
      <xmin>156</xmin>
      <ymin>97</ymin>
      <xmax>351</xmax>
      <ymax>270</ymax>
    </bndbox>
  </object>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>420</xmin>
      <ymin>171</ymin>
      <xmax>535</xmax>
      <ymax>486</ymax>
    </bndbox>
  </object>
</annotation>
```

### 5. GPS Coordinates (gps_coords.csv)
CSV file containing image ID and GPS coordinates:

```csv
image_id,latitude,longitude
000001,37.7749,-122.4194
000002,34.0522,-118.2437
000003,40.7128,-74.0060
...
```

### 6. Train/Val Split (optional)
If you want to specify custom training and validation splits, create two text files:
- `train.txt`: List of image IDs for training, one per line
- `val.txt`: List of image IDs for validation, one per line

If these files are not provided, the dataset will be randomly split according to the `VAL_SPLIT` parameter in the configuration.

## Data Normalization

### Depth Normalization
It's recommended to normalize depth values to the 0-1 range for training:

```python
# Example normalization (adjust based on your sensor's range)
min_depth, max_depth = 0.1, 10.0  # in meters
normalized_depth = (raw_depth - min_depth) / (max_depth - min_depth)
normalized_depth = np.clip(normalized_depth, 0, 1)
```

### GPS Normalization
GPS coordinates should be normalized to 0-1 range for the model:

```python
# Example normalization (adjust based on your geographic region)
# For a dataset in the USA
min_lat, max_lat = 24.0, 50.0  # Rough USA bounds
min_lon, max_lon = -125.0, -66.0  # Rough USA bounds

norm_lat = (lat - min_lat) / (max_lat - min_lat)
norm_lon = (lon - min_lon) / (max_lon - min_lon)
```

During inference, you'll need to convert the model's output back to geographic coordinates using the same normalization parameters.

## Using the Dataset

### Training with Real Data
Use the `--dataset-path` argument to specify your dataset directory:

```bash
python train.py --dataset-path /path/to/my_dataset
```

### Testing with Missing Modalities
The model can handle missing modalities gracefully:
- If a depth image is missing, a synthetic depth will be generated
- If a thermal image is missing, a synthetic thermal image will be generated
- If GPS coordinates are missing, default values [0.5, 0.5] will be used

This allows you to start with partial data and incrementally add more modalities.

## Data Collection Tips

### Camera Setup
- Mount RGB, depth, and thermal cameras with minimal offset
- Ensure synchronized capture across all sensors
- Calibrate the cameras for accurate alignment
- Record GPS position with each capture

### Preprocessing
- Register and align images from different modalities
- Crop and scale images as needed
- Convert depth data to a consistent scale
- Normalize thermal data based on your sensor's range

### Annotation Tools
- Use [LabelImg](https://github.com/tzutalin/labelImg) for creating Pascal VOC format annotations
- Consider using semi-automated annotation tools for larger datasets

## Troubleshooting

### Image Loading Issues
- Ensure all filenames match across modalities (same ID, different extensions)
- Check that image dimensions are consistent within each modality
- Verify that PNG files are properly saved (especially 16-bit depth maps)

### Missing Data Warnings
- The model will print warnings when falling back to synthetic data
- Check file paths and naming conventions if you see unexpected warnings

### GPS Coordinate Issues
- Ensure GPS coordinates are properly normalized to 0-1 range
- Check for consistency between image IDs and GPS data
- Remember that the model outputs normalized coordinates that must be converted back

## Converting Existing Datasets

Several public datasets can be converted to this format:

- **NYU Depth Dataset V2**: Contains RGB-D pairs
- **FLIR Thermal Dataset**: Contains aligned RGB and thermal images
- **KITTI**: Contains RGB, depth, and GPS but needs reformatting

For conversion scripts and additional guidance, check the project's issue tracker or contribute your own conversion utilities.

## Example Code for Custom Dataset Loading

For more details, check the `dataset.py` file in the repository and use `train_modified.py`, which includes the complete implementation of the `MultiModalDataset` class that handles loading and preprocessing real multi-modal data.