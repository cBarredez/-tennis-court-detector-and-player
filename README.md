# Tennis Serve Zone Analysis and Player Detection

## Project Description

This project focuses on detecting tennis players on a court and analyzing their movements, with a special emphasis on serve zones. It leverages advanced computer vision algorithms, including YOLOv8 for player detection and a SORT-based object tracking system to identify and track players throughout a match. Additionally, the project incorporates keypoint analysis techniques to determine if a player has entered the service box during a serve.

### Key Features

- **Player Detection**: Utilizes YOLOv8 model to identify and track players on the court.
- **Keypoint Analysis**: Detects specific points on the court to determine serve zones.
- **SORT Tracking**: Maintains continuous tracking of each player, even if detection is temporarily lost.
- **Video Processing**: Processes match videos and highlights areas of interest, such as court lines and service boxes.
- **Serve Fault Detection**: Automatically identifies if a player steps on the service line during a serve.

## Related Projects

This project was inspired by and builds upon the following repositories:

- [YOLOv8-DeepSORT-Object-Tracking](https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking)
- [Tennis Court Detector](https://github.com/yastrebksv/TennisCourtDetector) - Reference implementation for court detection
- [Tennis Analysis](https://github.com/example/tennis_analysis) - General tennis analysis techniques

## Requirements

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)
- Video files in the project directory

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/cBarredez/-tennis-court-detector-and-player.git
   cd tennis-court-detector-and-player
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained weights from [Google Drive](https://drive.google.com/drive/u/0/folders/1rKZ3enwbEEkmUnevNYLWZqOpQsGIlK6B) and place them in the `Weights/` directory:
   - `bestv3.pt` - YOLOv8 model for player detection
   - `keypoints_model_v2.pth` - ResNet101 model for keypoint detection
   - `yolo11x-pose.pt` - YOLO pose estimation model

## Usage

To process a tennis match video:

```bash
python tennis.py --input input_video.mp4 --output output_video.mp4
```

Replace `input_video.mp4` with your video file and `output_video.mp4` with your desired output filename.

## Project Structure

- `Tennis.ipynb`: Jupyter notebook with the main analysis code (Spanish)
- `Tennis_EN.py`: Python script with the complete implementation in English
- `tennis.py`: Main Python script for video processing
- `sort.py`: SORT (Simple Online and Realtime Tracking) implementation
- `requirements.txt`: List of Python dependencies
- `Weights/`: Directory containing pre-trained model weights
- `images/`: Directory containing demo and result images/GIFs

## Demo

![Tennis Court Detection Demo](images/ezgif-71e4ee9f3e4d1b.gif)

*Example of the tennis court detection and player tracking in action.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
