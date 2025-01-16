# Blood Cell Detection using YOLO and Streamlit

This project uses a YOLO (You Only Look Once) model to detect and classify Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets in images. A user-friendly interface has been developed using Streamlit, where users can upload an image for detection. The project is deployed on Hugging Face Spaces for easy access.

## Installation

To set up and run the project locally, follow the instructions below:

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/Narsireddy-Konapalli/blood-cell-detection.git
    cd blood-cell-detection
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    venv\Scripts\activate  # For Windows
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should include the following libraries:

    ```txt
    streamlit
    torch
    opencv-python
    pillow
    numpy
    matplotlib
    ```

4. The pre-trained YOLO model is included in the repository, so no need for separate download instructions.

5. Start the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Run the Streamlit app as described in the **Installation** section.
2. Once the app is running, go to the provided local URL in your browser (typically `http://localhost:8501`).
3. Upload an image containing blood cells (e.g., from a microscope).
4. The app will display the uploaded image with bounding boxes around the detected Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets. The detection results are shown with labels.

## How It Works

- The YOLO model is trained to detect blood cells in images.
- The Streamlit app provides an interface for users to upload images.
- The uploaded image is passed through the YOLO model, which detects and classifies RBCs, WBCs, and Platelets.
- The output image displays bounding boxes around the detected cells, with labels indicating the type of cell.

## Deployment

The project is deployed on Hugging Face Spaces for easy access and usage:

[Blood Cell Detection on Hugging Face Spaces](https://huggingface.co/spaces/narsireddy/BloodAnalyzer)

## Interface

The model's interface allows users to:

- Upload an image of a blood sample.
- Detect and classify Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets in the image.
- View the image with bounding boxes drawn around the detected cells, labeled with their respective types.

Here’s how the interface looks:

![Interface Screenshot](assets/screenshot.png)

## Acknowledgments

- YOLO (You Only Look Once) for object detection.
- Streamlit for building the web app interface.
- Hugging Face for hosting the deployment.
- OpenCV and PyTorch for image processing and model training.
