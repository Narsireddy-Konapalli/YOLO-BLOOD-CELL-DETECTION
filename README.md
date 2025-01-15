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
    git clone https://github.com/your-username/blood-cell-detection.git
    cd blood-cell-detection
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should include the following libraries:

    ```txt
    streamlit
    torch
    torchvision
    opencv-python
    pillow
    numpy
    matplotlib
    ```

4. Download the pre-trained YOLO model for blood cell detection.

    (Provide a link or instructions for the model if not included in the repository)

5. Start the Streamlit app:

    ```bash
    streamlit run app.py
    ```

### Requirements

- Python 3.x
- Streamlit
- PyTorch
- OpenCV
- Pillow
- NumPy
- Matplotlib

## Usage

1. Run the Streamlit app as described in the **Installation** section.
2. Once the app is running, go to the provided local URL in your browser (typically `http://localhost:8501`).
3. Upload an image containing blood cells (e.g., from a microscope).
4. The app will display the detected cells with labels for Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets, along with their count.

## How It Works

- The YOLO model is trained to detect blood cells in images.
- The Streamlit app provides an interface for users to upload images.
- The uploaded image is passed through the YOLO model, which detects and classifies RBCs, WBCs, and Platelets.
- The output image is displayed with bounding boxes around the detected cells, and the count of each type is shown.

## Deployment

The project is deployed on Hugging Face Spaces for easy access and usage:

[Blood Cell Detection on Hugging Face Spaces](https://huggingface.co/spaces/your-username/blood-cell-detection)

To deploy on your own Hugging Face Spaces:

1. Create an account on Hugging Face (if you haven't already).
2. Fork the repository or upload your project files to a new Space.
3. Follow the Hugging Face documentation for deploying Streamlit apps.

## Contributing

If you'd like to contribute to this project:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLO (You Only Look Once) for object detection.
- Streamlit for building the web app interface.
- Hugging Face for hosting the deployment.
- OpenCV and PyTorch for image processing and model training.

