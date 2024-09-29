
Img_Segment_Object_analysis.ipynb is created to launch the Streamlit testing app.

Upload the project on google drive at below path: 
MyDrive/Image_Segmentation/Project_root

Steps :
1. Mount google drive
	#mount google drive
	from google.colab import drive
	drive.mount('/content/drive')
2. Install Packages
	!pip install pytesseract
	!pip install streamlit -q
	!pip install yolov5 # install the missing module
	!sudo apt-get install tesseract-ocr
	!pip install pytesseract
	!pip install pytesseract
	!pip install Pillow
	!apt-get install tesseract-ocr
	!which tesseract
3. Get the IP address to launch streamlit
	!wget -q -O - ipv4.icanhazip.com
4. !npm install -g localtunnel@2.0.2
	Install localtunnel@2
5. Launch Streamlit
	!streamlit run /content/drive/MyDrive/Image_Segmentation/Project_root/streamlit_app/app.py & npx localtunnel --port 8501



Code and Folder Structure is as below:

Folder Structure:
project_root/
│
├── data/
│   ├── input_images/               # Directory for input images
│   ├── segmented_objects/          # Directory to save segmented object images
│   └── output/                     # Directory for output images and tables
│
├── models/
│   ├── segmentation_model.py       # Script for segmentation model
│   ├── identification_model.py     # Script for object identification model
│   ├── text_extraction_model.py    # Script for text/data extraction model
│   └── summarization_model.py      # Script for summarization model
│
├── utils/
│   ├── preprocessing.py            # Script for preprocessing functions
│   ├── postprocessing.py           # Script for postprocessing functions
│   ├── data_mapping.py             # Script for data mapping functions
│   └── visualization.py            # Script for visualization functions
│
├── streamlit_app/
│   ├── app.py                      # Main Streamlit application script
│   └── components/                 # Directory for Streamlit components
│
├── tests/
│   ├── test_segmentation.py        # Tests for segmentation
│   ├── test_identification.py      # Tests for identification
│   ├── test_text_extraction.py     # Tests for text extraction
│   └── test_summarization.py       # Tests for summarization
│
├── README.md                       # Project overview and setup instructions
├── requirements.txt                # Required Python packages
└── presentation.pptx               # Presentation slides summarizing the project
