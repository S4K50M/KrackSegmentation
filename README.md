<pre style="color: red;">
____  _____ _____ ____  ____  _____ 
 |  _ \| ____/ ____/ ___|/ __ \|  _ \ 
 | |_) |  _|| |   | |   | |  | | | | |
 |  __/| |__| |___| |___| |__| | |_| |
 |_|   |_____\_____\____|\____/|____/ 
                                      
 TERRAIN SEMANTIC SEGMENTATION SYSTEM
 ====================================
</pre>

This repository hosts a state-of-the-art semantic segmentation pipeline designed for complex terrain analysis. It utilizes a DINOv2 (ViT-B/14) backbone fused with a custom deep residual decoder to achieve high-fidelity segmentation across 11 terrain classes. The system includes a full training pipeline, an automated evaluation suite, and a Flask-based web interface integrated with AWS SageMaker for real-time inference.

[ LIVE DEMO ]
The model is currently deployed and accessible at:
http://43.204.148.104:8000

---

SYSTEM ARCHITECTURE
-------------------
The model (DinoTerrainMax) employs a dual-path architecture:
1.  Encoder: DINOv2 (Vision Transformer) extracts deep semantic features.
2.  Context: ASPP (Atrous Spatial Pyramid Pooling) captures multi-scale context.
3.  Decoder: A custom Residual Decoder with Attention Gates filters and refines features.
4.  Fusion: A dedicated Low-Level Pathway preserves edge details from the raw image.

Classes Supported:
0: Background    1: Trees           2: Lush Bushes
3: Dry Grass     4: Dry Bushes      5: Ground Clutter
6: Logs          7: Rocks           8: Landscape
9: Sky           10: Other

---

PREREQUISITES
-------------
- Python 3.9+
- CUDA-enabled GPU (Recommended for training)
- AWS Account (For deployment)

Required Libraries:
  pip install torch torchvision flask boto3 pillow numpy tqdm matplotlib

---

INSTALLATION & LOCAL SETUP
--------------------------

1. CLONE REPOSITORY
   git clone <repository_url>
   cd <repository_folder>

2. DATA PREPARATION
   Ensure your dataset follows this structure:
   /data
     /train
       /Color_Images (JPG/PNG)
       /Segmentation (PNG Masks)
     /val
       /Color_Images
       /Segmentation
     /test
       /Color_Images
       /Segmentation

3. TRAINING
   To train the model from scratch:
   python train.py

   This will:
   - Initialize the DINOv2 backbone.
   - Train for the configured epochs.
   - Save the best weights to 'stats_v2_final/best_model.pth'.
   - Automatically evaluate on the test set upon completion.

---

WEB INTERFACE (LOCAL)
---------------------
You can run the web interface locally without AWS if you have a GPU.

1. MODIFY APP.PY (Optional)
   If you do not have an active SageMaker endpoint, you will need to modify 
   app.py to load the model locally instead of calling boto3. 
   (See 'local_test.py' for reference on local inference).

2. RUN THE APP
   python app.py

   Access the interface at: http://127.0.0.1:8000

---

DEPLOYMENT (AWS SAGEMAKER)
--------------------------
To deploy the trained model to the cloud:

1. CONFIGURE AWS
   Ensure you have configured your AWS CLI with appropriate permissions:
   aws configure

2. UPDATE CONFIG
   Edit 'deploy_final.py' with your specific ARN and Region:
   REGION = "ap-south-1"
   ROLE_ARN = "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/<YOUR_ROLE>"

3. DEPLOY
   python deploy_final.py

   This script will:
   - Package the model and inference code into 'model.tar.gz'.
   - Upload the artifact to S3.
   - Provision a SageMaker Endpoint (ml.g4dn.xlarge).

4. STOPPING THE ENDPOINT
   To stop billing, delete the endpoint when not in use:
   python stop.py

---

METRICS & EVALUATION
--------------------
The system automatically calculates:
- Mean Intersection over Union (mIoU)
- Pixel Accuracy
- Per-Class IoU

During inference via the Web UI, users can optionally upload a Ground Truth 
mask to receive instant metric calculation alongside the visual prediction.

---

LICENSE
-------
This project is open-source.
