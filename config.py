"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np


# 基礎設定類別
# 不要直接使用這個類別。相反地，繼承它產生一個子類別，並覆寫你需要修改的配置。
class Config(object):
    """Base configuration 類別。對於一些客製化的自定義配置，請創建一個
    從這個類別繼承的子類別和覆寫需要修改屬性。
    """
    # 命名這個配置。例如，'COCO'，'Experiment 3'，...等。
    NAME = None  # Override in sub-classes

    # 使用GPU的數量。如果要使用CPU來進行模型訓練，請設定為 1
    GPU_COUNT = 1

    # 在每個GPU上訓練的圖像數量。一個有12GB記憶體的GPU通常可以
    # 處理2張1024x1024像素的圖片。
    # 根據你的GPU記憶體和圖像大小進行調整。使用GPU可以處理的最大數字以獲得最佳性能。
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    # 每個訓練循環的訓練步數 (training steps per epoch)
    # 這個配置並不需要匹配訓練集的大小。 Tensorboard在每個時代的末尾會更新保存，因此將其
    # 設置為更小的數字意味著更頻繁的TensorBoard更新。驗證統計數據也會在每個時代
    # 結束時計算出來，而且可能需要一段時間，所以不要將其設置的太小以免花費大量時間
    # 在驗證統計信息上。
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    # 在每個訓練循環結束時要運行的驗證步驟次數。
    # 更大的數字會提高了驗證統計的準確性，但會減慢訓練速度。
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.    
    # FPN金字塔網絡的每一層的步幅。這些值基於Resnet101的backbonbe。
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    # 圖像類別的數量(包括背景)
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels
    # 方形錨點的長度(以像素為單位)
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    # 每個cell錨點的比例(寬度/高度)
    # 值1表示方形錨，0.5表示寬錨點
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    # 錨點步幅
    # 如果為1，則在特徵圖中的每個單元格創建錨點。
    # 如果是2，則為每個其他單元格創建錨點，依此類推。
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    # 過濾RPN提議的非最大抑制閾值(Non-max suppression threshold)。
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    # 每個圖像有多少錨點用於RPN訓練
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    # 在非最大抑制過濾後感興趣區域(Region of Interest)留下來的個數
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    # 如果啟用，則將實例遮罩調整為較小的大小以減少記憶體的使用。
    # 建議當用來偵測高解析度的圖像時使用。
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    # 輸入圖像的大小修正
    # 調整圖像的大小，使最小的邊>= IMAGE_MIN_DIM，最長的邊<= IMAGE_MAX_DIM。
    # 如果兩個條件不能一起滿足，則強制執行IMAGE_MAX_DIM。
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # If True, pad images with zeros such that they're (max_dim by max_dim)
    # 如果為True，則用零填充圖像（max_dim by max_dim）
    IMAGE_PADDING = True  # 目前，不支持False選項

    # Image mean (RGB)
    # 圖像像素值的平均值 (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # 提供給分類器/遮罩決議的每個圖像的ROI數量
    # Mask-RCNN論文使用512個，但RPN往往沒有產生足夠的正面建議來填補這一點，
    # 並保持1：3的正負比例。可以通過調整RPN NMS閾值來增加提議(Region proposal)的數量。
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    # 用於訓練分類器/遮罩決議的正ROI百分比
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    # ROIs池
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    # 一個圖像中真實實例的最大數量
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    # RPN和最終檢測的邊界框細化標準差。
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    # 最大圖像物件實例的數量
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    # 接受為檢測到的實例的最小機率值
    # 低於此閾值的ROI將被跳過
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    # 檢測的非最大抑制閾值
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    # 學習速度和momentum
    # Mask RCNN論文使 lr=0.02, 但是在Tensorflow這樣的設定會導致權重爆炸。
    # 這可能是由於optimzer實踐的差異的導致。
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    # 權重衰減正規化
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    # 使用RPN ROIs或是外部生成的ROIs來進行訓練
    # 在大多數情況下保持這個設定。如果要訓練由代碼生成的ROI而不是來自RPN的ROI，
    # 則設置為False。例如，debug分類器頭，而不必訓練RPN。
    USE_RPN_ROIS = True

    def __init__(self):
        """Set values of computed attributes."""
        # 有效的批量大小
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # 輸入圖像大小
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # 從輸入圖像大小計算要修正作為網絡輸入的圖像大小
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

    # 打印出這個設定的所有key:value訊息
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
