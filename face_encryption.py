from face_encryption_method import FaceEncryptionMethod
import dlib
import numpy as np
from skimage import io
from copy import deepcopy

from face_encryption_model import FaceEncryptionModel


def _mosaic(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    对图像的指定区域进行马赛克处理
    :param img: 输入图像 (BGR格式)
    :param x: 区域左上角x坐标
    :param y: 区域左上角y坐标
    :param w: 区域宽度
    :param h: 区域高度
    :return: 马赛克处理后的图像
    """
    # 检查区域是否超出图像边界
    height, width = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, width - x), min(h, height - y)

    # 马赛克块大小
    block_size = 10

    # 遍历区域内的每个马赛克块
    for i in range(y, y + h, block_size):
        for j in range(x, x + w, block_size):
            # 当前块的结束位置（防止越界）
            i_end = min(i + block_size, y + h)
            j_end = min(j + block_size, x + w)

            # 取当前块左上角像素的颜色，填充整个块
            color = img[i, j]
            img[i:i_end, j:j_end] = color

    return img


class FaceEncryption:
    def __init__(
        self,
        method: FaceEncryptionMethod = FaceEncryptionMethod.MOSAIC,
        model: FaceEncryptionModel = FaceEncryptionModel.YOLO,
    ):
        self.method = method
        self.model = model
        if model == FaceEncryptionModel.YOLO:
            from ultralytics import YOLO

            self.detector = YOLO("yolov8n-face.pt")
        else:
            self.detector = dlib.get_frontal_face_detector()

    def encrypt(self, img_path: str) -> np.ndarray:
        img = io.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.method == FaceEncryptionMethod.NONE:
            return img
        if self.model == FaceEncryptionModel.YOLO:
            import supervision as sv

            result = self.detector(img)[0]
            dets = sv.Detections.from_ultralytics(result)
            if self.method == FaceEncryptionMethod.MOSAIC:
                for box in dets.xyxy:
                    x, y, w, h = (
                        int(box[0]),
                        int(box[1]),
                        int(box[2] - box[0]),
                        int(box[3] - box[1]),
                    )
                    img = _mosaic(img, x, y, w, h)
                return img
            for box in dets.xyxy:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                img[y1:y2, x1:x2] = 0
            return img
        else:
            dets = self.detector(img, 1)
            print(f"***  Number of faces detected: {len(dets)} ***")

            if len(dets) == 0:
                return img

            if self.method == FaceEncryptionMethod.MOSAIC:
                for face in dets:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    img = _mosaic(img, x, y, w, h)
                return img
            for face in dets:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                img[y : y + h, x : x + w] = 0
            return img


if __name__ == "__main__":
    import cv2

    imgpath = "test.jpg"
    face_encryption = FaceEncryption(
        model=FaceEncryptionModel.DLIB, method=FaceEncryptionMethod.CROP
    )
    img = face_encryption.encrypt(imgpath)
    cv2.imwrite("encrypted.jpg", img)

    face_encryption2 = FaceEncryption()
    img = face_encryption2.encrypt(imgpath)
    cv2.imwrite("encrypted2.jpg", img)
