### (如果图像中包含人脸)避免个人人脸信息泄露

install:
```bash
pip install -r requirements.txt
```

usage:
```python
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
```