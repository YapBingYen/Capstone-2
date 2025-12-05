import cv2
import os

def crop_cat_faces(input_dir, output_dir, cascade_path='haarcascade_frontalcatface.xml'):
    """
    Detects and crops cat faces from images in input_dir and saves them to output_dir.
    Keeps folder structure (breed or individual subfolders).
    Each cropped image is resized to 224x224 for model compatibility.
    """
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

                if len(faces) > 0:
                    # Choose the largest detected face
                    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    cropped = img[y:y+h, x:x+w]
                else:
                    # fallback: use resized original
                    cropped = img

                # ✅ Resize to 224x224 for model input
                cropped = cv2.resize(cropped, (224, 224))

                # Preserve subfolder structure
                rel_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, cropped)

    print(f"✅ Cropping complete! Cropped images saved in: {output_dir}")


# Example usage
input_folder = r"D:\Cursor AI projects\Capstone2.1\dataset\cats"
output_folder = r"D:\Cursor AI projects\Capstone2.1\dataset_cropped\cats"
cascade_file = r"D:\Cursor AI projects\Capstone2.1\haarcascade_frontalcatface.xml"

crop_cat_faces(input_folder, output_folder, cascade_file)
