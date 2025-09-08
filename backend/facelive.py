import cv2
import numpy as np
import faiss
import firebase_admin
from firebase_admin import credentials, firestore
from insightface.app import FaceAnalysis

# Initialize Firestore
try:
    cred = credentials.Certificate(r"eagleeye-e9d6e-firebase-adminsdk-fbsvc-53fc55a2ec.json")  # Update with your JSON path
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Firebase initialization error (may already be initialized): {e}")

db = firestore.client()

# Initialize FaceAnalysis model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def load_face_database():
    try:
        faces_ref = db.collection("faces").stream()
        records = []
        feature_vectors = []
        
        for doc in faces_ref:
            data = doc.to_dict()
            if "feature_vector" in data and "name" in data:
                raw_vector = data["feature_vector"]
                if isinstance(raw_vector, str):
                    print(f"Warning: feature_vector for doc {doc.id} is a string, skipping.")
                    continue
                
                try:
                    feature_vector = np.array(raw_vector, dtype=np.float32)
                except Exception as e:
                    print(f"Error converting feature_vector for doc {doc.id}: {e}")
                    continue

                if feature_vector.size == 0:
                    continue

                records.append((doc.id, data["name"], data.get("location", "Unknown")))
                feature_vectors.append(feature_vector)
        
        if not feature_vectors:
            print("No valid face vectors found in Firestore database.")
            return None, []

        feature_vectors = np.vstack(feature_vectors)
        faiss.normalize_L2(feature_vectors)

        index = faiss.IndexFlatIP(feature_vectors.shape[1])
        index.add(feature_vectors)

        print(f"Loaded {len(records)} faces into FAISS index.")
        return index, records
    except Exception as e:
        print(f"Error loading face database: {e}")
        return None, []

def recognize_live():
    index, records = load_face_database()
    if index is None:
        print("No faces in database or failed to load index.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    threshold = 0.3
    print("Starting live face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_width = frame.shape[1]

        # Detect faces on the original Frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_app.get(frame_rgb)

        # Flip frame for display after Detection
        flipped_frame = cv2.flip(frame, 1)

        print(f"Faces detected: {len(faces)}")

        for face in faces:
            matched_name = "Unknown"
            matched_location = "Unknown"
            best_similarity = 0.3

            feature_vector = face.embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(feature_vector)

            similarity, index_match = index.search(feature_vector, 1)
            best_similarity = similarity[0][0]

            x1, y1, x2, y2 = map(int, face.bbox)
            x1 = x2
            x2 = x1
    

            if best_similarity >= threshold:
                matched_id, matched_name, matched_location = records[index_match[0][0]]
                print(f"Recognized: {matched_name} from {matched_location} (Similarity: {best_similarity:.2f})")

            # Draw in Flipped frame
            cv2.rectangle(flipped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_text = f"{matched_name} - {matched_location} ({best_similarity:.2f})"
            cv2.putText(flipped_frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display_width = 800
        aspect_ratio = display_width / flipped_frame.shape[1]
        display_height = int(flipped_frame.shape[0] * aspect_ratio)
        resized_frame = cv2.resize(flipped_frame, (display_width, display_height))

        cv2.imshow("Live Face Recognition", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting live recognition.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_live()

