import math
import os
import pickle
import random
import time

import cv2
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class TrainFaceData:
    def __init__(self, train_dir, model_save_path, n_neighbors, knn_algo='ball_tree', verbose=True):
        self.train_dir = train_dir
        self.model_save_path = model_save_path
        self.n_neighbors = n_neighbors
        self.knn_algo = knn_algo
        self.verbose = verbose
        self.knn_clf = None
        self.X_val = []
        self.y_val = []
        self.X_test = []
        self.y_test = []
        self.X_train = []
        self.y_train = []
        self.accuracies = []

    def train(self):

        # 학습 세트의 각 사람을 루프
        for class_dir in os.listdir(self.train_dir):
            if not os.path.isdir(os.path.join(self.train_dir, class_dir)):
                continue
            img_paths = image_files_in_folder(os.path.join(self.train_dir, class_dir))
            valid_img_paths = []  # 얼굴이 인식된 이미지 경로를 저장할 리스트

            # 현재 사람에 대한 각 학습 이미지를 루프
            for img_path in img_paths:
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # 학습 이미지에 사람이 없거나 너무 많은 경우 이미지를 건너뜀
                    if self.verbose:
                        print("이미지 {}는 학습에 적합하지 않음: {}".format(img_path, "얼굴을 찾지 못함" if len(face_bounding_boxes) < 1 else "여러 얼굴을 찾음"))
                else:
                    valid_img_paths.append(img_path)

            # 인식된 얼굴이 있는 이미지만 shuffle하고 X에 추가
            random.shuffle(valid_img_paths)

            # 이미지를 학습, 검증 및 테스트 세트로 나누기 위한 비율 설정
            train_ratio = 0.7
            val_ratio = 0.2

            # 학습, 검증 및 테스트 세트로 나누기 위한 인덱스 계산
            num_train = int(valid_img_paths * train_ratio)
            num_val = int(valid_img_paths * val_ratio)

            # 이미지를 학습, 검증 및 테스트 세트로 나누기
            train_paths = valid_img_paths[:num_train]
            val_paths = valid_img_paths[num_train:num_train + num_val]
            test_paths = valid_img_paths[num_train + num_val:]

            # 이미지 데이터 증강 객체 생성
            datagen = ImageDataGenerator(
                rotation_range=20,  # 회전 범위 (degree)
                width_shift_range=0.2,  # 가로 이동 범위 (전체 가로의 20%)
                height_shift_range=0.2,  # 세로 이동 범위 (전체 세로의 20%)
                shear_range=0.2,  # 전단 변환 범위
                zoom_range=0.2,  # 확대/축소 범위
                horizontal_flip=True,  # 가로 반전 여부
                fill_mode='nearest'  # 이동 시 빈 공간 채우기 방식
            )

            # train 데이터 처리
            for img_path in train_paths:
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # 검증 이미지에 사람이 없거나 너무 많은 경우 이미지를 건너뜀
                    if self.verbose:
                        print("이미지 {}는 검증에 적합하지 않음: {}".format(img_path, "얼굴을 찾지 못함" if len(face_bounding_boxes) < 1 else "여러 얼굴을 찾음"))
                else:
                    # 현재 이미지의 얼굴 인코딩을 train 세트에 추가
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                    self.X_train.append(face_encoding)
                    self.y_train.append(class_dir)

                    # 데이터 증강
                    aug_iter = datagen.flow(np.expand_dims(image, 0), batch_size=1)
                    for i in range(4): # 원본 이미지 + 증강 이미지 4개 추가
                        aug_img = next(aug_iter)[0].astype('uint8')
                        aug_face_encoding = face_recognition.face_encodings(aug_img, known_face_locations=face_bounding_boxes)[0]
                        self.X_train.append(aug_face_encoding)
                        self.y_train.append(class_dir)

            # 검증 데이터 처리
            for img_path in val_paths:
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # 검증 이미지에 사람이 없거나 너무 많은 경우 이미지를 건너뜀
                    if self.verbose:
                        print("이미지 {}는 검증에 적합하지 않음: {}".format(img_path, "얼굴을 찾지 못함" if len(face_bounding_boxes) < 1 else "여러 얼굴을 찾음"))
                else:
                    # 현재 이미지의 얼굴 인코딩을 검증 세트에 추가
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                    self.X_val.append(face_encoding)
                    self.y_val.append(class_dir)

            # 테스트 데이터 처리
            for img_path in test_paths:
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # 테스트 이미지에 사람이 없거나 너무 많은 경우 이미지를 건너뜀
                    if self.verbose:
                        print("이미지 {}는 테스트에 적합하지 않음: {}".format(img_path, "얼굴을 찾지 못함" if len(face_bounding_boxes) < 1 else "여러 얼굴을 찾음"))
                else:
                    # 현재 이미지의 얼굴 인코딩을 테스트 세트에 추가
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                    self.X_test.append(face_encoding)
                    self.y_test.append(class_dir)

        # knn 분류기에서 가중치를 적용할 이웃의 수를 결정
        if self.n_neighbors is None:
            self.n_neighbors = int(round(math.sqrt(len(self.X_train))))
            if self.verbose:
                print("자동으로 선택된 n_neighbors:", self.n_neighbors)

        # knn 분류기 생성 및 학습
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.knn_algo, weights='distance')
        knn_clf.fit(self.X_train, self.y_train)

        # 학습된 knn 분류기 저장
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        return knn_clf

    def evaluate(self, knn_clf):
        y_pred = knn_clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        self.accuracies.append(acc)
        return acc

    def cross_validate(self, n_splits=5):
        # 결정 트리 분류기 객체 생성
        clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.knn_algo, weights='distance')

        # n-fold 교차 검증 수행
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        accuracies = []
        for train_index, test_index in kf.split(self.X_val):
            # 데이터 분할
            X_train, X_test = [self.X_val[i] for i in train_index], [self.X_val[i] for i in test_index]
            y_train, y_test = [self.y_val[i] for i in train_index], [self.y_val[i] for i in test_index]

            # 모델 학습
            clf.fit(X_train, y_train)

            # 예측 및 정확도 계산
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        # 교차 검증 정확도 출력
        print(f"{n_splits}-fold 교차 검증 정확도: {sum(accuracies) / len(accuracies):.4f}")

class StudentAttendance:
    def __init__(self, file_path):
        self.file_path = file_path

    # 출석 파일을 읽어 이미 출석된 사람의 이름 반환
    def load_attendance(self):
        if not os.path.exists(self.file_path):
            return set()
        
        with open(self.file_path, "r") as f:
            return set(line.strip() for line in f.readlines())

    # 출석 파일에 이름 기록
    def mark_attendance(self, name):
        #attendance_set = self.load_attendance()  # 기존 출석 이름 로드
        if name not in attendance_set:
            with open(self.file_path, "a") as f:
                f.write(f"{name}\n")
            attendance_set.add(name)

    def reset_attendance(self):
        with open(self.file_path, 'w') as file:
            file.write('')  # 파일 내용 비우기
        return set()

class FaceRecognition:
    def __init__(self) -> None:
        pass

    def recognize_face(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_distance = []
        #face_recognition_frames = {}

        # 입력된 얼굴 인코딩과 가장 가까운 이웃 사이의 거리 계산
        for face_encoding in face_encodings:
            closest_distances = knn_clf.kneighbors([face_encoding], n_neighbors=1)
            face_distance = closest_distances[0][0][0]
            is_recognized = closest_distances[0][0][0] <= 0.4  # 거리가 임계값 이하일 때만 True
            if is_recognized:
                name = knn_clf.predict([face_encoding])[0]
                face_names.append(name)
                #face_recognition_frames[name] = face_recognition_frames.get(name, 0) + 1
                #attendance.mark_attendance(name)
            # 학습되지 않은 얼굴인 경우
            else:
                face_names.append("Unknown")
        
        return face_locations, face_names, face_distance#, face_recognition_frames

# 학습 디렉토리 설정
#train_dir = "/Users/yewon/Desktop/linearAlgebra2_face_detection_datasets/team1"
attendance_file = "attendance.txt"

with open('/Users/yewon/Desktop/q/model_cleandata.h5', 'rb') as f:
    knn_clf = pickle.load(f) 

# KNN 모델 학습
# train_data = TrainFaceData(train_dir, model_save_path="model_35.h5", n_neighbors=None, knn_algo='ball_tree', verbose=False)
# knn_clf = train_data.train()

# train_data.cross_validate()

# 테스트 데이터로 모델 성능 평가
# test_accuracy = knn_clf.score(train_data.X_test, train_data.y_test)
# print(f"테스트 데이터 정확도: {test_accuracy:.4f}")

face_recognition_times = {}
already_attended = set()

# 출석 파일에서 이미 출석된 사람들의 이름 로드
attendance = StudentAttendance(attendance_file)
attendance.reset_attendance()
attendance_set = attendance.load_attendance()

process_this_frame = True

frame_threshold = 90  # 3초 동안의 프레임 수 (30fps 기준)

# 웹캠에서 비디오 캡처
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)

    if process_this_frame:
        recognization = FaceRecognition()
        face_locations, face_names, distance = recognization.recognize_face(frame)

        # 최초로 인식된 얼굴의 시간 기록
        current_time = time.time()
        for name in face_names:
            if name != "Unknown":
                current_names = face_names
                length = len(current_names)
                if name not in face_recognition_times:
                    face_recognition_times[name] = current_time
                    past_names = face_names

                else:
                    if(length != len(past_names)):
                        new_index = []
                        if length > len(past_names):
                            for i in range(length):
                                if current_names[i] not in past_names:
                                    # 현재 얼굴 리스트에는 있지만 과거 얼굴 리스트에는 없는 요소를 찾아서 과거 얼굴 리스트에 insert
                                    past_names.insert(i, current_names[i])

                        else:
                            for i in range(len(past_names)):
                                if past_names[-i] not in current_names:
                                    # 현재 얼굴 리스트에는 없지만 과거 얼굴 리스트에는 있는 요소를 찾아서 과거 얼굴 리스트에서 remove
                                    past_names.remove(past_names[-i])

                    # 현재 얼굴 리스트와 과거 얼굴 리스트를 비교하면서
                    for i in range(length):
                        # 현재 얼굴이 과거 얼굴과 같은지 확인
                        if past_names[i] == current_names[i]:
                            # 동일한 얼굴인 경우 시간을 측정하고 기록을 업데이트
                            if name not in face_recognition_times:
                                face_recognition_times[name] = current_time
                            elapsed_time = current_time - face_recognition_times[name]
                            if elapsed_time >= 3:  # 3초 이상 인식된 경우
                                if name not in already_attended:
                                    attendance.mark_attendance(name)
                                    already_attended.add(name)  # 출석부에 등록
                                    del face_recognition_times[name]  # 딕셔너리에서 기록 제거
                                else:
                                    #print(f"{name}은(는) 이미 출석부에 등록되어 있습니다.")
                                    del face_recognition_times[name]
                            past_names = face_names
                        else:                    
                            del face_recognition_times[name]  # 딕셔너리에서 기록 제거
            else:
                # 얼굴이 인식되지 않은 경우 시간값을 기록하지 않음                        
                pass

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f'{name} {(1-(round(distance,2)))*100}%', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 출석된 사람들의 이름을 화면 모퉁이에 표시
    y0, dy = 50, 20
    cv2.putText(frame, "Attendance", (5, y0 - dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    for i, name in enumerate(attendance_set):
        y = y0 + i * dy
        cv2.putText(frame, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
