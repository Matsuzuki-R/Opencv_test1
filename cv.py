import face_recognition
import cv2
import numpy as np

# これは、ウェブカメラからのライブビデオで顔認識を実行するデモです。少し複雑ですが、いくつかの基本的なパフォーマンス調整が含まれています：
#   1. ビデオフレームごとに1/4の解像度で処理（ただし、フル解像度で表示）
#   2. 顔を検出するのはビデオの2フレームごと

# この例は、ウェブカメラからの読み取り専用にOpenCV（cv2ライブラリ）が必要です。
# face_recognitionライブラリを使用するためにはOpenCVは必要ありません。特定のデモを実行したい場合にのみ必要です。インストールに問題がある場合は、OpenCVを必要としない他のデモを試してみてください。

# ウェブカメラ #0（デフォルトのもの）への参照を取得
video_capture = cv2.VideoCapture(0)

# サンプル画像をロードし、認識方法を学習させる
a_image = face_recognition.load_image_file("./a.jpg")
a_face_encoding = face_recognition.face_encodings(a_image)[0]

# 2番目のサンプル画像をロードし、認識方法を学習させる
b_image = face_recognition.load_image_file("./b.jpg")
b_face_encoding = face_recognition.face_encodings(b_image)[0]

# 3番目のサンプル画像をロードし、認識方法を学習させる
c_image = face_recognition.load_image_file("./c.jpg")
c_face_encoding = face_recognition.face_encodings(c_image)[0]

# 知っている顔のエンコーディングとその名前の配列を作成
known_face_encodings = [
    a_face_encoding,
    b_face_encoding,
    c_face_encoding
]
known_face_names = [
    "a_person",
    "b_person",
    "c_person"
]

# いくつかの変数を初期化
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # ビデオの単一フレームを取得
    ret, frame = video_capture.read()

    # 処理時間を節約するために、ビデオの2フレームごとにのみ処理
    if process_this_frame:
        # 顔認識処理を高速化するために、ビデオフレームを1/4サイズにリサイズ
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # 画像をBGR色（OpenCVが使用）からRGB色（face_recognitionが使用）に変換
        rgb_small_frame = small_frame[:, :, ::-1]

        code = cv2.COLOR_BGR2RGB
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)
        
        # 現在のビデオフレームでのすべての顔と顔エンコーディングを見つける
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 顔が既知の顔と一致するかどうかを確認
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # known_face_encodingsで一致が見つかった場合は、最初のものを使用
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # または、既知の顔で新しい顔に最も近い距離のものを使用
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # 結果を表示
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 顔の位置を元に戻す（検出したフレームは1/4サイズにスケーリングされているため）
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 顔の周りにボックスを描画
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 顔の下に名前のラベルを描画
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 結果の画像を表示
    cv2.imshow('Video', frame)

    # キーボードの'q'を押すと終了！
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ウェブカメラのハンドルを解放
video_capture.release()
cv2.destroyAllWindows()
