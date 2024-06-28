import cv2

def read_video(video_path) : 
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames



def save_video(output_video_frames, output_video_path): #프레임 리스트, 저장 경로
    fourcc = cv2.VideoWriter_fourcc(* 'XVID') # 영상 코덱 정의, 주로 xvid 사용
    out = cv2.VideoWriter(output_video_path, fourcc, 30, 
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) # 프레임 너비, 높이
    # VideoWriter : 비디오 작성기 객체
    # 30 : 프레임 속도
    for frame in output_video_frames:
        out.write(frame)
    out.release()



