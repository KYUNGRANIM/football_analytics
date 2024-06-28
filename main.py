from utils import read_video, save_video
from tracker import Tracker
import os
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner



def main():
    # 영상 읽기
    video_frames = read_video(r"C:\krpython\Final Project\football\pretrain\input_video\08fd33_4.mp4")

    # tracker 초기화
    tracker = Tracker(r"C:\krpython\Final Project\football\pretrain\model\best.pt")
    
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=r"C:\krpython\Final Project\football\pretrain\stub\track_stub.pkl")  # stub 파일 먼저 생성 후, 파일명 지정


    # interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positios(tracks['ball'])

    # 크롭 이미지를 저장할 디렉토리 경로
    """output_dir = r"C:\krpython\Final Project\football\pretrain\output_video"
    
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 선수 크롭이미지
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        # 크롭 이미지
        cropped_image = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]):int(bbox[2])] 

        # 크롭 이미지 저장 경로
        cropped_image_path = os.path.join(output_dir, "cropped_img.jpg")

        # 크롭 이미지 저장
        cv2.imwrite(cropped_image_path, cropped_image)
        break"""

    # team assign
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], 
                                                 track['bbox'],
                                                 player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # ball assign
    player_assiger = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assiger.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])

        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control= np.array(team_ball_control)


    # 객체 트랙 그리기
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)


    # 영상 저장
    save_video(output_video_frames, r"C:\krpython\Final Project\football\pretrain\output_video\output_video.mp4")

if __name__ == "__main__":
    main()
