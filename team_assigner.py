#kmeans를 사용하여 이미지 프레임 내에서 플레이어를 두 팀으로 나누는데 사용

#1. 필요 라이브러리
from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    #1. __init__ 메서드 : 초기화
    def __init__(self):
        self.team_colors = {} #팀 색상 지정
        self.player_team_dict = {}  #선수, 팀 id 매핑

    #2. 입력 이미지에서 kmeans 클러스터링 모델 생성
    # 입력 이미지를 2D 배열로 변환
    def get_clustering_model(self, image):
        #2D 배열로 변환
        image_2d = image.reshape(-1, 3) #(높이*너비, RGB 색상 값)
        #k-means 클러스터링 모델 생성
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)
        #kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0).fit(image_2d)
        #init="k-means++" : 최적의 클러스터로 묶을 가능성을 높인 알고리즘(중심점을 신중하게 선택)
        #n_init : 초기 중심점을 선택하는데 사용되는
        
        return kmeans

    #3. 플레이어 색상 추출로 팀 식별
    #bbox = (xmin, ymin, xmax, ymax)
    def get_player_color(self, frame, bbox):
        
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]

        # cluster model
        kmeans = self.get_clustering_model(top_half_image)

        # cluster label
        labels = kmeans.labels_

        # 이미지 모양으로 reshape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # 선수 cluster
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1-non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color


    #선수 두 팀으로 나누고 팀 색상 할당
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detections in player_detections.items():
            bbox = player_detections['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
    #특정 선수 팀 반환
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id +=1

        if player_id == 91:
            team_id =1

        self.player_team_dict[player_id] = team_id

        return team_id
    

