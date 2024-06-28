from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)
        return kmeans
    # k-means++ : 클러스터 중심점 초기화
    # n_init = 1 : 서로 다른 초기 중심점을 사용하여 몇번 실행할지 결정
    

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2), :]

        # clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # 라벨
        labels = kmeans.labels_

        # 라벨 -> 이미지로 배열 변경
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # 선수 클러스터
        corner_clusters = [clustered_image[0,0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key = corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color
    
    # 두 팀 색상 구하기    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            # bbox를 통해 선수별 색깔 추출
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # kmean 클러스팅으로 2개로 팀 구분
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1).fit(player_colors)

        # 팀 중심 색상을 저장함
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    
    # 선수의 팀 예측
    def get_player_team(self, frame, player_bbox, player_id):
        # 이미 할당된 팀 확인
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # 선수 색상 추출
        player_color = self.get_player_color(frame, player_bbox)

        # 팀 예측
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] # predict -> 2차원 배열이기 때문에 1차원 배열로 변환
        team_id += 1 # tema_id : 1부터 시작하게끔
        # 팀 할당 및 반환
        self.player_team_dict[player_id] = team_id

        return team_id
    




