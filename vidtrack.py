import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import pygame
import pygame.gfxdraw
import random

pygame.init()

# between 0 (easier) and 1 (harder)
difficulty = 0.04
fps = 30
fpsClock = pygame.time.Clock()
# keep ratio 1:1
size = width, height = 780, 780
black = "#000000"
white = "#FFFFFF"
green = "#00FF00"
pink = "#FF00FF"
red = "#FF0000"

screen = pygame.display.set_mode(size)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

class Enemy:
    def __init__(self, start_pos, size, speed, num) -> None:
        self.pos = start_pos.copy()
        self.width, self.height = size.copy()
        self.speed = speed.copy()
        self.num = num

    def get_pos_scaled(self):
        return [[(1-self.pos[0])*width, self.pos[1]*height], [(1-self.pos[0])*width, self.pos[1]*height + self.height*height], [(1-self.pos[0])*width  + self.width*width, self.pos[1]*height  + self.height*height], [(1-self.pos[0])*width + self.width*width, self.pos[1]*height]]
        

    def move(self):
        self.pos[0] += self.speed[0]
        self.pos[1] += self.speed[1]

    def draw_enemy(self):
        if (self.check_in_bounds()):
            pygame.draw.rect(screen, pink, ((1-self.pos[0])*width, self.pos[1]*height, self.width*width, self.height*height))
            return 0
        else:
            return self.num


    def check_in_bounds(self):
        if (self.pos[0] - self.width <= 1 and self.pos[0] + self.width >= 0 and self.pos[1] - self.height <= 1 and self.pos[1] + self.height >= 0):
            
            return True
        return False
    

class Point:
    def __init__(self) -> None:
       self.point = {"x": 0, "y": 0}
    
    def get_point(self):
        return self.point
    
    def update_x(self, newx):
        self.point["x"] = newx
    
    def update_y(self, newy):
        self.point["y"] = newy

class HitBox:
    def __init__(self) -> None:
        '''
        [p3 p1------------  , where ------------ is the connector to start point.
        p4 p2]
        ''' 
        self.hitBox = {"p1": [0,0], "p2": [0,0], "p3": [0,0], "p4": [0,0]}
    
    def get_hit_box(self):
        return self.hitBox

    def get_hit_box_points(self):
        return [self.scale_point(self.hitBox["p2"]), self.scale_point(self.hitBox["p1"]), self.scale_point(self.hitBox["p3"]), self.scale_point(self.hitBox["p4"])]
    
    def scale_point(self, point):
        return [(1-point[0])*width, point[1]*height]
    
    def update_hit_box(self, start, end, thiccness):
        norm_update = np.linalg.norm([float(start[1] - end[1]), float(end[0] - start[0])])*thiccness
        updateVertex = [float(start[1] - end[1])/norm_update, float(end[0] - start[0])/norm_update]
        self.hitBox["p1"] = [float(start[0] + updateVertex[0]), float(start[1]+updateVertex[1])]
        self.hitBox["p2"] = [float(start[0] - updateVertex[0]), float(start[1]-updateVertex[1])]
        self.hitBox["p3"] = [float(end[0] + updateVertex[0]), float(end[1]+updateVertex[1])]
        self.hitBox["p4"] = [float(end[0] - updateVertex[0]), float(end[1]-updateVertex[1])]


class Stick:
    def __init__(self) -> None:
        self.points = {"NOSE": Point(),
                "LEFT_SHOULDER": Point(),
                "RIGHT_SHOULDER": Point(),
                "LEFT_ELBOW": Point(),
                "RIGHT_ELBOW": Point(),
                "LEFT_WRIST": Point(),
                "RIGHT_WRIST": Point(),
                "LEFT_HIP": Point(),
                "RIGHT_HIP": Point(),
                "LEFT_KNEE": Point(),
                "RIGHT_KNEE": Point(),
                "LEFT_ANKLE": Point(),
                "RIGHT_ANKLE": Point()}
        self.hitBoxes = {"FACE": HitBox(),
                "LEFT_UARM": HitBox(),
                "RIGHT_UARM": HitBox(),
                "LEFT_LARM": HitBox(),
                "RIGHT_LARM": HitBox(),
                "LEFT_THIGH": HitBox(),
                "RIGHT_THIGH": HitBox(),
                "LEFT_LEG": HitBox(),
                "RIGHT_LEG": HitBox(),
                "TORSO": HitBox()}
        self.nose = 0
        self.left_shoulder = 0
        self.right_shoulder = 0
        self.left_elbow = 0
        self.right_elbow = 0
        self.left_wrist = 0
        self.right_wrist = 0
        self.left_hip = 0
        self.right_hip = 0
        self.left_knee = 0
        self.right_knee = 0
        self.left_ankle = 0
        self.right_ankle = 0
        self.neckx = 0
        self.necky = 0
        self.bootyx = 0
        self.bootyy = 0
        self.head_radius = height/10
        self.thicc = 20
    
    def get_points(self):
        return self.points.values()
    
    def get_hit_boxes(self):
        return self.hitBoxes.values()

    def update_points(self, results):
        for part, point in self.points.items():
            try:
                point.update_x(results.pose_landmarks.landmark[mp_pose.PoseLandmark[part]].x)
                point.update_y(results.pose_landmarks.landmark[mp_pose.PoseLandmark[part]].y)
            except:
                continue
        self.nose = self.points["NOSE"].get_point()
        self.left_shoulder = self.points["LEFT_SHOULDER"].get_point()
        self.right_shoulder = self.points["RIGHT_SHOULDER"].get_point()
        self.left_elbow = self.points["LEFT_ELBOW"].get_point()
        self.right_elbow = self.points["RIGHT_ELBOW"].get_point()
        self.left_wrist = self.points["LEFT_WRIST"].get_point()
        self.right_wrist = self.points["RIGHT_WRIST"].get_point()
        self.left_hip = self.points["LEFT_HIP"].get_point()
        self.right_hip = self.points["RIGHT_HIP"].get_point()
        self.left_knee = self.points["LEFT_KNEE"].get_point()
        self.right_knee = self.points["RIGHT_KNEE"].get_point()
        self.left_ankle = self.points["LEFT_ANKLE"].get_point()
        self.right_ankle = self.points["RIGHT_ANKLE"].get_point()
        self.neckx = 1 - (self.left_shoulder["x"] + self.right_shoulder["x"])/2
        self.necky = (self.left_shoulder["y"] + self.right_shoulder["y"])/2
        self.bootyx = 1 - (self.left_hip["x"] + self.right_hip["x"])/2
        self.bootyy = (self.left_hip["y"] + self.right_hip["y"])/2

        self.hitBoxes["FACE"].update_hit_box([self.nose["x"], self.nose["y"] - self.head_radius/height], [self.nose["x"], self.nose["y"] + self.head_radius/height], height/self.head_radius)
        self.hitBoxes["LEFT_UARM"].update_hit_box([1-self.neckx, self.necky], [self.left_elbow["x"], self.left_elbow["y"]], self.thicc*2)
        self.hitBoxes["RIGHT_UARM"].update_hit_box([1-self.neckx, self.necky], [self.right_elbow["x"], self.right_elbow["y"]], self.thicc*2)
        self.hitBoxes["LEFT_LARM"].update_hit_box([self.left_elbow["x"], self.left_elbow["y"]], [self.left_wrist["x"], self.left_wrist["y"]], self.thicc*2)
        self.hitBoxes["RIGHT_LARM"].update_hit_box([self.right_elbow["x"], self.right_elbow["y"]], [self.right_wrist["x"], self.right_wrist["y"]], self.thicc*2)
        self.hitBoxes["LEFT_THIGH"].update_hit_box([1-self.bootyx, self.bootyy], [self.left_knee["x"], self.left_knee["y"]], self.thicc*2)
        self.hitBoxes["RIGHT_THIGH"].update_hit_box([1-self.bootyx, self.bootyy], [self.right_knee["x"], self.right_knee["y"]], self.thicc*2)
        self.hitBoxes["LEFT_LEG"].update_hit_box([self.left_knee["x"], self.left_knee["y"]], [self.left_ankle["x"], self.left_ankle["y"]], self.thicc*2)
        self.hitBoxes["RIGHT_LEG"].update_hit_box([self.right_knee["x"], self.right_knee["y"]], [self.right_ankle["x"], self.right_ankle["y"]], self.thicc*2)
        self.hitBoxes["TORSO"].update_hit_box([1-self.neckx, self.necky], [1-self.bootyx, self.bootyy], self.thicc*2)

    def draw_parts(self):
        if (self.check_in_bounds(self.nose)):
            pygame.draw.circle(screen, white, (int((1 - self.nose["x"])*width), int(self.nose["y"]*height)), self.head_radius)
        
        if (self.check_in_bounds(self.left_shoulder) and self.check_in_bounds(self.right_shoulder)):
            if (self.check_in_bounds(self.left_elbow)):
                pygame.draw.line(screen, white, [self.neckx*width, self.necky*height], [int((1 - self.left_elbow["x"])*width), int(self.left_elbow["y"]*height)], self.thicc)
                if (self.check_in_bounds(self.left_wrist)):
                    pygame.draw.line(screen, white, [int((1 - self.left_elbow["x"])*width), int(self.left_elbow["y"]*height)], [int((1 - self.left_wrist["x"])*width), int(self.left_wrist["y"]*height)], self.thicc)
            if (self.check_in_bounds(self.right_elbow)):
                pygame.draw.line(screen, white, [self.neckx*width, self.necky*height], [int((1 - self.right_elbow["x"])*width), int(self.right_elbow["y"]*height)], self.thicc)
                if (self.check_in_bounds(self.right_wrist)):
                    pygame.draw.line(screen, white, [int((1 - self.right_elbow["x"])*width), int(self.right_elbow["y"]*height)], [int((1 - self.right_wrist["x"])*width), int(self.right_wrist["y"]*height)], self.thicc)
        if (self.check_in_bounds(self.left_hip) and self.check_in_bounds(self.right_hip)):
            if (self.check_in_bounds(self.left_knee)):
                pygame.draw.line(screen, white, [self.bootyx*width, self.bootyy*height], [int((1 - self.left_knee["x"])*width), int(self.left_knee["y"]*height)], self.thicc)
                if (self.check_in_bounds(self.left_ankle)):
                    pygame.draw.line(screen, white, [int((1 - self.left_knee["x"])*width), int(self.left_knee["y"]*height)], [int((1 - self.left_ankle["x"])*width), int(self.left_ankle["y"]*height)], self.thicc)
            if (self.check_in_bounds(self.right_knee)):
                pygame.draw.line(screen, white, [self.bootyx*width, self.bootyy*height], [int((1 - self.right_knee["x"])*width), int(self.right_knee["y"]*height)], self.thicc)
                if (self.check_in_bounds(self.right_ankle)):
                    pygame.draw.line(screen, white, [int((1 - self.right_knee["x"])*width), int(self.right_knee["y"]*height)], [int((1 - self.right_ankle["x"])*width), int(self.right_ankle["y"]*height)], self.thicc)

        if (self.check_in_bounds(self.left_shoulder) and self.check_in_bounds(self.right_shoulder) and self.check_in_bounds(self.left_hip) and self.check_in_bounds(self.right_hip)):
            pygame.draw.line(screen, white, [self.neckx*width, self.necky*height], [self.bootyx*width, self.bootyy*height], self.thicc)

    def draw_collision(self):
        face = self.hitBoxes["FACE"].get_hit_box_points()
        left_uarm = self.hitBoxes["LEFT_UARM"].get_hit_box_points()
        right_uarm = self.hitBoxes["RIGHT_UARM"].get_hit_box_points()
        left_larm = self.hitBoxes["LEFT_LARM"].get_hit_box_points()
        right_larm = self.hitBoxes["RIGHT_LARM"].get_hit_box_points()
        left_thigh = self.hitBoxes["LEFT_THIGH"].get_hit_box_points()
        right_thigh = self.hitBoxes["RIGHT_THIGH"].get_hit_box_points()
        left_leg = self.hitBoxes["LEFT_LEG"].get_hit_box_points()
        right_leg = self.hitBoxes["RIGHT_LEG"].get_hit_box_points()
        torso = self.hitBoxes["TORSO"].get_hit_box_points()

        if (self.check_in_bounds(self.nose)):
            pygame.draw.polygon(screen, green, face, 5)
        
        if (self.check_in_bounds(self.left_shoulder) and self.check_in_bounds(self.right_shoulder)):
            if (self.check_in_bounds(self.left_elbow)):
                pygame.draw.polygon(screen, green, left_uarm, 5)
                if (self.check_in_bounds(self.left_wrist)):
                    pygame.draw.polygon(screen, green, left_larm, 5)
            if (self.check_in_bounds(self.right_elbow)):
                pygame.draw.polygon(screen, green, right_uarm, 5)
                if (self.check_in_bounds(self.right_wrist)):
                    pygame.draw.polygon(screen, green, right_larm, 5)
        if (self.check_in_bounds(self.left_hip) and self.check_in_bounds(self.right_hip)):
            if (self.check_in_bounds(self.left_knee)):
                pygame.draw.polygon(screen, green, left_thigh, 5)
                if (self.check_in_bounds(self.left_ankle)):
                    pygame.draw.polygon(screen, green, left_leg, 5)
            if (self.check_in_bounds(self.right_knee)):
                pygame.draw.polygon(screen, green, right_thigh, 5)
                if (self.check_in_bounds(self.right_ankle)):
                    pygame.draw.polygon(screen, green, right_leg, 5)

        if (self.check_in_bounds(self.left_shoulder) and self.check_in_bounds(self.right_shoulder) and self.check_in_bounds(self.left_hip) and self.check_in_bounds(self.right_hip)):
            pygame.draw.polygon(screen, green, torso, 5)

    def check_in_bounds(self, part):
        if (part["x"] <= 1 and part["x"] >=0 and part["y"] <= 1 and part["y"] >=0):
            return True
        return False

class Game:
    def __init__(self, player) -> None:
        self.starting_box = [0.4, 0.25, 0.3, 0.7]
        self.player = player
        self.enemies = {}

    def get_player(self):
        return self.player

    def get_enemies(self):
        return self.enemies.values()

    def add_enemy(self, enemy, num_enemies):
        self.enemies[num_enemies] = enemy

    def rem_enemy(self, num):
        self.enemies.pop(num)

    def clear_enemies(self):
        self.enemies = {}

    def start_game(self):
        if (self.start_collision()):
            print("game started")
            return True
        else:
            self.draw_starting_box()
            return False

    def start_collision(self):
        stick_points = self.player.get_points()
        can_start = True
        for point in stick_points:
            p = point.get_point()
            if not (p["x"] <= self.starting_box[0] + self.starting_box[2] and p["x"] >= self.starting_box[0] and p["y"] <= self.starting_box[1] + self.starting_box[3] and p["y"] >= self.starting_box[1]):
                can_start = False
                break
            
        return can_start

    def player_enemy_collision(self):
        stick_hit_boxes = self.player.get_hit_boxes()
        is_colliding = False
        for hit_box in stick_hit_boxes:
            box = hit_box.get_hit_box_points()
            for enemy in self.enemies.values():
                is_colliding = True
                e = enemy.get_pos_scaled()
                axiss = [[e[3][0] - e[0][0], e[3][1] - e[0][1]], [e[3][0] - e[2][0], e[3][1] - e[2][1]], [e[0][0] - e[1][0], e[0][1] - e[1][1]], [e[0][0] - e[3][0], e[0][1] - e[3][1]]]
    
                for axis in axiss:
                    minbox = 1000000
                    maxbox = -1000000
                    for point in box:
                        proj = (point[0]*axis[0]+point[1]*axis[1])/(axis[0]**2+axis[1]**2)
                        projx = proj*axis[0]
                        projy = proj*axis[1]
                        contendor = projx*axis[0]+projy*axis[1]
                        if (contendor > maxbox):
                            maxbox = contendor
                        elif (contendor < minbox):
                            minbox = contendor
                    mine = 1000000
                    maxe = -1000000
                    for point in e:
                        proj = (point[0]*axis[0]+point[1]*axis[1])/(axis[0]**2+axis[1]**2)
                        projx = proj*axis[0]
                        projy = proj*axis[1]
                        contendor = projx*axis[0]+projy*axis[1]
                        if (contendor > maxe):
                            maxe = contendor
                        elif (contendor < mine):
                            mine = contendor
                    if not (minbox <= maxe and maxbox >= mine):
                        is_colliding = False
                        break

                if is_colliding:
                    return is_colliding

            
        return is_colliding

    def draw_starting_box(self):
        pygame.draw.rect(screen, red, (self.starting_box[0]*width, self.starting_box[1]*height, self.starting_box[2]*width, self.starting_box[3]*height))

game_on = False
game = Game(Stick())
enemy_configs = [[[0, 0.8], [0.1, 0.1], [0.02, 0]], [[1.1, 0.8], [0.1, 0.1], [-0.01, 0]], [[0, 0.3], [0.1, 0.1], [0.01, 0]], [[1.1, 0.3], [0.1, 0.1], [-0.01, 0]]]
num_enemies = 1
mincooldown = 3
maxcooldown = 5
# frame counter
fcCoolDown = 0

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    screen.fill(black)
    if not game_on:
        game_on = game.start_game()
        
    if game_on:
        num = random.random()
        if (num < difficulty and fcCoolDown/fps >= mincooldown):
            enemy_config = random.choice(enemy_configs).copy()
            game.add_enemy(Enemy(enemy_config[0], enemy_config[1], enemy_config[2], num_enemies), num_enemies)
            num_enemies += 1
            fcCoolDown = 0
        else:
            fcCoolDown += 1
        
        if fcCoolDown/fps >= maxcooldown:
            enemy_config = random.choice(enemy_configs).copy()
            game.add_enemy(Enemy(enemy_config[0], enemy_config[1], enemy_config[2], num_enemies), num_enemies)
            num_enemies += 1
            fcCoolDown = 0

    game.get_player().update_points(results)
    to_remove = []
    for enemy in game.get_enemies():
        enemy.move()
        dead = enemy.draw_enemy()
        if dead:
            to_remove.append(dead)
    for rem in to_remove:
        game.rem_enemy(rem)
    game.get_player().draw_parts()
    game.get_player().draw_collision()
    if (game.player_enemy_collision()):
        print("game over")
        game.clear_enemies()
        num_enemies = 0
        game_on = False
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    
    pygame.display.flip()
    fpsClock.tick(fps)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()