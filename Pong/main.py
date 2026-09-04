import pygame
from pygame.locals import *
import math
import random
import time
import os
import pickle
import neat

#init pygame engine
pygame.init()
WIN_WIDTH = 1000
WIN_HEIGHT = 800
# display = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
# pygame.display.set_caption("Pong AI")
clock = pygame.time.Clock()

STAT_FONT = pygame.font.SysFont("timesnewroman", 50)

class Paddle():
    def __init__(self,right):
        self.dimensions = (20,100)
        self.right = right
        if right:
            self.position = (WIN_WIDTH-30,(WIN_HEIGHT/2-self.dimensions[1])/2)
        else:
            self.position = (10,(WIN_HEIGHT/2-self.dimensions[1])/2)
            
    def update(self,verticalMotion):
        if self.position[1] <= 0:
            if verticalMotion>0:
                self.position = (self.position[0],self.position[1]+verticalMotion)
            else:
                self.position = (self.position[0],0)
        elif self.position[1] >= WIN_HEIGHT-self.dimensions[1]:
            if verticalMotion<0:
                self.position = (self.position[0],self.position[1]+verticalMotion)
            else:
                self.position = (self.position[0],WIN_HEIGHT-self.dimensions[1])
        else:
            self.position = (self.position[0],self.position[1]+verticalMotion)
    def draw(self,display):
        pygame.draw.rect(display, (255,255,255), pygame.Rect(self.position[0],self.position[1],self.dimensions[0],self.dimensions[1]))

class Ball():
    def __init__(self):
        self.radius = 10
        self.position = (WIN_WIDTH/2,WIN_HEIGHT/2)
        self.velocity = (random.randrange(-1,2,2)*5,random.randrange(-1,2,2)*5)
        
    def update(self,paddleL,paddleR):
        
        #Account ball wall bounces
        if self.position[1] >= WIN_HEIGHT-self.radius:
            self.position = (self.position[0],WIN_HEIGHT-self.radius)
            self.velocity = (self.velocity[0],-self.velocity[1])
        elif self.position[1] <= self.radius:
            self.position = (self.position[0],self.radius)
            self.velocity = (self.velocity[0],-self.velocity[1])
            
        self.collide(paddleL)
        self.collide(paddleR)
        
        #Update position
        self.position = tuple(map(sum, zip(self.position,self.velocity)))
    
    def reset(self):
        self.position = (WIN_WIDTH/2,WIN_HEIGHT/2)
        self.velocity = (random.randrange(-1,2,2)*5,random.randrange(-1,2,2)*5)
    
    def collide(self,paddle):
        if (self.position[0]+self.velocity[0]>=paddle.position[0]-self.radius and paddle.right) or (self.position[0]+self.velocity[0]<=paddle.position[0]+paddle.dimensions[0]+self.radius and not paddle.right):
            if paddle.dimensions[1]+self.radius >= self.position[1]-paddle.position[1] and -self.radius <= self.position[1]-paddle.position[1]:
                angle = (self.position[1]-(paddle.position[1]+paddle.dimensions[1]/2))/math.sqrt(math.pow(abs(self.position[1]-(paddle.position[1]+paddle.dimensions[1]/2)),2)+math.pow(abs(self.position[0]-(paddle.position[0] if paddle.right else (paddle.position[0]+paddle.dimensions[0]))),2))
                self.velocity = (-(self.velocity[0]+(2*(1-(abs(angle)/2))*self.velocity[0]/abs(self.velocity[0]))),abs(self.velocity[0])*angle*0.5)
    def draw(self, display):
        pygame.draw.circle(display,(255,255,255), self.position, self.radius)

def draw_background(display,scores,gen,numAlive):
    display.fill((0,0,0))
    score_label = STAT_FONT.render(str(scores[0])+"    "+str(scores[1]),1,(255,255,255))
    display.blit(score_label, (WIN_WIDTH/2 - score_label.get_width()/2, 10))
    
    gen_label = STAT_FONT.render("Gen: "+str(gen),1,(255,255,255))
    display.blit(gen_label, (WIN_WIDTH/3 - gen_label.get_width()/2, 10))
    
    alive_label = STAT_FONT.render("Alive: "+str(numAlive),1,(255,255,255))
    display.blit(alive_label, (WIN_WIDTH*2/3 - alive_label.get_width()/2, 10))
    
    algorithm_label = STAT_FONT.render("Algorithm",1,(255,0,0))
    neural_label = STAT_FONT.render("Neural Net",1,(0,0,255))
    
    display.blit(neural_label, (10, 10))
    display.blit(algorithm_label, (WIN_WIDTH - neural_label.get_width() - 10, 10))
    for i in range(0, WIN_HEIGHT+20,20):
        pygame.draw.line(display, (255,255,255), (WIN_WIDTH/2,i),(WIN_WIDTH/2,i-10))
    
gens = 1
def eval_genomes(genomes, config):
    global gens
    run = True
    
    with open("moggerL.pkl","rb") as f:
        model = pickle.load(f)
        f.close()
    
    opponent = neat.nn.FeedForwardNetwork.create(model, config)
    
#     UPARROW = False
#     DOWNARROW = False
    
    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    paddlesRight = []
    paddlesLeft = []
    scores = []
    ge = []
    balls = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        paddlesRight.append(Paddle(1))
        paddlesLeft.append(Paddle(0))
        scores.append((0,0))
        ge.append(genome)
        balls.append(Ball())
    
    while run == True and len(balls)>0:
        for event in pygame.event.get():
            if event.type==QUIT:
                run = False
                pygame.quit()
                quit()
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_UP:
#                     UPARROW = True
#                 elif event.key == pygame.K_DOWN:
#                     DOWNARROW = True
#             if event.type == pygame.KEYUP:
#                 if event.key == pygame.K_UP:
#                     UPARROW = False
#                 elif event.key == pygame.K_DOWN:
#                     DOWNARROW = False
            
#         if UPARROW:
#             verticalMotion = -10
#         elif DOWNARROW:
#             verticalMotion = 10
#         else:
#             verticalMotion = 0
        max=0
        index = 0  
        clock.tick(1440)
        
#         draw_background(display,scores[0],gens,len(balls))
#         balls[index].draw(display)
#         paddlesLeft[index].draw(display)
#         paddlesRight[index].draw(display)
        
        for ball in balls:
            i = balls.index(ball)
            ball.update(paddlesLeft[i],paddlesRight[i])
            
            outputR = nets[i].activate((paddlesRight[i].position[1], abs(paddlesRight[i].position[0] - ball.position[0]), abs(paddlesRight[i].position[1] - ball.position[1]),ball.velocity[0],ball.velocity[1]))
            outputL = opponent.activate((paddlesLeft[i].position[1], abs(paddlesLeft[i].position[0] - ball.position[0]), abs(paddlesLeft[i].position[1] - ball.position[1]),ball.velocity[0],ball.velocity[1]))
            
            if outputL[0]>0:
                paddlesLeft[i].update(10)
            elif outputL[1]>0:
                paddlesLeft[i].update(-10)
            
            if outputR[0]>0:
                paddlesRight[i].update(10)
            elif outputR[1]>0:
                paddlesRight[i].update(-10)
            
            if ball.collide(paddlesLeft[i]):
                ge[i].fitness+=20
            if ball.position[0]<0:
                scores[i]=(scores[i][0],scores[i][1]+1)
                ge[i].fitness-=10
                ball.reset()
            elif ball.position[0]>WIN_WIDTH:
                scores[i]=(scores[i][0]+1,scores[i][1])
                ge[i].fitness+=10
                ball.reset()
                
            if scores[i][0]>4 or scores[i][1]>4:
                nets.pop(i)
                ge.pop(i)
                paddlesRight.pop(i)
                paddlesLeft.pop(i)
                scores.pop(i)
                balls.pop(i)
            
        
        for genome in ge:
            if genome.fitness>max:
                index = ge.index(genome)
            genome.fitness+=0.15
        
        

#         pygame.display.update()
    
    gens+=1
    
def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))
    
    
    winner = p.run(eval_genomes,100)
    
    

    with open("moggerR.pkl","wb") as f:
        pickle.dump(winner,f)
        f.close()

#     genomes = [(1,genome)]
    
#     eval_genomes(genomes, config)
    
    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    
    pygame.quit()
    quit()
    
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)