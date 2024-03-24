import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from tqdm import tqdm

from PIL import Image

import os
import pandas as pd

from multiprocessing import Process
from threading import Thread

class BoidSimulation:

    def __init__(self, space_length = 100):

        self.positions:np.ndarray = None
        self.velocities:np.ndarray = None
        self.obstacles:np.ndarray = None
        self.last_forces:np.ndarray = None

        self.space_length = space_length


    def placeObstacles(self, N_obstacles, min_radius=1, max_radius=10, clear=True):

        if N_obstacles == 0:
            self.obstacles = None
            return
        
        new_obstacles = np.random.uniform(0, self.space_length, (N_obstacles, 3))
        new_obstacles[:, 2] = np.random.uniform(min_radius, max_radius, N_obstacles)

        if not clear and self.obstacles is not None:
            new_obstacles = np.vstack([self.obstacles, new_obstacles])

        self.obstacles = new_obstacles


    def placeBoids(self, N_boids, avg_speed):
        '''
        Ignore if a boid is placed on an obstacle. The forces will take care of it.
        
        '''
        assert(N_boids > 0)

        self.positions = np.random.uniform(0, self.space_length, (N_boids, 2))
        self.velocities = np.random.randn(N_boids, 2) * np.sqrt(avg_speed)

        self.last_forces = np.zeros_like(self.velocities)

    def calculateForces(self, separation, alignment, coherence, avoidance, visual_range, avoid_range, obstacles_enabled):
        
        # distances
        directions = self.positions - self.positions[:, None]
        distances = np.linalg.norm(directions, axis=2)

        distances[distances == 0] = np.inf

        directions = directions / distances[:,:,None]

        # apply visual distance and exclude 
        mask = distances < visual_range
        
        # separation, personal space to other boids
        closest_index = np.argmin(distances, axis=1)

        force_separation = - separation * directions[np.arange(directions.shape[0]),closest_index] * mask[np.arange(mask.shape[0]),closest_index][:,None]
        
        """
        #Evaluation purpose
        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('off')
        ax.set_xlim([0,100])
        ax.set_ylim([0,100])
        ax.quiver(self.positions[:,0], self.positions[:,1], force_separation[:,0], force_separation[:,1])
        plt.show()
        """

        num_close = np.count_nonzero(mask, axis=1)
        num_close_non_zero = num_close
        num_close_non_zero[num_close == 0] = 1
        
        # alignment, move in the same direction as the average in the visual range
        aligment_vector = (mask[:, :, None] * self.velocities).sum(axis=1) / num_close_non_zero[:, None]
        aligment_lengths = np.linalg.norm(aligment_vector, axis=1)
        aligment_lengths[aligment_lengths == 0] = 1
        force_alignment = alignment * aligment_vector / aligment_lengths[:,None]
        force_alignment[num_close == 0] = 0.0
        
        """
        #Evaluation purpose
        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('off')
        ax.set_xlim([0,100])
        ax.set_ylim([0,100])
        ax.quiver(self.positions[:,0], self.positions[:,1], force_alignment[:,0], force_alignment[:,1])
        plt.show()
        """
        
        # coherence, move towards the group center
        coherence_vector = (mask[:, :, None] * self.positions).sum(axis=1) / num_close_non_zero[:, None] - self.positions
        
        force_coherence = coherence * coherence_vector / np.linalg.norm(coherence_vector, axis=1)[:,None]
        force_coherence[num_close == 0] = 0.0

        """
        #Evaluation purpose
        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('off')
        ax.set_xlim([0,100])
        ax.set_ylim([0,100])
        ax.quiver(self.positions[:,0], self.positions[:,1], force_coherence[:,0], force_coherence[:,1])
        plt.show()
        """

        force_avoidance = 0

        # avoidance, move away from obstacles
        if obstacles_enabled:
            # avoidance, move away from obstacles
            obstacle_directions = self.obstacles[:, :2] - self.positions[:, None]
            obstacle_distances = np.linalg.norm(obstacle_directions, axis=2)

            obstacle_directions = obstacle_directions / obstacle_distances[:, :, None]

            obstacle_distances -= (self.obstacles[:, 2] + avoid_range)
            obstacle_distances[obstacle_distances < 0] = 0

            closest_obstacle_index = np.argmin(obstacle_distances, axis=1)
            force_avoidance = - avoidance * (obstacle_directions[np.arange(obstacle_directions.shape[0]), closest_obstacle_index]) * (np.min(obstacle_distances, axis=1) < avoid_range)[:, None]
        

        # avoid the edge
        # check if too close to an edge -> mask
        edge_mask = np.any(np.abs(self.positions - self.space_length/2.0) >= (self.space_length/2.0 - avoid_range), axis=1)

        # calculate the center direction
        center = np.array([self.space_length/2, self.space_length/2])
        center_directions = center - self.positions
        center_directions /= np.linalg.norm(center_directions, axis=1)[:,None]

        # weight force

        force_avoidance += avoidance * center_directions * edge_mask[:,None]

        """
        #Evaluation purpose
        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('off')
        ax.set_xlim([0,100])
        ax.set_ylim([0,100])
        ax.quiver(self.positions[:,0], self.positions[:,1], force_avoidance[:,0], force_avoidance[:,1])
        plt.show()
        """
                    
        self.last_forces =  force_coherence + force_avoidance + force_alignment + force_separation

        return self.last_forces


    def update(self, dt):
        self.positions += self.velocities *dt + 0.5*self.last_forces*dt*dt
                        
        self.velocities += self.last_forces*dt

        max_speed = self.space_length / 10

        speed_limit_factors = np.linalg.norm(self.velocities, axis=1) / max_speed
        speed_limit_factors[speed_limit_factors < 1] = 1
        self.velocities = self.velocities / speed_limit_factors[:, None]

        self.velocities = np.clip(self.velocities, -self.space_length / 10, self.space_length / 10)


    def simulate(self, separation=1.0, alignment=1.0, coherence=1.0, avoidance=1.0, dt=0.1, num_time_steps=100, visual_range=100, avoid_range=100, animate = False):
        obstacles_enabled = self.obstacles is not None

        boid_positions_per_time_step = np.zeros((num_time_steps, self.positions.shape[0], 2))
        boid_velocities_per_time_step = np.zeros((num_time_steps, self.positions.shape[0], 2))

        for i in tqdm(range(num_time_steps)):
            boid_positions_per_time_step[i] = self.positions
            boid_velocities_per_time_step[i] = self.velocities


            self.calculateForces(separation=separation, 
                                 alignment=alignment, 
                                 coherence=coherence,
                                 avoidance=avoidance,
                                 visual_range=visual_range, 
                                 avoid_range=avoid_range,
                                 obstacles_enabled=obstacles_enabled)
            
            self.update(dt=dt)

        if animate:
            fig, ax = plt.subplots(figsize=(10,10))
            if obstacles_enabled:
                for obstacle in self.obstacles:
                    circle = plt.Circle((obstacle[0], obstacle[1]), radius=obstacle[2])
                    ax.add_patch(circle)

            velocities_magnitudes = np.linalg.norm(boid_velocities_per_time_step[0], axis=1)
            velocities_normalized = boid_velocities_per_time_step[0] / np.reshape(velocities_magnitudes, (-1,1))
            
            scat = ax.quiver(boid_positions_per_time_step[0][:,0], 
                            boid_positions_per_time_step[0][:,1],
                            velocities_normalized[:,0],
                            velocities_normalized[:,1], scale=10, scale_units='inches')
            
            ax.set_xlim([0,self.space_length])
            ax.set_ylim([0,self.space_length])

            def update(frame):
                scat.set_offsets(boid_positions_per_time_step[frame])

                velocities_magnitudes = np.linalg.norm(boid_velocities_per_time_step[frame], axis=1)
                velocities_normalized = boid_velocities_per_time_step[frame]/ np.reshape(velocities_magnitudes, (-1,1))
                scat.set_UVC(velocities_normalized[:,0], 
                            velocities_normalized[:,1])

                return scat,

            ani = FuncAnimation(fig, update, frames=num_time_steps, blit=True)
            print("Animation finished. Video processing ...")
            display(HTML(ani.to_jshtml()))


    def finalStateImage(self, filename=None):
        pxls = np.floor(self.positions).astype(int)
        speed = np.linalg.norm(self.velocities, axis=1)
        tray = np.floor((pxls + 0.5) - self.velocities / speed[:, None]).astype(int)

        #plt.scatter(boid_simulation.positions[:, 0], boid_simulation.positions[:, 1])
        #plt.show()

        board = np.zeros((self.space_length, self.space_length, 3), dtype=np.uint8)

        # TODO make tray color depend on speed
        pxls = pxls[np.logical_and(np.all(pxls >= 0, axis=1), np.all(pxls < self.space_length, axis=1))]
        tray = tray[np.logical_and(np.all(tray >= 0, axis=1), np.all(tray < self.space_length, axis=1))]
        board[pxls[:, 0], pxls[:, 1], 0] = 255
        board[tray[:, 0], tray[:, 1], 1] = 255


        # TODO maybe do it parallel
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                radius = obstacle[2].astype(int)
                pos = obstacle[:2].astype(int)

                x_range = np.linspace(pos[0]-radius, pos[0]+radius, 2*radius+1, dtype=int)
                x_range = x_range[x_range >= 0]
                x_range = x_range[x_range < self.space_length]
                y_range = np.linspace(pos[1]-radius, pos[1]+radius, 2*radius+1, dtype=int)
                y_range = y_range[y_range >= 0]
                y_range = y_range[y_range < self.space_length]

                x, y = np.meshgrid(x_range, y_range)

                
                mask = np.linalg.norm(np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)]) - pos, axis=1) < radius
                mask = mask.reshape(x.shape)

                board[x[mask], y[mask], 2] = 255
 
        the_image = Image.fromarray(board)
        
        if filename is not None:
            the_image.save(filename)

        return the_image


def runSimulationStoreImage(space_size, boids, obst, sep, coh, ali, avo, vis_rang, av_rang, foldername, i):
    boid_simulation = BoidSimulation(space_size)
    boid_simulation.placeBoids(boids, 5)
    boid_simulation.placeObstacles(obst, 5, 10)
    boid_simulation.simulate(separation=sep,
                            coherence=coh,
                            alignment=ali,
                            avoidance=avo,
                            num_time_steps=500,
                            visual_range=vis_rang,
                            avoid_range=av_rang,
                            animate=False,
                            dt=0.1)

    boid_simulation.finalStateImage(f'{foldername}/images/img{i}.png')

def createDataset(foldername, N_simulations, space_size=64):
    
    image_folder = os.path.join(foldername, 'images')
    start_index = 0

    # TODO add a functionality to use the csv as input of the existing folder
    if os.path.exists(foldername):
        df = pd.read_csv(f'{foldername}/params.csv')
        N_boids, N_obstacles, L_separation, L_coherence, L_aligment, L_avoidance, visual_range, avoid_range = df.values.T

        N_boids = N_boids.astype(int)
        N_obstacles = N_obstacles.astype(int)

        start_index = len(os.listdir(image_folder))
    else:
        os.mkdir(foldername)
        os.mkdir(image_folder)

        N_boids = np.random.uniform(50, 500, N_simulations).astype(int)
        N_obstacles = np.random.uniform(0, 5, N_simulations).astype(int)

        L_separation = np.random.uniform(1, 4, N_simulations)
        L_coherence = np.random.uniform(0, 4, N_simulations)
        L_aligment = np.random.uniform(0, 8, N_simulations)

        L_avoidance = np.random.uniform(7, 25, N_simulations)

        visual_range = np.random.uniform(0, space_size, N_simulations)
        avoid_range = np.random.uniform(0, space_size/8, N_simulations)

        # save the csv
        df = pd.DataFrame({
            "N_boids": N_boids, 
            "N_obstacles": N_obstacles, 
            "L_separation": L_separation, 
            "L_coherence": L_coherence, 
            "L_aligment": L_aligment, 
            "L_avoidance": L_avoidance, 
            "visual_range": visual_range, 
            "avoid_range": avoid_range
        })

        df.to_csv(f'{foldername}/params.csv', index=False)


    processes: list[Thread] = []
    max_processes = 100
    for i, (boids, obst, sep, coh, ali, avo, vis_rang, av_rang) in enumerate(zip(N_boids[start_index:], N_obstacles[start_index:], L_separation[start_index:], L_coherence[start_index:], L_aligment[start_index:], L_avoidance[start_index:], visual_range[start_index:], avoid_range[start_index:]), start_index):
        
        if len(processes) == max_processes:
            processes.clear()

        proc = Thread(target=runSimulationStoreImage, args=(space_size, boids, obst, sep, coh, ali, avo, vis_rang, av_rang, foldername, i))
        proc.start()
        processes.append(proc)

        if len(processes) == max_processes:
            for proc in processes:
                proc.join()

import sys
if __name__ == '__main__':

    if len(sys.argv) > 2:
        folder = sys.argv[1]
        N_simulations = int(sys.argv[2])

        createDataset(folder, N_simulations)

    else:
        print("Please enter a foldername and the required dataset size.")