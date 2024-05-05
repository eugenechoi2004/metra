from dm_control import suite
import numpy as np
import cv2

env = suite.load(domain_name='quadruped', task_name='walk')

camera_id = 0
width, height = 640, 480
video_path = 'quad_walk.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
fps = 30
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
timestep = env.reset()

while not timestep.last():  
    action_spec = env.action_spec()
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
    timestep = env.step(action)
    print(timestep)
    pixels = env.physics.render(camera_id=camera_id, width=width, height=height, overlays=())
    print(pixels)
    bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    video_writer.write(bgr_pixels)

video_writer.release()
print(f'Video saved as {video_path}')
