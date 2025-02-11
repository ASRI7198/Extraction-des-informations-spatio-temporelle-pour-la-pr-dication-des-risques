import os
from Carla_session import Carla_session

image_save_path ='_data'
if not os.path.exists(os.path.join(image_save_path)):
    os.makedirs(os.path.join(image_save_path))

c = Carla_session()
c.drive_around(10)