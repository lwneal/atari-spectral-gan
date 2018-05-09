import imutil
from atari_dataloader import AtariDataloader

loader = AtariDataloader(name='Pong-v0', batch_size=1)
vid = imutil.VideoMaker('pongtest')
for _ in range(1000):
  x, y = next(loader)
  vid.write_frame(x, caption=str(y))
vid.finish()

