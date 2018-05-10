import imutil
from atari_dataloader import AtariDataloader

loader1 = AtariDataloader(name='Pong-v0', batch_size=1)
loader2 = AtariDataloader(name='Pong-v0', batch_size=4)
loader3 = AtariDataloader(name='Pong-v0', batch_size=9)
loader4 = AtariDataloader(name='Pong-v0', batch_size=16)
vid = imutil.VideoMaker('pongtest')

for _ in range(400):
  x, y = next(loader1)
  vid.write_frame(x, caption=str(y[0]))

for _ in range(300):
  x, y = next(loader2)
  vid.write_frame(x, caption=str(y[:2]))

for _ in range(200):
  x, y = next(loader3)
  vid.write_frame(x, caption=str(y[:4]))

for _ in range(100):
  x, y = next(loader4)
  vid.write_frame(x, caption=str(y[:8]))

vid.finish()

